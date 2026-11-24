#!/usr/bin/env python3
"""Multi-answer probabilistic CoT runner for BBH-MCQ-Hard.

Mirrors the binary StrategyQA flow but uses the new MCQ helpers in
``cot_prog/prompt/multi_answer_cot.py``:

    generate_combined_cot_mcq / generate_per_answer_cot_mcq
    write_mcq_program(... answer_aggregation_fn, statement_aggregation_fn, ...)

Each example is processed end-to-end:

    1. Generate per-letter CoT (combined or separate strategy).
    2. Assemble an N-way Categorical/Mixture probabilistic program.
    3. Compile + run inference; pick the candidate with the highest posterior.
    4. Checkpoint the per-example result.

Input format (matches experiments/data_prep/prep_bbh.py):

    [
      {
        "qid": "...",
        "question": "...",            # stem only, options NOT inline
        "choices": ["...", "..."],
        "choice_labels": ["A","B","C","D","..."],
        "answerKey": "A",
        "answer_text": "..."
      },
      ...
    ]

Usage:

    python experiments/runners/run_BBH_multi_cot.py \
      --model_name meta-llama/Llama-3.3-70B-Instruct-Turbo \
      --dataset_path bbh_data/temporal_sequences/multi_cot_test.json \
      --cot_strategy combined --pool_mode logitpool \
      --answer_aggregation_fn c --statement_aggregation_fn mean \
      --output_dir results/bbh_temporal_sequences_combined_test
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
import sys
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional

from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cot_prog import axes
from cot_prog.lang import ParseStringCot
from cot_prog.prompt.cot_text_parser import Step, CoTOutput
from cot_prog.prompt.multi_answer_cot import (
    run_mcq_external,
    write_mcq_program,
    format_choices_block,
    teacher_forcing_logprobs_parallel,
    _get_model,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_bbh(path: str) -> List[dict]:
    """Load BBH-style data (see experiments/data_prep/prep_bbh.py output)."""
    with open(path) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            data = [json.loads(line) for line in f if line.strip()]
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    out = []
    for i, ex in enumerate(data):
        out.append({
            "qid":           str(ex.get("qid", f"ex-{i:04d}")),
            "question":      ex.get("question", ""),
            "choices":       ex.get("choices") or [],
            "choice_labels": ex.get("choice_labels") or [],
            "answer_key":    str(ex.get("answerKey", ex.get("answer_key", ""))).strip(),
            "answer_text":   ex.get("answer_text", ""),
        })
    print(f"Loaded {len(out)} examples from {path}")
    return out


def sample_dataset(data: List[dict], n: Optional[int], seed: int) -> List[dict]:
    if not n or n >= len(data):
        return data
    return random.Random(seed).sample(data, n)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def _serialize_cot(cot) -> dict:
    out = {
        "raw_cot": getattr(cot, "raw_cot", ""),
        "steps": [s.to_dict() for s in cot.lines],
        "attempts": getattr(cot, "attempts", 1),
        "errors": cot.errors or [],
    }
    raw_combined = getattr(cot, "_raw_combined", None)
    if raw_combined:
        out["raw_combined"] = raw_combined
    init_score = getattr(cot, "_init_score", None)
    if init_score is not None:
        out["init_score"] = init_score
    # NOTE: tf_priors / tf_meta / tf_raw are NOT written per-letter — TF is
    # a single shared LLM call per example, so we save them once at the
    # top level of the example dict (process_example does that). Restore
    # paths fall back to top-level via reprocess_from_checkpoint.
    # Unified per-letter "all output tokens" logprobs:
    #   - Combined: header + body (the full '## For X (initial confidence:
    #     Y%): ...' section), so init-line tokens are part of cot_logprobs
    #     rather than a separate field.
    #   - Separate: the full response (init line + steps), so init-line
    #     tokens are at the start of cot_logprobs.
    # The parsed init confidence value is still on cot._init_score.
    cot_logprobs = getattr(cot, "_cot_logprobs", None)
    if cot_logprobs is not None:
        out["cot_logprobs"] = cot_logprobs
    # NEW: the {system, user, response} payload of the LLM call that
    # produced this CoT. For separate strategy this is distinct per letter
    # (one call per candidate); for combined-style it's the single shared
    # call that produced all N sections (same payload on every per-letter
    # cot — small duplication for clarity).
    cot_prompt = getattr(cot, "_cot_prompt", None)
    if cot_prompt is not None:
        out["cot_prompt"] = cot_prompt
    return out


def _restore_cot(serialized: dict, query: str) -> CoTOutput:
    """Rebuild a CoTOutput (with weights + auxiliary attrs) from a checkpoint."""
    steps = [Step(**s) for s in serialized.get("steps", [])]
    cot = CoTOutput(
        query=query,
        raw_cot=serialized.get("raw_cot", ""),
        lines=steps,
        attempts=int(serialized.get("attempts", 1)),
        errors=serialized.get("errors", []) or [],
    )
    if "raw_combined" in serialized:
        cot._raw_combined = serialized["raw_combined"]
    if "init_score" in serialized:
        cot._init_score = float(serialized["init_score"])
    if "tf_priors" in serialized:
        cot._tf_priors = serialized["tf_priors"]
    if "tf_meta" in serialized:
        cot._tf_meta = serialized["tf_meta"]
    if "tf_raw" in serialized:
        cot._tf_raw = serialized["tf_raw"]
        # If we have legacy tf_raw but no tf_meta, synthesize a partial
        # tf_meta so downstream consumers can use a uniform interface.
        if "tf_meta" not in serialized:
            cot._tf_meta = {
                "prompt":   None,
                "response": serialized["tf_raw"],
                "logprobs": None,
            }
    if "cot_logprobs" in serialized:
        cot._cot_logprobs = serialized["cot_logprobs"]
    if "init_score_logprobs" in serialized:
        cot._init_score_logprobs = serialized["init_score_logprobs"]
    if "cot_prompt" in serialized:
        cot._cot_prompt = serialized["cot_prompt"]
    return cot


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _run_execution_modes(
    prog, choice_labels, choices, *, sample_tries: int = 5000,
) -> Dict[str, Any]:
    """Run all 5 execution modes on a CotProg and return serialisable results.

    Mirrors the old ``cot_prog-main/run_exp/run_strategyqa_multi_cot.py``
    helper so the per-example checkpoint contains everything needed to
    render the rich HTML sections (Question/Predictions/Sample/Semantics/
    Circuit/Inference/Gradient/...).

    Returns a dict with keys:
        sample     : {attempts, executed, return_values}  or {error}
        semantics  : pretty-printed program semantics str  or {error}
        circuit    : {return_vars, num_literals}           or {error}
        inference  : {rv: {prob, Z}}                       or {error}
        gradient   : {rv: param_grads list}                or {error}
        predicted_label : winning letter from inference
        predicted_prob  : its probability
    """
    out: Dict[str, Any] = {}

    # 1. Sample
    try:
        sample_res = prog.sample(tries=sample_tries)
        out["sample"] = {
            "attempts":      sample_res.get("attempts"),
            "executed":      sample_res.get("executed"),
            "return_values": sample_res.get("return_values"),
        }
    except Exception as exc:
        out["sample"] = {"error": str(exc)}

    # 2. Semantics
    try:
        out["semantics"] = prog.semantics(pretty=True)
    except Exception as exc:
        out["semantics"] = {"error": str(exc)}

    # 3. Circuit + 4. Inference + 5. Gradient
    inference_results: Dict[str, Any] = {}
    gradient_results: Dict[str, Any] = {}
    try:
        circuit = prog.compile_circuit()
        out["circuit"] = {
            "return_vars":  sorted(circuit.return_vars),
            "num_literals": len(circuit.literal_map),
        }
        for rv in sorted(circuit.return_vars):
            try:
                inf = prog.inference(query={rv})
                inference_results[rv] = {
                    "prob": float(inf.out),
                    "Z":    float(inf.Z),
                }
            except Exception as exc:
                inference_results[rv] = {"error": str(exc)}
        # Optional Exc query
        try:
            exc_inf = prog.inference(query={"Exc"})
            inference_results["Exc"] = {
                "prob": float(exc_inf.out),
                "Z":    float(exc_inf.Z),
            }
        except Exception:
            pass

        for rv in sorted(circuit.return_vars):
            try:
                gi = prog.inference(query={rv}, compute_grad=True)
                gradient_results[rv] = list(gi.param_grads) if gi.param_grads else []
            except Exception as exc:
                gradient_results[rv] = {"error": str(exc)}
    except Exception as exc:
        out.setdefault("circuit",   {"error": str(exc)})
        out.setdefault("gradient",  {"error": str(exc)})
        # If circuit failed, inference_results is empty; surface the error.
        if not inference_results:
            inference_results = {"_error": str(exc)}

    out["inference"] = inference_results
    out["gradient"]  = gradient_results or out.get("gradient", {})

    # Pick predicted label by mapping return-var strings back to letters.
    prop_vals   = getattr(prog, "_prop_vals", {}) or {}
    return_vars = getattr(prog, "_return_vars", {}) or {}
    str_to_label: Dict[str, str] = {}
    for lbl, txt in zip(choice_labels, choices):
        str_to_label[txt.strip()] = lbl
    for lbl in choice_labels:
        str_to_label[lbl.strip()] = lbl

    best_label, best_p = None, -1.0
    for var_name, rv_name in return_vars.items():
        s = prop_vals.get(var_name, var_name).strip()
        lbl = str_to_label.get(s)
        if lbl is None:
            continue
        d = inference_results.get(rv_name, {})
        if isinstance(d, dict) and "prob" in d and d["prob"] > best_p:
            best_p = d["prob"]
            best_label = lbl

    out["predicted_label"] = best_label
    out["predicted_prob"]  = best_p
    return out


# Back-compat alias so external callers / older callers don't break.
def _run_inference(prog, choice_labels, choices) -> Dict[str, Any]:
    em = _run_execution_modes(prog, choice_labels, choices)
    return {
        "raw":             em.get("inference", {}),
        "predicted_label": em.get("predicted_label"),
        "predicted_prob":  em.get("predicted_prob", -1.0),
        # Surface the rich payload too so callers can opt-in.
        "execution_modes": em,
    }


# ---------------------------------------------------------------------------
# Per-example processing
# ---------------------------------------------------------------------------

def process_example(
    example: dict,
    *,
    model_name: str,
    cot_strategy: str,
    pool_mode: str,
    answer_aggregation_fn: str,
    statement_aggregation_fn: str,
    step_dependency_mode: str,
    num_retries: int,
    temperature: float = 0.0,
    verbal_conf: bool = True,
    verbal_step_deps: bool = True,
) -> dict:
    qid = example["qid"]
    question = example["question"]
    choices = example["choices"]
    choice_labels = example["choice_labels"]
    answer_key = example["answer_key"]

    result: Dict[str, Any] = {
        "qid": qid,
        "question": question,
        "choices": choices,
        "choice_labels": choice_labels,
        "answer_key": answer_key,
        "answer_text": example["answer_text"],
        "config": {
            "model_name": model_name,
            "cot_strategy": cot_strategy,
            "pool_mode": pool_mode,
            "answer_aggregation_fn": answer_aggregation_fn,
            "statement_aggregation_fn": statement_aggregation_fn,
            "step_dependency_mode": step_dependency_mode,
            "verbal_conf": verbal_conf,
            "verbal_step_deps": verbal_step_deps,
            "temperature": temperature,
        },
        "per_answer_traces": {},
        "predicted_label": None,
        "predicted_text": None,
        "error": None,
        "timestamp": datetime.utcnow().isoformat(),
    }

    final_cot, program_str, per_label_cots = run_mcq_external(
        query=question,
        choice_labels=choice_labels,
        choice_texts=choices,
        strategy=cot_strategy,
        pool_mode=pool_mode,
        answer_aggregation_fn=answer_aggregation_fn,
        statement_aggregation_fn=statement_aggregation_fn,
        step_dependency_mode=step_dependency_mode,
        model_name=model_name,
        num_retries=num_retries,
        temperature=temperature,
        debug=False,
    )

    for lbl, cot in per_label_cots.items():
        result["per_answer_traces"][lbl] = _serialize_cot(cot)

    # Surface the TF priors dict + raw response at the top level for easy
    # inspection / reuse.
    if per_label_cots:
        any_cot = next(iter(per_label_cots.values()))
        tf = getattr(any_cot, "_tf_priors", None)
        if tf is not None:
            result["tf_priors"] = tf
        # NEW: surface the full TF payload (prompt + response + logprobs)
        # at top level so the checkpoint has everything in one obvious place.
        tf_meta = getattr(any_cot, "_tf_meta", None)
        if tf_meta is not None:
            result["tf_meta"] = tf_meta
        # Keep tf_raw at top level for back-compat with existing readers.
        tf_raw = getattr(any_cot, "_tf_raw", None)
        if tf_raw is not None:
            result["tf_raw"] = tf_raw

    # Save the full CoT raw output.
    raw_combined = getattr(final_cot, "_raw_combined", None) or getattr(final_cot, "raw_cot", "")
    result["raw_cot"] = raw_combined
    # NEW: save the global '## Final Answer:' info — value, prob,
    # raw section text, AND per-token logprobs of the entire FA section.
    # Only present for combined-style strategies; separate has no FA.
    final_answer_info = getattr(final_cot, "_final_answer_info", None)
    if final_answer_info:
        result["final_answer_info"] = final_answer_info

    # Build ONE program for the chosen answer_aggregation_fn and run inference
    # once. The per-answer-prior-mode loop is gone: answer_aggregation_fn is the
    # single answer-prior axis now, so the checkpoint stores exactly the result
    # for the parameter set the user chose. The program parsing + execution
    # (write_mcq_program -> ParseStringCot -> _run_execution_modes) is identical
    # to the original code.
    cot_list = [per_label_cots[l] for l in choice_labels]

    try:
        ps = write_mcq_program(
            query=question,
            choice_labels=choice_labels,
            choice_texts=choices,
            cot_outputs=cot_list,
            step_dependency_mode=step_dependency_mode,
            pool_mode=pool_mode,
            answer_aggregation_fn=answer_aggregation_fn,
            statement_aggregation_fn=statement_aggregation_fn,
        )
        result["program"] = ps
        prog = ParseStringCot(ps)
        em = _run_execution_modes(prog, choice_labels, choices)
        # Save all 5 execution-mode results (matches the layout of the legacy
        # cot_prog-main runner so json_to_html.py can render the Sample /
        # Semantics / Circuit / Inference / Gradient sections).
        result["sample"]    = em.get("sample")
        result["semantics"] = em.get("semantics")
        result["circuit"]   = em.get("circuit")
        result["inference"] = em.get("inference")
        result["gradient"]  = em.get("gradient")
        result["predicted_label"] = em.get("predicted_label")
        if em.get("predicted_label"):
            idx = choice_labels.index(em["predicted_label"])
            result["predicted_text"] = choices[idx]
        else:
            result["predicted_text"] = None
    except Exception:
        result["error"] = traceback.format_exc()
        result["predicted_label"] = None
        result["predicted_text"] = None

    return result


# ---------------------------------------------------------------------------
# Reuse: rebuild programs from saved checkpoints (no LLM calls)
# ---------------------------------------------------------------------------

def refresh_tf_prior(ckpt_data: dict, llm, model_name: str,
                     num_retries: int = 3) -> dict:
    """Replace the cached slot-1 answer prior of one checkpoint with a fresh
    teacher-forcing logprob call (one LLM call); the CoT traces are kept.

    Writes ``tf_priors`` / ``tf_meta`` / ``tf_raw`` at the top level and drops
    any stale per-letter copies so ``_restore_cot`` picks up the new values.
    """
    labels = ckpt_data["choice_labels"]
    texts = ckpt_data["choices"]
    out = teacher_forcing_logprobs_parallel(
        query=ckpt_data["question"],
        candidate_labels=labels,
        candidate_texts=texts,
        llm=llm,
        model_name=model_name,
        num_retries=num_retries,
    )
    if out is None:
        u = 1.0 / max(len(labels), 1)
        tf_priors = {l: u for l in labels}
        tf_meta = {"prompt": None,
                   "response": "(TF call failed; uniform fallback)",
                   "logprobs": None}
    else:
        tf_priors, tf_meta = out
    ckpt_data = dict(ckpt_data)
    ckpt_data["tf_priors"] = tf_priors
    ckpt_data["tf_meta"] = tf_meta
    ckpt_data["tf_raw"] = tf_meta.get("response")
    traces = {}
    for lbl, t in (ckpt_data.get("per_answer_traces") or {}).items():
        t = dict(t)
        for k in ("tf_priors", "tf_meta", "tf_raw"):
            t.pop(k, None)
        traces[lbl] = t
    ckpt_data["per_answer_traces"] = traces
    return ckpt_data


def reprocess_from_checkpoint(
    ckpt_data: dict,
    *,
    pool_mode: str,
    answer_aggregation_fn: str,
    statement_aggregation_fn: str,
    step_dependency_mode: str,
) -> dict:
    """Rebuild and re-run inference for a saved per-example checkpoint with
    fresh (answer_aggregation_fn, statement_aggregation_fn, step_dependency_mode). No LLM calls.

    Reads ``per_answer_traces`` from the source ckpt, reconstructs each
    per-letter ``CoTOutput`` (incl. ``_init_score`` / ``_tf_priors``),
    then builds ONE program for the chosen ``answer_aggregation_fn`` and runs
    inference once — exactly as ``process_example`` does in the LLM path.
    """
    qid           = ckpt_data["qid"]
    question      = ckpt_data["question"]
    choices       = ckpt_data["choices"]
    choice_labels = ckpt_data["choice_labels"]
    answer_key    = ckpt_data["answer_key"]
    traces        = ckpt_data.get("per_answer_traces", {})
    top_tf        = ckpt_data.get("tf_priors")     # shared dict, optional
    top_tf_raw    = ckpt_data.get("tf_raw")        # shared raw response, optional
    top_tf_meta   = ckpt_data.get("tf_meta")       # shared full payload, optional

    cot_outputs = []
    for lbl in choice_labels:
        t = traces.get(lbl) or traces.get(lbl.upper()) or traces.get(lbl.lower())
        if t is None:
            raise KeyError(f"Missing per_answer_traces[{lbl!r}] in checkpoint {qid}")
        cot = _restore_cot(t, query=question)
        if not getattr(cot, "_tf_priors", None) and top_tf:
            cot._tf_priors = top_tf
        if not getattr(cot, "_tf_raw", None) and top_tf_raw:
            cot._tf_raw = top_tf_raw
        if not getattr(cot, "_tf_meta", None) and top_tf_meta:
            cot._tf_meta = top_tf_meta
        cot_outputs.append(cot)

    result: Dict[str, Any] = dict(ckpt_data)
    result["config"] = {
        **(ckpt_data.get("config", {})),
        "pool_mode":        pool_mode,
        "answer_aggregation_fn": answer_aggregation_fn,
        "statement_aggregation_fn":        statement_aggregation_fn,
        "step_dependency_mode":    step_dependency_mode,
        "reused":           True,
    }
    result.pop("modes", None)  # drop any legacy per-mode block from the source
    result["predicted_label"] = None
    result["predicted_text"] = None
    result["error"] = None
    result["timestamp"] = datetime.utcnow().isoformat()

    # Build ONE program for the chosen answer_aggregation_fn and run inference
    # once — no per-answer-prior-mode loop. Program parsing + execution
    # (write_mcq_program -> ParseStringCot -> _run_execution_modes) is identical
    # to the original code.
    try:
        ps = write_mcq_program(
            query=question,
            choice_labels=choice_labels,
            choice_texts=choices,
            cot_outputs=cot_outputs,
            step_dependency_mode=step_dependency_mode,
            pool_mode=pool_mode,
            answer_aggregation_fn=answer_aggregation_fn,
            statement_aggregation_fn=statement_aggregation_fn,
        )
        result["program"] = ps
        prog = ParseStringCot(ps)
        em = _run_execution_modes(prog, choice_labels, choices)
        result["sample"]    = em.get("sample")
        result["semantics"] = em.get("semantics")
        result["circuit"]   = em.get("circuit")
        result["inference"] = em.get("inference")
        result["gradient"]  = em.get("gradient")
        result["predicted_label"] = em.get("predicted_label")
        if em.get("predicted_label"):
            idx = choice_labels.index(em["predicted_label"])
            result["predicted_text"] = choices[idx]
        else:
            result["predicted_text"] = None
    except Exception:
        result["error"] = traceback.format_exc()
        result["predicted_label"] = None
        result["predicted_text"] = None

    return result


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def get_completed_qids(ckpt_dir: str) -> set:
    if not os.path.isdir(ckpt_dir):
        return set()
    done = set()
    for fname in os.listdir(ckpt_dir):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(ckpt_dir, fname)) as f:
                d = json.load(f)
            if d.get("error"):
                continue
            done.add(fname[:-5])
        except Exception:
            continue
    return done


def save_checkpoint(result: dict, ckpt_dir: str):
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, f"{result['qid']}.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)


def merge_checkpoints(ckpt_dir: str) -> List[dict]:
    rows = []
    if not os.path.isdir(ckpt_dir):
        return rows
    for fname in sorted(os.listdir(ckpt_dir)):
        if fname.endswith(".json"):
            with open(os.path.join(ckpt_dir, fname)) as f:
                rows.append(json.load(f))
    return rows


# ---------------------------------------------------------------------------
# Aggregate accuracy
# ---------------------------------------------------------------------------

def compute_accuracy(results: List[dict], n_total: int) -> Dict[str, Any]:
    """Accuracy of the run's single ``predicted_label`` per example.

    ``n_total`` is the dataset size of the run (examples sampled in LLM mode;
    source checkpoints, errored ones included, in reuse mode). Two numbers are
    reported, both in percent:

      ``accuracy``               = n_correct / n_total — an errored or missing
                                   example counts as WRONG. Same denominator as
                                   the vanilla runner, so numbers are comparable across
                                   configurations whose error counts differ.
      ``accuracy_on_predicted``  = n_correct / n_predicted — accuracy among the
                                   examples that produced a prediction. This
                                   is what the runner used to report alone;
                                   it is inflated whenever examples error.
    """
    n_correct = n_errored = n_predicted = 0
    for r in results:
        if r.get("error"):
            n_errored += 1
            continue
        pred = r.get("predicted_label")
        if pred is None:
            n_errored += 1
            continue
        n_predicted += 1
        if str(pred).strip().upper() == str(r["answer_key"]).strip().upper():
            n_correct += 1
    n_total = max(int(n_total), n_predicted + n_errored)
    n_missing = n_total - n_predicted - n_errored
    return {
        "n_total": n_total,
        "n_predicted": n_predicted,
        "n_errored": n_errored,
        "n_missing": n_missing,
        "n_correct": n_correct,
        "accuracy": (n_correct / n_total * 100) if n_total else 0.0,
        "accuracy_on_predicted": (n_correct / n_predicted * 100) if n_predicted else 0.0,
        "accuracy_unit": "percent",
        "accuracy_denominator": "n_total (errored and missing examples count as wrong)",
    }


def print_summary(summary: dict):
    print("\n" + "=" * 70)
    print(f"{'BBH-MCQ RESULTS':^70}")
    print("=" * 70)
    print(f"{'Correct':>8} | {'Predicted':>9} | {'Errors':>6} | {'Missing':>7} | "
          f"{'Dataset':>7} | {'Acc (dataset)':>13} | {'Acc (predicted)':>15}")
    print("-" * 70)
    print(f"{summary['n_correct']:>8} | {summary['n_predicted']:>9} | "
          f"{summary['n_errored']:>6} | {summary['n_missing']:>7} | "
          f"{summary['n_total']:>7} | {summary['accuracy']:>12.2f}% | "
          f"{summary['accuracy_on_predicted']:>14.2f}%")
    print("-" * 70)
    print("Acc (dataset) counts errored/missing examples as wrong; "
          "Acc (predicted) is over examples that produced a prediction.")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Run multi-answer CoT on BBH-MCQ-Hard")
    p.add_argument("--model_name", type=str, default="openai/gpt-4o")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_samples", type=int, default=None,
                   help="Optional cap on examples; if unset, run the whole dataset.")
    p.add_argument("--dataset_path", type=str, default=None,
                   help="Path to a BBH multi_cot.json (see experiments/data_prep/prep_bbh.py). "
                        "Required for fresh LLM runs; not needed when "
                        "--reuse_traces_from is set (the reuse path reads "
                        "examples directly from the source checkpoints).")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--num_retries", type=int, default=5)

    # Configuration axes (paper Table 7) — defined once in cot_prog/axes.py so
    # the runner and console can't drift. This adds: cot_strategy,
    # verbal_conf, verbal_step_deps (trace elicitation) and
    # answer_aggregation_fn, statement_aggregation_fn, pool_mode,
    # step_dependency_mode (program construction), each with its
    # Table 7 default. The old --answer_prior axis is gone — its source choice
    # is folded into answer_aggregation_fn (c=verbal, p=logit, cal/mean/max/min
    # = combine the two slots).
    axes.add_arguments(p)

    p.add_argument("--temperature", type=float, default=0.0,
                   help="LLM sampling temperature for the CoT-generation "
                        "calls (separate / combined). "
                        "Default 0.0 = greedy / deterministic. Set >0 (e.g. "
                        "0.7) for multiple independent samples per example. "
                        "The TF-priors call always runs at its built-in "
                        "default — those should stay stable across samples.")
    p.add_argument("--refresh_tf_priors", action="store_true",
                   help=("Reuse mode only: keep the cached CoT traces but "
                         "replace the slot-1 answer prior of every example "
                         "with a fresh teacher-forcing logprob call to "
                         "--model_name (one LLM call per example)."))
    p.add_argument("--reuse_traces_from", type=str, default=None,
                   help="Path to an existing output dir (with ckpts/) whose "
                        "per-example checkpoints contain CoT + tf_priors. When "
                        "set, this script skips ALL LLM calls and only rebuilds "
                        "the probabilistic program with the new "
                        "(answer_aggregation_fn, statement_aggregation_fn, step_dependency_mode), then runs "
                        "inference. Use to sweep configurations cheaply after "
                        "one base LLM-call run per (strategy, dataset).")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # --- Post-parse validation -----------------------------------------------
    # --dataset_path is only required for fresh LLM runs; the reuse path
    # loads examples directly from the source checkpoints and ignores it.
    if not args.reuse_traces_from and not args.dataset_path:
        sys.stderr.write(
            "ERROR: --dataset_path is required unless --reuse_traces_from is "
            "set.\n"
        )
        return 2

    print("="*70)
    print(f" Model         : {args.model_name}")
    print(f" Dataset       : {args.dataset_path or '(reuse mode — loaded from --reuse_traces_from)'}")
    print(f" Output dir    : {args.output_dir}")
    print(f" Strategy(#calls): {args.cot_strategy}")
    print(f" verbal_conf   : {args.verbal_conf}   verbal_step_deps: {args.verbal_step_deps}")
    print(f" Pool mode     : {args.pool_mode}")
    print(f" answer_aggregation_fn    : {args.answer_aggregation_fn}")
    print(f" statement_aggregation_fn : {args.statement_aggregation_fn}")
    print(f" step_dependency_mode     : {args.step_dependency_mode}")
    print(f" Seed          : {args.seed}, Num samples: {args.num_samples}")
    print("="*70 + "\n")

    ckpt_dir = os.path.join(args.output_dir, "ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---------------------------------------------------------------
    # REUSE MODE — skip the LLM phase entirely
    # ---------------------------------------------------------------
    if args.reuse_traces_from:
        src_ckpt_dir = os.path.join(args.reuse_traces_from, "ckpts")
        src_all = merge_checkpoints(src_ckpt_dir)
        src_ckpts = [c for c in src_all if not c.get("error")]
        n_dataset = len(src_all)   # errored source examples count as wrong
        print(f"Reusing traces from: {src_ckpt_dir} "
              f"({len(src_ckpts)} usable of {n_dataset} checkpoints)\n")

        completed = get_completed_qids(ckpt_dir)
        remaining_ckpts = [c for c in src_ckpts if str(c["qid"]) not in completed]
        print(f"Already done: {len(completed)}, remaining: {len(remaining_ckpts)}\n")

        refresh_llm = None
        if args.refresh_tf_priors:
            refresh_llm = _get_model(model_name=args.model_name)
            print(f"Refreshing slot-1 answer priors with a fresh teacher-forcing "
                  f"logprob call per example ({args.model_name}).\n")

        start = time.time()
        n_ok = n_err = 0
        pbar = tqdm(remaining_ckpts, desc="Reprocessing", unit="ex")
        for ckpt_data in pbar:
            qid = str(ckpt_data["qid"])
            pbar.set_postfix(qid=qid, ok=n_ok, err=n_err)
            try:
                if refresh_llm is not None:
                    ckpt_data = refresh_tf_prior(
                        ckpt_data, refresh_llm, args.model_name,
                        num_retries=args.num_retries)
                r = reprocess_from_checkpoint(
                    ckpt_data,
                    pool_mode=args.pool_mode,
                    answer_aggregation_fn=args.answer_aggregation_fn,
                    statement_aggregation_fn=args.statement_aggregation_fn,
                    step_dependency_mode=args.step_dependency_mode,
                )
                save_checkpoint(r, ckpt_dir)
                n_ok += 1
            except Exception:
                n_err += 1
                tqdm.write(f"[ERROR] qid={qid}: {traceback.format_exc().splitlines()[-1]}")
                err_row = {
                    "qid": qid,
                    "error": traceback.format_exc(),
                    "answer_key": ckpt_data.get("answer_key", ""),
                    "timestamp": datetime.utcnow().isoformat(),
                }
                save_checkpoint(err_row, ckpt_dir)

        elapsed = time.time() - start
        print(f"\nReprocessing done in {elapsed:.1f}s — ok={n_ok}, err={n_err}")

        all_results = merge_checkpoints(ckpt_dir)
        out_path = os.path.join(args.output_dir, "results.json")
        with open(out_path, "w") as f:
            json.dump({
                "metadata": {
                    "reused_from":      args.reuse_traces_from,
                    "tf_priors_refreshed_with": (args.model_name if args.refresh_tf_priors else None),
                    "pool_mode":        args.pool_mode,
                    "answer_aggregation_fn": args.answer_aggregation_fn,
                    "statement_aggregation_fn":        args.statement_aggregation_fn,
                    "step_dependency_mode":    args.step_dependency_mode,
                    "total_results":    len(all_results),
                    "n_dataset":        n_dataset,
                    "n_source_errored": n_dataset - len(src_ckpts),
                },
                "results": all_results,
            }, f, indent=2, default=str)
        print(f"Wrote {out_path}")

        summary = compute_accuracy(all_results, n_total=n_dataset)
        # Source checkpoints that had already failed at generation are not
        # re-processed; count them as errored (not missing) examples.
        n_src_err = n_dataset - len(src_ckpts)
        summary["n_source_errored"] = n_src_err
        shift = min(n_src_err, summary["n_missing"])
        summary["n_errored"] += shift
        summary["n_missing"] -= shift
        print_summary(summary)
        with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        return

    # ---------------------------------------------------------------
    # NORMAL MODE — full LLM pipeline (CoT + TF priors)
    # ---------------------------------------------------------------
    data = load_bbh(args.dataset_path)
    examples = sample_dataset(data, args.num_samples, args.seed)
    print(f"Sampled {len(examples)} examples (seed={args.seed}).\n")

    completed = get_completed_qids(ckpt_dir)
    remaining = [ex for ex in examples if ex["qid"] not in completed]
    print(f"Completed: {len(completed)}, Remaining: {len(remaining)}\n")

    start = time.time()
    n_ok = n_err = 0

    pbar = tqdm(remaining, desc="Processing", unit="ex")
    for ex in pbar:
        pbar.set_postfix(qid=ex["qid"], ok=n_ok, err=n_err)
        try:
            r = process_example(
                ex,
                model_name=args.model_name,
                cot_strategy=args.cot_strategy,
                pool_mode=args.pool_mode,
                answer_aggregation_fn=args.answer_aggregation_fn,
                statement_aggregation_fn=args.statement_aggregation_fn,
                step_dependency_mode=args.step_dependency_mode,
                num_retries=args.num_retries,
                temperature=args.temperature,
                verbal_conf=args.verbal_conf,
                verbal_step_deps=args.verbal_step_deps,
            )
            save_checkpoint(r, ckpt_dir)
            n_ok += 1
        except Exception as exc:
            n_err += 1
            tqdm.write(f"[ERROR] qid={ex['qid']}: {exc}")
            err_row = {
                "qid": ex["qid"],
                "error": traceback.format_exc(),
                "answer_key": ex["answer_key"],
                "timestamp": datetime.utcnow().isoformat(),
            }
            save_checkpoint(err_row, ckpt_dir)

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s — ok={n_ok}, err={n_err}")

    all_results = merge_checkpoints(ckpt_dir)
    out_path = os.path.join(args.output_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump({
            "metadata": {
                "model_name":       args.model_name,
                "dataset_path":     args.dataset_path,
                "cot_strategy":     args.cot_strategy,
                "pool_mode":        args.pool_mode,
                "answer_aggregation_fn": args.answer_aggregation_fn,
                "statement_aggregation_fn":        args.statement_aggregation_fn,
                "step_dependency_mode":    args.step_dependency_mode,
                "verbal_conf":      args.verbal_conf,
                "verbal_step_deps": args.verbal_step_deps,
                "temperature":      args.temperature,
                "seed":             args.seed,
                "num_samples":      args.num_samples,
                "total_results":    len(all_results),
                "n_dataset":        len(examples),
            },
            "results": all_results,
        }, f, indent=2, default=str)
    print(f"Wrote {out_path}")

    summary = compute_accuracy(all_results, n_total=len(examples))
    print_summary(summary)

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
