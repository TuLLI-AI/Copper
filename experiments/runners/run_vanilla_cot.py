"""Vanilla CoT pipeline — one LLM call per example, parsed into a sequential
probabilistic program, then compiled and run.

Compared to ``run_BBH_multi_cot.py`` (which runs the multi-answer-CoT
pipeline with per-letter advocacy chains and a Categorical posterior over
all answers), this runner is simpler:

  - ONE LLM call per example using the generic CoT instruction
    (binary tasks: GENERIC_COT_INSTRUCTION_BINARY, MCQ tasks:
    GENERIC_COT_INSTRUCTION_MCQ).
  - The parser turns the response into a CoTOutput; ``cot.write_program()``
    emits a sequential DSL program of the form::

        s1: str = "step 1 claim"
        ...
        sN: str = "<the answer letter or 'yes'/'no'>"
        P1: bool =~ Flip(s1, w1)
        ...
        P_N: bool =~ Flip(sN, w_N)
        return sN if P1 and ... and P_N else None

    For an MCQ example where the model picked "B", the only return values
    of ``prog()`` are "B" or None — the other letters never enter the
    program. The "final distribution" is therefore::

        P(picked_letter) = sample_count("B") / total_attempts
        P(other_or_none) = 1 - P(picked_letter)

    For binary (yes/no) tasks the complement bucket is the OTHER label
    (since "no" is a real candidate, not a void)::

        {"yes": p_yes, "no": 1 - p_yes}

  - Saved checkpoints follow the same field layout as ``run_BBH_multi_cot``
    (qid, question, choices, choice_labels, answer_key, answer_text,
    config, raw_cot, cot_logprobs, cot_prompt, error, timestamp), plus a
    ``vanilla`` block holding the program / sample / inference / etc.,
    plus ``predicted_label`` / ``predicted_text`` / ``predicted_distribution``.

Output dir layout:
    <output_dir>/
      ├── ckpts/<qid>.json
      ├── results.json
      └── summary.json

Usage:
    python experiments/runners/run_vanilla_cot.py \\
        --dataset_path bbh_data/penguins_in_a_table/multi_cot_val.json \\
        --output_dir   results/vanilla/<model_slug>/penguins_val \\
        --model_name   meta-llama/Llama-3.3-70B-Instruct-Turbo \\
        --num_samples  50 --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional

# Make the repo root importable when running this file directly
# (e.g. ``python experiments/runners/run_vanilla_cot.py ...``).
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Optional: orjson is faster
try:
    import orjson
    def _read_json(p):
        with open(p, "rb") as f: return orjson.loads(f.read())
except ImportError:
    def _read_json(p):
        with open(p) as f: return json.load(f)

from tqdm import tqdm

from cot_prog.lang import ParseStringCot
from cot_prog.prompt.cot_text_parser import CoTContreteSyntaxParser
from cot_prog.prompt.prompt_to_program import find_logps
from cot_prog.prompt.prompts import (
    GENERIC_COT_INSTRUCTION_BINARY,
    GENERIC_COT_INSTRUCTION_MCQ,
)
from cot_prog.prompt.multi_answer_cot import (
    _get_model,           # unified Together / OpenRouter / OpenAI routing
    _normalize_logprobs,  # both OpenAI and Together logprob shapes
)

# Reuse the multi_cot helper for sample/semantics/circuit/inference/gradient
from experiments.runners.run_BBH_multi_cot import _run_execution_modes


# ---------------------------------------------------------------------------
# Local fixup: the shared write_program() template emits `query = "..."` with
# the raw query text interpolated, which is fine for short single-line queries
# but breaks Python's AST parser as soon as the query contains newlines,
# apostrophes, or double quotes (e.g. BoolQ passages). We don't want to touch
# the shared template (run_BBH_multi_cot.py depends on the exact same writer),
# so we rewrite the single offending line locally before parsing.
# ---------------------------------------------------------------------------

import re as _re

# Matches `query = "..."` or `s12: str = "..."` (the two assignment shapes
# emitted by the shared write_program() / string_var() helpers, both of
# which interpolate raw text without escaping quotes).
_STR_ASSIGN_RE = _re.compile(
    r'^(?P<lhs>(?:query|s\d+\s*:\s*str))\s*=\s*"',
)


def _escape_query_line(program_str: str, query: str, cot=None) -> str:
    """Rewrite every ``query = "..."`` and ``s<N>: str = "..."`` line emitted
    by the shared writer with a ``repr()``-escaped string literal.

    The shared ``write_program()`` template and ``CoTOutputLine.string_var()``
    both build their assignment lines via raw ``f'... "{value}" ...'``
    interpolation, which produces invalid Python whenever the interpolated
    text contains a double quote, an apostrophe at an awkward position, or
    a newline (e.g. BoolQ passages, CoT claims that quote the question).

    We can't fix this in the shared writer without affecting
    ``run_BBH_multi_cot.py``, so we patch the program string locally here:

      * For each ``query =`` / ``s<N>: str =`` line, gather any continuation
        lines that spilled past the first newline (until the next section
        header or blank line).
      * Use the truth value from ``cot.query`` / ``cot.lines[i].claim`` (or
        ``.answer`` if claim is empty) to emit ``repr(value)``.
      * Fall back to a best-effort regex parse of the raw line if no ``cot``
        is provided.
    """
    # Build a lookup from "s<N>" -> intended string value, using the parsed
    # CoT lines (these hold the unescaped text we want to round-trip).
    step_values: Dict[str, str] = {}
    if cot is not None:
        for ln_obj in cot.lines:
            key = f"s{ln_obj.prefix}"
            val = ln_obj.claim if ln_obj.claim else ln_obj.answer
            step_values[key] = val

    lines = program_str.split("\n")
    out: List[str] = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        m = _STR_ASSIGN_RE.match(ln.lstrip())
        if m:
            lhs = m.group("lhs").replace(" ", "")
            # Walk forward across continuation lines until we hit a section
            # header, a blank line that precedes a section header, or the
            # NEXT assignment line (so consecutive ``s1: str = ...`` /
            # ``s2: str = ...`` lines aren't merged together).
            j = i + 1
            while j < len(lines):
                nxt = lines[j]
                if nxt.startswith("###") or nxt.startswith("## "):
                    break
                if _STR_ASSIGN_RE.match(nxt.lstrip()):
                    break
                if nxt == "" and j + 1 < len(lines) and (
                    lines[j + 1].startswith("###") or lines[j + 1].startswith("## ")
                ):
                    break
                j += 1
            # Pick the canonical value.
            if lhs == "query":
                value = query
                new_line = f"query = {value!r}"
            else:
                # lhs looks like "s12:str" after the space-strip — split on ":"
                key = lhs.split(":")[0]  # "s12"
                if key in step_values:
                    value = step_values[key]
                else:
                    # Fallback: take everything after the first `"` on the
                    # original line and strip the trailing `"` if present.
                    raw_after = ln.split('"', 1)[1] if '"' in ln else ""
                    if raw_after.endswith('"'):
                        raw_after = raw_after[:-1]
                    value = raw_after
                new_line = f"{key}: str = {value!r}"
            out.append(new_line)
            i = j
            continue
        out.append(ln)
        i += 1
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Local repair: fill in the answer step's formula/premises when the model
# omits the ``(ref. ...)`` clause on its final answer line. Without this,
# the shared ``write_program()`` silently skips the answer's ACOND (because
# its single-answer loop is gated on ``if line.formula:``), producing a
# program with no ``yield`` — and therefore no return values, no inference,
# no gradients. We can't change the shared writer (run_BBH_multi_cot uses
# it too), so we repair the parsed CoT in-place here.
# ---------------------------------------------------------------------------


def _patch_missing_answer_formula(cot) -> None:
    """If the answer step has ``formula==""`` and ``premises==[]``, fill it
    with the conjunction of every preceding non-answer step's prop var
    (``P1 and P2 and ... and P_{N-1}``) and the matching ``premises`` list.
    No-op when the answer step already has a formula or when there is no
    answer step.
    """
    if cot is None or not getattr(cot, "lines", None):
        return
    answer_lines = [ln for ln in cot.lines if ln.answer and ln.answer.strip()]
    if len(answer_lines) != 1:
        # Multi-answer or no-answer programs are handled by a different
        # branch of write_program() — leave them alone.
        return
    ans = answer_lines[0]
    if ans.formula:
        return  # already has a formula — nothing to fix
    preceding = [
        ln for ln in cot.lines
        if ln.prefix < ans.prefix and not (ln.answer and ln.answer.strip())
    ]
    if not preceding:
        return  # no premises to attach; let the writer skip as before
    ans.premises = [ln.prefix for ln in preceding]
    ans.formula  = " and ".join(ln.prop_var_name for ln in preceding)


# Matches a valid step-prop-var reference (P1, P12, ...) — the only
# tokens the SDD compiler will accept inside a formula.
_PROP_VAR_TOKEN = _re.compile(r"^P\d+$")


def _patch_invalid_premise_refs(cot) -> None:
    """Drop self-references and forward-references from each step's
    ``premises`` list, and rewrite ``formula`` accordingly.

    Why this is needed: small models (e.g. Gemma-3n-E4B-it) sometimes
    emit ``(ref. 1)`` on step 1 itself, or ``(ref. 5)`` on step 3. Both
    create cycles in the SDD (P_i depends on itself / on a not-yet-
    defined variable) and ``cot.check_program()`` rejects them. With
    ``num_retries=5`` the model usually keeps making the same mistake,
    so every attempt fails.

    The fix here silently filters those bogus references — equivalent
    to the model writing ``(ref. none)`` or a smaller premise set —
    *before* check_program runs. Self-/forward-referenced steps then
    behave like top-level (premise-free) steps in the program.
    """
    if cot is None or not getattr(cot, "lines", None):
        return
    import re
    for step in cot.lines:
        prefix = step.prefix
        if not step.premises:
            continue
        kept = [p for p in step.premises if p < prefix]   # only PRIOR steps
        if kept == list(step.premises):
            continue
        step.premises = kept
        # Rewrite the formula to drop the same prop-vars. Strip references
        # to invalid prefixes from the boolean expression while leaving
        # operators/structure intact.
        if step.formula:
            invalid = set(step.premises_in_formula(prefix)) if hasattr(step, "premises_in_formula") else set()
            # Simpler: rebuild from kept premises if they form an AND-chain
            if kept:
                step.formula = " and ".join(f"P{p}" for p in kept)
            else:
                step.formula = ""


def _patch_unrecognized_premise_refs(cot) -> None:
    """When the model writes ``(ref. passage)`` (or any free-text token) on
    a step instead of ``(ref. none)`` / ``(ref. 1, 2)``, the parser leaves
    ``premises=[]`` (no integers found) but ``formula="passage"`` (the
    boolean expression parser preserved the token).

    This is inconsistent: the shared ``write_program()`` then emits the
    step BOTH at the top level (because ``premises`` is empty) AND inside
    a conditional ``if passage: P1: bool =~ Flip(s1, ...)`` (because
    ``formula`` is non-empty), which duplicates ``P1`` and trips
    ``ValueError: Decision variable defined multiple times: P1``.

    BoolQ is the only BBH task where this fires often — the model is
    naturally tempted to write ``(ref. passage)`` because BoolQ examples
    have a passage prefix. Fix: clear the formula on any non-answer step
    whose ``premises`` is empty but ``formula`` references something
    other than valid ``P<N>`` step propositions. The step then behaves
    like a clean ``(ref. none)`` line.
    """
    if cot is None or not getattr(cot, "lines", None):
        return
    for ln in cot.lines:
        # Skip the answer step — its formula is fixed up by the helper above.
        if ln.answer and ln.answer.strip():
            continue
        if ln.premises:
            continue  # well-formed reference to numbered steps
        if not ln.formula:
            continue  # already a clean (ref. none) — nothing to do
        # Tokenize on whitespace / boolean operators; if every non-empty
        # token is a valid P<N> we leave it alone. Otherwise (any free-text
        # token like ``passage``) drop the formula entirely.
        toks = [t for t in _re.split(r"\s+|and|or|not|\(|\)", ln.formula) if t]
        if all(_PROP_VAR_TOKEN.match(t) for t in toks) and toks:
            continue
        ln.formula = ""


# ---------------------------------------------------------------------------
# Inline LLM call for vanilla CoT (mirrors generate_per_answer_cot_mcq from
# multi_answer_cot.py — same generator pattern, different prompt).
# Lives HERE (not in prompt_to_program.py) so we can attach per-token logprobs
# + the full prompt/response payload without modifying a shared utility.
# ---------------------------------------------------------------------------

def _generate_vanilla_cot(
    *,
    query: str,
    user_msg: str,
    instruction_template: str,
    llm,
    model_name: str,
    num_retries: int = 5,
    temperature: float = 0.0,
    top_p=None,
    strict: bool = False,
    debug: bool = False,
    remove_meta_tokens: bool = True,
    length_normalization: bool = True,
):
    """Make ONE LLM call with the vanilla CoT instruction, parse the response,
    and return a CoTOutput with per-token logprobs + prompt attached.

    Returns:
        (cot_program, last_raw_message) on success, or (None, last_raw_message)
        if all `num_retries` attempts failed.
    """
    error_msg = "No errors, first attempt"
    parsing_errors = []
    last_raw = None

    for attempt in range(1, num_retries + 1):
        system_prompt = instruction_template.format(errors=error_msg)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user",   "content": [{"type": "text", "text": user_msg}]},
        ]
        try:
            response = llm.chat.completions.create(
                model=model_name,
                temperature=temperature,
                messages=messages,
                top_p=top_p,
                logprobs=True,
            )
        except Exception as exc:
            error_msg = str(exc)
            parsing_errors.append(error_msg)
            print(f"[VANILLA] API error (attempt {attempt}/{num_retries}): {exc}",
                  file=sys.stderr)
            continue

        raw_message = response.choices[0].message.content
        last_raw = raw_message
        if debug:
            print(system_prompt); print(user_msg); print(raw_message); print("=" * 40)

        try:
            cot_parser = CoTContreteSyntaxParser()
            cot = cot_parser.parse(raw_message, query=query)
            # Repair common small-model format mistakes BEFORE the
            # check_program() pass — otherwise we'd burn all num_retries
            # on output the parser could have accepted with a tiny fix.
            _patch_invalid_premise_refs(cot)        # self-/forward-refs
            _patch_unrecognized_premise_refs(cot)   # (ref. passage) etc.
            _patch_missing_answer_formula(cot)      # final answer step
            err = cot.check_program(strict=strict)
            if err:
                raise ValueError(err)
            # find_logps splits the response token stream by newlines and
            # asserts the chunk count matches len(cot.lines). Small models
            # (Gemma-3n-E4B-it) sometimes emit an extra trailing newline
            # or a minimal closing line like "10. (final answer: yes)"
            # which makes the chunk count drift off by 1 from the parsed
            # step count — even though the CoT text itself is fine. Don't
            # blow up the whole attempt for that — try alignment first,
            # fall back to neutral weights (0.95 per step) on AssertionError
            # so the program still builds and inference can proceed.
            try:
                step_probs = find_logps(
                    response.choices[0], cot.lines,
                    remove_meta_tokens=remove_meta_tokens,
                    length_normalization=length_normalization,
                )
                cot._weights_from_logprobs = True
            except AssertionError as logp_err:
                # Don't add to parsing_errors — this is a recoverable
                # mismatch, not a parse failure. Log to stderr so the
                # cause is visible in run logs.
                print(f"[VANILLA] logp alignment off (attempt {attempt}, "
                      f"using uniform 0.95 fallback): {logp_err}",
                      file=sys.stderr)
                step_probs = [0.95] * len(cot.lines)
                cot._weights_from_logprobs = False
            cot.add_weights(step_probs)
        except Exception as exc:
            error_msg = str(exc)
            parsing_errors.append(error_msg)
            print(f"[VANILLA] Parse error (attempt {attempt}/{num_retries}): {exc}",
                  file=sys.stderr)
            continue

        # Attach LLM-call payload — full token stream + prompt/response
        try:
            full_tokens = _normalize_logprobs(response.choices[0])
            cot._cot_logprobs = [
                {"token":   getattr(t, "token",   None),
                 "logprob": getattr(t, "logprob", None),
                 "top_logprobs": [
                     {"token":   getattr(tl, "token",   None),
                      "logprob": getattr(tl, "logprob", None)}
                     for tl in (getattr(t, "top_logprobs", None) or [])
                 ]}
                for t in (full_tokens or [])
            ] or None
        except Exception:
            cot._cot_logprobs = None
        cot._cot_prompt = {
            "system":   system_prompt,
            "user":     user_msg,
            "response": raw_message,
        }
        cot.attempts = attempt
        cot.errors   = parsing_errors
        return cot, raw_message

    # Surface parsing_errors so process_example can save them — without
    # this list, "_generate_vanilla_cot returned None after N attempts"
    # gives no clue which exception caused each retry.
    return None, (last_raw, parsing_errors)


# ---------------------------------------------------------------------------
# Helpers (some duplicated from multi_cot for self-containment)
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> List[dict]:
    """Load multi_cot_*.json (preferred) or baseline_*.json."""
    data = _read_json(path)
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
    return out


def sample_dataset(data: List[dict], n: Optional[int], seed: int) -> List[dict]:
    if n is None or n >= len(data):
        return data
    rng = random.Random(seed)
    return rng.sample(data, n)


def is_mcq(example: dict) -> bool:
    """Treat as MCQ if there are >= 3 choices, OR exactly 2 non-yes/no choices."""
    labels  = example.get("choice_labels") or []
    choices = example.get("choices") or []
    if len(labels) < 2:
        return False
    if len(labels) >= 3:
        return True
    yn = {"yes", "no"}
    if len(labels) == 2 and {c.strip().lower() for c in choices} == yn:
        return False  # binary yes/no
    return True


def build_user_message(example: dict, mcq: bool) -> str:
    """Question + (choices block if MCQ) + 'Answer (step-by-step):'."""
    q = example["question"].strip()
    if not mcq:
        return f"Question: {q}\nAnswer (step-by-step):\n"
    labels  = example["choice_labels"]
    choices = example["choices"]
    block = "\n".join(f"{l}. {t}" for l, t in zip(labels, choices))
    return f"Question: {q}\n{block}\nAnswer (step-by-step):\n"


def predicted_label_from_sample(sample_result: dict, choice_labels: List[str]) -> Optional[str]:
    """Pick the most-frequent non-None return value from prog.sample().

    The program returns either the letter the model picked or None. We
    prefer the letter (if any), even if None has more samples (None just
    means "the chain didn't fire" — not a meaningful prediction). If the
    picked letter is not one of `choice_labels`, return it anyway (the
    accuracy comparison will then fail correctly).
    """
    rv = (sample_result or {}).get("return_values") or {}
    # Sort non-None by count desc; if any non-None exists, return it.
    candidates = [(k, v) for k, v in rv.items() if k is not None and k != "None"]
    if not candidates:
        return None
    return sorted(candidates, key=lambda kv: -kv[1])[0][0]


def final_distribution(
    sample_result: dict,
    picked: Optional[str],
    *,
    mcq: bool,
) -> Dict[str, float]:
    """Convert sample histogram into a 2-bucket view of the predicted label.

    For binary (yes/no) tasks the complement bucket is the OTHER yes/no
    label, so the distribution is ``{yes: p, no: 1-p}`` (or vice versa) —
    never ``other_or_none``, since "no" is a real candidate, not a void.

    For MCQ tasks the program returns either the picked letter or None
    (the chain didn't fire), so the complement is genuinely "anything
    other than the picked letter, including None" — kept as
    ``other_or_none``.
    """
    rv = (sample_result or {}).get("return_values") or {}
    total = sum(rv.values()) or 1

    if not mcq:
        # Binary: always emit yes/no distribution.
        p_yes = (rv.get("yes") or 0) / total
        p_no  = (rv.get("no")  or 0) / total
        # If the program never returned yes/no (shouldn't happen, but be
        # defensive), fall back to the picked label's mass.
        if p_yes + p_no == 0 and picked in {"yes", "no"}:
            p_picked = (rv.get(picked) or 0) / total
            other = "no" if picked == "yes" else "yes"
            return {picked: p_picked, other: 1.0 - p_picked}
        # Renormalize the yes+no mass to drop the None bucket so the two
        # numbers always sum to 1 (the None mass just means "chain didn't
        # fire" — not informative for a yes/no decision).
        denom = p_yes + p_no
        if denom > 0:
            p_yes /= denom
            p_no  /= denom
        return {"yes": p_yes, "no": p_no}

    # MCQ: keep the picked-vs-other_or_none view.
    if picked is None:
        return {"other_or_none": 1.0}
    p_picked = (rv.get(picked) or 0) / total
    return {picked: p_picked, "other_or_none": 1.0 - p_picked}


# ---------------------------------------------------------------------------
# Per-example processing
# ---------------------------------------------------------------------------

def process_example(example: dict, *, model_name: str, num_retries: int,
                    sample_tries: int = 5000) -> dict:
    """One LLM call → one program → one inference. Returns a ckpt dict."""
    qid           = example["qid"]
    question      = example["question"]
    choices       = example["choices"]
    choice_labels = example["choice_labels"]
    gold_letter   = example["answer_key"]
    mcq           = is_mcq(example)

    result: Dict[str, Any] = {
        "qid":           qid,
        "question":      question,
        "choices":       choices,
        "choice_labels": choice_labels,
        "answer_key":    gold_letter,
        "answer_text":   example.get("answer_text", ""),
        "config": {
            "model_name":     model_name,
            "cot_strategy":   "vanilla",
            "question_type":  "mcq" if mcq else "binary",
        },
        "predicted_label":        None,
        "predicted_text":         None,
        "predicted_distribution": None,
        "error":     None,
        "timestamp": datetime.utcnow().isoformat(),
    }

    user_msg = build_user_message(example, mcq)
    instruction_template = (GENERIC_COT_INSTRUCTION_MCQ if mcq
                            else GENERIC_COT_INSTRUCTION_BINARY)

    try:
        llm = _get_model(model_name=model_name)   # routes to OpenAI / Together / OpenRouter
        cot, last_raw = _generate_vanilla_cot(
            query=question,
            user_msg=user_msg,
            instruction_template=instruction_template,
            llm=llm,
            model_name=model_name,
            num_retries=num_retries,
            temperature=0.0,
            strict=False,
        )
        if cot is None:
            # On failure _generate_vanilla_cot returns (None, (raw, errors)).
            # On success it returns (cot, raw_message). Unpack defensively.
            if isinstance(last_raw, tuple) and len(last_raw) == 2:
                last_raw_text, parse_errors = last_raw
            else:
                last_raw_text, parse_errors = last_raw, []
            err_summary = "\n".join(
                f"  attempt {i+1}: {e}" for i, e in enumerate(parse_errors or [])
            ) or "  (no error messages captured)"
            result["error"] = (
                f"_generate_vanilla_cot returned None after {num_retries} "
                f"attempts. Per-attempt errors:\n{err_summary}\n\n"
                f"Last raw response (first 800 chars):\n"
                f"{(last_raw_text or '')[:800]}"
            )
            return result

        # Save the LLM-call payload (prompt_to_program now attaches
        # _cot_logprobs and _cot_prompt to the returned cot — see the
        # patch in cot_prog/prompt/prompt_to_program.py).
        result["raw_cot"]      = cot.raw_cot
        result["attempts"]     = cot.attempts
        result["errors"]       = cot.errors or []
        cot_logprobs = getattr(cot, "_cot_logprobs", None)
        if cot_logprobs is not None:
            result["cot_logprobs"] = cot_logprobs
        cot_prompt = getattr(cot, "_cot_prompt", None)
        if cot_prompt is not None:
            result["cot_prompt"] = cot_prompt

        # Repair: when the model omits ``(ref. ...)`` on the final answer
        # step (it sometimes writes only ``... (final answer: A)``), the
        # parser leaves ``formula=""`` and ``premises=[]`` on that step.
        # The shared ``write_program()`` then silently skips the step's
        # ACOND, producing a program with NO ``yield`` — so ``prog.sample()``
        # always returns None, ``inference`` only computes ``Exc``, and
        # ``gradient`` is empty. Fill in a sane default (premises = every
        # preceding non-answer step) so the answer branch is emitted.
        # We patch in-place on the local ``cot.lines`` only — no shared
        # parser/writer code is touched.
        _patch_missing_answer_formula(cot)
        # Repair: when the model writes ``(ref. passage)`` (BoolQ-specific
        # tic) or any free-text token instead of step numbers, the parser
        # leaves the step with empty premises but a non-empty free-text
        # formula. The shared writer then duplicates the step's prop-var
        # definition (top-level + inside a conditional), which trips
        # ``ValueError: Decision variable defined multiple times``. Strip
        # the bogus formula so the step behaves like a clean ``(ref. none)``.
        _patch_unrecognized_premise_refs(cot)

        # Per-step parsed metadata (claim, answer, weight, confidence,
        # premises) — same shape as multi_cot's per_answer_traces[L].steps,
        # so the same downstream tools can consume it.
        result["steps"] = [s.to_dict() for s in cot.lines]

        # Build the DSL program
        program_str = cot.write_program()
        # The shared write_program() emits `query = "..."` with the raw query
        # interpolated, which breaks Python parsing when the query contains
        # newlines, apostrophes, or unescaped double quotes (e.g. BoolQ
        # passages). Rewrite that single line locally with a properly escaped
        # repr() so we don't touch the shared template (which run_BBH_multi_cot
        # also uses).
        program_str = _escape_query_line(program_str, question, cot=cot)
        result["program"] = program_str

        # Compile + run inference (sample / semantics / circuit / inference / gradient)
        prog = ParseStringCot(program_str)
        em = _run_execution_modes(prog, choice_labels, choices,
                                  sample_tries=sample_tries)
        result["sample"]    = em.get("sample")
        result["semantics"] = em.get("semantics")
        result["circuit"]   = em.get("circuit")
        result["inference"] = em.get("inference")
        result["gradient"]  = em.get("gradient")

        # Extract predicted letter from SYMBOLIC INFERENCE (exact marginal),
        # not from the Monte Carlo sample histogram. _run_execution_modes
        # already runs prog.inference(query={rv}) for every return variable
        # and stores the argmax label in em["predicted_label"]. This is the
        # same path multi_cot uses, and it gives an exact marginal instead
        # of a 5000-sample estimate.
        #
        # The vanilla program returns whatever the model wrote (a letter
        # for MCQ tasks, a word like "yes"/"no" for binary). Resolve both
        # shapes here:
        #   - predicted_label : the letter (matches answer_key)
        #   - predicted_text  : the option text (matches answer_text)
        # _run_execution_modes already maps text→letter via str_to_label,
        # so em["predicted_label"] is always a letter when the program
        # yields a recognized return value.
        picked = em.get("predicted_label")
        # Fallback to the sample histogram only if inference produced
        # nothing (e.g. the program's return_vars dict is empty or every
        # candidate's marginal is exactly zero). This keeps degenerate
        # programs scoreable; in practice it should rarely fire.
        if picked is None:
            picked = predicted_label_from_sample(em.get("sample") or {}, choice_labels)
        picked_letter, picked_text = None, None
        if picked is not None:
            if picked in choice_labels:
                # Program returned a letter (typical for MCQ)
                picked_letter = picked
                picked_text   = choices[choice_labels.index(picked)]
            elif picked in choices:
                # Program returned the option text (typical for binary yes/no)
                picked_text   = picked
                picked_letter = choice_labels[choices.index(picked)]
            else:
                # Free-form return (e.g., a substring) — leave both as raw
                picked_letter = picked
                picked_text   = picked
        result["predicted_label"] = picked_letter
        result["predicted_text"]  = picked_text
        result["predicted_distribution"] = final_distribution(
            em.get("sample") or {}, picked, mcq=mcq,
        )

    except Exception:
        result["error"] = traceback.format_exc()

    return result


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(result: dict, ckpt_dir: str):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"{result['qid']}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)


def get_completed_qids(ckpt_dir: str) -> set:
    if not os.path.isdir(ckpt_dir):
        return set()
    done = set()
    for fname in os.listdir(ckpt_dir):
        if not fname.endswith(".json"):
            continue
        try:
            d = _read_json(os.path.join(ckpt_dir, fname))
            if d.get("error"):
                continue
            done.add(fname[:-5])
        except Exception:
            continue
    return done


def merge_checkpoints(ckpt_dir: str) -> List[dict]:
    rows = []
    if not os.path.isdir(ckpt_dir):
        return rows
    for fname in sorted(os.listdir(ckpt_dir)):
        if fname.endswith(".json"):
            try:
                rows.append(_read_json(os.path.join(ckpt_dir, fname)))
            except Exception:
                pass
    return rows


def compute_accuracy(results: List[dict], n_total: int) -> Dict[str, Any]:
    """Accuracy with n_total as denominator (failures count as wrong).

    A prediction counts as correct if EITHER:
      - ``predicted_label`` (case-insensitive) equals ``answer_key``  (letter match), OR
      - ``predicted_text``  (case-insensitive) equals ``answer_text`` (option-text match).

    The dual check makes the function robust to either of two layouts:
      * MCQ: answer_key="A", answer_text="04/18/1969", predicted_label="A".
      * Binary: answer_key="A", answer_text="yes", predicted_label="A" (post-fix)
                — but older ckpts may have predicted_label="yes" stored.
    """
    n_correct = n_errored = n_predicted = 0
    for r in results:
        if r.get("error"):
            n_errored += 1
            continue
        n_predicted += 1
        gold_key  = (r.get("answer_key")  or "").strip().upper()
        gold_text = (r.get("answer_text") or "").strip().lower()
        pred_lab  = (r.get("predicted_label") or "")
        pred_text = (r.get("predicted_text")  or "")
        ok = False
        if isinstance(pred_lab, str) and pred_lab.strip().upper() == gold_key and gold_key:
            ok = True
        elif isinstance(pred_text, str) and pred_text.strip().lower() == gold_text and gold_text:
            ok = True
        # Final fallback: predicted_label might be the option text (older
        # ckpts where the text->letter mapping wasn't applied).
        elif isinstance(pred_lab, str) and pred_lab.strip().lower() == gold_text and gold_text:
            ok = True
        if ok:
            n_correct += 1
    n_missing = max(0, n_total - n_predicted - n_errored)
    return {
        "vanilla": {
            "n_correct":   n_correct,
            "n_predicted": n_predicted,
            "n_errored":   n_errored,
            "n_missing":   n_missing,
            "n_total":     n_total,
            "accuracy":    (n_correct / n_total) if n_total else 0.0,
        }
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset_path", type=str, required=True,
                   help="multi_cot_*.json or baseline_*.json with the schema "
                        "produced by experiments/data_prep/prep_bbh.py.")
    p.add_argument("--output_dir",   type=str, required=True)
    p.add_argument("--model_name",   type=str,
                   default="openai/gpt-4o")
    p.add_argument("--num_samples",  type=int, default=None,
                   help="Cap on examples (default: full dataset).")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--num_retries",  type=int, default=5)
    p.add_argument("--sample_tries", type=int, default=5000,
                   help="Tries for prog.sample() — how many MC samples to "
                        "draw when computing the empirical return distribution.")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    print("="*70)
    print(f" Vanilla CoT pipeline")
    print(f" Model        : {args.model_name}")
    print(f" Dataset      : {args.dataset_path}")
    print(f" Output dir   : {args.output_dir}")
    print(f" Num samples  : {args.num_samples}  (seed={args.seed})")
    print(f" Sample tries : {args.sample_tries}")
    print("="*70 + "\n")

    ckpt_dir = os.path.join(args.output_dir, "ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)

    data     = load_dataset(args.dataset_path)
    n_total  = len(data)
    examples = sample_dataset(data, args.num_samples, args.seed)
    print(f"Dataset: {n_total} examples; running {len(examples)} (after sampling)\n")

    completed = get_completed_qids(ckpt_dir)
    remaining = [ex for ex in examples if ex["qid"] not in completed]
    print(f"Already done: {len(completed)}, remaining: {len(remaining)}\n")

    n_ok = n_err = 0
    pbar = tqdm(remaining, desc="vanilla CoT", unit="ex")
    start = time.time()
    for ex in pbar:
        pbar.set_postfix(qid=ex["qid"], ok=n_ok, err=n_err)
        try:
            r = process_example(ex,
                                model_name=args.model_name,
                                num_retries=args.num_retries,
                                sample_tries=args.sample_tries)
            save_checkpoint(r, ckpt_dir)
            if r.get("error"):
                n_err += 1
            else:
                n_ok += 1
        except Exception:
            n_err += 1
            tqdm.write(f"[ERROR] qid={ex['qid']}: {traceback.format_exc().splitlines()[-1]}")
            err_row = {"qid": ex["qid"], "error": traceback.format_exc(),
                       "answer_key": ex.get("answer_key", ""),
                       "timestamp": datetime.utcnow().isoformat()}
            save_checkpoint(err_row, ckpt_dir)

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s — ok={n_ok}, err={n_err}")

    # Aggregate
    all_results = merge_checkpoints(ckpt_dir)
    n_dataset = len(data)   # full dataset size (for the denominator)
    summary = compute_accuracy(all_results, n_dataset)
    s = summary["vanilla"]
    print()
    print(f"Vanilla CoT accuracy: {s['accuracy']*100:.2f}%  "
          f"({s['n_correct']}/{s['n_total']})  "
          f"errored={s['n_errored']}  missing={s['n_missing']}")

    out_results = os.path.join(args.output_dir, "results.json")
    with open(out_results, "w") as f:
        json.dump({
            "metadata": {
                "model_name":   args.model_name,
                "dataset_path": args.dataset_path,
                "cot_strategy": "vanilla",
                "seed":         args.seed,
                "num_samples":  args.num_samples,
                "total_results": len(all_results),
            },
            "results": all_results,
        }, f, indent=2, default=str)
    out_summary = os.path.join(args.output_dir, "summary.json")
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_results}")
    print(f"Wrote {out_summary}")


if __name__ == "__main__":
    main()
