#!/usr/bin/env python3
"""
Run 3 direct reasoning baselines on a JSON dataset using OpenAI or OpenRouter.

Baselines
---------
1) direct answer without CoT
2) CoT
3) Tree of Thoughts (simple propose-score-select beam search)

Supported input formats
-----------------------
Format A:
[
  {"question": "...", "gold answer": "..."},
  ...
]

Format B:
{
  "metadata": {...},
  "results": [
    {
      "question": "...",
      "original_answer": true,
      ...
    },
    ...
  ]
}

For StrategyQA-style booleans, the loader maps:
- true  -> "yes"
- false -> "no"

OpenRouter usage
----------------
export OPENROUTER_API_KEY=your_key
python reasoning_baselines.py --input questions.json --output_dir outputs --model gpt-4o

OpenAI usage
------------
export OPENAI_API_KEY=your_key
python reasoning_baselines.py --input questions.json --output_dir outputs --model gpt-4o
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

import os

# API keys come from the environment. Set ONE of:
#   export OPENROUTER_API_KEY=sk-or-v1-...    # for gpt-4o via OpenRouter
#   export OPENAI_API_KEY=sk-...              # for direct OpenAI access
#   export TOGETHER_API_KEY=...               # for Llama-3.3 etc.
# (See LLMClient.__init__ in this file for the routing logic.)


# =========================
# Utilities
# =========================

def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


_FINAL_TAG_RE = re.compile(r"<final>(.*?)</final>", re.DOTALL | re.IGNORECASE)


def extract_final_answer(text: str) -> str:
    if not text:
        return ""
    m = _FINAL_TAG_RE.search(text)
    raw = m.group(1).strip() if m else text.strip()
    # Normalize yes/no answers: "Yes, because ..." → "Yes"
    first_word = raw.split(",")[0].split(".")[0].split(";")[0].strip()
    if first_word.lower() in ("yes", "no"):
        return first_word.lower()
    return raw


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\.,!\?;:\"'`$begin:math:text$$end:math:text$$begin:math:display$$end:math:display$\{\}]", "", s)
    return s


def exact_match(pred: str, gold: str) -> int:
    p = normalize_text(pred)
    g = normalize_text(gold)
    if p == g:
        return 1
    # For yes/no questions: "yes, because ..." should match "yes"
    if g in ("yes", "no") and p.startswith(g):
        return 1
    return 0


def contains_match(pred: str, gold: str) -> int:
    p = normalize_text(pred)
    g = normalize_text(gold)
    return int(g in p or p in g)


def convert_gold_answer(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if value is None:
        return ""
    return str(value).strip()


def load_examples(path: str) -> List[Dict[str, str]]:
    """
    Supports both:
    - [{"question": ..., "gold answer": ...}, ...]
    - {"results": [{"question": ..., "original_answer": ...}, ...], ...}
    """
    raw = read_json(path)

    if isinstance(raw, list):
        examples = []
        for i, ex in enumerate(raw):
            if not isinstance(ex, dict):
                raise ValueError(f"Example {i} is not a JSON object.")
            if "question" not in ex:
                raise ValueError(f"Example {i} missing key: 'question'")
            # Support both "gold answer" (baseline format) and "answer" (StrategyQA raw format)
            if "gold answer" in ex:
                gold = ex["gold answer"]
            elif "answer" in ex:
                gold = ex["answer"]
            else:
                raise ValueError(f"Example {i} missing 'gold answer' or 'answer' key")
            examples.append(
                {
                    "question": str(ex["question"]),
                    "gold_answer": convert_gold_answer(gold),
                }
            )
        return examples

    if isinstance(raw, dict) and "results" in raw:
        results = raw["results"]
        if not isinstance(results, list):
            raise ValueError("'results' must be a list.")
        examples = []
        for i, ex in enumerate(results):
            if not isinstance(ex, dict):
                raise ValueError(f"Result {i} is not a JSON object.")
            if "question" not in ex:
                raise ValueError(f"Result {i} missing key: 'question'")
            if "original_answer" not in ex:
                raise ValueError(f"Result {i} missing key: 'original_answer'")
            examples.append(
                {
                    "question": str(ex["question"]),
                    "gold_answer": convert_gold_answer(ex["original_answer"]),
                }
            )
        return examples

    raise ValueError(
        "Unsupported JSON format. Expected either a list of "
        "{question, gold answer} objects, or a dict with a 'results' field "
        "containing {question, original_answer}."
    )


# =========================
# Model client
# =========================

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

MODEL_MAP_OPENAI = {
    "gpt-4o": "gpt-4o-2024-11-20",
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "chatgpt-4o": "chatgpt-4o-latest",
    "gpt-5": "gpt-5-2025-08-07",
    "gpt-5-mini": "gpt-5-mini-2025-08-07",
    "gpt-5-nano": "gpt-5-nano-2025-08-07",
    "gpt-5-pro": "gpt-5-pro-2025-10-06",
}

MODEL_MAP_OPENROUTER = {
    "gpt-4o": "openai/gpt-4o-2024-11-20",
    "gpt-4o-mini": "openai/gpt-4o-mini-2024-07-18",
    "chatgpt-4o": "openai/chatgpt-4o-latest",
    "gpt-5": "openai/gpt-5-2025-08-07",
    "gpt-5-mini": "openai/gpt-5-mini-2025-08-07",
    "gpt-5-nano": "openai/gpt-5-nano-2025-08-07",
    "gpt-5-pro": "openai/gpt-5-pro-2025-10-06",
}


@dataclass
class GenerationConfig:
    temperature: float = 0.2
    max_completion_tokens: int = 1024
    timeout: int = 120
    retries: int = 5
    retry_sleep: float = 2.0


class LLMClient:
    def __init__(self, model: str, config: GenerationConfig) -> None:
        self.config = config

        together_key = os.getenv("TOGETHER_API_KEY", "").strip()
        openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        openai_key = os.getenv("OPENAI_API_KEY", "").strip()

        if together_key and "/" in model:
            from together import Together
            self.provider = "together"
            self.client = Together()
            self.model = model
        elif openrouter_key:
            self.provider = "openrouter"
            self.client = OpenAI(
                api_key=openrouter_key,
                base_url=OPENROUTER_BASE_URL,
            )
            self.model = MODEL_MAP_OPENROUTER.get(model, model)
        elif openai_key:
            self.provider = "openai"
            self.client = OpenAI(api_key=openai_key)
            self.model = MODEL_MAP_OPENAI.get(model, model)
        else:
            raise ValueError(
                "Missing API key. Set TOGETHER_API_KEY, OPENROUTER_API_KEY, or OPENAI_API_KEY."
            )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_completion_tokens: Optional[int] = None,
    ) -> str:
        sys_role = "developer" if self.provider == "openai" and "gpt-5" in self.model else "system"
        params: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": sys_role, "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        # Together uses max_tokens, OpenAI/OpenRouter uses max_completion_tokens
        if self.provider == "together":
            params["max_tokens"] = (
                self.config.max_completion_tokens
                if max_completion_tokens is None
                else max_completion_tokens
            )
        else:
            params["max_completion_tokens"] = (
                self.config.max_completion_tokens
                if max_completion_tokens is None
                else max_completion_tokens
            )

        if "gpt-5" in self.model:
            params["reasoning_effort"] = "minimal"
            params["verbosity"] = "low"
        else:
            params["temperature"] = (
                self.config.temperature if temperature is None else temperature
            )

        last_err = None
        for attempt in range(self.config.retries):
            try:
                create_kwargs = {**params}
                if self.provider != "together":
                    create_kwargs["timeout"] = self.config.timeout
                response = self.client.chat.completions.create(**create_kwargs)
                content = response.choices[0].message.content
                return content if content is not None else ""
            except Exception as e:
                last_err = e
                if attempt < self.config.retries - 1:
                    time.sleep(self.config.retry_sleep * (attempt + 1))
                else:
                    raise RuntimeError(
                        f"Generation failed after {self.config.retries} attempts: {e}"
                    ) from e

        raise RuntimeError(f"Unexpected generation failure: {last_err}")


# =========================
# Prompts
# =========================

BASE_SYSTEM = "You are a careful reasoning assistant. Follow the user instructions exactly."

DIRECT_PROMPT = """Answer the following question directly.
Do not show any chain-of-thought or explanation.
Return your final answer only inside tags: <final> ... </final>

Question:
{question}
"""

COT_PROMPT = """Solve the following question step by step.
Then give the final answer inside tags: <final> ... </final>

Question:
{question}
"""

COMBINED_COT_PROMPT = """Given the following question, consider the possible answers and generate step-by-step reasoning to support each side. Then, based on all the reasoning, give your final answer.

## For yes:
[Provide step-by-step reasoning supporting "yes"]

## For no:
[Provide step-by-step reasoning supporting "no"]

## Final Answer:
After considering both sides, give your final answer inside tags: <final> ... </final>

Question:
{question}
"""

TOT_PROPOSE_PROMPT = """You are doing tree-of-thought reasoning.
Given the question and the current partial reasoning path, propose {n_candidates} diverse, short next-step thoughts.
Each thought should be one useful step, not a full solution.
Return JSON only in the form:
{{"thoughts": ["thought 1", "thought 2", ...]}}

Question:
{question}

Current reasoning path:
{current_path}
"""

TOT_SCORE_PROMPT = """You are scoring a candidate reasoning path.
Given the question and the reasoning path so far, rate how promising this path is for reaching the correct answer.
Return JSON only in the form:
{{"score": <float between 0 and 1>, "reason": "very short reason"}}

Question:
{question}

Reasoning path:
{path}
"""

TOT_FINALIZE_PROMPT = """You are given a question and a candidate reasoning path.
Use it to produce the final answer.
Return the final answer inside tags: <final> ... </final>

Question:
{question}

Reasoning path:
{path}
"""

CORRECTNESS_PRED_PROMPT = """You are judging whether a model's predicted answer mentions / is equivalent to the gold (reference) answer for a multiple-choice or short-answer question.

Output JSON only in this format:
{{"predicted_correct": <0 or 1>, "confidence": <float between 0 and 1>, "reason": "very short reason"}}

Rules for judging equivalence:
- Output 1 if the predicted answer clearly mentions or matches the gold answer; otherwise 0.
- Letters and the choice text they refer to are equivalent. E.g. if the gold is "A" and the question's choice (A) is "Carol", then any of "A", "(A)", "A.", "A. Carol", or "Carol" should be judged correct.
- Different surface forms of the same value are equivalent. E.g. "12/25/1937" == "December 25, 1937"; "yes" == "Yes, because ...".
- Extra reasoning text in the predicted answer is fine as long as the conclusion matches the gold.
- An empty or off-topic prediction is incorrect.

Question:
{question}

Gold answer:
{gold_answer}

Predicted answer:
{pred_answer}
"""


# =========================
# ToT baseline
# =========================

@dataclass
class ThoughtNode:
    path: List[str]
    score: float


def safe_json_loads(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


class TreeOfThoughtSolver:
    def __init__(
        self,
        client: LLMClient,
        max_depth: int = 3,
        beam_width: int = 2,
        n_candidates: int = 3,
        temperature: float = 0.7,
    ) -> None:
        self.client = client
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.n_candidates = n_candidates
        self.temperature = temperature

    def propose(self, question: str, current_path: List[str]) -> List[str]:
        prompt = TOT_PROPOSE_PROMPT.format(
            question=question,
            current_path=(
                "\n".join(f"{i+1}. {step}" for i, step in enumerate(current_path))
                if current_path
                else "<empty>"
            ),
            n_candidates=self.n_candidates,
        )
        raw = self.client.generate(
            BASE_SYSTEM,
            prompt,
            temperature=self.temperature,
            max_completion_tokens=512,
        )
        try:
            obj = safe_json_loads(raw)
            thoughts = obj.get("thoughts", [])
            thoughts = [str(t).strip() for t in thoughts if str(t).strip()]
            return thoughts[: self.n_candidates]
        except Exception:
            thoughts = [
                line.strip(" -0123456789.")
                for line in raw.splitlines()
                if line.strip()
            ]
            return thoughts[: self.n_candidates]

    def score_path(self, question: str, path: List[str]) -> Tuple[float, str]:
        prompt = TOT_SCORE_PROMPT.format(
            question=question,
            path="\n".join(f"{i+1}. {step}" for i, step in enumerate(path)),
        )
        raw = self.client.generate(
            BASE_SYSTEM,
            prompt,
            temperature=0.0,
            max_completion_tokens=256,
        )
        try:
            obj = safe_json_loads(raw)
            score = float(obj.get("score", 0.0))
            score = max(0.0, min(1.0, score))
            reason = str(obj.get("reason", ""))
            return score, reason
        except Exception:
            return 0.5, "fallback parse"

    def finalize(self, question: str, path: List[str]) -> str:
        prompt = TOT_FINALIZE_PROMPT.format(
            question=question,
            path="\n".join(f"{i+1}. {step}" for i, step in enumerate(path)),
        )
        return self.client.generate(
            BASE_SYSTEM,
            prompt,
            temperature=0.2,
            max_completion_tokens=512,
        )

    def solve(self, question: str, gold: str = "") -> Dict[str, Any]:
        beam = [ThoughtNode(path=[], score=1.0)]
        search_trace: List[Dict[str, Any]] = []

        for depth in range(self.max_depth):
            expanded: List[ThoughtNode] = []
            for node in beam:
                thoughts = self.propose(question, node.path)
                scored_children = []
                for thought in thoughts:
                    child_path = node.path + [thought]
                    score, reason = self.score_path(question, child_path)
                    expanded.append(ThoughtNode(path=child_path, score=score))
                    scored_children.append(
                        {
                            "thought": thought,
                            "score": score,
                            "reason": reason,
                        }
                    )
                search_trace.append(
                    {
                        "depth": depth,
                        "parent_path": node.path,
                        "children": scored_children,
                    }
                )

            if not expanded:
                break

            expanded.sort(key=lambda x: x.score, reverse=True)
            beam = expanded[: self.beam_width]

        best = max(beam, key=lambda x: x.score)
        raw = self.finalize(question, best.path)
        final_answer = extract_final_answer(raw)
        correctness = predict_correctness(self.client, question, final_answer, gold)

        return {
            "raw_output": raw,
            "pred_answer": final_answer,
            "best_path": best.path,
            "best_score": best.score,
            "search_trace": search_trace,
            **correctness,
        }


# =========================
# Baseline runners
# =========================

def predict_correctness(
    client: LLMClient,
    question: str,
    pred_answer: str,
    gold_answer: str = "",
) -> Dict[str, Any]:
    """LLM-as-judge: 1 if `pred_answer` mentions / is equivalent to `gold_answer`, else 0.

    The judge sees the question, the gold answer, and the predicted answer,
    and returns a 0/1 judgment plus a confidence and short reason.
    """
    prompt = CORRECTNESS_PRED_PROMPT.format(
        question=question,
        gold_answer=gold_answer,
        pred_answer=pred_answer,
    )
    raw = client.generate(
        BASE_SYSTEM,
        prompt,
        temperature=0.0,
        max_completion_tokens=256,
    )
    try:
        obj = safe_json_loads(raw)
        predicted_correct = int(obj.get("predicted_correct", 0))
        predicted_correct = 1 if predicted_correct else 0
        confidence = float(obj.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))
        reason = str(obj.get("reason", ""))
        return {
            "correctness_raw_output": raw,
            "predicted_correct": predicted_correct,
            "correctness_confidence": confidence,
            "correctness_reason": reason,
        }
    except Exception:
        return {
            "correctness_raw_output": raw,
            "predicted_correct": 0,
            "correctness_confidence": 0.0,
            "correctness_reason": "fallback parse",
        }


def run_direct(client: LLMClient, question: str, gold: str = "") -> Dict[str, Any]:
    raw = client.generate(
        BASE_SYSTEM,
        DIRECT_PROMPT.format(question=question),
        temperature=0.0,
        max_completion_tokens=256,
    )
    pred_answer = extract_final_answer(raw)
    correctness = predict_correctness(client, question, pred_answer, gold)
    return {
        "raw_output": raw,
        "pred_answer": pred_answer,
        **correctness,
    }


def run_cot(client: LLMClient, question: str, gold: str = "") -> Dict[str, Any]:
    raw = client.generate(
        BASE_SYSTEM,
        COT_PROMPT.format(question=question),
        temperature=0.2,
        max_completion_tokens=1024,
    )
    pred_answer = extract_final_answer(raw)
    correctness = predict_correctness(client, question, pred_answer, gold)
    return {
        "raw_output": raw,
        "pred_answer": pred_answer,
        **correctness,
    }


def run_combined_cot(client: LLMClient, question: str, gold: str = "") -> Dict[str, Any]:
    raw = client.generate(
        BASE_SYSTEM,
        COMBINED_COT_PROMPT.format(question=question),
        temperature=0.2,
        max_completion_tokens=2048,
    )
    pred_answer = extract_final_answer(raw)
    correctness = predict_correctness(client, question, pred_answer, gold)
    return {
        "raw_output": raw,
        "pred_answer": pred_answer,
        **correctness,
    }


# =========================
# Evaluation
# =========================

def evaluate_predictions(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-row scores into a summary.

    The headline `accuracy` is the LLM-as-judge equivalence rate
    (does the prediction mention / match the gold answer?).  Strict-string
    `exact_match` and permissive `contains_match` are kept as references.
    """
    n = len(rows)
    em = sum(r.get("exact_match", 0) for r in rows) / max(n, 1)
    cm = sum(r.get("contains_match", 0) for r in rows) / max(n, 1)
    judge_acc = sum(int(r.get("predicted_correct", 0) == 1) for r in rows) / max(n, 1)

    return {
        "count": n,
        "accuracy": judge_acc,        # ← LLM-judge headline accuracy
        "exact_match": em,            # strict normalized string match (reference)
        "contains_match": cm,         # permissive substring match (reference)
    }


def process_dataset(
    data: List[Dict[str, str]],
    client: LLMClient,
    output_dir: str,
    max_examples: Optional[int] = None,
    seed: int = 0,
    tot_max_depth: int = 3,
    tot_beam_width: int = 2,
    tot_n_candidates: int = 3,
    baselines: Optional[List[str]] = None,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Seeded random sampling — matches run_strategyqa_multi_cot.py's sample_dataset()
    if max_examples is not None and max_examples < len(data):
        rng = random.Random(seed)
        data = rng.sample(data, max_examples)
    else:
        random.seed(seed)

    # Determine which baselines to run
    if baselines is None or "all" in baselines:
        run_set = {"direct", "cot", "combined_cot", "tree_of_thoughts"}
    else:
        run_set = set(baselines)

    print(f"Running baselines: {', '.join(sorted(run_set))}\n")

    tot_solver = None
    if "tree_of_thoughts" in run_set:
        tot_solver = TreeOfThoughtSolver(
            client=client,
            max_depth=tot_max_depth,
            beam_width=tot_beam_width,
            n_candidates=tot_n_candidates,
        )

    direct_rows = []
    cot_rows = []
    combined_cot_rows = []
    tot_rows = []

    for idx, ex in enumerate(data):
        question = ex["question"]
        gold = ex["gold_answer"]

        if "direct" in run_set:
            print(f"[{idx+1}/{len(data)}] Running direct baseline...")
            direct = run_direct(client, question, gold)
            direct_row = {
                "id": idx,
                "question": question,
                "gold_answer": gold,
                **direct,
            }
            direct_row["exact_match"] = exact_match(direct_row["pred_answer"], gold)
            direct_row["contains_match"] = contains_match(direct_row["pred_answer"], gold)
            direct_rows.append(direct_row)

        if "cot" in run_set:
            print(f"[{idx+1}/{len(data)}] Running CoT baseline...")
            cot = run_cot(client, question, gold)
            cot_row = {
                "id": idx,
                "question": question,
                "gold_answer": gold,
                **cot,
            }
            cot_row["exact_match"] = exact_match(cot_row["pred_answer"], gold)
            cot_row["contains_match"] = contains_match(cot_row["pred_answer"], gold)
            cot_rows.append(cot_row)

        if "combined_cot" in run_set:
            print(f"[{idx+1}/{len(data)}] Running Combined CoT baseline...")
            combined_cot = run_combined_cot(client, question, gold)
            combined_cot_row = {
                "id": idx,
                "question": question,
                "gold_answer": gold,
                **combined_cot,
            }
            combined_cot_row["exact_match"] = exact_match(combined_cot_row["pred_answer"], gold)
            combined_cot_row["contains_match"] = contains_match(combined_cot_row["pred_answer"], gold)
            combined_cot_rows.append(combined_cot_row)

        if "tree_of_thoughts" in run_set:
            print(f"[{idx+1}/{len(data)}] Running ToT baseline...")
            tot = tot_solver.solve(question, gold)
            tot_row = {
                "id": idx,
                "question": question,
                "gold_answer": gold,
                **tot,
            }
            tot_row["exact_match"] = exact_match(tot_row["pred_answer"], gold)
            tot_row["contains_match"] = contains_match(tot_row["pred_answer"], gold)
            tot_rows.append(tot_row)

        # Checkpoint after each example
        if direct_rows:
            write_jsonl(output_path / "direct_predictions.jsonl", direct_rows)
        if cot_rows:
            write_jsonl(output_path / "cot_predictions.jsonl", cot_rows)
        if combined_cot_rows:
            write_jsonl(output_path / "combined_cot_predictions.jsonl", combined_cot_rows)
        if tot_rows:
            write_jsonl(output_path / "tot_predictions.jsonl", tot_rows)

    summary = {}
    if direct_rows:
        summary["direct"] = evaluate_predictions(direct_rows)
    if cot_rows:
        summary["cot"] = evaluate_predictions(cot_rows)
    if combined_cot_rows:
        summary["combined_cot"] = evaluate_predictions(combined_cot_rows)
    if tot_rows:
        summary["tree_of_thoughts"] = evaluate_predictions(tot_rows)

    summary["config"] = {
        "num_examples": len(data),
        "model": client.model,
        "provider": client.provider,
        "baselines": sorted(run_set),
        "tot_max_depth": tot_max_depth,
        "tot_beam_width": tot_beam_width,
        "tot_n_candidates": tot_n_candidates,
    }

    write_json(output_path / "summary.json", summary)
    print("\nDone.")
    print(json.dumps(summary, indent=2))


# =========================
# CLI
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input JSON file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs.")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model alias or full model name.")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_completion_tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--retry_sleep", type=float, default=2.0)
    parser.add_argument("--tot_max_depth", type=int, default=3)
    parser.add_argument("--tot_beam_width", type=int, default=2)
    parser.add_argument("--tot_n_candidates", type=int, default=3)
    parser.add_argument(
        "--baselines",
        type=str,
        nargs="+",
        default=["all"],
        choices=["all", "direct", "cot", "combined_cot", "tree_of_thoughts"],
        help="Which baselines to run (default: all)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data = load_examples(args.input)
    client = LLMClient(
        model=args.model,
        config=GenerationConfig(
            temperature=args.temperature,
            max_completion_tokens=args.max_completion_tokens,
            timeout=args.timeout,
            retries=args.retries,
            retry_sleep=args.retry_sleep,
        ),
    )

    process_dataset(
        data=data,
        client=client,
        output_dir=args.output_dir,
        max_examples=args.max_examples,
        seed=args.seed,
        tot_max_depth=args.tot_max_depth,
        tot_beam_width=args.tot_beam_width,
        tot_n_candidates=args.tot_n_candidates,
        baselines=args.baselines,
    )


if __name__ == "__main__":
    main()