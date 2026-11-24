#!/usr/bin/env python3
"""Convert a BBH-MCQ-Hard subtask from HuggingFace into the formats used by
``run_qasc_multi_cot.py`` and ``baseline/reasoning_baselines.py``.

Source dataset: https://huggingface.co/datasets/Joschka/big_bench_hard

The repo provides a clean schema for every MCQ subtask:

    question : str        # stem only (no options inline)
    choices  : {label: [str], text: [str]}
                          # labels look like 'A)', 'B)', ... (trailing paren)
    target   : str        # bare letter, e.g. 'A'

We strip the trailing ')' from labels and emit two files per subtask under
``<out_dir>/<task>/``:

    multi_cot.json   list of {qid, question, choices, choice_labels,
                              answerKey, answer_text}    (QASC-style)
    baseline.json    list of {qid, question, gold answer}
                              (question contains the options inline;
                               gold answer is the LETTER)

Usage:
    python experiments/data_prep/prep_bbh.py \\
        --task logical_deduction_seven_objects \\
        --out_dir bbh_data
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import List


def clean_label(lbl: str) -> str:
    """Strip surrounding parens/brackets from a BBH choice label.

    BBH stores labels like ``'A)'`` — we want just ``'A'`` so it lines up
    with the parser's expectation that the section header reads
    ``## For A. <text>:``.
    """
    s = lbl.strip()
    s = s.lstrip("([").rstrip(").]:")
    return s.strip()


def parse_target_letter(target: str, valid_labels: List[str]) -> str:
    """Pull a single letter out of ``target``, validating against label set."""
    s = target.strip()
    # Try exact-match first, then strip any wrapping like '(A)' / 'A)' / 'A.'
    for cand in (s, clean_label(s)):
        m = re.match(r"^[A-Z]$", cand)
        if m and cand in valid_labels:
            return cand
    raise ValueError(f"target {target!r} not in labels {valid_labels}")


def render_question_with_options(stem: str, labels: List[str], texts: List[str]) -> str:
    block = "\n".join(f"({l}) {t}" for l, t in zip(labels, texts))
    return f"{stem}\nOptions:\n{block}"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True,
                   help="BBH subtask name, e.g. logical_deduction_seven_objects")
    p.add_argument("--out_dir", default="bbh_data",
                   help="Top-level dir; one subdirectory per task")
    p.add_argument("--hf_repo", default="Joschka/big_bench_hard")
    p.add_argument("--split", default=None,
                   help="HF split. If unset, picks the only split exposed for "
                        "this task (Joschka/big_bench_hard names the split "
                        "after the task itself).")
    p.add_argument("--val_size", type=int, default=50,
                   help="Number of validation examples (deterministic via "
                        "--split_seed). Default: 50")
    p.add_argument("--test_size", type=int, default=200,
                   help="Number of test examples (deterministic via "
                        "--split_seed). Default: 200")
    p.add_argument("--split_seed", type=int, default=42,
                   help="RNG seed used to deterministically shuffle examples "
                        "before splitting into val/test. Default: 42")
    args = p.parse_args()

    try:
        from datasets import load_dataset, get_dataset_split_names
    except ImportError:
        print("ERROR: pip install datasets", file=sys.stderr)
        return 1

    candidate_splits = []
    if args.split:
        candidate_splits.append(args.split)
    candidate_splits.append(args.task)
    candidate_splits.append("train")
    try:
        for s in get_dataset_split_names(args.hf_repo, args.task):
            if s not in candidate_splits:
                candidate_splits.append(s)
    except Exception:
        pass

    last_err = None
    ds = None
    for split in candidate_splits:
        try:
            print(f"Loading {args.hf_repo} :: {args.task} (split={split})")
            ds = load_dataset(args.hf_repo, args.task, split=split)
            break
        except ValueError as e:
            last_err = e
            continue
    if ds is None:
        print(f"ERROR: could not load any split. Last error: {last_err}",
              file=sys.stderr)
        return 1
    print(f"  -> {len(ds)} examples")
    print(f"  columns: {ds.column_names}")

    multi_cot: list = []
    baseline: list = []
    skipped = 0
    for i, ex in enumerate(ds):
        stem = ex.get("question") or ex.get("input") or ""
        choices_raw = ex.get("choices")
        target = ex.get("target") or ex.get("answer") or ""

        if isinstance(choices_raw, dict):
            raw_labels = choices_raw.get("label", [])
            texts = choices_raw.get("text", [])
        elif isinstance(choices_raw, list):
            raw_labels = [c.get("label", "") if isinstance(c, dict) else "" for c in choices_raw]
            texts = [c.get("text", "") if isinstance(c, dict) else str(c) for c in choices_raw]
        else:
            raw_labels, texts = [], []

        labels = [clean_label(l) for l in raw_labels]
        if not labels or len(labels) != len(texts):
            print(f"  [skip qid={i}] no/mismatched choices: labels={raw_labels} texts={texts}")
            skipped += 1
            continue

        try:
            gold_letter = parse_target_letter(target, labels)
        except ValueError as e:
            print(f"  [skip qid={i}] {e}")
            skipped += 1
            continue

        try:
            answer_text = texts[labels.index(gold_letter)]
        except ValueError:
            print(f"  [skip qid={i}] gold {gold_letter!r} not in {labels}")
            skipped += 1
            continue

        qid = f"{args.task}-{i:04d}"
        multi_cot.append({
            "qid": qid,
            "question": stem.strip(),
            "choices": texts,
            "choice_labels": labels,
            "answerKey": gold_letter,
            "answer_text": answer_text,
        })
        baseline.append({
            "qid": qid,
            "question": render_question_with_options(stem.strip(), labels, texts),
            "gold answer": gold_letter,
        })

    out_subdir = os.path.join(args.out_dir, args.task)
    os.makedirs(out_subdir, exist_ok=True)

    # ---- Write the FULL files (all kept examples) ----
    mc_path = os.path.join(out_subdir, "multi_cot.json")
    bl_path = os.path.join(out_subdir, "baseline.json")
    with open(mc_path, "w") as f:
        json.dump(multi_cot, f, indent=2, ensure_ascii=False)
    with open(bl_path, "w") as f:
        json.dump(baseline, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(multi_cot)} examples ({skipped} skipped)")
    print(f"  full: {mc_path}")
    print(f"  full: {bl_path}")

    # ---- Deterministic val/test split ----
    # Shuffle a list of indices with random.Random(split_seed), take the
    # first val_size as val and the next test_size as test. Same seed and
    # same source data always produce the same partition.
    import random
    n_total = len(multi_cot)
    val_size = max(0, int(args.val_size))
    test_size = max(0, int(args.test_size))

    if val_size + test_size > n_total:
        print(f"  WARN: val_size({val_size}) + test_size({test_size}) > available "
              f"({n_total}). Truncating: keep {min(val_size, n_total)} val, "
              f"{max(0, n_total - val_size)} test.")
        val_size = min(val_size, n_total)
        test_size = max(0, n_total - val_size)

    rng = random.Random(args.split_seed)
    indices = list(range(n_total))
    rng.shuffle(indices)
    val_idx = indices[:val_size]
    test_idx = indices[val_size:val_size + test_size]

    splits = {
        "val":  val_idx,
        "test": test_idx,
    }
    for split_name, idx_list in splits.items():
        mc_split = [multi_cot[i] for i in idx_list]
        bl_split = [baseline[i]  for i in idx_list]
        mc_p = os.path.join(out_subdir, f"multi_cot_{split_name}.json")
        bl_p = os.path.join(out_subdir, f"baseline_{split_name}.json")
        with open(mc_p, "w") as f:
            json.dump(mc_split, f, indent=2, ensure_ascii=False)
        with open(bl_p, "w") as f:
            json.dump(bl_split, f, indent=2, ensure_ascii=False)
        print(f"  {split_name:>4}: {mc_p}  ({len(mc_split)} examples)")
        print(f"  {split_name:>4}: {bl_p}  ({len(bl_split)} examples)")

    print(f"Split seed: {args.split_seed} (val={val_size}, test={test_size})")
    return 0 if multi_cot else 2  # exit 2 if literally nothing was kept


if __name__ == "__main__":
    raise SystemExit(main())
