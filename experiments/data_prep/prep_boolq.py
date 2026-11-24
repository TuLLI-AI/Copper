#!/usr/bin/env python3
"""Prep BoolQ into the bbh_data/<task>/{baseline,multi_cot}_{val,test}.json
schema. Yes/no → 2-way MCQ with labels A=yes, B=no.

  val  = 200 examples sampled from HF train split (seed=42)
  test = 500 examples sampled from HF validation split (seed=42)

BoolQ questions need the passage to be answerable, so we embed the passage
into the question text:
   Passage: <passage>

   Question: <question>?
"""
import json, os, random
from datasets import load_dataset

OUT_DIR = "bbh_data/boolq"
TASK    = "boolq"
SEED    = 42
N_VAL   = 200
N_TEST  = 500

CHOICES       = ["yes", "no"]
CHOICE_LABELS = ["A",   "B"]
def gold_label(ans_bool): return "A" if ans_bool else "B"
def gold_text(ans_bool):  return "yes" if ans_bool else "no"

def _format_q(ex):
    """Embed passage so the question is self-contained.

    BoolQ questions are statement-style ('does ethanol take more energy...');
    add a trailing '?' if missing for cleanliness.
    """
    q = ex["question"].strip()
    if not q.endswith("?"):
        q = q + "?"
    return f"Passage: {ex['passage'].strip()}\n\nQuestion: {q}"

def to_baseline(ex, qid):
    body = _format_q(ex) + "\nOptions:\n(A) yes\n(B) no"
    return {
        "qid":         qid,
        "question":    body,
        "gold answer": gold_label(ex["answer"]),
    }

def to_multi_cot(ex, qid):
    return {
        "qid":           qid,
        "question":      _format_q(ex),
        "choices":       list(CHOICES),
        "choice_labels": list(CHOICE_LABELS),
        "answerKey":     gold_label(ex["answer"]),
        "answer_text":   gold_text(ex["answer"]),
    }

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ds = load_dataset("google/boolq")
    train = list(ds["train"])
    valid = list(ds["validation"])

    rng = random.Random(SEED)
    val_idx  = sorted(rng.sample(range(len(train)), N_VAL))
    test_idx = sorted(rng.sample(range(len(valid)), N_TEST))
    val      = [train[i] for i in val_idx]
    test     = [valid[i] for i in test_idx]

    splits = {"val": val, "test": test}
    for split, rows in splits.items():
        baseline, multi = [], []
        for i, ex in enumerate(rows):
            qid = f"{TASK}-{i:04d}"
            baseline.append(to_baseline(ex, qid))
            multi.append(to_multi_cot(ex, qid))
        with open(os.path.join(OUT_DIR, f"baseline_{split}.json"), "w") as f:
            json.dump(baseline, f, indent=2)
        with open(os.path.join(OUT_DIR, f"multi_cot_{split}.json"), "w") as f:
            json.dump(multi, f, indent=2)
        print(f"  wrote {split}: baseline_{split}.json + multi_cot_{split}.json ({len(rows)} ex)")

    # combined val+test
    all_rows = splits["val"] + splits["test"]
    baseline_all, multi_all = [], []
    for i, ex in enumerate(all_rows):
        qid = f"{TASK}-{i:04d}"
        baseline_all.append(to_baseline(ex, qid))
        multi_all.append(to_multi_cot(ex, qid))
    with open(os.path.join(OUT_DIR, "baseline.json"), "w") as f:
        json.dump(baseline_all, f, indent=2)
    with open(os.path.join(OUT_DIR, "multi_cot.json"), "w") as f:
        json.dump(multi_all, f, indent=2)
    print(f"  wrote combined: baseline.json + multi_cot.json ({len(all_rows)} ex)")

    # Smoke check
    print()
    print("--- baseline_val[0] (truncated) ---")
    row = json.load(open(os.path.join(OUT_DIR, "baseline_val.json")))[0]
    row_show = {**row, "question": row["question"][:400] + ("..." if len(row["question"]) > 400 else "")}
    print(json.dumps(row_show, indent=2))
    print("--- multi_cot_val[0] (truncated) ---")
    row = json.load(open(os.path.join(OUT_DIR, "multi_cot_val.json")))[0]
    row_show = {**row, "question": row["question"][:400] + ("..." if len(row["question"]) > 400 else "")}
    print(json.dumps(row_show, indent=2))
    for split in ("val", "test"):
        rows = json.load(open(os.path.join(OUT_DIR, f"multi_cot_{split}.json")))
        n_yes = sum(1 for r in rows if r["answerKey"] == "A")
        n_no  = sum(1 for r in rows if r["answerKey"] == "B")
        print(f"  {split}: yes={n_yes}  no={n_no}  total={len(rows)}")

if __name__ == "__main__":
    main()
