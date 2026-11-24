#!/usr/bin/env python3
"""Prep StrategyQA into the bbh_data/<task>/{baseline,multi_cot}_{val,test}.json
schema. Yes/no → 2-way MCQ with labels A=yes, B=no.

  val  = 200 examples sampled deterministically from the HF train split
  test = full HF test split (687 examples)
"""
import json, os, random
from datasets import load_dataset

OUT_DIR = "bbh_data/strategyqa"
TASK    = "strategyqa"
SEED    = 42
N_VAL   = 200

CHOICES       = ["yes", "no"]
CHOICE_LABELS = ["A",   "B"]
def gold_label(ans_bool):
    return "A" if ans_bool else "B"
def gold_text(ans_bool):
    return "yes" if ans_bool else "no"

def to_baseline(ex, qid):
    q = ex["question"].strip()
    body = q + "\nOptions:\n(A) yes\n(B) no"
    return {
        "qid":         qid,
        "question":    body,
        "gold answer": gold_label(ex["answer"]),
    }

def to_multi_cot(ex, qid):
    return {
        "qid":           qid,
        "question":      ex["question"].strip(),
        "choices":       list(CHOICES),
        "choice_labels": list(CHOICE_LABELS),
        "answerKey":     gold_label(ex["answer"]),
        "answer_text":   gold_text(ex["answer"]),
    }

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ds = load_dataset("ChilleD/StrategyQA")
    train = list(ds["train"])
    test  = list(ds["test"])

    # Stratified sample so val matches the yes/no ratio of test.
    test_yes_frac = sum(1 for ex in test if ex["answer"]) / len(test)
    n_val_yes     = round(N_VAL * test_yes_frac)
    n_val_no      = N_VAL - n_val_yes
    yes_pool = [i for i, ex in enumerate(train) if ex["answer"]]
    no_pool  = [i for i, ex in enumerate(train) if not ex["answer"]]
    rng = random.Random(SEED)
    val_idx = sorted(
        rng.sample(yes_pool, n_val_yes) + rng.sample(no_pool, n_val_no)
    )
    val = [train[i] for i in val_idx]
    print(f"  val target ratio: yes={n_val_yes}, no={n_val_no} "
          f"(test yes_frac={test_yes_frac:.3f})")

    splits = {"val": val, "test": test}
    for split, rows in splits.items():
        baseline = []
        multi    = []
        for i, ex in enumerate(rows):
            qid = f"{TASK}-{i:04d}"
            baseline.append(to_baseline(ex, qid))
            multi.append(to_multi_cot(ex, qid))
        with open(os.path.join(OUT_DIR, f"baseline_{split}.json"), "w") as f:
            json.dump(baseline, f, indent=2)
        with open(os.path.join(OUT_DIR, f"multi_cot_{split}.json"), "w") as f:
            json.dump(multi, f, indent=2)
        print(f"  wrote {split}: baseline_{split}.json + multi_cot_{split}.json ({len(rows)} ex)")

    # Convenience: also write split-less aggregates (val+test combined) the way
    # bbh_data has baseline.json / multi_cot.json. Keep order = val then test.
    all_rows = splits["val"] + splits["test"]
    baseline_all = []
    multi_all    = []
    for i, ex in enumerate(all_rows):
        qid = f"{TASK}-{i:04d}"  # renumbered so qids are unique here too
        baseline_all.append(to_baseline(ex, qid))
        multi_all.append(to_multi_cot(ex, qid))
    with open(os.path.join(OUT_DIR, "baseline.json"), "w") as f:
        json.dump(baseline_all, f, indent=2)
    with open(os.path.join(OUT_DIR, "multi_cot.json"), "w") as f:
        json.dump(multi_all, f, indent=2)
    print(f"  wrote combined: baseline.json + multi_cot.json ({len(all_rows)} ex)")

    # Smoke check: print one row from each output
    print()
    print("--- baseline_val[0] ---")
    print(json.dumps(json.load(open(os.path.join(OUT_DIR, "baseline_val.json")))[0], indent=2))
    print("--- multi_cot_val[0] ---")
    print(json.dumps(json.load(open(os.path.join(OUT_DIR, "multi_cot_val.json")))[0], indent=2))
    # Class balance sanity
    for split in ("val", "test"):
        rows = json.load(open(os.path.join(OUT_DIR, f"multi_cot_{split}.json")))
        n_yes = sum(1 for r in rows if r["answerKey"] == "A")
        n_no  = sum(1 for r in rows if r["answerKey"] == "B")
        print(f"  {split}: yes={n_yes}  no={n_no}  total={len(rows)}")

if __name__ == "__main__":
    main()
