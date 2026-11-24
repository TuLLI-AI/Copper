#!/usr/bin/env python3
"""Generate an HTML report from baseline reasoning results.

Usage:
    python baseline/baselines_to_html.py baseline/outputs/

Reads *_predictions.jsonl and summary.json from the output directory,
generates per-example HTML pages and an index page.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List


def _normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\.,!\?;:\"'`\[\]\(\)\{\}]", "", s)
    return s


def _recompute_exact_match(pred: str, gold: str) -> int:
    """Recompute exact match with yes/no prefix handling."""
    p = _normalize_text(pred)
    g = _normalize_text(gold)
    if p == g:
        return 1
    if g in ("yes", "no") and p.startswith(g):
        return 1
    return 0

# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def _esc(text: str) -> str:
    """Escape HTML special characters."""
    if not text:
        return ""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _match_badge(is_match: int) -> str:
    if is_match:
        return '<span style="background:#48bb78;color:white;padding:2px 8px;border-radius:4px;font-size:.85rem;">Correct</span>'
    return '<span style="background:#f56565;color:white;padding:2px 8px;border-radius:4px;font-size:.85rem;">Wrong</span>'


# ---------------------------------------------------------------------------
# Per-example HTML
# ---------------------------------------------------------------------------

BASELINE_COLORS = {
    "direct": "#4299e1",
    "cot": "#48bb78",
    "combined_cot": "#d69e2e",
    "tree_of_thoughts": "#9f7aea",
}

BASELINE_LABELS = {
    "direct": "Direct Reasoning",
    "cot": "Chain-of-Thought",
    "combined_cot": "Combined CoT (both sides)",
    "tree_of_thoughts": "Tree of Thoughts",
}


def build_example_html(example: Dict[str, Any]) -> str:
    """Build HTML for a single example across all baselines."""
    qid = example.get("id", "?")
    question = example.get("question", "")
    gold = example.get("gold_answer", "")

    cards = []
    for baseline_key in ["direct", "cot", "combined_cot", "tree_of_thoughts"]:
        bl = example.get(baseline_key)
        if bl is None:
            continue

        color = BASELINE_COLORS.get(baseline_key, "#718096")
        label = BASELINE_LABELS.get(baseline_key, baseline_key)
        pred = bl.get("pred_answer", "")
        em = bl.get("exact_match", 0)
        raw = bl.get("raw_output", "")

        # Extra info for ToT
        extra = ""
        if baseline_key == "tree_of_thoughts":
            best_path = bl.get("best_path", [])
            best_score = bl.get("best_score", "?")
            if best_path:
                path_html = "\n".join(
                    f"  {i+1}. {_esc(step)}" for i, step in enumerate(best_path)
                )
                extra = f"""
  <h4 style="margin-top:10px;">Best Path (score={best_score})</h4>
  <pre style="background:#f7fafc;padding:10px;border-radius:6px;font-size:.82rem;white-space:pre-wrap;">{path_html}</pre>"""

        cards.append(f"""
<div class="card" style="border-left: 4px solid {color}; margin-bottom: 16px;">
  <h3 style="margin-top:0;color:{color};">{_esc(label)} {_match_badge(em)}</h3>
  <p><strong>Predicted:</strong> {_esc(pred)} &nbsp; <strong>Gold:</strong> {_esc(gold)}</p>{extra}
  <details>
    <summary style="cursor:pointer;color:#4a5568;font-size:.9rem;">Raw LLM Output</summary>
    <pre style="background:#1a202c;color:#e2e8f0;padding:12px;border-radius:6px;font-size:.82rem;line-height:1.5;white-space:pre-wrap;word-wrap:break-word;margin-top:8px;">{_esc(raw)}</pre>
  </details>
</div>""")

    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Baseline #{qid}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f7fafc; color: #2d3748; }}
  .card {{ background: white; border-radius: 8px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,.1); }}
  a {{ color: #4299e1; }}
</style></head><body>
<p><a href="index.html">&larr; Back to index</a></p>
<h1>Example #{qid}</h1>
<div class="card" style="margin-bottom:16px;">
  <h2 style="margin-top:0;">Question</h2>
  <p style="font-size:1.1rem;">{_esc(question)}</p>
  <p><strong>Gold answer:</strong> {_esc(gold)}</p>
</div>
{"".join(cards)}
</body></html>"""


# ---------------------------------------------------------------------------
# Index HTML
# ---------------------------------------------------------------------------

def build_index_html(examples: List[Dict[str, Any]], summary: Dict[str, Any], html_dir: str) -> str:
    """Build the index HTML page."""

    # Summary table
    summary_rows = []
    for bl_key in ["direct", "cot", "combined_cot", "tree_of_thoughts"]:
        stats = summary.get(bl_key)
        if stats is None:
            continue
        label = BASELINE_LABELS.get(bl_key, bl_key)
        em = stats.get("exact_match", 0)
        n = stats.get("count", 0)
        correct = int(round(em * n))
        summary_rows.append(
            f"<tr><td>{_esc(label)}</td>"
            f"<td>{correct}/{n} ({em*100:.1f}%)</td>"
            f"<td>{stats.get('contains_match', 0)*100:.1f}%</td></tr>"
        )

    summary_table = f"""
<table style="width:100%;border-collapse:collapse;margin-bottom:20px;">
  <thead>
    <tr style="background:#edf2f7;text-align:left;">
      <th style="padding:8px 12px;">Baseline</th>
      <th style="padding:8px 12px;">Exact Match</th>
      <th style="padding:8px 12px;">Contains Match</th>
    </tr>
  </thead>
  <tbody>{"".join(summary_rows)}</tbody>
</table>"""

    # Per-example rows
    example_rows = []
    for ex in examples:
        qid = ex.get("id", "?")
        question = ex.get("question", "")[:80]
        gold = ex.get("gold_answer", "")

        cells = []
        for bl_key in ["direct", "cot", "combined_cot", "tree_of_thoughts"]:
            bl = ex.get(bl_key)
            if bl is None:
                cells.append("<td>-</td>")
                continue
            em = bl.get("exact_match", 0)
            pred = bl.get("pred_answer", "")[:20]
            color = "#48bb78" if em else "#f56565"
            cells.append(f'<td style="color:{color};">{_esc(pred)}</td>')

        example_rows.append(
            f'<tr><td><a href="{qid}.html">#{qid}</a></td>'
            f'<td>{_esc(question)}...</td>'
            f'<td>{_esc(gold)}</td>'
            f'{"".join(cells)}</tr>'
        )

    config = summary.get("config", {})

    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Baseline Results</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f7fafc; color: #2d3748; }}
  .card {{ background: white; border-radius: 8px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,.1); margin-bottom: 16px; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th, td {{ padding: 6px 10px; border-bottom: 1px solid #e2e8f0; text-align: left; font-size: .85rem; }}
  a {{ color: #4299e1; }}
</style></head><body>
<h1>Baseline Reasoning Results</h1>
<p>Model: <strong>{_esc(config.get('model', '?'))}</strong> &nbsp;|&nbsp;
   Examples: <strong>{config.get('num_examples', '?')}</strong> &nbsp;|&nbsp;
   Baselines: {_esc(', '.join(config.get('baselines', [])))}</p>

<div class="card">
  <h2 style="margin-top:0;">Summary</h2>
  {summary_table}
</div>

<div class="card">
  <h2 style="margin-top:0;">Per-Example Results</h2>
  <table>
    <thead>
      <tr style="background:#edf2f7;">
        <th>ID</th><th>Question</th><th>Gold</th>
        <th>Direct</th><th>CoT</th><th>Combined CoT</th><th>ToT</th>
      </tr>
    </thead>
    <tbody>{"".join(example_rows)}</tbody>
  </table>
</div>
</body></html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_combined_predictions(output_dir: str) -> List[Dict[str, Any]]:
    """Load combined_predictions.json if it exists."""
    path = os.path.join(output_dir, "combined_predictions.json")
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)

    # Fallback: load individual jsonl files and merge by id
    baselines = {}
    for bl_key, fname in [
        ("direct", "direct_predictions.jsonl"),
        ("cot", "cot_predictions.jsonl"),
        ("combined_cot", "combined_cot_predictions.jsonl"),
        ("tree_of_thoughts", "tot_predictions.jsonl"),
    ]:
        fpath = os.path.join(output_dir, fname)
        if not os.path.isfile(fpath):
            continue
        rows = []
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        baselines[bl_key] = {r["id"]: r for r in rows}

    # Merge
    all_ids = set()
    for bl_rows in baselines.values():
        all_ids.update(bl_rows.keys())

    examples = []
    for idx in sorted(all_ids):
        ex = {"id": idx}
        for bl_key, bl_rows in baselines.items():
            row = bl_rows.get(idx)
            if row:
                ex["question"] = row.get("question", ex.get("question", ""))
                ex["gold_answer"] = row.get("gold_answer", ex.get("gold_answer", ""))
                ex[bl_key] = {
                    k: v for k, v in row.items()
                    if k not in ("id", "question", "gold_answer")
                }
        examples.append(ex)

    return examples


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(f"Usage: python {sys.argv[0]} <output_dir>")
        sys.exit(1)

    output_dir = sys.argv[1]
    html_dir = os.path.join(output_dir, "html")
    os.makedirs(html_dir, exist_ok=True)

    # Load summary
    summary_path = os.path.join(output_dir, "summary.json")
    summary = {}
    if os.path.isfile(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)

    # Load examples
    examples = load_combined_predictions(output_dir)
    print(f"Loaded {len(examples)} examples from {output_dir}")

    # Recompute exact_match with yes/no prefix fix
    for ex in examples:
        gold = ex.get("gold_answer", "")
        for bl_key in ["direct", "cot", "combined_cot", "tree_of_thoughts"]:
            bl = ex.get(bl_key)
            if bl and "pred_answer" in bl:
                bl["exact_match"] = _recompute_exact_match(bl["pred_answer"], gold)

    # Recompute summary stats
    for bl_key in ["direct", "cot", "combined_cot", "tree_of_thoughts"]:
        rows = [ex[bl_key] for ex in examples if bl_key in ex]
        if rows:
            n = len(rows)
            em = sum(r.get("exact_match", 0) for r in rows)
            summary[bl_key] = summary.get(bl_key, {})
            summary[bl_key]["count"] = n
            summary[bl_key]["exact_match"] = em / n
            print(f"  {bl_key}: {em}/{n} ({em/n*100:.1f}%)")

    # Generate per-example HTML
    errors = 0
    for ex in examples:
        qid = ex.get("id", "unknown")
        try:
            html = build_example_html(ex)
            with open(os.path.join(html_dir, f"{qid}.html"), "w") as f:
                f.write(html)
        except Exception as e:
            errors += 1
            print(f"  Error on #{qid}: {e}")

    # Generate index
    idx_html = build_index_html(examples, summary, html_dir)
    with open(os.path.join(html_dir, "index.html"), "w") as f:
        f.write(idx_html)

    print(f"Generated {len(examples)} HTML pages + index (errors={errors})")
    print(f"Open: {os.path.join(html_dir, 'index.html')}")


if __name__ == "__main__":
    main()
