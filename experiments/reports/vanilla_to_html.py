"""Render vanilla_cot checkpoints as readable HTML.

Companion to ``json_to_html.py`` (which targets the run_BBH_multi_cot
schema with ``per_answer_traces`` and ``modes``). This version targets
the flatter vanilla_cot schema where ``program`` / ``sample`` /
``inference`` / ``gradient`` / ``raw_cot`` / ``cot_logprobs`` /
``cot_prompt`` live at the top level of each ckpt.

Modes
-----

  Single ckpt:
      python experiments/reports/vanilla_to_html.py path/to/<qid>.json

  Cell directory (``<cell>/ckpts/`` + ``summary.json``):
      python experiments/reports/vanilla_to_html.py path/to/<task>_<split>

  Sweep root (one or more <model>/<task>_<split> cells):
      python experiments/reports/vanilla_to_html.py path/to/vanilla_cot --top

  Add ``--open`` to launch the resulting page in the default browser.

Output placement:
  - single ckpt → ``<basename>.html`` next to the .json (or ``--out``)
  - cell dir    → ``<cell>/html/index.html`` + ``<cell>/html/ckpts/*.html``
  - sweep root  → ``<root>/index.html`` linking to each cell's index
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
import webbrowser
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------

_CSS = """
:root { --fg:#1f2328; --muted:#6e7781; --bg:#ffffff; --code-bg:#f6f8fa;
        --border:#d0d7de; --accent:#0969da; --good:#1a7f37; --bad:#cf222e;
        --warn:#9a6700; }
* { box-sizing: border-box; }
body { font: 14px/1.45 -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       color: var(--fg); background: var(--bg); margin: 0; padding: 24px;
       max-width: 1200px; margin-left: auto; margin-right: auto; }
h1 { font-size: 22px; margin: 0 0 8px; }
h2 { font-size: 17px; margin: 22px 0 8px; border-bottom: 1px solid var(--border);
     padding-bottom: 4px; }
h3 { font-size: 14px; color: var(--muted); margin: 12px 0 6px; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
.muted { color: var(--muted); }
.good  { color: var(--good); font-weight: 600; }
.bad   { color: var(--bad);  font-weight: 600; }
.warn  { color: var(--warn); }
dl.kv { display: grid; grid-template-columns: max-content auto;
        column-gap: 14px; row-gap: 4px; margin: 8px 0 14px; }
dl.kv dt { color: var(--muted); }
dl.kv dd { margin: 0; }
table { border-collapse: collapse; margin: 8px 0 16px; width: 100%; }
th, td { border: 1px solid var(--border); padding: 6px 10px; text-align: left;
         vertical-align: top; }
th { background: var(--code-bg); font-weight: 600; }
tr:hover td { background: #fafbfc; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
pre { background: var(--code-bg); border: 1px solid var(--border);
      border-radius: 6px; padding: 10px 12px; overflow-x: auto;
      font: 12.5px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace;
      white-space: pre-wrap; word-wrap: break-word; }
code { background: var(--code-bg); padding: 1px 5px; border-radius: 4px;
       font: 12px ui-monospace, SFMono-Regular, Menlo, monospace; }
.choice { display: inline-block; padding: 2px 8px; margin: 2px 4px 2px 0;
          border: 1px solid var(--border); border-radius: 12px;
          background: var(--code-bg); font-size: 12.5px; }
.choice.gold      { border-color: var(--good); background: #dafbe1; }
.choice.predicted { border-color: var(--accent); background: #ddf4ff; }
.choice.both      { border-color: var(--good); background: #dafbe1;
                    box-shadow: 0 0 0 2px var(--accent) inset; }
details { margin: 8px 0; }
summary { cursor: pointer; user-select: none; color: var(--accent);
          font-weight: 600; }
.bar { display: inline-block; height: 10px; background: var(--accent);
       border-radius: 2px; vertical-align: middle; margin-left: 4px; }
.error-banner { background: #ffebe9; border: 1px solid var(--bad);
                color: #82061a; padding: 8px 12px; border-radius: 6px;
                margin: 12px 0; }
.scroll { max-height: 320px; overflow-y: auto; }
.token { display: inline-block; padding: 1px 4px; margin: 1px;
         border-radius: 3px; font: 11.5px ui-monospace; }
"""


def _esc(s: Any) -> str:
    return html.escape(str(s) if s is not None else "")


def _truncate(s: Optional[str], n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[: n - 1] + "…"


def _is_correct(d: dict) -> Optional[bool]:
    """True if the prediction matches gold by either letter or text.
    None when there's no gold / no prediction (so we don't paint
    everything red on test sets that lack labels)."""
    gold_l = (d.get("answer_key") or "").strip().upper()
    gold_t = (d.get("answer_text") or "").strip().lower()
    pred_l = (d.get("predicted_label") or "").strip().upper()
    pred_t = (d.get("predicted_text")  or "").strip().lower()
    if not (gold_l or gold_t):
        return None
    if not (pred_l or pred_t):
        return False
    if pred_l and gold_l and pred_l == gold_l:
        return True
    if pred_t and gold_t and pred_t == gold_t:
        return True
    return False


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _render_choices(d: dict) -> str:
    choices = d.get("choices") or []
    labels  = d.get("choice_labels") or []
    gold_l  = (d.get("answer_key") or "").strip().upper()
    gold_t  = (d.get("answer_text") or "").strip().lower()
    pred_l  = (d.get("predicted_label") or "").strip().upper()
    pred_t  = (d.get("predicted_text")  or "").strip().lower()
    chips = []
    for lab, txt in zip(labels, choices):
        is_gold = (lab.strip().upper() == gold_l) or (txt.strip().lower() == gold_t)
        is_pred = (lab.strip().upper() == pred_l) or (txt.strip().lower() == pred_t)
        klass = "choice"
        if is_gold and is_pred: klass += " both"
        elif is_gold: klass += " gold"
        elif is_pred: klass += " predicted"
        chips.append(f'<span class="{klass}"><b>{_esc(lab)}.</b> {_esc(txt)}</span>')
    return "\n".join(chips) or '<span class="muted">(none)</span>'


def _render_distribution(dist: Optional[dict]) -> str:
    if not dist:
        return '<span class="muted">(none)</span>'
    rows = []
    for k, v in dist.items():
        try: w = float(v)
        except: w = 0.0
        bar_px = int(round(140 * max(0.0, min(1.0, w))))
        rows.append(f"<tr><td>{_esc(k)}</td>"
                    f"<td class='num'>{w:.4f}</td>"
                    f'<td><span class="bar" style="width:{bar_px}px"></span></td></tr>')
    return f"<table><tr><th>label</th><th>prob</th><th></th></tr>{''.join(rows)}</table>"


def _render_steps_table(steps: List[dict]) -> str:
    if not steps:
        return '<span class="muted">(none)</span>'
    head = ["#","claim","answer","premises","formula","weight","confidence"]
    rows = []
    for s in steps:
        rows.append(
            "<tr>"
            f"<td>{_esc(s.get('prefix'))}</td>"
            f"<td>{_esc(_truncate(s.get('claim') or '', 200))}</td>"
            f"<td>{_esc(s.get('answer') or '')}</td>"
            f"<td><code>{_esc(s.get('premises'))}</code></td>"
            f"<td><code>{_esc(s.get('formula'))}</code></td>"
            f"<td class='num'>{_esc(s.get('weight'))}</td>"
            f"<td class='num'>{_esc(s.get('confidence'))}</td>"
            "</tr>"
        )
    return ("<table><tr>" + "".join(f"<th>{h}</th>" for h in head)
            + "</tr>" + "".join(rows) + "</table>")


def _render_sample(sample: Optional[dict]) -> str:
    if not sample:
        return '<span class="muted">(none)</span>'
    rv = sample.get("return_values") or {}
    atts = sample.get("attempts"); execd = sample.get("executed")
    head = f"<p class='muted'>attempts={_esc(atts)}, executed={_esc(execd)}</p>"
    if not rv:
        return head + '<span class="muted">(no return values)</span>'
    total = sum(float(v) for v in rv.values()) or 1.0
    rows = []
    for k, v in sorted(rv.items(), key=lambda kv: -float(kv[1])):
        c = int(float(v))
        frac = c / total
        bar = int(round(140 * frac))
        rows.append(f"<tr><td>{_esc(k)}</td>"
                    f"<td class='num'>{c}</td>"
                    f"<td class='num'>{frac*100:.1f}%</td>"
                    f"<td><span class='bar' style='width:{bar}px'></span></td></tr>")
    return head + ("<table><tr><th>return value</th><th>count</th>"
                   "<th>frac</th><th></th></tr>" + "".join(rows) + "</table>")


def _render_inference(inf: Optional[dict]) -> str:
    if not inf:
        return '<span class="muted">(none)</span>'
    rows = []
    for k, v in inf.items():
        if isinstance(v, dict) and isinstance(v.get("prob"), (int, float)):
            rows.append(f"<tr><td>{_esc(k)}</td>"
                        f"<td class='num'>{v['prob']:.4g}</td>"
                        f"<td class='num'>{v.get('Z')}</td>"
                        f"<td>{_esc(v.get('error') or '')}</td></tr>")
        else:
            rows.append(f"<tr><td>{_esc(k)}</td><td colspan=3>{_esc(v)}</td></tr>")
    return ("<table><tr><th>return value</th><th>prob</th><th>Z</th>"
            "<th>error</th></tr>" + "".join(rows) + "</table>")


def _render_gradient(g: Optional[dict]) -> str:
    if not g:
        return '<span class="muted">(none — likely no return-values were emitted)</span>'
    return f"<pre>{_esc(json.dumps(g, indent=2, default=str))}</pre>"


def _render_logprobs(lp: Optional[List[dict]]) -> str:
    if not lp:
        return '<span class="muted">(none)</span>'
    spans = []
    import math
    for tok in lp:
        t = (tok.get("token") or "").replace("\n", "↵\n")
        try:
            logp = float(tok.get("logprob"))
        except Exception:
            logp = 0.0
        clamped = max(-12.0, min(0.0, logp))
        # Lightness: 100% (high prob) → 80% (low prob)
        lightness = 100 + int(round(20 * (clamped / 12.0)))
        title = f"logp={logp:.4f}  p={math.exp(logp):.3g}"
        spans.append(
            f'<span class="token" title="{_esc(title)}" '
            f'style="background:hsl(50,100%,{lightness}%)">{_esc(t)}</span>'
        )
    return f"<div class='scroll'>{''.join(spans)}</div>"


# ---------------------------------------------------------------------------
# Page templates
# ---------------------------------------------------------------------------

def render_ckpt(d: dict, *, title: Optional[str] = None,
                back_link: Optional[str] = None) -> str:
    qid    = d.get("qid", "?")
    cfg    = d.get("config") or {}
    model  = cfg.get("model_name", "?")
    strat  = cfg.get("cot_strategy", "?")
    qtype  = cfg.get("question_type", "?")
    err    = d.get("error")
    correct = _is_correct(d)

    parts: List[str] = []
    if back_link:
        parts.append(f'<div class="muted"><a href="{_esc(back_link)}">← back to index</a></div>')
    parts.append(f"<h1>{_esc(qid)}</h1>")
    parts.append("<dl class='kv'>")
    parts.append(f"<dt>model</dt><dd><code>{_esc(model)}</code></dd>")
    parts.append(f"<dt>strategy</dt><dd><code>{_esc(strat)}</code></dd>")
    parts.append(f"<dt>question type</dt><dd><code>{_esc(qtype)}</code></dd>")
    parts.append(f"<dt>gold</dt><dd>{_esc(d.get('answer_key') or '–')} "
                 f"<span class='muted'>({_esc(d.get('answer_text') or '')})</span></dd>")
    parts.append(f"<dt>predicted</dt><dd>{_esc(d.get('predicted_label') or '–')} "
                 f"<span class='muted'>({_esc(d.get('predicted_text') or '')})</span></dd>")
    if correct is True:
        parts.append("<dt>correct?</dt><dd><span class='good'>YES</span></dd>")
    elif correct is False:
        parts.append("<dt>correct?</dt><dd><span class='bad'>no</span></dd>")
    else:
        parts.append("<dt>correct?</dt><dd><span class='muted'>(no gold)</span></dd>")
    parts.append("</dl>")

    if err:
        parts.append(f'<div class="error-banner"><b>error:</b><br><pre>{_esc(err)}</pre></div>')

    parts.append("<h2>Question</h2>")
    parts.append(f"<pre>{_esc(d.get('question') or '')}</pre>")
    parts.append("<h2>Choices</h2>")
    parts.append(_render_choices(d))

    parts.append("<h2>Predicted distribution</h2>")
    parts.append(_render_distribution(d.get("predicted_distribution")))

    parts.append("<h2>Raw CoT</h2>")
    parts.append(f"<pre>{_esc(d.get('raw_cot') or '')}</pre>")

    if d.get("steps"):
        parts.append("<h2>Parsed steps</h2>")
        parts.append(_render_steps_table(d.get("steps")))

    if d.get("program"):
        parts.append("<h2>Compiled program</h2>")
        parts.append(f"<pre>{_esc(d.get('program') or '')}</pre>")

    if d.get("sample"):
        parts.append("<h2>Sample histogram</h2>")
        parts.append(_render_sample(d.get("sample")))

    if d.get("inference"):
        parts.append("<h2>Inference</h2>")
        parts.append(_render_inference(d.get("inference")))

    if "gradient" in d:
        parts.append("<h2>Gradient</h2>")
        parts.append(_render_gradient(d.get("gradient")))

    if d.get("semantics"):
        parts.append("<h2>Program semantics</h2>")
        parts.append(f"<pre>{_esc(d.get('semantics'))}</pre>")

    if d.get("circuit"):
        parts.append("<h2>Compiled circuit</h2>")
        parts.append(f"<pre>{_esc(json.dumps(d.get('circuit'), indent=2, default=str))}</pre>")

    if d.get("cot_prompt"):
        parts.append("<h2>Prompt &amp; response</h2>")
        cp = d["cot_prompt"]
        parts.append("<details><summary>system prompt</summary>"
                     f"<pre>{_esc(cp.get('system') or '')}</pre></details>")
        parts.append("<details><summary>user message</summary>"
                     f"<pre>{_esc(cp.get('user') or '')}</pre></details>")
        parts.append("<details><summary>raw response</summary>"
                     f"<pre>{_esc(cp.get('response') or '')}</pre></details>")

    if d.get("cot_logprobs"):
        parts.append("<h2>Per-token logprobs</h2>")
        parts.append(
            "<details><summary>show tokens (color-coded by logprob)</summary>"
            f"{_render_logprobs(d.get('cot_logprobs'))}</details>")

    if d.get("attempts") and int(d.get("attempts") or 1) > 1:
        parts.append(f"<p class='warn'>parsed in {_esc(d.get('attempts'))} attempts</p>")
    if d.get("errors"):
        parts.append("<details><summary>parser retry errors</summary>"
                     f"<pre>{_esc(json.dumps(d.get('errors'), indent=2))}</pre></details>")

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{_esc(title or qid)}</title>
<style>{_CSS}</style></head><body>{''.join(parts)}</body></html>"""


def render_cell_index(cell_dir: str, html_dir: str, ckpt_pages: List[tuple],
                      summary_json: Optional[dict]) -> str:
    name = os.path.relpath(cell_dir).replace("/", " / ")
    rows = []
    for fname, d in ckpt_pages:
        c = _is_correct(d)
        if d.get("error"):
            klass = "bad"; marker = "✗"
        elif c is True:
            klass = "good"; marker = "✓"
        elif c is False:
            klass = "bad";  marker = "✗"
        else:
            klass = "muted"; marker = "·"
        err_short = ""
        if d.get("error"):
            err_short = (d["error"] or "").strip().split("\n")[-1][:120]
        question_short = _truncate((d.get("question") or "").replace("\n", " "), 140)
        rows.append(
            "<tr>"
            f"<td><a href='ckpts/{_esc(fname)}'>{_esc(d.get('qid','?'))}</a></td>"
            f"<td class='{klass}'>{marker}</td>"
            f"<td>{_esc(d.get('answer_key') or '')}</td>"
            f"<td>{_esc(d.get('predicted_label') or '')}</td>"
            f"<td>{_esc(question_short)}</td>"
            f"<td class='muted'>{_esc(err_short)}</td>"
            "</tr>"
        )

    summ_html = ""
    if summary_json:
        v = summary_json.get("vanilla") or summary_json
        summ_html = (
            "<dl class='kv'>"
            f"<dt>accuracy</dt><dd><b>{(v.get('accuracy') or 0)*100:.2f}%</b> "
            f"({v.get('n_correct')}/{v.get('n_total')})</dd>"
            f"<dt>predicted</dt><dd>{v.get('n_predicted')}</dd>"
            f"<dt>errored</dt><dd>{v.get('n_errored')}</dd>"
            f"<dt>missing</dt><dd>{v.get('n_missing')}</dd>"
            "</dl>"
        )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{_esc(name)}</title>
<style>{_CSS}</style></head><body>
<h1>{_esc(name)}</h1>
{summ_html}
<table>
<tr><th>qid</th><th></th><th>gold</th><th>pred</th><th>question</th><th>error</th></tr>
{''.join(rows)}
</table>
</body></html>"""


def render_top_index(root: str, cells: List[dict]) -> str:
    rows = []
    for c in cells:
        v = c["summary"] or {}
        acc = v.get("accuracy")
        acc_disp = f"{acc*100:.2f}%" if isinstance(acc, (int, float)) else "—"
        rows.append(
            "<tr>"
            f"<td>{_esc(c['model'])}</td>"
            f"<td><a href='{_esc(c['index_rel'])}'>{_esc(c['task'])}</a></td>"
            f"<td>{_esc(c['split'])}</td>"
            f"<td class='num'><b>{acc_disp}</b></td>"
            f"<td class='num'>{v.get('n_correct','')}/{v.get('n_total','')}</td>"
            f"<td class='num'>{v.get('n_errored','')}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>vanilla_cot results</title>
<style>{_CSS}</style></head><body>
<h1>{_esc(os.path.abspath(root))}</h1>
<p class='muted'>{len(cells)} cells</p>
<table>
<tr><th>model</th><th>task</th><th>split</th><th>accuracy</th>
    <th>correct/total</th><th>errored</th></tr>
{''.join(rows)}
</table>
</body></html>"""


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------

def _load(p: str) -> Optional[dict]:
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def render_single_file(in_path: str, out_path: Optional[str] = None) -> str:
    d = _load(in_path)
    if d is None:
        sys.exit(f"ERROR: cannot load {in_path}")
    out_path = out_path or os.path.splitext(in_path)[0] + ".html"
    with open(out_path, "w") as f:
        f.write(render_ckpt(d, title=os.path.basename(in_path)))
    return out_path


def render_cell(cell_dir: str) -> str:
    ck_dir = os.path.join(cell_dir, "ckpts")
    if not os.path.isdir(ck_dir):
        sys.exit(f"ERROR: {ck_dir} not found")
    html_dir = os.path.join(cell_dir, "html")
    out_ck   = os.path.join(html_dir, "ckpts")
    os.makedirs(out_ck, exist_ok=True)

    pages: List[tuple] = []
    for fname in sorted(f for f in os.listdir(ck_dir) if f.endswith(".json")):
        d = _load(os.path.join(ck_dir, fname))
        if d is None:
            continue
        out_html_name = fname[:-5] + ".html"
        with open(os.path.join(out_ck, out_html_name), "w") as f:
            f.write(render_ckpt(d, title=fname, back_link="../index.html"))
        pages.append((out_html_name, d))

    summary_json = _load(os.path.join(cell_dir, "summary.json"))
    idx_path = os.path.join(html_dir, "index.html")
    with open(idx_path, "w") as f:
        f.write(render_cell_index(cell_dir, html_dir, pages, summary_json))
    return idx_path


def render_top(root: str) -> str:
    cells = []
    for model in sorted(os.listdir(root)):
        m_dir = os.path.join(root, model)
        if not os.path.isdir(m_dir): continue
        for cell in sorted(os.listdir(m_dir)):
            c_dir = os.path.join(m_dir, cell)
            if not os.path.isdir(c_dir): continue
            if not os.path.isdir(os.path.join(c_dir, "ckpts")): continue
            idx = render_cell(c_dir)
            if cell.endswith("_test"):  task, split = cell[:-5], "test"
            elif cell.endswith("_val"): task, split = cell[:-4], "val"
            else: task, split = cell, ""
            sj = _load(os.path.join(c_dir, "summary.json")) or {}
            cells.append({
                "model": model, "task": task, "split": split,
                "summary": sj.get("vanilla") or sj,
                "index_rel": os.path.relpath(idx, root),
            })
    landing = os.path.join(root, "index.html")
    with open(landing, "w") as f:
        f.write(render_top_index(root, cells))
    return landing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="ckpt .json | cell dir | sweep root")
    ap.add_argument("--out", default=None,
                    help="(single-file) output HTML path")
    ap.add_argument("--top", action="store_true",
                    help="treat path as a sweep root (<root>/<model>/<cell>) "
                         "and write a landing index linking each cell")
    ap.add_argument("--open", action="store_true",
                    help="open the result in the default browser")
    args = ap.parse_args()

    p = args.path
    if os.path.isfile(p) and p.endswith(".json"):
        out = render_single_file(p, args.out)
    elif os.path.isdir(p) and args.top:
        out = render_top(p)
    elif os.path.isdir(p) and os.path.isdir(os.path.join(p, "ckpts")):
        out = render_cell(p)
    else:
        sys.exit(
            f"ERROR: don't know how to handle {p!r}.\n"
            "  - .json file              → render single ckpt\n"
            "  - cell dir (with ckpts/)  → render index + per-ckpt pages\n"
            "  - sweep root with --top   → landing page over every cell")

    print(f"wrote {out}")
    if args.open:
        webbrowser.open("file://" + os.path.abspath(out))


if __name__ == "__main__":
    main()
