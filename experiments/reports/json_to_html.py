#!/usr/bin/env python3
"""Convert run_BBH_multi_cot.py per-example JSON checkpoints to HTML.

Renders one self-contained HTML page per example with:
  - Question + multiple-choice options (with gold answer highlighted).
  - Per-prior predictions (teacher_forcing / verbalized / uniform) and
    whether each one is correct.
  - Inference posteriors per letter (bar chart).
  - Per-letter CoT (with init_score, step-level token weights, verbalized
    confidences, and the answer step highlighted).
  - The raw CoT model response (combined / separate concatenated).
  - The teacher-forcing model output (tf_raw) and the parsed tf_priors.
  - The generated probabilistic program (syntax-highlighted minimal).
  - Configuration footer (model, strategy, pool/combo/aggr/evidence rules).

Usage
-----
Single ckpt:
    python experiments/reports/json_to_html.py path/to/ckpts/q-0001.json

Whole ckpts directory (one HTML per example + an index):
    python experiments/reports/json_to_html.py path/to/ckpts/ -o html_output/

Whole results.json (one HTML per row + index):
    python experiments/reports/json_to_html.py path/to/results.json -o html_output/

By default, output dir is the same as the input file's parent / "html/".
"""

from __future__ import annotations

import argparse
import html as _html
import json
import re
import os
import sys
from typing import Any, Dict, List, Optional

# Readable parameter names ("Step 1 for C: stated confidence"), shared with
# the console. The repo root is two levels up from this file.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    from cot_prog.program_labels import answer_label, answer_names, param_label, param_labels
except Exception:      # report still renders with the program's own names
    answer_label = answer_names = param_label = param_labels = None


# ── CSS ──────────────────────────────────────────────────────────────────────

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
       background: #f0f2f5; color: #1a1a2e; padding: 24px; line-height: 1.5;
       max-width: 1200px; margin: 0 auto; }
h1 { font-size: 1.5rem; margin-bottom: 6px; }
h2 { font-size: 1.15rem; margin: 18px 0 8px; color: #2d3748;
     border-bottom: 2px solid #e2e8f0; padding-bottom: 4px; }
h3 { font-size: 1rem; margin: 12px 0 6px; color: #4a5568; }
.card { background: #fff; border-radius: 10px; padding: 18px;
        margin-bottom: 14px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
.q-text { font-size: 1.05rem; white-space: pre-wrap; }
.choices { margin-top: 10px; }
.choice { padding: 6px 10px; border-radius: 6px; margin: 3px 0;
          font-size: .95rem; border: 1px solid #e2e8f0; }
.choice.gold { background: #c6f6d5; border-color: #38a169; font-weight: 600; }
.tag { display: inline-block; padding: 3px 10px; border-radius: 12px;
       font-size: .78rem; font-weight: 600; margin-right: 4px; }
.tag-green { background: #c6f6d5; color: #276749; }
.tag-red   { background: #fed7d7; color: #9b2c2c; }
.tag-blue  { background: #bee3f8; color: #2a4365; }
.tag-gray  { background: #e2e8f0; color: #4a5568; }
.tag-yellow{ background: #fef3c7; color: #92400e; }
.preds-row { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 8px; }
.pred-card { padding: 10px 14px; border-radius: 8px; text-align: center;
             min-width: 180px; flex: 1; border: 2px solid; }
.pred-card.correct  { border-color: #38a169; background: #f0fff4; }
.pred-card.incorrect{ border-color: #e53e3e; background: #fff5f5; }
.pred-card.unknown  { border-color: #cbd5e0; background: #f7fafc; }
.pred-label { font-size: .75rem; color: #718096; text-transform: uppercase;
              letter-spacing: .04em; }
.pred-val { font-size: 1.1rem; font-weight: 700; margin-top: 2px; }
.bar-row { display: flex; align-items: center; gap: 10px; margin: 4px 0; }
.bar-label { width: 70px; font-weight: 600; font-family: monospace;
             font-size: .92rem; }
.bar-track { flex: 1; height: 22px; background: #e2e8f0;
             border-radius: 4px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 4px;
            background: linear-gradient(90deg, #4299e1, #3182ce); }
.bar-fill.gold { background: linear-gradient(90deg, #48bb78, #2f855a); }
.bar-val { width: 60px; font-family: monospace; font-size: .85rem;
           text-align: right; }
.bar-track.gold { box-shadow: inset 0 0 0 2px #38a169; }
.bar-rank { width: 26px; text-align: right; font-size: .78rem;
            color: #718096; font-weight: 600; }
.section-cot { background: #fafafa; padding: 12px; border-radius: 6px;
               margin: 6px 0; border-left: 3px solid #cbd5e0; }
.section-cot.gold { border-left-color: #38a169; }
.section-header { font-weight: 600; margin-bottom: 6px; color: #2d3748; }
.step { display: flex; gap: 8px; padding: 5px 8px; border-radius: 4px;
        margin: 2px 0; font-size: .9rem; }
.step:hover { background: #edf2f7; }
.step-num { width: 22px; flex-shrink: 0; font-family: monospace;
            font-weight: 700; color: #4a5568; }
.step-text { flex: 1; }
.step-meta { font-size: .76rem; color: #718096; margin-top: 1px; }
.step-answer { background: #faf5ff; border: 1px dashed #805ad5; }
pre.code { background: #1a202c; color: #e2e8f0; padding: 14px;
           border-radius: 8px; overflow-x: auto; font-size: .82rem;
           line-height: 1.5; white-space: pre-wrap; word-wrap: break-word;
           font-family: 'Menlo','Consolas',monospace; }
pre.raw { background: #f7fafc; color: #2d3748; padding: 12px;
          border-radius: 6px; overflow-x: auto; font-size: .85rem;
          line-height: 1.5; white-space: pre-wrap; word-wrap: break-word;
          font-family: 'Menlo','Consolas',monospace;
          border: 1px solid #e2e8f0; max-height: 500px; }
.tf-card { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
@media (max-width: 800px) { .tf-card { grid-template-columns: 1fr; } }
details > summary { cursor: pointer; font-weight: 600; padding: 4px 0; }
details > summary:hover { color: #2b6cb0; }
.config-row { display: flex; gap: 10px; flex-wrap: wrap; font-size: .85rem;
              margin-top: 6px; color: #4a5568; }
.config-row span { background: #edf2f7; padding: 3px 10px; border-radius: 12px;
                   font-family: monospace; }
table { border-collapse: collapse; width: 100%; font-size: .9rem; }
th, td { padding: 7px 10px; border-bottom: 1px solid #e2e8f0;
         text-align: left; }
th { background: #f7fafc; font-weight: 600; }
.row-correct td { background: #f0fff4; }
"""


# ── Helpers ──────────────────────────────────────────────────────────────────

def _esc(s: Any) -> str:
    return _html.escape(str(s)) if s is not None else ""


def _pct(x: float) -> str:
    try:
        return f"{float(x) * 100:.1f}%"
    except Exception:
        return "—"


def _fmt(x: Any, nd: int = 4) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "—"


def _bar(label: str, value: float, gold: bool = False, max_val: float = 1.0,
         rank: Optional[int] = None) -> str:
    pct = max(0.0, min(1.0, float(value) / max_val if max_val else 0.0))
    rank_html = f'<div class="bar-rank">#{rank}</div>' if rank is not None else ''
    return (
        f'<div class="bar-row">'
        f'  <div class="bar-label">{_esc(label)}</div>'
        f'{rank_html}'
        f'  <div class="bar-track{" gold" if gold else ""}">'
        f'    <div class="bar-fill{" gold" if gold else ""}" style="width: {pct*100:.1f}%"></div>'
        f'  </div>'
        f'  <div class="bar-val">{_fmt(value)}</div>'
        f'</div>'
    )


# ── Section renderers ───────────────────────────────────────────────────────

def render_question(ex: Dict[str, Any]) -> str:
    q = _esc(ex.get("question", ""))
    labels = ex.get("choice_labels", []) or []
    choices = ex.get("choices", []) or []
    gold = (ex.get("answer_key") or "").strip().upper()
    parts = [
        '<div class="card">',
        f'  <h1>Question {_esc(ex.get("qid",""))}</h1>',
        f'  <div class="q-text">{q}</div>',
        '  <div class="choices">',
    ]
    for lbl, txt in zip(labels, choices):
        cls = "choice gold" if lbl.strip().upper() == gold else "choice"
        parts.append(f'    <div class="{cls}"><b>{_esc(lbl)}.</b> {_esc(txt)}</div>')
    parts.append('  </div>')
    parts.append(
        f'  <div style="margin-top:10px"><span class="tag tag-green">'
        f'gold = {_esc(gold)} ({_esc(ex.get("answer_text",""))})</span></div>'
    )
    parts.append('</div>')
    return "\n".join(parts)


def _modes_view(ex: Dict[str, Any]) -> Dict[str, Any]:
    """Present a checkpoint as ``{label: exec_dict}`` for rendering.

    Flat format (one result per cell) -> a single entry keyed by the run's
    ``answer_aggregation_fn``. Legacy per-mode ``modes`` dicts pass through.
    """
    m = ex.get("modes")
    if m:
        return m
    if any(ex.get(k) for k in ("program", "inference", "sample", "error")):
        label = (ex.get("config", {}) or {}).get("answer_aggregation_fn") or "result"
        entry = {k: ex.get(k) for k in
                 ("program", "sample", "semantics", "circuit", "inference", "gradient")}
        if ex.get("error"):
            entry["error"] = ex["error"]
        return {label: entry}
    return {}


def _pred_view(ex: Dict[str, Any]):
    """``(predicted_label, predicted_text)`` as ``{label: scalar}`` dicts
    (flat single-result or legacy per-mode)."""
    pl = ex.get("predicted_label")
    pt = ex.get("predicted_text")
    if isinstance(pl, dict):
        return pl, (pt if isinstance(pt, dict) else {})
    label = (ex.get("config", {}) or {}).get("answer_aggregation_fn") or "result"
    return ({label: pl} if pl is not None else {}), ({label: pt} if pt is not None else {})


def render_predictions(ex: Dict[str, Any]) -> str:
    gold = (ex.get("answer_key") or "").strip().upper()
    pred_label, pred_text = _pred_view(ex)
    parts = ['<div class="card">', '<h2>Predictions</h2>',
             '<div class="preds-row">']
    for mode in pred_label:
        pl = pred_label.get(mode)
        pt = pred_text.get(mode)
        if pl is None:
            cls, status = "unknown", "—"
        else:
            ok = pl.strip().upper() == gold
            cls = "correct" if ok else "incorrect"
            status = "✓" if ok else "✗"
        parts.append(
            f'<div class="pred-card {cls}">'
            f'  <div class="pred-label">{_esc(mode)}</div>'
            f'  <div class="pred-val">{_esc(pl) if pl else "—"} {status}</div>'
            f'  <div style="font-size:.78rem;color:#718096">{_esc(pt) if pt else ""}</div>'
            f'</div>'
        )
    parts.append('</div>')
    parts.append('</div>')
    return "\n".join(parts)


def _build_rv_to_letter_map(mode_entry: Dict[str, Any], labels: List[str]) -> Dict[str, str]:
    """Map return-variable names (e.g. ``A_s4``) to MCQ letters (e.g. ``B``).

    The program text saved on each mode_entry has the form::

        s4:  str = "A"          # step strings encode letters
        s7:  str = "B"
        ...
        if A_x:
            return s4
        elif B_x:
            return s7
        ...

    Inference results are keyed by ``<answer_var>_<return_value>`` such as
    ``A_s4`` (where the leading ``A_`` is just the categorical answer
    variable name and is identical for every return — NOT the letter).
    The actual letter is the contents of ``s4`` as a step string.

    This function parses both pieces and returns the right mapping.
    Falls back to splitting on ``_`` when the program isn't available.
    """
    prog = (mode_entry or {}).get("program") or ""
    # Map step name -> step string contents (e.g. "s4" -> "A")
    step_strs = dict(re.findall(
        r'^(s\d+)\s*:\s*str\s*=\s*"([^"]*)"', prog, re.MULTILINE,
    ))
    # rv_name "A_s4" → letter is the step string at s4
    rv_map: Dict[str, str] = {}
    inf = (mode_entry or {}).get("inference") or {}
    label_set = {l.strip().upper() for l in labels}
    for rv_name in inf:
        if rv_name == "Exc":
            rv_map[rv_name] = "Exc"
            continue
        # Split off the answer-var prefix (e.g. "A_s4" → ans="A", rest="s4")
        if "_" in rv_name:
            _ans, _rest = rv_name.split("_", 1)
            text = step_strs.get(_rest)
            if text and text.strip().upper() in label_set:
                rv_map[rv_name] = text.strip().upper()
                continue
        # Fallback: legacy convention where rv_name was already <letter>_<txt>
        head = rv_name.split("_", 1)[0]
        if head.upper() in label_set:
            rv_map[rv_name] = head.upper()
        else:
            rv_map[rv_name] = rv_name
    return rv_map


def render_posteriors(ex: Dict[str, Any]) -> str:
    """For each mode, show the per-letter posterior bar."""
    modes = _modes_view(ex)
    labels = ex.get("choice_labels", []) or []
    gold = (ex.get("answer_key") or "").strip().upper()
    parts = ['<div class="card">', '<h2>Posterior P(answer = X | evidence)</h2>']
    for mode_name, m in modes.items():
        if not isinstance(m, dict):
            continue
        if "error" in m:
            parts.append(
                f'<h3>{_esc(mode_name)}</h3>'
                f'<pre class="raw" style="color:#9b2c2c">{_esc(m["error"][:1500])}</pre>'
            )
            continue
        inf = m.get("inference", {})
        parts.append(f'<h3>{_esc(mode_name)}</h3>')
        rv_to_letter = _build_rv_to_letter_map(m, labels)
        rows = []
        for var_name, d in (inf or {}).items():
            if not isinstance(d, dict) or "prob" not in d:
                continue
            if var_name == "Exc":
                # Show Exc separately below the per-letter bars
                continue
            letter = rv_to_letter.get(var_name, var_name)
            rows.append((letter, float(d["prob"])))
        rows.sort(key=lambda r: -r[1])
        for letter, prob in rows:
            parts.append(_bar(letter, prob, gold=(letter.upper() == gold)))
        # Append the Exc row last (always small but worth showing)
        exc = (inf or {}).get("Exc")
        if isinstance(exc, dict) and "prob" in exc:
            parts.append(_bar("Exc", float(exc["prob"]), gold=False))
    parts.append('</div>')
    return "\n".join(parts)


def render_tf(ex: Dict[str, Any]) -> str:
    """Show the teacher-forcing raw response + parsed priors."""
    tf_priors = ex.get("tf_priors")
    tf_raw = ex.get("tf_raw")
    tf_meta = ex.get("tf_meta") or {}
    if not tf_priors and not tf_raw:
        return ""
    gold = (ex.get("answer_key") or "").strip().upper()
    parts = [
        '<div class="card">',
        "<h2>Teacher-forcing answer prior (slot 1) — token logprob of "
        "'correct' vs 'incorrect' per option</h2>",
        '<div class="tf-card">',
    ]
    # left column — parsed priors (+ per-option logprob diagnostics when saved)
    per_letter = tf_meta.get("per_letter") or {}
    parts.append('<div>')
    parts.append('<h3>Parsed P(option is correct) — sorted high → low</h3>')
    if tf_priors:
        items = sorted(tf_priors.items(), key=lambda kv: -float(kv[1]))
        for k, (lbl, p) in enumerate(items, start=1):
            parts.append(_bar(lbl, float(p),
                              gold=(lbl.strip().upper() == gold),
                              rank=k))
            d = per_letter.get(lbl)
            if isinstance(d, dict) and "lp_correct" in d:
                def _f(v):
                    return f"{v:.3f}" if isinstance(v, (int, float)) else "—"
                parts.append(
                    '<div style="font-size:.75rem;color:#718096;'
                    'margin:-4px 0 6px 30px">'
                    f'wrote <b>{_esc(str(d.get("chosen", "?")))}</b> · '
                    f'lp(correct)={_f(d.get("lp_correct"))} · '
                    f'lp(incorrect)={_f(d.get("lp_incorrect"))} · '
                    f'{_esc(str(d.get("path", "")))}</div>')
    else:
        parts.append('<p style="color:#718096">(no priors saved)</p>')
    parts.append('</div>')
    # right column — raw model response
    parts.append('<div>')
    parts.append('<h3>Raw model response (TF call)</h3>')
    if tf_raw:
        parts.append(f'<pre class="raw">{_esc(tf_raw)}</pre>')
    else:
        parts.append('<p style="color:#718096">(no raw response saved)</p>')
    parts.append('</div>')
    parts.append('</div></div>')
    return "\n".join(parts)


def render_per_letter_cot(ex: Dict[str, Any]) -> str:
    """Show the per-letter CoT sections with steps + weights + confidences."""
    traces = ex.get("per_answer_traces", {}) or {}
    labels = ex.get("choice_labels", []) or []
    gold = (ex.get("answer_key") or "").strip().upper()
    if not traces:
        return ""
    parts = ['<div class="card">', '<h2>Per-letter chain-of-thought</h2>']
    for lbl in labels:
        t = traces.get(lbl) or traces.get(lbl.upper()) or traces.get(lbl.lower())
        if t is None:
            continue
        is_gold = lbl.strip().upper() == gold
        cls = "section-cot gold" if is_gold else "section-cot"
        init = t.get("init_score")
        steps = t.get("steps") or []
        # Pull the per-section answer-step's verbalized confidence
        verb_conf = None
        for s in steps:
            if s.get("answer"):
                verb_conf = s.get("confidence")
        parts.append(f'<div class="{cls}">')
        gold_tag = ' <span class="tag tag-green">✓ gold</span>' if is_gold else ''
        init_tag = (
            f' <span class="tag tag-blue">init {_pct(init)}</span>'
            if init is not None else ''
        )
        verb_tag = (
            f' <span class="tag tag-yellow">verb {_pct(verb_conf)}</span>'
            if verb_conf is not None else ''
        )
        # Choice text from the parent example for nicer header.
        choice_text = ""
        for cl, ct in zip(labels, ex.get("choices") or []):
            if cl == lbl:
                choice_text = ct; break
        parts.append(
            f'<div class="section-header">## For {_esc(lbl)} '
            f'<span style="color:#718096;font-weight:400">— {_esc(choice_text)}</span>'
            f'{init_tag}{verb_tag}{gold_tag}</div>'
        )
        for s in steps:
            text = s.get("claim") or s.get("answer") or ""
            ans = s.get("answer") or ""
            refs = s.get("premises") or []
            weight = s.get("weight")
            conf = s.get("confidence")
            step_cls = "step step-answer" if ans else "step"
            meta_parts = []
            if refs:
                meta_parts.append(f'(ref. {", ".join(map(str, refs))})')
            elif s.get("formula") is not None:
                meta_parts.append("(ref. none)")
            if ans:
                meta_parts.append(f'(final answer: {_esc(ans)})')
            if conf is not None:
                meta_parts.append(f'(confidence: {_pct(conf)})')
            if weight is not None:
                meta_parts.append(f'[w={_fmt(weight, 3)}]')
            parts.append(
                f'<div class="{step_cls}">'
                f'  <div class="step-num">{_esc(s.get("prefix",""))}.</div>'
                f'  <div>'
                f'    <div class="step-text">{_esc(text)}</div>'
                f'    <div class="step-meta">{" ".join(_esc(m) for m in meta_parts)}</div>'
                f'  </div>'
                f'</div>'
            )
        parts.append('</div>')
    parts.append('</div>')
    return "\n".join(parts)


def render_cot_prompts(ex: Dict[str, Any]) -> str:
    """Show the per-letter CoT prompt (system + user) that produced each call.

    For separate strategy this surfaces N distinct prompts. For combined-
    style strategies all letters share one prompt, so we collapse to a
    single card to avoid 6× duplication.
    """
    traces = ex.get("per_answer_traces", {}) or {}
    labels = ex.get("choice_labels", []) or []
    if not traces:
        return ""
    # Collect (label, prompt_dict) pairs
    pairs = []
    for lbl in labels:
        t = traces.get(lbl) or traces.get(lbl.upper()) or traces.get(lbl.lower())
        if not t:
            continue
        p = t.get("cot_prompt")
        if not isinstance(p, dict):
            continue
        pairs.append((lbl, p))
    if not pairs:
        return ""
    # If every prompt is byte-identical, collapse to one card.
    distinct = {(p.get("system") or "", p.get("user") or "") for _, p in pairs}
    parts = ['<div class="card">', '<h2>CoT prompts</h2>']
    if len(distinct) == 1:
        # combined-style — one shared prompt
        sys_p, usr_p = next(iter(distinct))
        parts.append('<p style="color:#718096;font-size:.85rem;margin-top:0">'
                     'Single shared prompt produced all per-letter sections '
                     '(combined strategy).</p>')
        parts.append(f'<h3>system</h3><pre class="raw">{_esc(sys_p)}</pre>')
        parts.append(f'<h3>user</h3><pre class="raw">{_esc(usr_p)}</pre>')
    else:
        # separate strategy — N distinct prompts
        parts.append('<p style="color:#718096;font-size:.85rem;margin-top:0">'
                     f'{len(pairs)} distinct prompts (separate strategy — one '
                     'LLM call per candidate letter).</p>')
        for lbl, p in pairs:
            parts.append(f'<details><summary><strong>letter {_esc(lbl)}</strong> '
                         f'— prompt for this call</summary>')
            parts.append(f'<h4>system</h4><pre class="raw">{_esc(p.get("system",""))}</pre>')
            parts.append(f'<h4>user</h4><pre class="raw">{_esc(p.get("user",""))}</pre>')
            parts.append('</details>')
    parts.append('</div>')
    return "\n".join(parts)


def render_raw_cot(ex: Dict[str, Any]) -> str:
    raw = ex.get("raw_cot")
    if not raw:
        return ""
    parts = ['<div class="card">', '<h2>Raw CoT model response</h2>']
    parts.append(f'<pre class="raw">{_esc(raw)}</pre>')
    parts.append('</div>')
    return "\n".join(parts)


def render_program(ex: Dict[str, Any]) -> str:
    modes = _modes_view(ex)
    if not modes:
        return ""
    parts = ['<div class="card">', '<h2>Probabilistic program</h2>']
    for mode_name, m in modes.items():
        prog = m.get("program")
        if not prog:
            continue
        parts.append(f'<details><summary>{_esc(mode_name)}</summary>')
        parts.append(f'<pre class="code">{_esc(prog)}</pre>')
        parts.append('</details>')
    parts.append('</div>')
    return "\n".join(parts)


def render_config(ex: Dict[str, Any]) -> str:
    cfg = ex.get("config") or {}
    if not cfg:
        return ""
    items = []
    for k in ("model_name", "cot_strategy", "pool_mode",
              "answer_aggregation_fn", "statement_aggregation_fn", "step_dependency_mode",
              "verbal_conf", "verbal_step_deps", "reused"):
        if k in cfg:
            items.append(f"<span>{_esc(k)}={_esc(cfg[k])}</span>")
    if not items:
        return ""
    return (
        '<div class="card"><h2>Configuration</h2>'
        f'<div class="config-row">{"".join(items)}</div></div>'
    )


_MODES_FOR_RENDER = ("teacher_forcing", "verbalized", "uniform")


def render_sample(ex: Dict[str, Any]) -> str:
    """Empirical sample bar charts per mode."""
    modes = _modes_view(ex)
    charts = []
    for mode, _m in modes.items():
        s = (_m or {}).get("sample")
        if not s:
            continue
        if isinstance(s, dict) and "error" in s:
            charts.append(
                f'<div><h3>{_esc(mode)}</h3>'
                f'<pre class="raw" style="color:#9b2c2c">{_esc(str(s["error"])[:1500])}</pre></div>'
            )
            continue
        rv = (s.get("return_values") or {})
        if not rv:
            continue
        attempts = int(s.get("attempts") or 0)
        executed = int(s.get("executed") or 0)
        total    = sum(rv.values()) or 1
        max_ct   = max(rv.values()) or 1
        bars = []
        for label, count in sorted(rv.items(), key=lambda x: -x[1]):
            count = int(count)
            pct = count / total * 100
            h = max(2, int(100 * count / max_ct))
            ll = label.lower()
            if "yes" in ll:
                bg, col = "linear-gradient(180deg,#48bb78,#38a169)", "#276749"
            elif "no" in ll:
                bg, col = "linear-gradient(180deg,#fc8181,#e53e3e)", "#9b2c2c"
            else:
                bg, col = "#e2e8f0", "#a0aec0"
            bars.append(
                f'<div style="text-align:center;flex:1;">'
                f'  <div style="background:{bg};height:{h}px;border-radius:4px 4px 0 0;position:relative;">'
                f'    <span style="position:absolute;top:-20px;left:0;right:0;font-size:.78rem;font-weight:700;">{count:,}</span>'
                f'  </div>'
                f'  <div style="font-size:.78rem;font-weight:600;margin-top:4px;color:{col};">{_esc(label)} ({pct:.1f}%)</div>'
                f'</div>'
            )
        charts.append(
            f'<div><h3>{_esc(mode)} <span style="color:#718096;font-size:.85rem">tries={attempts}, exec={executed}</span></h3>'
            f'<div style="display:flex;align-items:end;gap:18px;height:130px;padding:10px 16px 0;">{"".join(bars)}</div></div>'
        )
    if not charts:
        return ""
    return (
        '<div class="card"><h2>Empirical Sample</h2>'
        + "".join(charts) +
        '</div>'
    )


def render_semantics(ex: Dict[str, Any]) -> str:
    """Pretty-printed program semantics (first available mode wins)."""
    modes = _modes_view(ex)
    for mode, _m in modes.items():
        sem = (_m or {}).get("semantics")
        if isinstance(sem, str) and sem.strip():
            return (
                f'<div class="card">'
                f'  <h2>Program Semantics <span style="color:#718096;font-size:.85rem">({_esc(mode)})</span></h2>'
                f'  <pre class="raw">{_esc(sem)}</pre>'
                f'</div>'
            )
    return ""


def render_circuit(ex: Dict[str, Any]) -> str:
    """Compiled circuit summary (first available mode wins)."""
    modes = _modes_view(ex)
    for mode, _m in modes.items():
        circ = (_m or {}).get("circuit")
        if not isinstance(circ, dict) or "error" in circ:
            continue
        rvs   = circ.get("return_vars") or []
        n_lit = circ.get("num_literals", "?")
        # Pull Z from inference of any rv
        inf = (_m or {}).get("inference") or {}
        z = None
        if isinstance(inf, dict):
            for v in inf.values():
                if isinstance(v, dict) and "Z" in v:
                    z = v["Z"]; break
        z_str = f"{z:.4f}" if isinstance(z, (int, float)) else "?"
        return (
            '<div class="card">'
            f' <h2>Compiled Circuit (SDD) <span style="color:#718096;font-size:.85rem">({_esc(mode)})</span></h2>'
            ' <table style="border-collapse:collapse">'
            '   <tr><th style="text-align:left;padding:4px 12px;border-bottom:1px solid #e2e8f0">Property</th>'
            '       <th style="text-align:left;padding:4px 12px;border-bottom:1px solid #e2e8f0">Value</th></tr>'
            f'  <tr><td style="padding:4px 12px">Return variables</td>'
            f'      <td style="padding:4px 12px"><code>{_esc(", ".join(rvs))}</code></td></tr>'
            f'  <tr><td style="padding:4px 12px">Number of literals</td>'
            f'      <td style="padding:4px 12px">{_esc(n_lit)}</td></tr>'
            f'  <tr><td style="padding:4px 12px">Partition function Z</td>'
            f'      <td style="padding:4px 12px">≈ {z_str}</td></tr>'
            ' </table>'
            '</div>'
        )
    return ""


def render_gradient(ex: Dict[str, Any]) -> str:
    """Sensitivity / gradient analysis (∂P/∂w_i per parameter)."""
    modes = _modes_view(ex)
    sections = []
    for mode, _m in modes.items():
        grad_data = (_m or {}).get("gradient")
        if not isinstance(grad_data, dict) or "error" in grad_data:
            continue
        program = (_m or {}).get("program") or ""
        labels = param_labels(program) if param_labels else {}
        names = answer_names(program) if answer_names else {}

        def _readable(name):
            return param_label(labels, name) if param_label else name

        def _answer(rv):
            var = str(rv)[2:] if str(rv).startswith("A_") else str(rv)
            return answer_label(names, var) if answer_label else var
        for rv, grads in sorted(grad_data.items()):
            if isinstance(grads, dict) and "error" in grads:
                continue
            if not isinstance(grads, list) or not grads:
                continue
            pos = sorted([(n, v) for n, v in grads if v > 0], key=lambda x: -x[1])
            neg = sorted([(n, v) for n, v in grads if v < 0], key=lambda x:  x[1])
            max_abs = max((abs(v) for _, v in grads), default=0.001) or 0.001
            def _bars(items, color, sign):
                out = []
                for name, val in items:
                    w = abs(val) / max_abs * 100
                    out.append(
                        f'<div style="display:flex;align-items:center;gap:8px;margin:2px 0;font-size:.82rem">'
                        f'  <span style="min-width:250px" title="{_esc(name)} in the program">{_esc(_readable(name))}</span>'
                        f'  <div style="flex:1;height:12px;background:#edf2f7;border-radius:3px;position:relative">'
                        f'    <div style="width:{w:.1f}%;height:12px;background:{color};border-radius:3px"></div>'
                        f'  </div>'
                        f'  <span style="min-width:64px;text-align:right;color:{color}">{sign}{val:.4f}</span>'
                        f'</div>'
                    )
                return "".join(out) or '<p style="color:#718096;font-size:.85rem">(none)</p>'
            sections.append(
                '<div class="card">'
                f' <h2>Sensitivity of the answer: {_esc(_answer(rv))} '
                f' <span style="color:#718096;font-size:.85rem">({_esc(mode)})</span></h2>'
                ' <div style="display:grid;grid-template-columns:1fr 1fr;gap:18px;">'
                f'  <div><h3 style="color:#38a169;margin:0 0 6px">Raises this answer</h3>{_bars(pos, "#38a169", "+")}</div>'
                f'  <div><h3 style="color:#e53e3e;margin:0 0 6px">Lowers this answer</h3>{_bars(neg, "#e53e3e", "")}</div>'
                ' </div>'
                '</div>'
            )
    return "\n".join(sections)


# ── Top-level renderers ─────────────────────────────────────────────────────

def example_to_html(ex: Dict[str, Any], title: str = "") -> str:
    title = title or f"BBH MCQ — {ex.get('qid','example')}"
    body_sections = [
        render_question(ex),
        render_predictions(ex),
        render_posteriors(ex),
        render_tf(ex),
        render_per_letter_cot(ex),
        render_program(ex),
        render_cot_prompts(ex),
        render_raw_cot(ex),
        render_sample(ex),
        render_semantics(ex),
        render_circuit(ex),
        render_gradient(ex),
        render_config(ex),
    ]
    if ex.get("error"):
        body_sections.insert(1, (
            '<div class="card" style="border:2px solid #e53e3e">'
            '<h2 style="color:#9b2c2c">Error</h2>'
            f'<pre class="raw" style="color:#9b2c2c">{_esc(ex["error"])}</pre>'
            '</div>'
        ))
    return (
        '<!DOCTYPE html>\n'
        '<html lang="en"><head>\n'
        f'  <meta charset="utf-8"><title>{_esc(title)}</title>\n'
        f'  <style>{CSS}</style>\n'
        '</head><body>\n'
        + "\n".join(body_sections) +
        '\n</body></html>\n'
    )


def save_example_html(ex: Dict[str, Any], out_path: str) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    html_str = example_to_html(ex)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_str)
    return out_path


def render_index(items: List[Dict[str, Any]], out_dir: str) -> str:
    """Build a small index page listing every example with its predictions."""
    rows = []
    for it in items:
        ex = it["ex"]
        gold = (ex.get("answer_key") or "").strip().upper()
        pl, _ = _pred_view(ex)
        cells = []
        for mode in pl:
            v = pl.get(mode)
            if v is None:
                cells.append("—")
            else:
                ok = v.strip().upper() == gold
                cells.append(f'{_esc(v)} {"✓" if ok else "✗"}')
        any_correct = any(
            (v or "").strip().upper() == gold for v in pl.values()
        )
        rows.append(
            f'<tr class="{"row-correct" if any_correct else ""}">'
            f'<td><a href="{_esc(it["filename"])}">{_esc(ex.get("qid",""))}</a></td>'
            f'<td>{_esc((ex.get("question") or "")[:120])}…</td>'
            f'<td>{_esc(gold)}</td>'
            f'<td>{" / ".join(cells) if cells else "—"}</td>'
            f'</tr>'
        )
    body = (
        '<!DOCTYPE html>\n<html lang="en"><head>\n'
        '<meta charset="utf-8"><title>BBH MCQ — index</title>\n'
        f'<style>{CSS}</style>\n</head><body>\n'
        '<div class="card"><h1>BBH MCQ — examples index</h1>'
        f'<p style="color:#718096">{len(items)} examples</p></div>\n'
        '<div class="card"><table>\n'
        '<tr><th>QID</th><th>Question</th><th>Gold</th>'
        '<th>predicted</th></tr>\n'
        + "\n".join(rows) +
        '</table></div>\n</body></html>\n'
    )
    out = os.path.join(out_dir, "index.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(body)
    return out


# ── CLI ──────────────────────────────────────────────────────────────────────

def _load_examples(path: str) -> List[Dict[str, Any]]:
    """Load examples from a single JSON file or all *.json in a directory."""
    if os.path.isdir(path):
        items = []
        for fname in sorted(os.listdir(path)):
            if not fname.endswith(".json"):
                continue
            try:
                with open(os.path.join(path, fname)) as f:
                    items.append(json.load(f))
            except Exception as exc:
                print(f"  [skip] {fname}: {exc}", file=sys.stderr)
        return items
    # Single file — could be a per-example ckpt or a results.json
    with open(path) as f:
        d = json.load(f)
    if isinstance(d, list):
        return d
    if isinstance(d, dict) and isinstance(d.get("results"), list):
        return d["results"]
    return [d]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="Path to a per-example .json, a results.json, "
                                  "or a ckpts/ directory.")
    p.add_argument("-o", "--output_dir", default=None,
                   help="Output dir (default: <input parent>/html/).")
    p.add_argument("--no-index", action="store_true",
                   help="Skip the index.html when processing many examples.")
    args = p.parse_args()

    examples = _load_examples(args.input)
    if not examples:
        print("No examples found.", file=sys.stderr)
        return 1

    if args.output_dir is None:
        parent = os.path.dirname(os.path.abspath(args.input)) \
                 if os.path.isfile(args.input) else os.path.abspath(args.input)
        args.output_dir = os.path.join(parent, "html")
    os.makedirs(args.output_dir, exist_ok=True)

    written: List[Dict[str, str]] = []
    for ex in examples:
        if not isinstance(ex, dict):
            continue
        qid = str(ex.get("qid", "ex"))
        # safe filename
        safe_qid = "".join(c if c.isalnum() or c in "-_" else "_" for c in qid)
        out = os.path.join(args.output_dir, f"{safe_qid}.html")
        save_example_html(ex, out)
        written.append({"filename": f"{safe_qid}.html", "ex": ex})

    print(f"Wrote {len(written)} example pages to {args.output_dir}")

    if len(written) > 1 and not args.no_index:
        idx = render_index(written, args.output_dir)
        print(f"Wrote index: {idx}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
