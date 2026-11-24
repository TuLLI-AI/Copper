import json
import re
import streamlit as st
import traceback
import sys
import numpy as np
import pandas as pd
import altair as alt
import argparse
from code_editor import code_editor
from cot_prog.lang import ParseStringCot
from cot_prog.prompt import prompt_to_program
from cot_prog.prompt.multi_answer_cot import run_multi_external, run_mcq_external, _get_model
from cot_prog.program_labels import (
    _PROG_STR_RE,
    answer_names as _answer_names,
    answer_label as _answer_label,
    param_labels as _param_labels,
    param_label as _param_label,
)

PROMPT_PARSER =  argparse.ArgumentParser()
PROMPT_PARSER.add_argument('--client', type=str, default='openai',choices=['openai','together'])
PROMPT_PARSER.add_argument('--max_data', type=int, default=100000000)
PROMPT_PARSER.add_argument('--model_name', type=str, default='gpt-4o')
PROMPT_PARSER.add_argument('--num_retries', type=int, default=4)
PROMPT_PARSER.add_argument('--split', type=str, default='train')
PROMPT_PARSER.add_argument('--query_field', type=str, default='question')
PROMPT_PARSER.add_argument('--temperature', type=float, default=0.0)
PROMPT_PARSER.add_argument('--no_length_norm', type=bool,default=False,action=argparse.BooleanOptionalAction)
PROMPT_PARSER.add_argument('--meta_tokens', type=bool,default=False,action=argparse.BooleanOptionalAction)
PROMPT_PARSER.add_argument('--debug_prompt', type=bool,default=False,action=argparse.BooleanOptionalAction)

DEFAULT_CODE = """
# query: Is Mixed martial arts totally original from Roman Colosseum games?
# 1. The Roman Colosseum games involved various forms of combat and entertainment, including gladiatorial contests.
# 2. Mixed martial arts (MMA) is a modern combat sport that combines techniques from various martial arts disciplines.
# 3. MMA has influences from many historical and cultural combat practices, not just Roman Colosseum games.
# 4. The Colosseum games were primarily spectacles for entertainment, often involving life-or-death scenarios.
# 5. MMA is a regulated sport with rules to ensure the safety of participants, unlike the Colosseum games.
# 6. Based on 1 and 2, MMA has different origins and influences compared to the Colosseum games.
# 7. Based on 3, 4, and 5, MMA is not totally original from Roman Colosseum games.
# 8. Based on 6 and 7, the answer is no.

query: str = "Is Mixed martial arts totally original from Roman Colosseum games?"

## step strings
s1: str = "The Roman Colosseum games involved various forms of combat and entertainment, including gladiatorial contests."
s2: str  = "Mixed martial arts (MMA) is a modern combat sport that combines techniques from various martial arts disciplines."
s3: str = "MMA has influences from many historical and cultural combat practices, not just Roman Colosseum games."
s4: str = "The Colosseum games were primarily spectacles for entertainment, often involving life-or-death scenarios."
s5: str = "MMA is a regulated sport with rules to ensure the safety of participants, unlike the Colosseum games."
s6: str = "MMA has different origins and influences compared to the Colosseum games."
s7: str = "MMA is not totally original from Roman Colosseum games."
s8: str = "no"

## step propositions 
P1: bool =~ Flip(s1,0.6639751890192626)
P2: bool =~ Flip(s2,0.8782548278626059)
P3: bool =~ Flip(s3,0.4392634219466587)
P4: bool =~ Flip(s4,0.505012840730937)
P5: bool =~ Flip(s5,0.6736885262052589)

if P1 and P2:
    P6: bool =~ Flip(s6,0.5358896299288175)
else:
    raise ValueError("Cannot establish steps P1 and P2")

if P3 and P4 and P5:
    P7: bool =~ Flip(s7,0.8578690018532449)
else:
    raise ValueError("Cannot establish steps 3-5")

if P6 and P7:
    P8 =~ Flip(s8,0.9882047124007578)
    yield P8
else:
    raise ValueError("Cannot established steps 6 and 7")

"""


MULTI = """## multi
s1: str = "Paris is the most populous city in France"
s2: str = "Paris is therefore the capital of France."
s3: str = "Lyon is the capital of France"

P1: bool =~ Flip(s1,0.9)
P2: bool =~ Flip(s2, 0.8)
P3: bool =~ Flip(s3, 0.9)

if P1 and P2:
    return  s2
elif not (P1 and P2) and P3:
    return s3
else: 
    raise ValueError('wrong reasoning')

"""





MULTI_AGGR_MIXTURE = """
s1: str = "Paris is the most populous city in France"
s2: str = "Paris is therefore the capital of France."
s3: str = "Lyon is the capital of France"

## answer probabilities
P_A: bool =~ Flip("Paris", 0.87)
L_A: bool =~ Flip("Lyon", 0.45)
P, L  =~ Categorical("Answer", Mixture(P_A.prob, L_A.prob))

## evidence for Paris
P1:  bool =~ Flip(s1,0.9)
P2:  bool =~ Flip(s2, 0.8)
P_E: bool =~ Factor("P_E", LogitPool(P1.prob, P2.prob))

## evidence for Lyon
P3: bool  =~ Flip(s3, 0.9)
L_E: bool =~ Factor("L_E", LogitPool(P3.prob))

observe((P and P_E) or (L and L_E))

if P:
    return s2
elif L:
    return s3
else:
    raise ValueError('Unable to find answer')
"""

# An MCQ-shaped program — same Categorical/Mixture pattern as MULTI_AGGR_MIXTURE
# but generalized to N=4. Mirrors what `run_mcq_external` will emit for a BBH
# question. Useful as a click-and-run example in the editor.
BBH_MCQ_AGGR = """
### original query
query: str = "Three people - Alice, Bob, and Carol - finished a race. (1) Carol finished before Bob. (2) Alice did not finish first. Who finished first?"

### step strings (per-candidate, globally re-indexed)
s1: str = "Constraint (2) says Alice did not finish first."
s2: str = "Alice"
s3: str = "Carol finished before Bob, so Bob is not first."
s4: str = "Bob"
s5: str = "Alice excluded by (2)."
s6: str = "Bob excluded by (1)."
s7: str = "Only Carol remains."
s8: str = "Carol"
s9: str = "Carol is uniquely first; D is wrong."
s10: str = "Cannot be determined"

## answer priors (final_answer_line + initial confidences from the LLM)
A_pr: bool =~ Flip("A", 0.05, [0.05])
B_pr: bool =~ Flip("B", 0.10, [0.05])
C_pr: bool =~ Flip("C", 0.80, [0.85])
D_pr: bool =~ Flip("D", 0.05, [0.05])

A_x, B_x, C_x, D_x =~ Categorical("answer", Mixture(A_pr.cal, B_pr.cal, C_pr.cal, D_pr.cal))

## evidence for A
P1:  bool =~ Flip(s1, 0.85)
A_E: bool =~ Factor("A_E", P1.mean)

## evidence for B
P3:  bool =~ Flip(s3, 0.90)
B_E: bool =~ Factor("B_E", P3.mean)

## evidence for C
P5:  bool =~ Flip(s5, 0.92)
P6:  bool =~ Flip(s6, 0.90)
P7:  bool =~ Flip(s7, 0.88)
C_E: bool =~ Factor("C_E", LogitPool(P5.mean, P6.mean, P7.mean))

## evidence for D
P9:  bool =~ Flip(s9, 0.65)
D_E: bool =~ Factor("D_E", P9.mean)

observe((A_x and A_E) or (B_x and B_E) or (C_x and C_E) or (D_x and D_E))

if A_x:
    return s2
elif B_x:
    return s4
elif C_x:
    return s8
elif D_x:
    return s10
else:
    raise ValueError('Unable to find answer')
"""

# Example programs offered at the bottom of the page: one per prompt strategy.
EXAMPLE_PROGRAMS = {
    "Vanilla CoT (single chain)": DEFAULT_CODE,
    "Multi-answer CoT": MULTI,
    "Bayesian multi-answer CoT (two answers)": MULTI_AGGR_MIXTURE,
    "Bayesian multi-answer CoT (multiple choice)": BBH_MCQ_AGGR,
}


def _load_example():
    """Put the chosen example program into the editor."""
    choice = st.session_state.get("example_choice")
    if choice:
        st.session_state["running_code"] = EXAMPLE_PROGRAMS[choice]


STRATEGIES = {
    "vanilla CoT"      : "generic",
    "multi answer CoT" : "multi_answer",
    #"disjunctive CoT"  : "generic"
    "Bayesian"         : "generic"
}
MODEL_LIST = {
    "gpt-4o" : "openai",
    "gpt-4.1": "openai",
    "gpt-4o-mini" : "openai",
    #"Qwen/Qwen3-Next-80B-A3B-Instruct" : "together",
    #"Qwen/Qwen3.5-397B-A17B" : "together",
    #"openai/gpt-oss-120b" : "together",
    #"moonshotai/Kimi-K2-Instruct-0905" : "together",
    #"moonshotai/Kimi-K2.5" : "together",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": "together",
    "mistralai/Mixtral-8x7B-Instruct-v0.1" : "together",
    "deepseek-ai/DeepSeek-V3.1" : "together",
    "Qwen/Qwen2.5-7B-Instruct-Turbo" : "together",
    "meta-llama/Meta-Llama-3-8B-Instruct-Lite" : "together",
    "arize-ai/qwen-2-1.5b-instruct" : "together",
    "google/gemma-3n-E4B-it": "together"
}
    

_MCQ_CHOICE_RE = __import__("re").compile(
    r"^\s*\(?([A-Za-z])\)?\s*[\.\):\-]\s*(.+?)\s*$"
)

# BBH-flavoured constraint-deduction example — pre-filled when the user
# selects "multiple_choice" in the Bayesian config so they see the expected
# format and can run end-to-end with a single click.
BBH_MCQ_EXAMPLE_QUERY = (
    "Three people - Alice, Bob, and Carol - finished a race. "
    "(1) Carol finished before Bob. (2) Alice did not finish first. "
    "Who finished first?"
)
BBH_MCQ_EXAMPLE_CHOICES = (
    "A. Alice\n"
    "B. Bob\n"
    "C. Carol\n"
    "D. Cannot be determined"
)


def _parse_mcq_choices(text: str):
    """Parse a textarea value into (choice_labels, choice_texts).

    Accepts lines like:
        A. heart
        (A) heart
        A) heart
        A: heart
    Returns ([], []) when ``text`` is empty / unparsable (binary fallback).
    """
    labels, texts = [], []
    if not text or not text.strip():
        return [], []
    for line in text.splitlines():
        if not line.strip():
            continue
        m = _MCQ_CHOICE_RE.match(line)
        if not m:
            continue
        labels.append(m.group(1).upper())
        texts.append(m.group(2).strip())
    return (labels, texts) if len(labels) >= 2 else ([], [])


# ---------------------------------------------------------------------------
# Chain-of-thought view
# ---------------------------------------------------------------------------
_COT_HEADER_RE = re.compile(
    r'^#+\s*For\s+(?P<label>.+?)\s*\(initial confidence:\s*(?P<conf>[\d.]+)\s*%?\)\s*:?\s*$', re.I)
_COT_ALT_HEADER_RE = re.compile(r'^#+\s*(?P<label>answer\s+\d+|final answer[^:]*)\s*:?\s*$', re.I)
_COT_STEP_RE = re.compile(r'^\s*(?P<num>\d+)\.\s+(?P<body>.*)$')
_COT_REF_RE = re.compile(r'\(\s*ref\.\s*(?P<v>[^)]*)\)', re.I)
_COT_CONF_RE = re.compile(r'\(\s*confidence:\s*(?P<v>[\d.]+)\s*%?\s*\)', re.I)
_COT_FINAL_RE = re.compile(r'\(\s*final answer:\s*(?P<v>[^)]*)\)', re.I)
_PROG_FLIP_RE = re.compile(r'^\w+\s*:\s*bool\s*=~\s*(?:Flip|Bernoulli)\(\s*(s\d+)\s*,\s*([\d.eE+-]+)')


def _norm_text(text):
    return " ".join(str(text).replace('\\"', '"').split()).strip().rstrip(".")


def _token_probs_from_program(program):
    """Map each step statement in a generated program to its token probability."""
    strings, probs = {}, {}
    for line in (program or "").splitlines():
        m = _PROG_STR_RE.match(line.strip())
        if m:
            strings[m.group(1)] = _norm_text(m.group(2))
            continue
        m = _PROG_FLIP_RE.match(line.strip())
        if m and m.group(1) in strings:
            try:
                probs[strings[m.group(1)]] = float(m.group(2))
            except ValueError:
                pass
    return probs


def _lookup_token_prob(token_probs, claim, answer=""):
    """Find a step's token probability in the program.

    The program stores a cleaned statement: a leading "Based on 1, 2, ..."
    is dropped, and a vanilla answer line is stored as just the answer. So an
    exact match is tried first, then the stored statement as the ending of
    the displayed one, then the bare answer.
    """
    key = _norm_text(claim)
    if key in token_probs:
        return token_probs[key]
    endings = [k for k in token_probs if len(k) >= 12 and key.endswith(k)]
    if endings:
        return token_probs[max(endings, key=len)]
    ans = _norm_text(answer).strip("'\"")
    if ans and ans in token_probs:
        return token_probs[ans]
    return None


def _parse_cot_sections(raw, program):
    """Split the raw model output into per-answer sections of numbered steps."""
    token_probs = _token_probs_from_program(program)
    sections, current = [], None
    for line in (raw or "").splitlines():
        if not line.strip():
            continue
        head = _COT_HEADER_RE.match(line.strip())
        alt = None if head else _COT_ALT_HEADER_RE.match(line.strip())
        if head or alt:
            current = {
                "title": (head or alt).group("label").strip(),
                "initial": float(head.group("conf")) if head else None,
                "steps": [], "notes": [],
            }
            sections.append(current)
            continue
        if current is None:
            current = {"title": "Reasoning", "initial": None, "steps": [], "notes": []}
            sections.append(current)
        step = _COT_STEP_RE.match(line)
        if not step:
            current["notes"].append(line.strip())
            continue
        body = step.group("body")
        ref = _COT_REF_RE.search(body)
        conf = _COT_CONF_RE.search(body)
        final = _COT_FINAL_RE.search(body)
        claim = body
        for rx in (_COT_REF_RE, _COT_CONF_RE, _COT_FINAL_RE):
            claim = rx.sub("", claim)
        claim = claim.strip()
        conf_val = float(conf.group("v")) if conf else None
        if conf_val is not None and conf_val <= 1.0:
            conf_val *= 100.0          # "0.8" style -> percent
        tok = _lookup_token_prob(token_probs, claim, final.group("v") if final else "")
        ref_val = ref.group("v").strip() if ref else ""
        current["steps"].append({
            "Step": int(step.group("num")),
            "Statement": claim,
            "Answer": final.group("v").strip() if final else "",
            "Depends on steps": "" if ref_val.lower() == "none" else ref_val,
            "Stated confidence": conf_val,
            "Token probability": None if tok is None else round(tok * 100.0, 1),
        })
    return sections


def _render_cot(view):
    """Show the model's chain of thought as one table per candidate answer."""
    st.subheader("Chain of thought")
    st.caption(f"Query: {view['query']}")
    sections = _parse_cot_sections(view.get("raw"), view.get("program"))
    show_conf = view.get("verbal_conf", True)
    show_refs = view.get("verbal_step_deps", True)
    if not any(sec["steps"] for sec in sections):
        st.code(view.get("raw") or "(no model output)")

    # Answer priors: what the program starts from for each candidate answer,
    # before any reasoning step is taken into account.
    tf_priors = view.get("tf_priors") or {}
    prior_rows = []
    for sec in sections:
        logit = tf_priors.get(sec["title"])
        if sec["initial"] is None and logit is None:
            continue
        prior_rows.append({
            "Answer": sec["title"],
            "Stated initial confidence": sec["initial"],
            "Logit prior": None if logit is None else round(float(logit) * 100.0, 1),
        })
    if prior_rows:
        st.markdown("**Answer priors**")
        df_prior = pd.DataFrame(prior_rows)
        prior_cols = ["Answer"]
        if show_conf and df_prior["Stated initial confidence"].notna().any():
            prior_cols.append("Stated initial confidence")
        if df_prior["Logit prior"].notna().any():
            prior_cols.append("Logit prior")
        st.dataframe(
            df_prior[prior_cols],
            hide_index=True,
            column_config={
                "Stated initial confidence": st.column_config.ProgressColumn(
                    "Stated initial confidence", format="%.0f%%", min_value=0, max_value=100),
                "Logit prior": st.column_config.ProgressColumn(
                    "Logit prior", format="%.1f%%", min_value=0, max_value=100),
            },
        )
        st.markdown("**Reasoning for each answer**")
    for sec in sections:
        title = f"**{sec['title']}**"
        if sec["initial"] is not None and show_conf:
            title += f" (initial confidence: {sec['initial']:g}%)"
        st.markdown(title)
        for note in sec["notes"]:
            st.write(note)
        if not sec["steps"]:
            continue
        df = pd.DataFrame(sec["steps"])
        cols = ["Step", "Statement"]
        if df["Answer"].astype(bool).any():
            cols.append("Answer")
        if show_refs:
            cols.append("Depends on steps")
        if show_conf and df["Stated confidence"].notna().any():
            cols.append("Stated confidence")
        if df["Token probability"].notna().any():
            cols.append("Token probability")
        st.dataframe(
            df[cols],
            hide_index=True,
            column_config={
                "Statement": st.column_config.TextColumn("Statement", width="large"),
                "Stated confidence": st.column_config.ProgressColumn(
                    "Stated confidence", format="%.0f%%", min_value=0, max_value=100),
                "Token probability": st.column_config.ProgressColumn(
                    "Token probability", format="%.1f%%", min_value=0, max_value=100),
            },
        )
    if view.get("tf_priors"):
        st.caption("An empty token probability on an answer line is expected: the "
                   "program does not use that line as evidence. The answer enters "
                   "through the answer priors above.")
    with st.expander("Raw model output"):
        st.code(view.get("raw") or "")
        st.json({"attempts": view.get("attempts"), "errors": view.get("errors")}, expanded=False)


# Pre-filled in the query box (the worked example used in the prompts).
EXAMPLE_QUERY = "Could the members of the Police perform lawful arrests?"


def main(argv):
    config = PROMPT_PARSER.parse_args(argv)
    
    st.title("Probabilistic LLM Programming")
    user_input = st.text_area(
        "Enter your (optional) query here",
        value=EXAMPLE_QUERY,
    )

    col1, col2 = st.columns(2)
    step_dependency_mode = "flat"
    cot_strategy = pool_mode = answer_aggregation_fn = statement_aggregation_fn = None
    verbal_conf = verbal_step_deps = True
    question_type = "binary"     # 'binary' or 'multiple_choice' — set inside the Bayesian popover
    mcq_choices_text = ""        # raw MCQ option block; only used when question_type == 'multiple_choice'

    with col1:

        # strategy = st.pills(
        #     "Prompt Strategy",
        #     options=STRATEGIES,
        #     default='vanilla CoT'
        # )
        strategy = st.pills("Prompt Strategy", ["vanilla CoT", "multi answer CoT", "Bayesian"], default='vanilla CoT')
        if strategy == "Bayesian":
            with st.popover("⚙️ Bayesian Config"):
                # NEW: explicit toggle so we never have to guess MCQ vs binary
                # from textarea contents.
                question_type = st.radio(
                    "Question type",
                    ["binary", "multiple_choice"],
                    horizontal=True,
                    help=(
                        "binary: yes/no questions (e.g. StrategyQA).\n"
                        "multiple_choice: N-way MCQ (e.g. BBH-MCQ-Hard) — "
                        "you'll be asked for the option list below."
                    ),
                )
                # Trace-elicitation axes (Table 7, Fig. 10A) — shape the LLM
                # call. cot_strategy == Table 7 '#calls' (combined=1, separate=N).
                cot_strategy = st.selectbox("CoT Strategy (#calls: combined=1, separate=N)",
                                            ["combined", "separate"])
                verbal_conf = st.radio(
                    "verbal_conf: use the model's stated confidences",
                    ["enabled", "disabled"], horizontal=True,
                    help=("Table 7 'verbal conf.'. enabled: the '(initial confidence: X%)' "
                          "and '(confidence: X%)' tags are used as the verbal signal. "
                          "disabled: they are ignored and only the logit-based "
                          "setting `p` can be chosen below. The prompt asks the model "
                          "for the tags either way."),
                ) == "enabled"
                verbal_step_deps = st.radio(
                    "verbal_step_deps: use the (ref. i, j) step references",
                    ["enabled", "disabled"], horizontal=True,
                    help=("Table 7 'verbal step deps.'. In this release only the "
                          "`flat` premise-dependency is available, so the references "
                          "do not change the program; this choice controls whether "
                          "they are shown in the chain-of-thought view."),
                ) == "enabled"
                # Program-construction axes (Table 7, Fig. 10B).
                pool_mode = "logitpool"  # the only pool mode in the release
                # answer_aggregation_fn folds in the old answer-prior source:
                # c=verbal init, p=TF-logit, cal/mean/max/min=combine the two
                # slots (TF-logit, verbal init). Default c (the paper's setting).
                # Without the verbal signal only the logit slot exists, so the
                # options that read or combine the verbal slot are not offered.
                if verbal_conf:
                    answer_options = ["c", "p", "cal", "mean", "max", "min"]
                    statement_options = ["mean", "p", "c", "max", "min", "pool"]
                else:
                    answer_options = ["p"]
                    statement_options = ["p"]
                    st.caption("verbal_conf is disabled: only the logit-based "
                               "setting `p` is available for both aggregation choices.")
                answer_aggregation_fn = st.selectbox(
                    "answer_aggregation_fn", answer_options,
                    help=("Per-candidate prior. Slots: (TF-logit, verbal init). "
                          "c=verbal only; p=token-logprob prior from the "
                          "correct/incorrect teacher-forcing call; "
                          "cal/mean/max/min=combine."),
                )
                statement_aggregation_fn = st.selectbox(
                    "statement_aggregation_fn", statement_options,
                    help=("Per-step. Slots: (logprob, verbal conf). c=verbal only; "
                          "p=logit only; mean/max/min=combine; pool=LogitPool."),
                )
                step_dependency_mode = st.selectbox(
                    "Premise-dependency",
                    ["flat"],
                    help=(
                        "How to wire each step's `(ref. ...)` premises into the "
                        "program:\n"
                        "• flat: every evidence step is independent; the per-"
                        "candidate pool is over all of them."
                    ),
                )

                if question_type == "multiple_choice":
                    # Pre-fill with a BBH-flavoured constraint-deduction example
                    # so the user can see the expected format at a glance.
                    mcq_choices_text = st.text_area(
                        "Choices (one per line; 'A. text', '(A) text', 'A) text', or 'A: text')",
                        value=BBH_MCQ_EXAMPLE_CHOICES,
                        height=160,
                        help=(
                            "One option per line. The leading letter becomes the "
                            "candidate label that the model commits to via "
                            "(final answer: <letter>)."
                        ),
                    )
                    st.caption(
                        "Example query (paste into the box at top): "
                        f"\n> {BBH_MCQ_EXAMPLE_QUERY}"
                    )

        instruction = STRATEGIES[strategy]

    with col2:

        model_name = st.selectbox(
            "Prompting model",
            list(MODEL_LIST.keys()),
            #index=None,
            placeholder=config.model_name,
        )

    generate_clicked = st.button("Generate Program")
    if generate_clicked and not user_input.strip():
        st.warning(f"Enter a query first, for example: {EXAMPLE_QUERY}")
    if generate_clicked and user_input.strip():
        if strategy == 'Bayesian':
            if question_type == "multiple_choice":
                mcq_labels, mcq_texts = _parse_mcq_choices(mcq_choices_text)
                if not mcq_labels:
                    st.error(
                        "Question type is 'multiple_choice' but no valid options "
                        "were parsed from the Choices box. Use one option per "
                        "line, e.g. `A. heart`."
                    )
                    return
                # MCQ path — N-way Categorical/Mixture
                cot, cot_code, _per_label = run_mcq_external(
                    query=user_input.strip(),
                    choice_labels=mcq_labels,
                    choice_texts=mcq_texts,
                    strategy=(cot_strategy or "combined"),
                    pool_mode=(pool_mode or "logitpool"),
                    answer_aggregation_fn=(answer_aggregation_fn or "c"),
                    statement_aggregation_fn=(statement_aggregation_fn or "mean"),
                    step_dependency_mode=(step_dependency_mode or "flat"),
                    model_name=model_name,
                )
            else:
                # Binary path — same combination + evidence knobs as MCQ
                cot, cot_code = run_multi_external(
                    query=user_input.strip(),
                    strategy=(cot_strategy or "combined"),
                    model_name=model_name,
                    pool_mode=(pool_mode or "logitpool"),
                    answer_aggregation_fn=(answer_aggregation_fn or "c"),
                    statement_aggregation_fn=(statement_aggregation_fn or "mean"),
                    candidates=['yes', 'no'],
                    step_dependency_mode=(step_dependency_mode or "flat"),
                )
            #cot_code = ""

        else:
            cot = prompt_to_program(
                query=user_input.strip(),
                model_name=model_name,
                instruction=instruction, #config.instruction,
                temperature=0.0,
                client=MODEL_LIST[model_name],
                # Same key routing as the Bayesian path (OpenAI, OpenRouter or Together).
                llm=_get_model(model_name=model_name),
                strict=False,
            )
            cot_code = cot.write_program()

        if cot is not None:
            st.session_state["running_code"] = cot_code #cot.write_program()
            # Kept in the session so the view stays when Execute is pressed.
            st.session_state["last_cot"] = {
                "query": user_input,
                "raw": cot.raw_cot,
                "attempts": cot.attempts,
                "errors": cot.errors,
                "program": cot_code,
                "verbal_conf": verbal_conf,
                "verbal_step_deps": verbal_step_deps,
                "tf_priors": getattr(cot, "_tf_priors", None),
            }

    if st.session_state.get("last_cot"):
        _render_cot(st.session_state["last_cot"])
        st.subheader("Program")

    code = code_editor(
        code=st.session_state.get("running_code",DEFAULT_CODE),
        lang="python",
        allow_reset=True,
        response_mode="debounce",
        theme="dark",
        options={"showLineNumbers": True},
    )

    output_options = st.multiselect(
        "Output options",
        ["sample", "semantics", "circuit", "gradient"],
        default=[],
    )
    
    if st.button("Execute"):
        to_execute = code['text'] if code["text"] else st.session_state.get("running_code",DEFAULT_CODE)
        prog = ParseStringCot(to_execute)
        answer_names = _answer_names(to_execute)   # s5 -> "yes"
        param_labels = _param_labels(to_execute)   # P5 -> "Step 1 for C"

        with st.spinner(text="executing program..."):
            # The exact inference result always comes first on the page.
            result_box = st.container()

            if "semantics" in output_options:
                expander = st.expander("Program semantics")
                expander.code(prog.semantics(pretty=True))

            if "sample" in output_options:
                expander = st.expander("Empirical sample")
                sample_out = prog.sample(tries=5000)
                
                attempts = sample_out["attempts"]
                executed = sample_out["executed"]
                exc_rate = executed / attempts
                non_executed = (attempts - executed) / attempts

                df_bar = pd.DataFrame({
                    "prob": [exc_rate, non_executed],
                    "Termination": ["executed","exception"],
                })
                expander.bar_chart(df_bar, x="Termination", y="prob")

                if sample_out.get("return_values") and len(sample_out["return_values"]) > 1:
                    denom = sum(sample_out["return_values"].values())
                    r_value = {r : c/denom for r,c in sample_out["return_values"].items()}
                    r = r_value.keys()
                    
                    df_ret = pd.DataFrame({
                        "prob"   : [r_value[x] for x in r],
                        "return value" : r,
                    })
                    expander.bar_chart(df_ret, x="return value", y="prob")
                    
            try:
                circuit = prog.compile_circuit()
            except Exception:
                circuit = None
                result_box.code(traceback.format_exc())

            if circuit is not None:
                compute_grad = True if "gradient" in output_options else False

                ##
                if True:   # exact inference is always run and shown

                    inference = result_box
                    
                    out_var = []
                    out_val = []
                    grads = {}

                    for return_var in circuit.return_vars: 
                        #count = circuit.inference(marginals={return_var})
                        count = prog.inference(query={return_var}, compute_grad=compute_grad)
                        out_var.append(_answer_label(answer_names, return_var.replace("A_","")))
                        out_val.append(count.out)

                        if count.param_grads is not None: 
                            grads[return_var] = count.param_grads

                    #exception = circuit.inference(marginals={"Exc"})
                    exception = prog.inference(query={"Exc"}, compute_grad=compute_grad)
                    out_var.append("No answer")
                    out_val.append(exception.out)

                    if True:
                        df = pd.DataFrame({"answer": out_var, "prob": out_val})
                        df["prob_pct"] = (df["prob"] * 100).round(1)
                        inference.subheader("Inference result")
                        answers_only = df[df["answer"] != "No answer"]
                        if len(answers_only) and answers_only["prob"].max() > 0:
                            best = answers_only.loc[answers_only["prob"].idxmax()]
                            inference.markdown(
                                f"Most probable answer: **{best['answer']}** "
                                f"({best['prob_pct']:.1f}%)")
                        inference.dataframe(
                            df[["answer", "prob_pct"]],
                            column_config={
                                "prob_pct": st.column_config.ProgressColumn(
                                    "probability", format="%.1f%%", min_value=0, max_value=100
                                    ),
                                    "class": "Class",
                            },
                            hide_index=True,
                        )
                    if grads: 
                        grad_expander = st.expander("Sensitivity analysis")

                        all_grad_rows = []
                        for return_var, param_grads in grads.items():
                            for param_name, grad_val in param_grads:
                                if grad_val != 0.0:
                                    all_grad_rows.append({
                                        "return_var": return_var.replace("A_", ""),
                                        "parameter": _param_label(param_labels, param_name),
                                        "label": _param_label(param_labels, param_name, short=True),
                                        "name in program": param_name,
                                        "gradient": grad_val,
                                        "sensitivity": abs(grad_val),
                                        #"|∂/∂param|": abs(grad_val),
                                    })

                        df_grads = pd.DataFrame(all_grad_rows)
                        
                        if "return_var" in df_grads.columns:
                            for rv in df_grads["return_var"].unique():
                                grad_expander.subheader(
                                    f"Sensitivity of the answer: {_answer_label(answer_names, rv)}")
                                grad_expander.caption(
                                    "How much the probability of this answer changes when "
                                    "each parameter changes. Positive gradient: the parameter "
                                    "supports this answer; negative: it works against it.")
                                subset = df_grads[df_grads["return_var"] == rv].sort_values("sensitivity", ascending=False)

                                # Largest effect first; short labels shown in full.
                                chart_df = subset.assign(effect=subset["gradient"].map(
                                    lambda g: "raises this answer" if g > 0 else "lowers this answer"))
                                grad_expander.altair_chart(
                                    alt.Chart(chart_df).mark_bar().encode(
                                        x=alt.X("sensitivity:Q", title="sensitivity (size of the gradient)"),
                                        y=alt.Y("label:N", sort="-x", title=None,
                                                axis=alt.Axis(labelLimit=0, labelOverlap=False)),
                                        color=alt.Color(
                                            "effect:N", title=None,
                                            scale=alt.Scale(
                                                domain=["raises this answer", "lowers this answer"],
                                                range=["#2a78d5", "#e34948"]),
                                            legend=alt.Legend(orient="top")),
                                        tooltip=["parameter", "name in program", "gradient"],
                                    # One fixed-height row per bar, so every label is drawn.
                                    ).properties(height=alt.Step(26)),
                                    use_container_width=True,
                                )
                                grad_expander.dataframe(
                                    subset[["parameter", "name in program", "gradient", "sensitivity"]].reset_index(drop=True),
                                    hide_index=True,
                                )


                if "circuit" in output_options: 
                    dot_rep = circuit.dot()
                    graphviz = st.expander("Compiled circuit")
                    graphviz.graphviz_chart(dot_rep)

    # Example programs: optional, so they sit at the end of the page. Picking
    # one loads it into the editor above; then press Execute.
    st.divider()
    st.selectbox(
        "Example programs (optional): load one into the editor",
        list(EXAMPLE_PROGRAMS.keys()),
        index=None,
        placeholder="choose an example...",
        key="example_choice",
        on_change=_load_example,
    )

if __name__ == "__main__":
    main(sys.argv[1:]) 
