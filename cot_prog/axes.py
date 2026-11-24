"""Single source of truth for the configuration axes of the Bayesian
multi-answer CoT framework.

This module encodes **paper Table 7 (p.22) and the §4 footnote (p.8)** —
the canonical vocabulary. (The p.22 footnote used a different, now-retired
set of names: ``combination_rule`` / ``aggr_rule`` / ``evidence_mode`` /
``answer_prior``. Those names are gone; this is a hard cut.)

Every consumer — the CLI runner, the Streamlit console and the launcher —
imports its option lists, defaults, and help text
from here, so they can never drift apart again.

------------------------------------------------------------------------
Two groups (mirrors Table 7's own split)
------------------------------------------------------------------------
A. Trace elicitation (Fig. 10A) — shape the LLM call / parsed trace.
   These are fixed *at generation time* and baked into ``ckpts/*.json``.
   They are NOT swept by ``--reuse_traces_from``.

B. Program construction (Fig. 10B) — assemble the parsed trace into the
   probabilistic program. These are what the reuse path varies on
   top of a cached trace; they make no LLM calls.

------------------------------------------------------------------------
The two slots (read this before touching the aggregation functions)
------------------------------------------------------------------------
Every Flip is ``Flip(name, primary, [secondary])``. Compiler attributes:
``.p`` = primary, ``.p2`` = secondary, ``.cal`` = DisCalibration(p, p2),
``.mean/.max/.min`` = arithmetic, ``.pool`` = LogitPool(p, p2).

CANONICAL SLOT ORDER (matches Table 7: ``p`` = logit, ``c`` = verbal):
    slot 1 (.p)  = logit signal
    slot 2 (.p2) = verbal signal

  * per-step Flip   (statement_aggregation_fn):
        slot 1 = step token logprob          (logit weight)
        slot 2 = parsed "(confidence: X%)"   (verbal)
  * per-candidate prior Flip (answer_aggregation_fn):
        slot 1 = TF-logit: token logprob of 'correct' vs 'incorrect' from
                 the teacher-forcing call          (cot._tf_priors)
        slot 2 = parsed "(initial confidence: X%)"      (verbal _init_score)

FALLBACK when only one slot is available (e.g. verbal_conf disabled at
generation, or no TF prior): set BOTH slots to that value -> Flip(name, x, [x]).
Never leave the second slot empty: cal/mean/max/min/pool are binary and
``cal`` raises "DisCalibration requires 2 args" on a single-slot Flip.
With (x, x) every function collapses cleanly to x. If neither slot is
available, the answer prior falls back to uniform (0.5, 0.5) -> 1/N.

Option -> attribute mapping used by the program writer:
    "p"  -> ".p"     (logit only)
    "c"  -> ".p2"    (verbal only)            # emitted as p2, see writer
    "cal"/"mean"/"max"/"min"/"pool" -> ".<name>"   (combine both slots)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# --------------------------------------------------------------------------
# Axis descriptor
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Axis:
    name: str                    # canonical identifier (CLI flag = --<name>)
    group: str                   # "elicitation" | "program"
    choices: tuple               # allowed values; () means free/bool
    default: object              # Table 7 default (bolded in the paper)
    help: str                    # one-line help shown in CLI / console
    is_bool: bool = False        # enabled/disabled toggle (BooleanOptionalAction)
    depends_on: Optional[str] = None   # cross-group dependency note
    todo: Optional[str] = None         # not-yet-implemented note


# --------------------------------------------------------------------------
# A. Trace-elicitation axes (baked into the checkpoint at generation time)
# --------------------------------------------------------------------------
ELICITATION_AXES: dict[str, Axis] = {
    "cot_strategy": Axis(
        name="cot_strategy",
        group="elicitation",
        choices=("combined", "separate"),
        default="combined",
        help=("Table 7 '#calls'. combined = 1 call arguing every candidate "
              "(#calls=1); separate = one call per candidate (#calls=N)."),
    ),
    "verbal_conf": Axis(
        name="verbal_conf",
        group="elicitation",
        choices=("enabled", "disabled"),
        default="enabled",
        help=("Table 7 'verbal conf.'. Whether the prompt elicits "
              "'(initial confidence: X%)' and per-step '(confidence: X%)' "
              "tags, which become the verbal (slot-2) signal."),
        is_bool=True,
        todo=("'disabled' path not built yet — needs a non-verbal prompt "
              "template (the verbal template is currently always used). "
              "Default 'enabled' is fully wired."),
    ),
    "verbal_step_deps": Axis(
        name="verbal_step_deps",
        group="elicitation",
        choices=("enabled", "disabled"),
        default="enabled",
        help=("Table 7 'verbal step deps.'. Whether the prompt elicits "
              "'(ref. i, j)' premise references. Only has an EFFECT when "
              "step_dependency_mode != flat."),
        is_bool=True,
        depends_on="step_dependency_mode",
        todo=("'disabled' path not built yet — needs a prompt template that "
              "omits the (ref. ...) requirement. Default 'enabled' is wired."),
    ),
}


# --------------------------------------------------------------------------
# B. Program-construction axes (swept on top of cached traces)
# --------------------------------------------------------------------------
ANSWER_AGG_CHOICES = ("c", "p", "cal", "mean", "max", "min")
STATEMENT_AGG_CHOICES = ("c", "p", "mean", "max", "min", "pool")

PROGRAM_AXES: dict[str, Axis] = {
    "answer_aggregation_fn": Axis(
        name="answer_aggregation_fn",
        group="program",
        choices=ANSWER_AGG_CHOICES,
        default="c",
        help=("Per-candidate prior Flip. Slots: (TF-logit, verbal initial "
              "confidence). c = verbal only; p = logit only; cal = calibrated "
              "combo; mean/max/min = arithmetic combo. Neither slot -> 1/N."),
    ),
    "statement_aggregation_fn": Axis(
        name="statement_aggregation_fn",
        group="program",
        choices=STATEMENT_AGG_CHOICES,
        default="mean",
        help=("Per-step Flip inside the pool. Slots: (token logprob, verbal "
              "step confidence). c = verbal only; p = logit only; "
              "mean/max/min = arithmetic combo; pool = LogitPool of the two."),
    ),
    "pool_mode": Axis(
        name="pool_mode",
        group="program",
        choices=("logitpool",),
        default="logitpool",
        help=("How per-step factors combine into one evidence factor per "
              "candidate. logitpool = sigmoid of the mean step logit "
              "(the only mode used in the paper)."),
    ),
    "step_dependency_mode": Axis(
        name="step_dependency_mode",
        group="program",
        choices=("flat",),
        default="flat",
        help=("How per-step Flips condition on each other. flat = independent "
              "(Table 7's 'gated' variant is not implemented in this release)."),
        depends_on="verbal_step_deps",
    ),
}


# --------------------------------------------------------------------------
# Combined view + helpers
# --------------------------------------------------------------------------
ALL_AXES: dict[str, Axis] = {**ELICITATION_AXES, **PROGRAM_AXES}


def defaults(group: Optional[str] = None) -> dict:
    """Table 7 default value for every axis (optionally one group)."""
    return {
        n: a.default
        for n, a in ALL_AXES.items()
        if group is None or a.group == group
    }


def attr_for(option: str) -> str:
    """Map an aggregation-fn option to the Flip attribute the writer emits.

    'c' selects the verbal (second) slot, which the compiler exposes as
    ``.p2``; everything else is the attribute of the same name.
    """
    return "p2" if option == "c" else option


def add_arguments(parser, group: Optional[str] = None) -> None:
    """Populate an argparse parser from the axis registry.

    Bool axes (verbal_*) use BooleanOptionalAction so both --verbal_conf
    and --no-verbal_conf work; their default is True (== 'enabled').
    """
    from argparse import BooleanOptionalAction

    for name, ax in ALL_AXES.items():
        if group is not None and ax.group != group:
            continue
        if ax.is_bool:
            parser.add_argument(
                f"--{name}",
                action=BooleanOptionalAction,
                default=(ax.default == "enabled"),
                help=ax.help.replace("%", "%%"),  # argparse treats % as a format char
            )
        else:
            parser.add_argument(
                f"--{name}",
                type=str,
                choices=list(ax.choices),
                default=ax.default,
                help=ax.help.replace("%", "%%"),  # argparse treats % as a format char
            )


def validate(config: dict) -> list[str]:
    """Return a list of human-readable warnings for an axis configuration
    (unknown values, or dependency axes that will have no effect)."""
    warnings: list[str] = []
    for name, ax in ALL_AXES.items():
        if name not in config:
            continue
        val = config[name]
        if not ax.is_bool and ax.choices and val not in ax.choices:
            warnings.append(f"{name}={val!r} not in {ax.choices}")
    # Dependency: step_dependency_mode needs verbal_step_deps to matter.
    if config.get("step_dependency_mode", "flat") != "flat" and \
            config.get("verbal_step_deps", True) in (False, "disabled"):
        warnings.append(
            "step_dependency_mode != flat but verbal_step_deps disabled — "
            "no (ref.) edges were elicited, so gating has nothing to gate on.")
    return warnings
