"""Multi-answer CoT pipeline: classify question type, discover candidates,
generate per-answer evidence chains, and assemble a MULTI_AGGR probabilistic program."""
import os
import sys
import math
import re
import json
import logging
import datetime
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union

from openai import OpenAI
from together.client import Together

from cot_prog.prompt.cot_text_parser import CoTOutput, CoTContreteSyntaxParser, sanitize_refs
from cot_prog.axes import attr_for


def _is_qwen_model(name: Optional[str]) -> bool:
    """Substring check for Qwen models across all providers.

    Together names look like ``Qwen/Qwen2.5-7B-Instruct-Turbo``; OpenRouter
    names look like ``qwen/qwen-2.5-7b-instruct``. Substring match on
    ``qwen`` (case-insensitive) covers both without false positives —
    neither OpenAI nor Llama nor Gemma model strings contain ``qwen``.
    """
    return bool(name) and "qwen" in str(name).lower()
from cot_prog.prompt.prompts import (
    DISCOVERY_COT_INSTRUCTION,
    PER_ANSWER_COT_INSTRUCTION,
    PER_ANSWER_COT_INSTRUCTION_VERBAL,
    USER_TEMPLATE,
    COMBINED_COT_INSTRUCTION,
    COMBINED_USER_TEMPLATE,
)

LLMType = Union[Together, OpenAI]

# Bare model names that belong to OpenAI ("gpt-4o", "o1-mini", ...).
_OPENAI_NAME_PREFIXES = ("gpt-", "chatgpt-", "o1", "o3", "o4")


def _openai_client() -> OpenAI:
    """Direct OpenAI client that also accepts OpenRouter-style names.

    OpenAI itself only knows bare names, so "openai/gpt-4o" is sent as
    "gpt-4o". This lets the same --model_name work with either key.
    """
    client = OpenAI()
    orig_create = client.chat.completions.create

    def create_with_bare_name(*args, **kwargs):
        m = kwargs.get("model")
        if isinstance(m, str) and m.startswith("openai/"):
            kwargs["model"] = m[len("openai/"):]
        return orig_create(*args, **kwargs)

    client.chat.completions.create = create_with_bare_name
    return client

logger = logging.getLogger(__name__)

__all__ = [
    "classify_question_type",
    "teacher_forcing_logprobs_parallel",
    "multi_answer_cot_to_program",
    "MultiAnswerCoTOutput",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BINARY_PATTERNS = [
    r"\byes\s*/\s*no\b",
    r"\byes\s+or\s+no\b",
    r"\btrue\s*/\s*false\b",
    r"\btrue\s+or\s+false\b",
    r"\banswer\s+only\s+yes\b",
    r"\banswer\s+yes\s+or\s+no\b",
]
_BINARY_RE = re.compile("|".join(_BINARY_PATTERNS), re.IGNORECASE)

_BINARY_START_WORDS = (
    "is ", "are ", "was ", "were ", "do ", "does ", "did ",
    "can ", "could ", "should ", "would ", "has ", "have ", "had ",
    "will ", "shall ",
)


def classify_question_type(query: str) -> str:
    """Classify a question as ``'binary'`` (yes/no) or ``'open_ended'``.

    Uses lightweight keyword heuristics -- no LLM call required.

    Arguments:
        query: The raw question string.
    Returns:
        ``'binary'`` or ``'open_ended'``.
    """
    q = query.strip()
    if _BINARY_RE.search(q):
        return "binary"
    ql = q.lower().lstrip()
    for w in _BINARY_START_WORDS:
        if ql.startswith(w):
            return "binary"
    return "open_ended"


def _sanitize_name(name: str) -> str:
    """Turn an answer string into a valid Python identifier fragment."""
    s = re.sub(r"[^A-Za-z0-9_]", "_", name.strip())
    s = re.sub(r"_+", "_", s).strip("_")
    if not s or s[0].isdigit():
        s = "A" + s
    return s


def _openrouter_client() -> OpenAI:
    """OpenAI-compatible client for OpenRouter with provider pinning.

    OpenRouter load-balances each model across backend providers, and many
    providers silently drop ``logprobs``/``top_logprobs`` (returning
    ``choice.logprobs = None``), which breaks the step-weight pipeline with
    "Unrecognized logprobs format". Injecting
    ``provider: {require_parameters: true}`` into every request makes
    OpenRouter route ONLY to providers that support all requested parameters.
    """
    import os
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    client = OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")
    orig_create = client.chat.completions.create

    def create_with_provider_pin(*args, **kwargs):
        # OpenRouter needs vendor-prefixed names: "gpt-4o" -> "openai/gpt-4o".
        m = kwargs.get("model")
        if isinstance(m, str) and "/" not in m and m.startswith(_OPENAI_NAME_PREFIXES):
            kwargs["model"] = "openai/" + m
        eb = dict(kwargs.get("extra_body") or {})
        prov = dict(eb.get("provider") or {})
        prov.setdefault("require_parameters", True)
        eb["provider"] = prov
        kwargs["extra_body"] = eb
        return orig_create(*args, **kwargs)

    client.chat.completions.create = create_with_provider_pin
    return client


def _get_model(client: Optional[str] = None, model_name: Optional[str] = None) -> LLMType:
    """Build an LLM client.

    Provider-aware routing — looks at the model name first, then falls back
    on whichever API key is set:

      1. ``gpt-*`` / ``openai/*`` / known OpenAI names → OpenAI direct
         (``OPENAI_API_KEY``) if set, else OpenRouter, else Together.
      2. ``vendor/model`` style (``meta-llama/...``, ``Qwen/...``, etc.)
         → Together (``TOGETHER_API_KEY``) if set, else OpenRouter,
         else vanilla OpenAI.
      3. No model-name hint → first available provider:
         Together → OpenRouter → OpenAI.

    The ``client`` arg is a legacy override accepting strings like
    ``"together"`` or ``"openai"``.
    """
    import os
    if client and "together" in str(client).lower():
        return Together()
    if client and "openrouter" in str(client).lower():
        key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        return _openrouter_client()

    together_key   = os.environ.get("TOGETHER_API_KEY", "").strip()
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    openai_key     = os.environ.get("OPENAI_API_KEY", "").strip()

    name = str(model_name or "")
    looks_like_openai = bool(
        name and (
            name.startswith("gpt-")
            or name.startswith("openai/")
            or name in {"gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-4-turbo",
                        "gpt-3.5-turbo", "o1", "o1-mini", "o1-preview"}
        )
    )
    # Together model names look like "meta-llama/..." or "Qwen/..." — vendor/model
    # but NOT "openai/...".
    looks_like_together = bool(
        name and "/" in name and not name.startswith("openai/")
    )

    # Provider hint from the model name takes priority over key-availability.
    if looks_like_openai:
        if openai_key:
            return _openai_client()
        if openrouter_key:
            return _openrouter_client()
        # Last resort — Together will probably 404 on gpt-*, but at least
        # the user gets a clear error pointing at the missing key.
        if together_key:
            return Together()
        return _openai_client()

    if looks_like_together:
        if together_key:
            return Together()
        if openrouter_key:
            return _openrouter_client()
        return _openai_client()

    # No model-name hint — fall back to whatever key is available.
    if openrouter_key:
        return _openrouter_client()
    if together_key:
        return Together()
    return _openai_client()


# ---------------------------------------------------------------------------
# Step 1: Discover candidate answers
# ---------------------------------------------------------------------------

def discover_candidates(
        query: str,
        llm: LLMType,
        model_name: str,
        n: int = 5,
        temperature: float = 0.7,
        top_k: int = 2,
) -> List[str]:
    """Sample *n* CoT responses and extract unique candidate answers.

    Arguments:
        query: The question to answer.
        llm: The LLM client.
        model_name: Model name for the API call.
        n: Number of independent samples.
        temperature: Sampling temperature (>0 for diversity).
        top_k: Maximum number of distinct candidates to return.
    Returns:
        A list of up to *top_k* distinct answer strings, ordered by
        frequency (most common first).
    """
    user_msg = f"Question: {query}\nAnswer (step-by-step):\n"
    system_prompt = DISCOVERY_COT_INSTRUCTION.format(errors="No errors, first attempt")
    answers: List[str] = []

    for i in range(n):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_msg}]},
        ]
        try:
            response = llm.chat.completions.create(
                model=model_name,
                temperature=temperature,
                messages=messages,
                logprobs=False,
            )
            raw = response.choices[0].message.content or ""
            logger.debug(f"[DISCOVERY {i + 1}/{n}] Raw LLM output:\n{raw}\n{'=' * 40}")
            m = CoTContreteSyntaxParser.FINAL_RE.search(raw)
            if m:
                ans = m.group("answer").strip().lower()
                if ans:
                    answers.append(ans)
                    logger.debug(f"[DISCOVERY {i + 1}/{n}] Parsed answer: {ans}")
            else:
                logger.debug(f"[DISCOVERY {i + 1}/{n}] No (final answer: ...) found in output!")
        except Exception as exc:
            logger.warning("Discovery sample failed: %s", exc)
            logger.debug(f"[DISCOVERY {i + 1}/{n}] Exception: {exc}")
            continue

    if not answers:
        raise ValueError(
            "Could not discover any candidate answers from the LLM. "
            "Try increasing n or adjusting the query."
        )

    counts = Counter(answers)
    top = [ans for ans, _ in counts.most_common(top_k)]
    # Restore original casing by picking the first occurrence for each
    seen: Dict[str, str] = {}
    for raw_ans in answers:
        key = raw_ans.lower()
        if key not in seen:
            seen[key] = raw_ans
    return [seen.get(a, a) for a in top]


# ---------------------------------------------------------------------------
# Step 2: Generate per-answer structured CoT
# ---------------------------------------------------------------------------

def _find_logps_import():
    """Lazy import of find_logps to avoid circular dependency."""
    from cot_prog.prompt.prompt_to_program import find_logps
    return find_logps


def generate_per_answer_cot(
        query: str,
        candidate: str,
        llm: LLMType,
        model_name: str,
        num_retries: int = 5,
        verbalization: bool = False,
        remove_meta_tokens: bool = True,
        length_normalization: bool = True,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        strict: bool = True,
        debug: bool = False,
) -> Optional[CoTOutput]:
    """Generate a structured CoT evidence chain for *candidate*.

    Follows the same retry-with-error-feedback loop as
    ``prompt_to_program()``.

    Returns:
        A :class:`CoTOutput` with logprob weights, or ``None`` on failure.
    """
    find_logps = _find_logps_import()

    # template = (
    #     PER_ANSWER_COT_INSTRUCTION_VERBAL if verbalization
    #     else PER_ANSWER_COT_INSTRUCTION
    # )
    template = PER_ANSWER_COT_INSTRUCTION_VERBAL
    user_msg = USER_TEMPLATE.format(query=query, candidate=candidate)
    error_msg = "No errors, first attempt"
    attempts = 0
    parsing_errors: List[str] = []
    cot_program: Optional[CoTOutput] = None

    while True:
        attempts += 1
        if attempts > num_retries:
            break

        logger.debug(f"[PER_ANSWER '{candidate}'] Attempt {attempts}/{num_retries}")

        system_prompt = template.format(
            candidate_answer=candidate,
            errors=error_msg,
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_msg}]},
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
            logger.debug(f"[PER_ANSWER '{candidate}'] API error: {exc}")
            continue

        raw_message = response.choices[0].message.content
        logger.debug(f"[PER_ANSWER '{candidate}'] Raw output:\n{raw_message}\n{'=' * 40}")
        cot_parser = CoTContreteSyntaxParser()

        if debug:
            print(system_prompt)
            print(user_msg)
            print(raw_message)
            print("========================")

        # Extract the leading "(initial confidence: X%)" header (if any)
        # — mirrors the MCQ separate path. The cleaned text is parsed; the
        # full text + tokens are saved for inspection.
        from types import SimpleNamespace
        init_score = None
        cleaned = raw_message
        init_end_char = 0
        m_init = re.search(
            r"^\s*\(?\s*initial\s+confidence\s*:?\s*(\d+(?:\.\d+)?)\s*%?\s*\)?\s*$",
            raw_message or "", flags=re.IGNORECASE | re.MULTILINE,
        )
        if m_init:
            try:
                pct = float(m_init.group(1))
                init_score = max(0.05, min(0.99, pct / 100.0))
            except Exception:
                init_score = None
            pre = raw_message[:m_init.start()]
            post = raw_message[m_init.end():]
            cleaned = (pre + post).lstrip()
            init_end_char = m_init.end()
            while init_end_char < len(raw_message) and \
                    raw_message[init_end_char] in " \t\r\n":
                init_end_char += 1

        # Drop blank/non-numbered lines that the strict parser rejects
        # (preamble, trailing summary). Same line filter as MCQ separate.
        STEP_PREFIX_RE = re.compile(r'^\s*\d+\s*[\.\)\-:]')
        kept_lines = [ln for ln in cleaned.splitlines() if STEP_PREFIX_RE.match(ln)]
        if kept_lines:
            cleaned = "\n".join(kept_lines)

        try:
            cot_program = cot_parser.parse(cleaned, query=query)
            program_error = cot_program.check_program(strict=strict)
            if program_error:
                raise ValueError(program_error)
        except Exception as exc:
            error_msg = str(exc)
            parsing_errors.append(error_msg)
            logger.debug(f"[PER_ANSWER '{candidate}'] Parse error: {exc}")
            cot_program = None
            continue

        # Slice tokens past the init-line so find_logps step alignment
        # matches the parsed cot.lines.
        full_tokens = _normalize_logprobs(response.choices[0])
        drop_n = 0
        if init_end_char > 0:
            char_pos = 0
            for i, tok in enumerate(full_tokens):
                char_pos += len(getattr(tok, "token", "") or "")
                if char_pos >= init_end_char:
                    drop_n = i + 1
                    break
            seg_tokens = full_tokens[drop_n:]
            mock_choice = SimpleNamespace(
                logprobs=SimpleNamespace(content=seg_tokens)
            )
        else:
            seg_tokens = full_tokens
            mock_choice = response.choices[0]

        # Extract logprobs
        try:
            step_probs = find_logps(
                mock_choice,
                cot_program.lines,
                remove_meta_tokens=remove_meta_tokens,
                length_normalization=length_normalization,
            )
            cot_program.add_weights(step_probs)
        except Exception as exc:
            error_msg = f"Logprob extraction failed: {exc}"
            parsing_errors.append(error_msg)
            logger.debug(f"[PER_ANSWER '{candidate}'] Logprob error: {exc}")
            cot_program = None
            continue

        # Restore raw_cot to the FULL original response (init line included)
        # so the saved checkpoint reflects what the model emitted.
        cot_program.raw_cot = raw_message
        # Save per-token logprobs for the FULL response.
        # Save the FULL response's per-token logprobs as one unified
        # cot_logprobs field — already includes the leading
        # '(initial confidence: X%)' line tokens; no separate
        # init_score_logprobs needed.
        try:
            _attach_cot_logprobs(cot_program, full_tokens)
        except Exception:
            cot_program._cot_logprobs = None
        _attach_cot_prompt(cot_program, system=system_prompt, user=user_msg, response=raw_message)
        if init_score is not None:
            cot_program._init_score = init_score

        logger.debug(f"[PER_ANSWER '{candidate}'] Success on attempt {attempts}")
        cot_program.attempts = attempts
        cot_program.errors = parsing_errors
        return cot_program

    # All retries exhausted
    logger.warning(
        "Failed to generate CoT for candidate '%s' after %d attempts",
        candidate, num_retries,
    )
    return None


# ---------------------------------------------------------------------------
# Step 2b: Combined CoT strategy
# ---------------------------------------------------------------------------

def _normalize_logprobs(choice):
    """Normalize logprobs from OpenAI or Together format into a flat list of
    objects with ``.token``, ``.logprob``, and ``.top_logprobs`` attributes
    (the last is a list of objects with ``.token`` and ``.logprob``).

    OpenAI: ``choice.logprobs.content`` — list of objects already shaped this way.
    Together: ``choice.logprobs.tokens`` + ``choice.logprobs.token_logprobs``
    (legacy completion shape) plus optional ``choice.logprobs.top_logprobs``
    as ``List[Optional[Dict[str, float]]]`` (token -> logprob per position).
    """
    from types import SimpleNamespace

    if hasattr(choice.logprobs, "content") and choice.logprobs.content:
        # OpenAI shape. Normally each entry already has .token / .logprob
        # / .top_logprobs as object attributes, in which case we pass
        # through. BUT for some Together-hosted models (e.g.
        # google/gemma-3n-E4B-it) the Together Python SDK v1.x doesn't
        # define ``content`` in its ``LogprobsPart`` schema, so the SDK
        # stores the raw API response as a list of *dicts*. Downstream
        # code expects attribute access (``.token``, ``.logprob``), which
        # silently returns None on a dict — making logprobs look "all
        # null" when the data is actually there. Normalize dict items
        # into SimpleNamespace here so the pass-through path works for
        # both clients.
        out = []
        for c in choice.logprobs.content:
            if isinstance(c, dict):
                tok = c.get("token")
                lp = c.get("logprob")
                top_raw = c.get("top_logprobs") or []
                top_norm = []
                for alt in top_raw:
                    if isinstance(alt, dict):
                        a_tok, a_lp = alt.get("token"), alt.get("logprob")
                    else:
                        a_tok = getattr(alt, "token", None)
                        a_lp = getattr(alt, "logprob", None)
                    if a_tok is not None and a_lp is not None:
                        top_norm.append(SimpleNamespace(token=a_tok, logprob=float(a_lp)))
                out.append(SimpleNamespace(
                    token=tok,
                    logprob=float(lp) if lp is not None else -20.0,
                    top_logprobs=top_norm,
                ))
            else:
                out.append(c)
        return out

    if hasattr(choice.logprobs, "tokens") and hasattr(choice.logprobs, "token_logprobs"):
        tokens = choice.logprobs.tokens
        token_lps = choice.logprobs.token_logprobs
        # Together's top_logprobs (when requested) is a list of dicts.
        top_lps_raw = getattr(choice.logprobs, "top_logprobs", None) or []
        result = []
        for i in range(len(tokens)):
            top_at_i = top_lps_raw[i] if i < len(top_lps_raw) else None
            top_normalized = []
            if isinstance(top_at_i, dict):
                for tok_str, lp_val in top_at_i.items():
                    if lp_val is None:
                        continue
                    top_normalized.append(SimpleNamespace(token=tok_str, logprob=float(lp_val)))
            elif isinstance(top_at_i, list):
                # Some clients wrap each alt as an object with .token/.logprob
                for alt in top_at_i:
                    a_tok = alt.token if hasattr(alt, "token") else alt.get("token")
                    a_lp = alt.logprob if hasattr(alt, "logprob") else alt.get("logprob")
                    if a_tok is not None and a_lp is not None:
                        top_normalized.append(SimpleNamespace(token=a_tok, logprob=float(a_lp)))
            result.append(SimpleNamespace(
                token=tokens[i],
                logprob=token_lps[i] if token_lps[i] is not None else -20.0,
                top_logprobs=top_normalized,
            ))
        return result

    raise ValueError("Unrecognized logprobs format — neither OpenAI nor Together")


# Per-call scratchpad: when a per-letter generator exhausts num_retries
# without producing a parseable CoT, it stashes the failed attempts
# (system_prompt, user_msg, [(attempt, kind, error, raw_message), ...])
# here. run_mcq_external then surfaces this dict in the RuntimeError it
# raises, so the per-example error checkpoint contains the raw model
# output and the parser's complaints — essential for diagnosing why
# strict-grammar parsing kept rejecting (especially relevant for gpt-4o,
# which tends to ignore the format hints).
_LAST_PER_ANSWER_FAILURES: Dict[str, Dict] = {}


def _attach_cot_prompt(cot, *, system: str, user: str, response: Optional[str] = None):
    """Attach the system + user prompt AND the model's response that
    produced this CoT.

    Stored on ``cot._cot_prompt`` as
    ``{"system": str, "user": str, "response": str}`` so ``_serialize_cot``
    can persist a full record of the LLM call. For separate strategy this
    is distinct per letter (N calls, N responses); for combined-style it's
    the single shared call (same prompt + same response duplicated on every
    per-letter cot — small cost for clarity, callers can de-dup at render
    time).
    """
    try:
        cot._cot_prompt = {"system": system, "user": user, "response": response}
    except Exception:
        cot._cot_prompt = None


def _attach_init_score_logprobs(cot, tokens):
    """Attach the per-token logprobs for the section-header tokens that
    encode '(initial confidence: X%)' (combined strategy) or the leading
    '(initial confidence: X%)' line (separate strategy).

    Stored on ``cot._init_score_logprobs`` as a list of
    ``{"token": str, "logprob": float}`` so ``_serialize_cot`` can persist
    them. ``find_logps`` excludes these tokens from step-weight alignment;
    saving them separately preserves the model's confidence signal at the
    init-score header itself.
    """
    try:
        cot._init_score_logprobs = [
                                       {"token": getattr(t, "token", None),
                                        "logprob": getattr(t, "logprob", None)}
                                       for t in (tokens or [])
                                   ] or None
    except Exception:
        cot._init_score_logprobs = None


def _attach_cot_logprobs(cot, tokens):
    """Attach a serializable per-token logprob list to ``cot``.

    Stored on ``cot._cot_logprobs`` so ``_serialize_cot`` (in
    ``run_BBH_multi_cot.py``) can persist it into the per-example
    checkpoint. Each entry is ``{"token": str, "logprob": float}`` —
    just the chosen-token logprob (no top_logprobs to keep checkpoint
    size manageable).

    For combined-style strategies, ``tokens`` is the section's slice of
    the full response token stream. For separate strategy it's the full
    per-letter response.

    Sets ``cot._cot_logprobs`` to ``None`` if the token list is empty or
    serialization fails (rather than raising).
    """
    try:
        cot._cot_logprobs = [
                                {"token": getattr(t, "token", None),
                                 "logprob": getattr(t, "logprob", None)}
                                for t in (tokens or [])
                            ] or None
    except Exception:
        cot._cot_logprobs = None


def _split_combined_response(raw_text, logprob_tokens, candidates):
    """Split a combined LLM response into per-candidate segments.

    The response is expected to have ``## For {candidate}:`` section headers.
    Returns a list of ``(segment_text, segment_tokens)`` tuples, one per
    candidate, in the order they appear.

    Arguments:
        raw_text: The full raw response text.
        logprob_tokens: The flat list of token objects from
            ``response.choices[0].logprobs.content``.
        candidates: List of candidate answer strings (e.g. ``["yes", "no"]``).

    Raises:
        ValueError: When section headers cannot be found for all candidates.
    """
    # Find section header positions in the raw text
    # header_pattern = re.compile(
    #     r'^##\s*[Ff]or\s+(' + '|'.join(re.escape(c) for c in candidates) + r')\s*:',
    #     re.MULTILINE,
    # )
    # header_pattern = re.compile(
    #     r'^##\s*[Ff]or\s+(' + '|'.join(re.escape(c) for c in candidates) + r')\s*'
    #     r'(?:\(initial confidence:\s*([\d.]+)\)\s*)?:',
    #     re.MULTILINE,
    # )
    # header_pattern = re.compile(
    #     r'^##\s*[Ff]or\s+(' + '|'.join(re.escape(c) for c in candidates) + r')\s*'
    #     r'(?:\(initial confidence:\s*([\d.]+)%\)\s*)?:',
    #     re.MULTILINE,
    # )
    header_pattern = re.compile(
        r'^##\s*[Ff]or\s+('
        + '|'.join(re.escape(c) for c in candidates)
        + r')\s*(?:\(initial\s+confidence:\s*([\d.]+)%\))?\s*:',
        re.MULTILINE,
    )
    headers = list(header_pattern.finditer(raw_text))
    if len(headers) < 2:
        raise ValueError(
            f"Expected 2 section headers (## For ...:) but found {len(headers)}. "
            f"Raw text starts with: {raw_text[:200]!r}"
        )

    # Determine the order of candidates as they appear in the response
    ordered_candidates = [h.group(1).strip().lower() for h in headers[:2]]
    # Tolerant parse of init_confidence — fall back to None per-section
    # when the model didn't emit the (initial confidence: X%) annotation.
    ordered_scores = []
    for h in headers[:2]:
        try:
            ordered_scores.append(float(h.group(2)) / 100.0)
        except (TypeError, ValueError):
            ordered_scores.append(None)

    # Look for ## Final Answer: section
    final_answer_pattern = re.compile(
        r'^##\s*[Ff]inal\s+[Aa]nswer\s*(.*):', re.MULTILINE
    )
    final_answer_match = final_answer_pattern.search(raw_text)
    final_answer_boundary = final_answer_match.start() if final_answer_match else len(raw_text)

    # Extract text segments (between headers, excluding Final Answer)
    text_segments = []
    for i in range(2):
        start = headers[i].end()
        end = headers[i + 1].start() if i + 1 < len(headers) else final_answer_boundary
        text_segments.append(raw_text[start:end].strip())

    # Extract final answer info (including the ALL tokens covering the
    # '## Final Answer:' section, for downstream inspection).
    final_answer_info = None
    if final_answer_match:
        final_section_text = raw_text[final_answer_match.end():].strip()
        fa_match = re.search(r'\(\s*final answer:\s*(\w+)\s*\)', final_section_text, re.IGNORECASE)
        if fa_match:
            final_answer_value = fa_match.group(1).strip().lower()
            # Find the logprob of the final answer token
            fa_char_pos = final_answer_match.end() + fa_match.start()
            char_offset_fa = 0
            final_answer_logprob = None
            fa_section_token_start = None
            for i, tok in enumerate(logprob_tokens):
                token_text = tok.token if hasattr(tok, 'token') else tok['token']
                token_end = char_offset_fa + len(token_text)
                if fa_section_token_start is None and token_end > final_answer_match.start():
                    fa_section_token_start = i
                if (final_answer_logprob is None and token_end > fa_char_pos
                        and token_text.strip().lower() in ("yes", "no")):
                    final_answer_logprob = tok.logprob if hasattr(tok, 'logprob') else tok['logprob']
                char_offset_fa = token_end
            if fa_section_token_start is None:
                fa_section_token_start = len(logprob_tokens)
            fa_section_tokens = logprob_tokens[fa_section_token_start:]

            if final_answer_logprob is not None:
                final_answer_prob = max(min(math.exp(final_answer_logprob), 1.0 - 1e-7), 1e-7)
            else:
                final_answer_prob = 0.5  # fallback
            final_answer_info = {
                "value": final_answer_value,
                "prob": final_answer_prob,
                "raw": final_section_text,
                "logprobs": [
                    {"token": getattr(t, "token", None),
                     "logprob": getattr(t, "logprob", None)}
                    for t in fa_section_tokens
                ],
            }

    # Split the token stream at the second header boundary.
    # Walk through tokens, tracking cumulative character offset,
    # to find where the second header starts.
    second_header_start = headers[1].start()
    second_header_end = headers[1].end()

    char_offset = 0
    split_token_start = None  # first token of second header
    split_token_end = None  # first token AFTER second header
    for i, tok in enumerate(logprob_tokens):
        token_text = tok.token if hasattr(tok, 'token') else tok['token']
        token_start = char_offset
        token_end = char_offset + len(token_text)

        if split_token_start is None and token_end > second_header_start:
            split_token_start = i
        if split_token_end is None and token_start >= second_header_end:
            split_token_end = i
            break
        char_offset = token_end

    if split_token_start is None:
        raise ValueError("Could not locate second section header in token stream")
    if split_token_end is None:
        # Header extends to the last tokens checked — content starts after
        split_token_end = len(logprob_tokens)
        # Walk forward to find the first newline after header
        for i in range(split_token_start, len(logprob_tokens)):
            tok = logprob_tokens[i]
            t = tok.token if hasattr(tok, 'token') else tok['token']
            if '\n' in t:
                split_token_end = i + 1
                break

    # Also find where the first header ends in the token stream
    first_header_end_char = headers[0].end()
    first_content_token = 0
    char_offset = 0
    for i, tok in enumerate(logprob_tokens):
        token_text = tok.token if hasattr(tok, 'token') else tok['token']
        if char_offset + len(token_text) > first_header_end_char:
            # Skip past newline after header
            t = token_text
            if '\n' in t:
                first_content_token = i + 1
            else:
                first_content_token = i
            break
        char_offset += len(token_text)

    tokens_part1 = logprob_tokens[first_content_token:split_token_start]

    # If there's a Final Answer section, exclude its tokens from part2
    if final_answer_match:
        fa_start_char = final_answer_match.start()
        fa_token_start = len(logprob_tokens)
        char_offset_search = 0
        for i, tok in enumerate(logprob_tokens):
            token_text = tok.token if hasattr(tok, 'token') else tok['token']
            if char_offset_search + len(token_text) > fa_start_char:
                fa_token_start = i
                break
            char_offset_search += len(token_text)
        tokens_part2 = logprob_tokens[split_token_end:fa_token_start]
    else:
        tokens_part2 = logprob_tokens[split_token_end:]

    # Header-token slices for each candidate. For candidate 0 the header
    # spans tokens [0..first_content_token); for candidate 1 it's
    # [split_token_start..split_token_end). Saved separately so the
    # ## For X (initial confidence: Y%): line's logprobs can be inspected
    # without polluting find_logps step alignment.
    header_tokens_1 = logprob_tokens[:first_content_token]
    header_tokens_2 = logprob_tokens[split_token_start:split_token_end]

    # Return 5-tuples in candidate order: (candidate, segment_text,
    # body_tokens, init_score, header_tokens)
    result = []
    for ci, oc in enumerate(ordered_candidates):
        score = ordered_scores[ci] if ordered_scores else None
        body = tokens_part1 if ci == 0 else tokens_part2
        head = header_tokens_1 if ci == 0 else header_tokens_2
        result.append((oc, text_segments[ci], body, score, head))

    return result, final_answer_info


def generate_combined_cot(
        query: str,
        candidates: List[str],
        llm: LLMType,
        model_name: str = "gpt-4o",
        num_retries: int = 5,
        remove_meta_tokens: bool = True,
        length_normalization: bool = True,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        strict: bool = False,
        debug: bool = False,
) -> Optional[List[CoTOutput]]:
    """Generate CoTs for ALL candidates in a single LLM call.

    Uses ``COMBINED_COT_INSTRUCTION`` which asks the model to produce
    ``## For {candidate}:`` sections in one response. The token-level
    logprobs are split at the section boundary and processed per-segment.

    Arguments:
        query: The original question.
        candidates: List of candidate answer strings (e.g. ``["yes", "no"]``).
        llm: LLM client (OpenAI-compatible).
        model_name: Model name for API calls.
        num_retries: Maximum retry attempts.
        remove_meta_tokens: Exclude metadata tokens from logprob computation.
        length_normalization: Use geometric mean of token probs per step.
        temperature: Sampling temperature.
        top_p: Nucleus sampling parameter.
        strict: Strict structural checks on CoT output.
        debug: Print intermediate outputs.

    Returns:
        A list of ``CoTOutput`` objects (one per candidate, in candidate order),
        or ``None`` if generation fails after all retries.
    """
    from types import SimpleNamespace
    from cot_prog.prompt.prompt_to_program import find_logps

    assert len(candidates) == 2, "Combined strategy currently requires exactly 2 candidates"

    user_msg = COMBINED_USER_TEMPLATE.format(query=query)
    attempts = 0
    parsing_errors: List[str] = []
    error_msg = "No errors, first attempt"
    raw_combined_text = None

    while True:
        attempts += 1
        if attempts > num_retries:
            break

        logger.debug(f"[COMBINED] Attempt {attempts}/{num_retries}")

        system_prompt = COMBINED_COT_INSTRUCTION.format(
            candidate_a=candidates[0],
            candidate_b=candidates[1],
            errors=error_msg,
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_msg}]},
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
            logger.debug(f"[COMBINED] API error: {exc}")
            continue

        raw_message = response.choices[0].message.content
        raw_combined_text = raw_message
        logger.debug(f"[COMBINED] Raw output:\n{raw_message}\n{'=' * 40}")

        if debug:
            print(system_prompt)
            print(user_msg)
            print(raw_message)
            print("========================")

        # Split response into per-candidate segments
        try:
            all_tokens = _normalize_logprobs(response.choices[0])
            segments, final_answer_info = _split_combined_response(raw_message, all_tokens, candidates)
        except Exception as exc:
            error_msg = f"Section splitting failed: {exc}"
            parsing_errors.append(error_msg)
            logger.debug(f"[COMBINED] Split error: {exc}")
            continue

        # Parse each segment independently
        cot_parser = CoTContreteSyntaxParser()
        cot_outputs = []
        parse_ok = True

        for seg_candidate, seg_text, seg_tokens, init_score, header_tokens in segments:
            try:
                cot = cot_parser.parse(seg_text, query=query)
                cot._init_score = init_score
                program_error = cot.check_program(strict=strict)
                if program_error:
                    raise ValueError(f"[{seg_candidate}] {program_error}")
            except Exception as exc:
                error_msg = f"Parse error for '{seg_candidate}': {exc}"
                parsing_errors.append(error_msg)
                logger.debug(f"[COMBINED] Parse error ({seg_candidate}): {exc}")
                parse_ok = False
                break

            # Extract logprobs using mock Choice object
            try:
                mock_choice = SimpleNamespace(
                    logprobs=SimpleNamespace(content=seg_tokens)
                )
                step_probs = find_logps(
                    mock_choice,
                    cot.lines,
                    remove_meta_tokens=remove_meta_tokens,
                    length_normalization=length_normalization,
                )
                cot.add_weights(step_probs)
                # Save the FULL section's per-token logprobs (header + body)
                # as one unified `cot_logprobs` field — no separate
                # init_score_logprobs. Analysts can identify the init-line
                # span by looking for '(initial confidence: X%)' inside the
                # saved tokens or by reading cot._init_score directly.
                _attach_cot_logprobs(cot, list(header_tokens) + list(seg_tokens))
                _attach_cot_prompt(cot, system=system_prompt, user=user_msg, response=raw_message)
            except Exception as exc:
                error_msg = f"Logprob extraction failed for '{seg_candidate}': {exc}"
                parsing_errors.append(error_msg)
                logger.debug(f"[COMBINED] Logprob error ({seg_candidate}): {exc}")
                parse_ok = False
                break

            cot_outputs.append((seg_candidate, cot))

        if not parse_ok:
            continue

        # Reorder to match the candidates list order
        cot_map = {c: cot for c, cot in cot_outputs}
        result = []
        for cand in candidates:
            cand_lower = cand.lower()
            if cand_lower not in cot_map:
                error_msg = f"Missing section for candidate '{cand}'"
                parsing_errors.append(error_msg)
                logger.debug(f"[COMBINED] {error_msg}")
                parse_ok = False
                break
            cot = cot_map[cand_lower]
            cot.attempts = attempts
            cot.errors = parsing_errors
            result.append(cot)

        if not parse_ok:
            continue

        logger.debug(f"[COMBINED] Success on attempt {attempts}")
        # Store the raw combined text on each CoTOutput. (per-letter
        # _final_answer_info was previously attached and is no longer
        # needed — but we DO want to surface the global '## Final Answer:'
        # info ONCE, including its per-token logprobs, at the trace level.
        # Attach to result[0] only so process_example can pick it up
        # without duplicating across all letters.)
        for cot in result:
            cot._raw_combined = raw_combined_text
        if result and final_answer_info:
            result[0]._final_answer_info = final_answer_info
        return result

    logger.warning("Failed to generate combined CoT after %d attempts", num_retries)
    return None


# ---------------------------------------------------------------------------
# Step 3: Assemble MULTI_AGGR probabilistic program
# ---------------------------------------------------------------------------

def _serialize_cot(cot) -> dict:
    """Convert a CoTOutput to a JSON-serialisable dict."""
    return {
        "raw_cot": cot.raw_cot,
        "steps": [s.to_dict() for s in cot.lines],
        "attempts": cot.attempts,
        "errors": cot.errors or [],
    }


def _logit(p: float) -> float:
    """Logit function with clamping to avoid infinities."""
    p = max(min(p, 1.0 - 1e-7), 1e-7)
    return math.log(p / (1.0 - p))


def run_multi_external(
        query,
        strategy,
        pool_mode,
        answer_aggregation_fn,
        statement_aggregation_fn,
        candidates,
        model_name,
        num_retries: int = 5,
        debug: bool = False,
        step_dependency_mode: str = "flat",
):
    # Unified routing — supports Together, OpenRouter (for openai/gpt-*),
    # and vanilla OpenAI based on which env var is set + the model name.
    llm = _get_model(model_name=model_name)

    per_answer_cots = {}
    # Note: pool_mode == "verbalized" was removed; verbalized is only an
    # answer_prior mode now. Keep the verbalization-prompt path off here.
    use_verbalization = False

    if strategy == "separate":
        for cand in candidates:
            cot = generate_per_answer_cot(
                query=query,
                candidate=cand,
                llm=llm,
                model_name=model_name,
                num_retries=num_retries,
                verbalization=use_verbalization,
                remove_meta_tokens=True,
                length_normalization=True,
                temperature=0.0,
                strict=False,
                debug=False,
            )
            per_answer_cots[cand] = cot

        logger.debug("#################")
        logger.debug(per_answer_cots.keys())

    elif strategy == "combined":
        cot_list = generate_combined_cot(
            query=query,
            candidates=candidates,
            llm=llm,
            model_name=model_name,
            num_retries=num_retries,
            remove_meta_tokens=True,
            length_normalization=True,
            temperature=0.0,
            strict=False,
            debug=False,
        )
        if cot_list is None:
            raise RuntimeError(
                f"Failed to generate combined CoT after {num_retries} retries"
            )
        for cand, cot in zip(candidates, cot_list):
            per_answer_cots[cand] = cot
    else:
        raise ValueError(f'Unknown strategy: {strategy}')

    result = {
        "qid": -1,
        "term": "",
        "description": "",
        "question": query,
        "question_type": "binary",
        "original_answer": "unk",
        "candidates": candidates,
        "discovery_traces": None,
        "per_answer_traces": {},
        "teacher_forcing_priors": {},
        "cot_strategy": strategy,
        "modes": {},
        "predicted_answer": {},
        "error": None,
        "timestamp": "now",  # datetime.utcnow().isoformat(),
    }
    for cond in candidates:
        result["per_answer_traces"][cand] = _serialize_cot(per_answer_cots[cand])

    cot_list = [per_answer_cots[c] for c in candidates]

    # Teacher-forcing (slot-1 / "logit") answer prior — parallel binary
    # judgment over {correct, incorrect}. ONE LLM call, independent of the
    # CoT analysis: the model labels every candidate correct/incorrect and
    # the prior is the renormalised token logprob of 'correct' vs 'incorrect'
    # at each candidate. Returns ({candidate: P(correct)}, tf_meta) and is
    # attached to every per-answer CoT so _answer_prior_slots can read it via
    # getattr(cot, "_tf_priors").
    #
    # ALWAYS compute it at generation time so both answer-prior slots —
    # TF-logit and verbal init — are cached in the trace.
    # answer_aggregation_fn (program construction) then selects/combines them.
    tf_priors_dict = None
    tf_result = teacher_forcing_logprobs_parallel(
        query=query,
        candidate_labels=list(candidates),
        llm=llm,
        model_name=model_name,
        num_retries=num_retries,
        debug=debug,
    )
    tf_meta = None
    if tf_result is None:
        uniform_p = 1.0 / max(len(candidates), 1)
        tf_priors_dict = {c: uniform_p for c in candidates}
        tf_meta = {
            "prompt": None,
            "response": "(TF call failed; uniform fallback)",
            "logprobs": None,
        }
        print(
            "[run_multi_external] WARNING: TF parser returned no priors; "
            f"falling back to uniform ({uniform_p:.3f}) per candidate.",
            file=sys.stderr,
        )
    else:
        tf_priors_dict, tf_meta = tf_result
    for cand in candidates:
        per_answer_cots[cand]._tf_priors = tf_priors_dict
        per_answer_cots[cand]._tf_meta = tf_meta
        # Back-compat: keep the legacy string attribute populated so old
        # readers (HTML renderer, restore_cot fallback) still work.
        per_answer_cots[cand]._tf_raw = tf_meta.get("response") if tf_meta else None
    result["teacher_forcing_priors"] = tf_priors_dict
    result["tf_meta"] = tf_meta
    result["tf_raw"] = tf_meta.get("response") if tf_meta else None

    ### write program
    program_str = write_multi_aggr_program(
        query=query,
        candidates=candidates,
        cot_outputs=cot_list,
        tf_priors=None,  # legacy arg — _tf_priors is read from each cot
        step_dependency_mode=step_dependency_mode,
        pool_mode=pool_mode,
        answer_aggregation_fn=answer_aggregation_fn,
        statement_aggregation_fn=statement_aggregation_fn,
    )
    # Same bug as the MCQ path: mutating cot_list[0].raw_cot would clobber
    # the first per-candidate CoT's section text. Shallow-copy and assign on
    # the copy.
    import copy as _copy
    final_cot = _copy.copy(cot_list[0])
    combined_raw = getattr(cot_list[0], "_raw_combined", None)
    if combined_raw:
        final_cot.raw_cot = combined_raw
    else:
        final_cot.raw_cot = f"#answer 1:\n{cot_list[0].raw_cot}\n#answer 2:\n{cot_list[1].raw_cot}"

    return (final_cot, program_str)


def _answer_prior_slots(
        cot: CoTOutput,
        candidate: Optional[str] = None,
        n_candidates: int = 2,
        tf_prior: Optional[float] = None,
) -> tuple:
    """Return the canonical ``(slot1_logit, slot2_verbal)`` pair for a
    candidate's answer-prior Flip.

    This replaces the old ``_compute_answer_prior(mode=...)``. The
    answer-prior source is no longer a separate axis: ``answer_aggregation_fn``
    (applied later by the program writer via :func:`cot_prog.axes.attr_for`)
    selects/combines the two slots.

    CANONICAL SLOT ORDER (Table 7: ``p`` = logit, ``c`` = verbal):
        slot 1 (.p)  = TF-logit: renormalised token logprob of 'correct' vs
                       'incorrect' for this candidate from the teacher-forcing
                       call (``cot._tf_priors[candidate]``, or ``tf_prior``)
        slot 2 (.p2) = verbal initial confidence (``cot._init_score``;
                       falls back to the answer step's ``(confidence: X%)``)

    NOTE: this FIXES the old slot inversion — ``teacher_forcing`` previously
    returned ``(verbal_init, tf_logit)``, i.e. primary=verbal, which is
    backwards versus the per-step convention and Table 7. Here slot 1 is
    always the logit.

    FALLBACK: if only one slot is available, duplicate it into both so the
    emitted ``Flip(name, x, [x])`` collapses cleanly under any
    ``answer_aggregation_fn`` (``cal``/``mean``/``max``/``min`` are binary and
    ``cal`` raises "DisCalibration requires 2 args" on a single-slot Flip).
    If neither is available, fall back to uniform ``1/N`` in both slots.

    Arguments:
        cot: The per-answer CoTOutput.
        candidate: The candidate answer string or MCQ letter; used to look up
            ``cot._tf_priors``.
        n_candidates: N, for the uniform fallback (binary = 2).
        tf_prior: Legacy fallback TF prior when ``cot._tf_priors`` is absent.
    Returns:
        ``(slot1_logit, slot2_verbal)`` — both clamped to (0, 1).
    """
    def _clamp(x: float) -> float:
        return max(min(float(x), 1.0 - 1e-7), 1e-7)

    # --- slot 1: TF-logit -------------------------------------------------
    tf = None
    tf_priors_dict = getattr(cot, "_tf_priors", None)
    if tf_priors_dict is not None and candidate is not None:
        for key in (candidate, str(candidate).lower(), str(candidate).upper()):
            if key in tf_priors_dict:
                tf = float(tf_priors_dict[key])
                break
    if tf is None:
        tf = tf_prior

    # --- slot 2: verbal initial confidence --------------------------------
    verb = getattr(cot, "_init_score", None)
    if verb is None:
        ans_steps = [s for s in cot.lines if s.answer]
        if ans_steps and getattr(ans_steps[-1], "confidence", None) is not None:
            verb = float(ans_steps[-1].confidence)

    # --- assemble with duplicate-slot fallback ----------------------------
    if tf is not None and verb is not None:
        return (_clamp(tf), _clamp(verb))
    if tf is not None:
        return (_clamp(tf), _clamp(tf))
    if verb is not None:
        return (_clamp(verb), _clamp(verb))
    u = 1.0 / max(n_candidates, 1)
    return (u, u)


def write_multi_aggr_program(
        query: str,
        candidates: List[str],
        cot_outputs: List[CoTOutput],
        tf_priors: Optional[Dict[str, float]] = None,
        step_dependency_mode: str = "flat",
        pool_mode: str = "logitpool",
        answer_aggregation_fn='c',
        statement_aggregation_fn='mean',
) -> str:
    """Generate a MULTI_AGGR-style probabilistic program.

    Arguments:
        query: The original question.
        candidates: List of 2 answer candidate strings.
        cot_outputs: Corresponding list of CoTOutput objects.
        tf_priors: Legacy per-candidate TF-prior fallback used by
            :func:`_answer_prior_slots` when ``cot._tf_priors`` is absent.
        answer_aggregation_fn: Table 7 axis selecting/combining the answer-prior
            slots ``(TF-logit, verbal init)``: c / p / cal / mean / max / min.
        statement_aggregation_fn: Table 7 axis for the per-step slots
            ``(logprob, verbal conf)``: c / p / mean / max / min / pool.
        step_dependency_mode: How to structure evidence steps.
            ``'flat'``: all steps independent, pool all (the only mode).
    Returns:
        A string in the probabilistic programming language.
    """
    assert len(candidates) == 2, "Currently only top_k=2 is supported"
    assert len(cot_outputs) == 2, "Need exactly 2 CoTOutputs"

    lines: List[str] = []
    lines.append(f'### original query')
    # Same escaping as the multiple-choice writer below.
    q_escaped = query.replace('"', '\\"').replace("\n", " ")
    lines.append(f'query: str = "{q_escaped}"')
    lines.append("")

    # --- Collect all steps with global re-indexing ---
    global_counter = 1
    # Per-candidate: list of (global_id, step, is_answer)
    candidate_steps: List[List[tuple]] = []
    # Track answer string variable for each candidate
    answer_string_vars: List[str] = []
    # Per-candidate: local step prefix → global id
    local_to_global_maps: List[Dict[int, int]] = []

    #### initial estimates of answer scores
    init_answer_scores = []

    lines.append("### step strings")
    for ci, (cand, cot) in enumerate(zip(candidates, cot_outputs)):
        steps_info = []
        answer_svar = None
        local_to_global: Dict[int, int] = {}
        for step in cot.lines:
            gid = global_counter
            global_counter += 1
            local_to_global[step.prefix] = gid
            text = step.claim if step.claim else step.answer
            # Escape quotes in text
            text_escaped = text.replace('"', '\\"')
            lines.append(f's{gid}: str = "{text_escaped}"')
            steps_info.append((gid, step, bool(step.answer)))
            if step.answer:
                answer_svar = f"s{gid}"

        candidate_steps.append(steps_info)
        answer_string_vars.append(answer_svar or f"s{steps_info[-1][0]}")
        local_to_global_maps.append(local_to_global)

    lines.append("")

    # --- Answer priors ---
    safe_names = [_sanitize_name(c) for c in candidates]
    tf_priors = tf_priors or {}
    # Canonical (slot1_logit, slot2_verbal) per candidate; answer_aggregation_fn
    # then selects/combines via the Flip attribute below.
    priors = [
        _answer_prior_slots(cot, candidate=cand, n_candidates=len(candidates),
                            tf_prior=tf_priors.get(cand))
        for cand, cot in zip(candidates, cot_outputs)
    ]

    lines.append("## answer priors")
    for i, (sname, cand, prior) in enumerate(zip(safe_names, candidates, priors)):
        lines.append(f'{sname}_A: bool =~ Flip("{cand}", {prior[0]}, [{prior[1]}])')

    # ``c`` selects the verbal (second) slot, exposed by the compiler as .p2.
    _ans_attr = attr_for(answer_aggregation_fn)
    lines.append(
        f'A =~ Categorical("{candidates[0]}", '
        f'Combine({safe_names[0]}_A.{_ans_attr}, {safe_names[1]}_A.{_ans_attr}))'
    )
    lines.append("")

    # --- Evidence sections ---
    evidence_var_names: List[str] = []
    # Per-candidate list of constraint variable names for the observe statement
    dep_constraints: List[List[str]] = []

    for ci, (sname, cand) in enumerate(zip(safe_names, candidates)):
        lines.append(f"## evidence for {cand}")
        l2g = local_to_global_maps[ci]

        # Collect evidence (non-answer) steps
        evidence_steps = [
            (gid, step) for gid, step, is_ans in candidate_steps[ci]
            if not is_ans
        ]
        evidence_gids = [gid for gid, _ in evidence_steps]
        evidence_gid_set = set(evidence_gids)
        constraints: List[str] = []

        for gid, step in evidence_steps:
            if step.confidence is not None:
                lines.append(f"P{gid}: bool =~ Flip(s{gid}, {step.weight},[{step.confidence}])")
            elif statement_aggregation_fn == "c":
                # "c" reads only the verbalized confidence; without the tag
                # there is nothing to read, so fail with a clear message.
                raise ValueError(
                    f"Step {step.prefix} of candidate '{cand}' (P{gid} in the program) has no (confidence: X%) tag, "
                    "but --statement_aggregation_fn c uses only the verbalized "
                    "step confidence. Use another setting or regenerate the trace."
                )
            else:
                lines.append(f"P{gid}: bool =~ Flip(s{gid}, {step.weight})")
        pool_gids = evidence_gids

        # Aggregate evidence — depends on pool_mode
        ev_name = f"{sname}_E"
        if len(pool_gids) >= 2:
            if pool_mode == "logitpool":
                # Standard: unweighted logit average. See note above re: "c".
                _attr = "p2" if statement_aggregation_fn == "c" else statement_aggregation_fn
                prob_refs = ", ".join(f"P{gid}.{_attr}" for gid in pool_gids)
                lines.append(f'{ev_name}: bool =~ Factor("{ev_name}", LogitPool({prob_refs}))')

            else:
                raise ValueError(f"Unknown pool_mode: {pool_mode!r}")

        elif len(pool_gids) == 1:
            _attr = "p2" if statement_aggregation_fn == "c" else statement_aggregation_fn
            lines.append(f'{ev_name}: bool =~ Factor("{ev_name}", P{pool_gids[0]}.{_attr})')
        else:
            lines.append(f'{ev_name}: bool =~ Flip("{ev_name}", {priors[ci]})')

        evidence_var_names.append(ev_name)
        dep_constraints.append(constraints)
        lines.append("")

    # --- Observe ---
    # Build observe expression: each candidate's evidence + dependency constraints
    obs_parts = []
    for ci, (ev_name, constrs) in enumerate(
            zip(evidence_var_names, dep_constraints)
    ):
        a_cond = "A" if ci == 0 else "(not A)"
        parts = [a_cond, ev_name] + constrs
        obs_parts.append(" and ".join(parts))
    lines.append(f"observe(({obs_parts[0]}) or ({obs_parts[1]}))")
    lines.append("")

    # --- Return branches ---
    lines.append(f"if A:")
    lines.append(f"    return {answer_string_vars[0]}")
    lines.append(f"elif not A:")
    lines.append(f"    return {answer_string_vars[1]}")
    lines.append(f"else:")
    lines.append(f"    raise ValueError('Unable to find answer')")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class MultiAnswerCoTOutput:
    """Result of the multi-answer CoT pipeline.

    Attributes:
        query: The original question.
        question_type: ``'binary'`` or ``'open_ended'``.
        candidates: Discovered candidate answer strings.
        per_answer_cots: Mapping from candidate -> CoTOutput.
        program: The generated MULTI_AGGR program string.
        discovery_raw: Raw discovery-phase LLM outputs (for debugging).
    """
    query: str
    question_type: str
    candidates: List[str]
    per_answer_cots: Dict[str, CoTOutput]
    program: str
    discovery_raw: Optional[List[str]] = None

    def write_program(self) -> str:
        """Return the generated probabilistic program."""
        return self.program

    @property
    def raw_cot(self) -> str:
        parts = []
        for cand, cot in self.per_answer_cots.items():
            parts.append(f"--- Evidence for '{cand}' ---\n{cot.raw_cot}")
        return "\n\n".join(parts)

    @property
    def attempts(self) -> int:
        return sum(c.attempts for c in self.per_answer_cots.values())

    @property
    def errors(self) -> List[str]:
        all_errors: List[str] = []
        for c in self.per_answer_cots.values():
            if c.errors:
                all_errors.extend(c.errors)
        return all_errors


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def multi_answer_cot_to_program(
        query: Union[str, dict],
        query_field: Optional[str] = "question",
        llm: Optional[LLMType] = None,
        model_name: Optional[str] = "gpt-4o",
        client: Optional[str] = None,
        num_retries: Optional[int] = 5,
        debug: Optional[bool] = False,
        remove_meta_tokens: Optional[bool] = True,
        length_normalization: Optional[bool] = True,
        temperature: Optional[float] = 0.0,
        top_p: Optional[float] = None,
        strict: Optional[bool] = True,
        n_discovery: Optional[int] = 5,
        discovery_temperature: Optional[float] = 0.7,
        top_k: Optional[int] = 2,
        verbalization: Optional[bool] = False,
        step_dependency_mode: Optional[str] = "flat",
        cot_strategy: Optional[str] = "separate",
        **kwargs,
) -> Optional[MultiAnswerCoTOutput]:
    """Main entry point for the multi-answer CoT strategy.

    Pipeline:
        1. Classify question type (binary vs open-ended).
        2. Discover candidate answers (or use ``['yes', 'no']``).
        3. Generate structured per-answer CoT evidence chains.
        4. Assemble a probabilistic program (binary or MCQ N-way).

    Candidate-input format
    ----------------------
    Internally the function works in two modes:

      * Binary (N == 2): each candidate is a single answer string —
        e.g. ``["yes", "no"]``.
      * MCQ (N > 2):     each candidate may be either

            (a) a plain answer-text string, e.g.
                ``["heart", "liver", "kidney", "lung"]``
                Letters are auto-assigned by position
                (``A``, ``B``, ``C``, ``D``).

            (b) a ``(label, text)`` 2-tuple/list, e.g.
                ``[("A", "heart"), ("B", "liver"), ("C", "kidney"), ("D", "lung")]``
                Use this when you want explicit letters.

            (c) a dict with ``"label"`` and ``"text"`` keys, e.g.
                ``[{"label": "A", "text": "heart"}, ...]``
                Same as (b), more readable.

    The 'discover' path (open-ended question) returns plain strings, so it
    always uses (a). The recommended call shape from external code is (b)
    or (c), since those preserve the letter↔text mapping that the parser
    and the (final answer: <letter>) check rely on.

    Arguments:
        query: The question (string or dict with *query_field* key).
        query_field: Key to extract query from if *query* is a dict.
        llm: Optional pre-built LLM client.
        model_name: Model name for the API.
        client: Client type (``'openai'`` or ``'together'``).
        num_retries: Max retries per per-answer CoT generation.
        debug: Print intermediate outputs.
        remove_meta_tokens: Passed to ``find_logps``.
        length_normalization: Passed to ``find_logps``.
        temperature: Temperature for per-answer CoT generation.
        top_p: Nucleus sampling parameter.
        strict: Strict structural checks on CoT output.
        n_discovery: Number of samples for candidate discovery.
        discovery_temperature: Temperature for discovery sampling.
        top_k: Number of top candidates to keep.
        verbalization: If True, prompt LLM to self-report confidence.
        kwargs: May include ``answer_aggregation_fn`` / ``statement_aggregation_fn``
            (Table 7 program-construction axes; defaults c / mean) and
            ``pool_mode``.

    Returns:
        A :class:`MultiAnswerCoTOutput`, or ``None`` on failure.
    """
    # Build LLM client
    if llm is None:
        llm = _get_model(client=client or model_name)

    # Extract query string
    raw_query = query
    query_str = query[query_field] if isinstance(query, dict) else query

    # Step 1: Classify
    question_type = classify_question_type(query_str)
    if debug:
        print(f"Question type: {question_type}")

    # Step 2: Discover candidates
    if question_type == "binary":
        candidates = ["yes", "no"]
        discovery_raw = None
    else:
        candidates = discover_candidates(
            query=query_str,
            llm=llm,
            model_name=model_name,
            n=n_discovery,
            temperature=discovery_temperature,
            top_k=top_k,
        )
        discovery_raw = candidates[:]  # snapshot
        if debug:
            print(f"Discovered candidates: {candidates}")

    if len(candidates) < 2:
        logger.warning("Only %d candidate(s) found; need at least 2", len(candidates))
        return None

    # Dispatch on candidate count:
    #   N == 2 → binary path (write_multi_aggr_program with Combine)
    #   N >  2 → MCQ path    (write_mcq_program with Mixture, treating each
    #                         candidate string as both the letter label and
    #                         the choice text)
    is_mcq = len(candidates) > 2

    # Step 3: Generate per-answer CoTs (strategy-dependent)
    per_answer_cots: Dict[str, CoTOutput] = {}

    if is_mcq:
        # ----- Normalize the MCQ candidate input into (labels, texts) -----
        # Accepts three shapes (see docstring for examples):
        #   - plain strings       : ["heart", "liver", ...]   → auto-letter A/B/C
        #   - (label, text) tuple : [("A", "heart"), ...]
        #   - {label, text} dict  : [{"label": "A", "text": "heart"}, ...]
        choice_labels, choice_texts = _normalize_mcq_candidates(candidates)
        if debug:
            print(f"MCQ candidates normalized: "
                  f"labels={choice_labels} texts={choice_texts}")
        choices_block = format_choices_block(choice_labels, choice_texts)

        if cot_strategy == "separate":
            for lbl, txt in zip(choice_labels, choice_texts):
                cot = generate_per_answer_cot_mcq(
                    query=query_str, choices_block=choices_block,
                    choice_label=lbl, choice_text=txt,
                    llm=llm, model_name=model_name, num_retries=num_retries,
                    remove_meta_tokens=remove_meta_tokens,
                    length_normalization=length_normalization,
                    temperature=temperature, top_p=top_p,
                    strict=strict, debug=debug,
                )
                if cot is None:
                    logger.warning("Could not generate MCQ CoT for '%s'", lbl)
                    return None
                per_answer_cots[lbl] = cot

        elif cot_strategy == "combined":
            cot_list = generate_combined_cot_mcq(
                query=query_str, choices_block=choices_block,
                choice_labels=choice_labels, choice_texts=choice_texts,
                llm=llm, model_name=model_name, num_retries=num_retries,
                remove_meta_tokens=remove_meta_tokens,
                length_normalization=length_normalization,
                temperature=temperature, top_p=top_p,
                strict=strict, debug=debug,
            )
            if cot_list is None:
                logger.warning("Combined MCQ CoT generation failed")
                return None
            for lbl, cot in zip(choice_labels, cot_list):
                per_answer_cots[lbl] = cot

        else:
            raise ValueError(f"Unknown cot_strategy: {cot_strategy!r}")

    else:
        # ---------- Binary (N == 2) path — original behaviour ----------
        if cot_strategy == "separate":
            for cand in candidates:
                cot = generate_per_answer_cot(
                    query=query_str,
                    candidate=cand,
                    llm=llm,
                    model_name=model_name,
                    num_retries=num_retries,
                    verbalization=verbalization,
                    remove_meta_tokens=remove_meta_tokens,
                    length_normalization=length_normalization,
                    temperature=temperature,
                    top_p=top_p,
                    strict=strict,
                    debug=debug,
                )
                if cot is None:
                    logger.warning("Could not generate CoT for candidate '%s'", cand)
                    return None
                per_answer_cots[cand] = cot

        elif cot_strategy == "combined":
            cot_list = generate_combined_cot(
                query=query_str,
                candidates=candidates,
                llm=llm,
                model_name=model_name,
                num_retries=num_retries,
                remove_meta_tokens=remove_meta_tokens,
                length_normalization=length_normalization,
                temperature=temperature,
                top_p=top_p,
                strict=strict,
                debug=debug,
            )
            if cot_list is None:
                logger.warning("Combined CoT generation failed")
                return None
            for cand, cot in zip(candidates, cot_list):
                per_answer_cots[cand] = cot

        else:
            raise ValueError(f"Unknown cot_strategy: {cot_strategy!r}")

    # Step 3b: Teacher-forcing priors (binary only — MCQ caches TF inside
    # run_mcq_external). ONE parallel correct/incorrect call over all
    # candidates; the slot-1 prior per candidate is the renormalised token
    # logprob of 'correct' vs 'incorrect'. ALWAYS computed so the answer
    # prior has both slots available; answer_aggregation_fn selects later.
    tf_priors = {}
    if not is_mcq:
        tf_result = teacher_forcing_logprobs_parallel(
            query=query_str,
            candidate_labels=list(candidates),
            llm=llm,
            model_name=model_name,
            num_retries=num_retries,
            debug=debug,
        )
        if tf_result is not None:
            tf_priors, tf_meta = tf_result
            for cand in candidates:
                per_answer_cots[cand]._tf_priors = tf_priors
                per_answer_cots[cand]._tf_meta = tf_meta
                per_answer_cots[cand]._tf_raw = tf_meta.get("response")
        else:
            logger.warning("TF prior call failed; answer prior falls back to "
                           "the verbal slot / uniform")
        if debug:
            print(f"Teacher-forcing priors: {tf_priors}")

    # Step 4: Assemble program — dispatch on N
    pool_mode = kwargs.get("pool_mode", "logitpool")
    if is_mcq:
        program = write_mcq_program(
            query=query_str,
            choice_labels=choice_labels,
            choice_texts=choice_texts,
            cot_outputs=[per_answer_cots[lbl] for lbl in choice_labels],
            step_dependency_mode=step_dependency_mode,
            pool_mode=pool_mode,
            answer_aggregation_fn=kwargs.get("answer_aggregation_fn", "c"),
            statement_aggregation_fn=kwargs.get("statement_aggregation_fn", "mean"),
        )
        # Surface the auto-derived labels back through `candidates` so the
        # downstream MultiAnswerCoTOutput knows what the answer labels are.
        candidates = choice_labels
    else:
        program = write_multi_aggr_program(
            query=query_str,
            candidates=candidates,
            cot_outputs=[per_answer_cots[c] for c in candidates],
            tf_priors=tf_priors,
            step_dependency_mode=step_dependency_mode,
            pool_mode=pool_mode,
            answer_aggregation_fn=kwargs.get("answer_aggregation_fn", "c"),
            statement_aggregation_fn=kwargs.get("statement_aggregation_fn", "mean"),
        )

    if debug:
        print("Generated program:")
        print(program)

    return MultiAnswerCoTOutput(
        query=query_str,
        question_type=question_type,
        candidates=candidates,
        per_answer_cots=per_answer_cots,
        program=program,
        discovery_raw=discovery_raw,
    )


# ===========================================================================
# Multiple-choice (MCQ) support — adds N-way Categorical/Mixture programs.
# Used by run_BBH_multi_cot.py for tasks like BBH where each example has N
# choice letters (A, B, C, ...) and one is correct.
#
# Conventions:
#   * `choice_labels`  : list of letters, e.g. ["A","B","C","D"]
#   * `choice_texts`   : list of corresponding option texts
#   * `candidate_letter` is what's stored as the (final answer: X) value
# ===========================================================================

def format_choices_block(labels, texts) -> str:
    """Format a choices block for the prompt:
        A. heart
        B. liver
        ...
    """
    return "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))


def _normalize_mcq_candidates(candidates):
    """Normalize a candidate list into ``(choice_labels, choice_texts)``.

    Accepted input shapes (any one of these works in
    :func:`multi_answer_cot_to_program` and via the discovery path):

      * plain answer-text strings — letters are auto-assigned by position::

            ["heart", "liver", "kidney", "lung"]
            -> labels=["A","B","C","D"], texts=["heart","liver","kidney","lung"]

      * ``(label, text)`` 2-tuples / 2-lists::

            [("A", "heart"), ("B", "liver"), ("C", "kidney"), ("D", "lung")]
            -> labels=["A","B","C","D"], texts=["heart","liver","kidney","lung"]

      * dicts with ``"label"`` and ``"text"`` keys::

            [{"label": "A", "text": "heart"}, {"label": "B", "text": "liver"}, ...]

    Mixing shapes within a single list is allowed but discouraged.

    Returns:
        ``(labels, texts)`` — both ``List[str]`` of equal length, with
        ``labels`` in upper-case so the parser regex matches.
    """
    labels: List[str] = []
    texts: List[str] = []
    auto_idx = 0
    for c in candidates:
        if isinstance(c, dict) and "label" in c and "text" in c:
            labels.append(str(c["label"]).strip().upper())
            texts.append(str(c["text"]).strip())
        elif isinstance(c, (tuple, list)) and len(c) == 2:
            labels.append(str(c[0]).strip().upper())
            texts.append(str(c[1]).strip())
        else:
            # Plain string — auto-assign next letter (A, B, C, ...).
            labels.append(chr(ord("A") + auto_idx))
            texts.append(str(c).strip())
        auto_idx += 1
    return labels, texts


def _split_combined_response_mcq(raw_text, logprob_tokens, choice_labels):
    """Split an N-section MCQ combined response into per-letter segments.

    Each section is expected to be marked with ``## For <letter>:`` (a trailing
    ``(initial confidence: X%)`` is also accepted and parsed).

    Returns:
        (segments, final_answer_info) where:
          - segments is a list of length N: (label, segment_text, segment_tokens, init_score)
            in the order the labels appear in the response.
          - final_answer_info is a dict with the global ``## Final Answer:`` line
            (or None if the model didn't emit one).
    """
    label_pat = "|".join(re.escape(l) for l in choice_labels)
    header_re = re.compile(
        r'^##\s*[Ff]or\s+(' + label_pat + r')'
                                          r'(?:\s*\(\s*initial\s+confidence:\s*([\d.]+)%\s*\))?\s*[:\.]',
        re.MULTILINE,
    )
    headers = list(header_re.finditer(raw_text))
    if len(headers) < len(choice_labels):
        raise ValueError(
            f"Expected {len(choice_labels)} section headers but found {len(headers)}. "
            f"Raw text starts: {raw_text[:200]!r}"
        )
    headers = headers[:len(choice_labels)]
    ordered_labels = [h.group(1).strip() for h in headers]
    ordered_scores = []
    for h in headers:
        try:
            ordered_scores.append(float(h.group(2)) / 100.0)
        except (TypeError, ValueError):
            ordered_scores.append(None)

    # Locate the (optional) ## Final Answer: section
    fa_re = re.compile(r'^##\s*[Ff]inal\s+[Aa]nswer\b[^\n]*', re.MULTILINE)
    fa_match = fa_re.search(raw_text)
    fa_boundary = fa_match.start() if fa_match else len(raw_text)

    # Per-section text segments (everything between consecutive headers,
    # excluding the Final Answer section).
    text_segments = []
    for i, h in enumerate(headers):
        start = h.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else fa_boundary
        text_segments.append(raw_text[start:end].strip())

    # Extract the global Final Answer info (letter + token logprob + ALL
    # tokens spanning the '## Final Answer:' section).
    final_answer_info = None
    if fa_match:
        fa_section = raw_text[fa_match.end():].strip()
        fa_letter_re = re.search(
            r'\(\s*final\s+answer:\s*([A-Za-z]+)\s*\)',
            fa_section, re.IGNORECASE,
        )
        if fa_letter_re:
            fa_value = fa_letter_re.group(1).strip().upper()
            fa_char_pos = fa_match.end() + fa_letter_re.start(1)
            offset = 0
            fa_logprob = None
            valid_letters = {l.upper() for l in choice_labels}
            # Walk tokens to also capture the FULL token range covering
            # everything from the '## Final Answer' header to the end of
            # the response (so the per-token logprobs for the entire FA
            # section can be saved).
            fa_section_token_start = None
            for i, tok in enumerate(logprob_tokens):
                t = tok.token if hasattr(tok, "token") else tok["token"]
                tok_end = offset + len(t)
                if fa_section_token_start is None and tok_end > fa_match.start():
                    fa_section_token_start = i
                if (fa_logprob is None and tok_end > fa_char_pos
                        and t.strip().upper() in valid_letters):
                    fa_logprob = tok.logprob if hasattr(tok, "logprob") else tok["logprob"]
                offset = tok_end
            if fa_section_token_start is None:
                fa_section_token_start = len(logprob_tokens)
            fa_section_tokens = logprob_tokens[fa_section_token_start:]
            if fa_logprob is not None:
                fa_prob = max(min(math.exp(fa_logprob), 1.0 - 1e-7), 1e-7)
            else:
                fa_prob = 0.5
            final_answer_info = {
                "value": fa_value,
                "prob": fa_prob,
                "raw": fa_section,
                "logprobs": [
                    {"token": getattr(t, "token", None),
                     "logprob": getattr(t, "logprob", None)}
                    for t in fa_section_tokens
                ],
            }

    # Walk tokens to find the start char-offset of each header / final answer
    # block, so we can slice the token stream into per-section subsequences.
    char_offset = 0
    boundary_chars = [h.start() for h in headers] + [fa_boundary]
    boundary_token_idx = [None] * len(boundary_chars)
    for i, tok in enumerate(logprob_tokens):
        t = tok.token if hasattr(tok, "token") else tok["token"]
        next_off = char_offset + len(t)
        for bi, bc in enumerate(boundary_chars):
            if boundary_token_idx[bi] is None and next_off > bc:
                boundary_token_idx[bi] = i
        if all(x is not None for x in boundary_token_idx):
            break
        char_offset = next_off
    # Fall-through fill (rare: very short final segment)
    for bi, x in enumerate(boundary_token_idx):
        if x is None:
            boundary_token_idx[bi] = len(logprob_tokens)

    # For each section, take tokens from "after this header line ends" up to
    # the next boundary. We approximate "after header" by walking forward
    # until we cross a newline.
    def _after_header(start_idx):
        for j in range(start_idx, len(logprob_tokens)):
            t = (logprob_tokens[j].token if hasattr(logprob_tokens[j], "token")
                 else logprob_tokens[j]["token"])
            if "\n" in t:
                return j + 1
        return len(logprob_tokens)

    segs = []
    for i in range(len(headers)):
        header_start = boundary_token_idx[i]
        body_start = _after_header(header_start)
        body_end = boundary_token_idx[i + 1] if i + 1 < len(headers) else boundary_token_idx[-1]
        # 5-tuple: (label, body_text, body_tokens, init_score, header_tokens).
        # header_tokens are the tokens spanning '## For X (initial confidence: Y%):'
        # — useful for inspecting the model's confidence at the init-score
        # header itself (saved separately on each cot via
        # _attach_init_score_logprobs; not used by find_logps for step
        # alignment).
        header_tokens = logprob_tokens[header_start:body_start]
        body_tokens = logprob_tokens[body_start:body_end]
        segs.append((
            ordered_labels[i], text_segments[i], body_tokens,
            ordered_scores[i], header_tokens,
        ))

    return segs, final_answer_info


def generate_per_answer_cot_mcq(
        query: str,
        choices_block: str,
        choice_label: str,
        choice_text: str,
        llm: LLMType,
        model_name: str,
        num_retries: int = 5,
        remove_meta_tokens: bool = True,
        length_normalization: bool = True,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        strict: bool = False,
        debug: bool = False,
) -> Optional[CoTOutput]:
    """Generate a CoT arguing for ONE specific choice letter on an MCQ.

    Mirrors ``generate_per_answer_cot`` but uses MCQ prompts and templates.
    """
    from cot_prog.prompt.prompts import (
        PER_ANSWER_COT_INSTRUCTION_MCQ_VERBAL,
        MCQ_PER_ANSWER_USER_TEMPLATE,
    )
    find_logps = _find_logps_import()

    target = f"{choice_label}. {choice_text}"
    user_msg = MCQ_PER_ANSWER_USER_TEMPLATE.format(
        query=query, choices_block=choices_block, candidate=target,
    )
    error_msg = "No errors, first attempt"
    parsing_errors: List[str] = []
    # Keep every attempt's raw response so we can dump it on total failure,
    # which is the only way to diagnose why a strict-grammar parser kept
    # rejecting the model's output (especially relevant for gpt-4o which
    # tends to ignore the format hints).
    failed_attempts: List[Dict[str, str]] = []

    for attempt in range(1, num_retries + 1):
        logger.debug(f"[MCQ_PER_ANSWER '{choice_label}'] Attempt {attempt}/{num_retries}")
        system_prompt = PER_ANSWER_COT_INSTRUCTION_MCQ_VERBAL.format(
            candidate_answer=target,
            candidate_letter=choice_label,
            errors=error_msg,
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_msg}]},
        ]
        try:
            response = llm.chat.completions.create(
                model=model_name, temperature=temperature,
                messages=messages, top_p=top_p, logprobs=True,
            )
        except Exception as exc:
            error_msg = str(exc);
            parsing_errors.append(error_msg)
            logger.debug(f"[MCQ_PER_ANSWER '{choice_label}'] API error: {exc}")
            failed_attempts.append({"attempt": attempt, "kind": "api_error",
                                    "error": str(exc), "raw": ""})
            continue

        raw_message = response.choices[0].message.content
        # Always log the first 600 chars on each attempt — useful for
        # diagnosing strict-grammar parse failures on non-llama models.
        logger.debug(f"[MCQ_PER_ANSWER '{choice_label}'] Raw (attempt {attempt}, "
              f"first 600 chars):\n{(raw_message or '')[:600]}\n{'=' * 40}")
        if debug:
            print(raw_message);
            print("=" * 40)

        try:
            from types import SimpleNamespace
            # Extract the leading "(initial confidence: X%)" header (if
            # present) BEFORE handing the rest to the strict parser. This
            # is the model's prior on the candidate, written before any
            # reasoning. Used as cot._init_score downstream (mirrors the
            # combined-strategy section header's init_score).
            init_score = None
            cleaned = raw_message
            init_end_char = 0  # char offset where the init line ends
            m_init = re.search(
                r"^\s*\(?\s*initial\s+confidence\s*:?\s*(\d+(?:\.\d+)?)\s*%?\s*\)?\s*$",
                raw_message or "", flags=re.IGNORECASE | re.MULTILINE,
            )
            if m_init:
                try:
                    pct = float(m_init.group(1))
                    init_score = max(0.05, min(0.99, pct / 100.0))
                except Exception:
                    init_score = None
                # Strip the matched line (and surrounding blank lines) so
                # the rest can be parsed as a clean numbered-step CoT.
                pre = raw_message[:m_init.start()]
                post = raw_message[m_init.end():]
                cleaned = (pre + post).lstrip()
                # Track where the init line + any leading whitespace ended
                # in the original raw_message so we can slice the matching
                # prefix off the token stream below.
                init_end_char = m_init.end()
                # Skip any extra leading whitespace/newlines that we
                # .lstrip()-ed off; advance init_end_char past them.
                while init_end_char < len(raw_message) and \
                        raw_message[init_end_char] in " \t\r\n":
                    init_end_char += 1

            # Strict CoT parser requires EVERY line to start with a
            # numbered-step prefix (HEADER_RE = r'^\s*(\d+)(\s*[\.\)\-:]\s*)').
            # gpt-4o sometimes adds:
            #   (a) blank lines between steps,
            #   (b) trailing summary paragraphs after the last numbered step,
            #   (c) preamble like "Sure, let me think step by step:"
            # All of these crash the strict parser. Filter them out: keep
            # only lines that start with `\d+[.\)\-:]`. (Token-stream
            # slicing below keys on init_end_char only, which is fine —
            # find_logps does substring matching against step text, so
            # a few extra non-step tokens get harmlessly attributed to
            # whichever step they fall into.)
            STEP_PREFIX_RE = re.compile(r'^\s*\d+\s*[\.\)\-:]')
            kept_lines = [ln for ln in cleaned.splitlines()
                          if STEP_PREFIX_RE.match(ln)]
            if kept_lines:
                cleaned = "\n".join(kept_lines)

            cot_parser = CoTContreteSyntaxParser()
            cot = cot_parser.parse(cleaned, query=query)
            err = cot.check_program(strict=strict)
            if err:
                raise ValueError(err)

            # Build a token-stream view that matches `cleaned`: walk the
            # full token stream, accumulating char count, and drop the
            # tokens whose right-edge falls before init_end_char. This
            # keeps find_logps's step alignment in sync with cot.lines.
            full_tokens = _normalize_logprobs(response.choices[0])
            drop_n = 0
            if init_end_char > 0:
                char_pos = 0
                for i, tok in enumerate(full_tokens):
                    char_pos += len(getattr(tok, "token", "") or "")
                    if char_pos >= init_end_char:
                        drop_n = i + 1
                        break
                seg_tokens = full_tokens[drop_n:]
                mock_choice = SimpleNamespace(
                    logprobs=SimpleNamespace(content=seg_tokens)
                )
            else:
                seg_tokens = full_tokens
                mock_choice = response.choices[0]

            step_probs = find_logps(
                mock_choice, cot.lines,
                remove_meta_tokens=remove_meta_tokens,
                length_normalization=length_normalization,
            )
            cot.add_weights(step_probs)
            # Restore raw_cot to the FULL original response (including the
            # `(initial confidence: X%)` header line) so it's visible in
            # the saved checkpoint. The parser already consumed the cleaned
            # version above for step alignment; we just override what gets
            # serialized.
            cot.raw_cot = raw_message
            # Save the FULL response's per-token logprobs as one unified
            # cot_logprobs field — includes the leading
            # '(initial confidence: X%)' line tokens; no separate
            # init_score_logprobs needed.
            try:
                _attach_cot_logprobs(cot, full_tokens)
            except Exception:
                cot._cot_logprobs = None
            _attach_cot_prompt(cot, system=system_prompt, user=user_msg, response=raw_message)
            if init_score is not None:
                cot._init_score = init_score
            cot.attempts = attempt
            cot.errors = parsing_errors
            return cot
        except Exception as exc:
            error_msg = str(exc);
            parsing_errors.append(error_msg)
            logger.debug(f"[MCQ_PER_ANSWER '{choice_label}'] Parse error: {exc}")
            failed_attempts.append({"attempt": attempt, "kind": "parse_error",
                                    "error": str(exc),
                                    "raw": raw_message or ""})
            continue

    # All num_retries attempts exhausted. Stash the failed attempts on a
    # module-level scratchpad so run_mcq_external can include them in the
    # raised RuntimeError (and thus in the per-example error checkpoint).
    _LAST_PER_ANSWER_FAILURES[choice_label] = {
        "system_prompt": system_prompt,
        "user_msg": user_msg,
        "attempts": failed_attempts,
    }
    return None


def generate_combined_cot_mcq(
        query: str,
        choices_block: str,
        choice_labels: List[str],
        choice_texts: List[str],
        llm: LLMType,
        model_name: str,
        num_retries: int = 5,
        remove_meta_tokens: bool = True,
        length_normalization: bool = True,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        strict: bool = False,
        debug: bool = False,
) -> Optional[List[CoTOutput]]:
    """Generate one CoT per choice in a SINGLE LLM call (combined strategy)."""
    from types import SimpleNamespace
    from cot_prog.prompt.prompt_to_program import find_logps
    from cot_prog.prompt.prompts import (
        COMBINED_COT_INSTRUCTION_MCQ,
        MCQ_USER_TEMPLATE,
    )

    n = len(choice_labels)
    user_msg = MCQ_USER_TEMPLATE.format(query=query, choices_block=choices_block)
    error_msg = "No errors, first attempt"
    parsing_errors: List[str] = []

    for attempt in range(1, num_retries + 1):
        logger.debug(f"[MCQ_COMBINED] Attempt {attempt}/{num_retries}")
        system_prompt = COMBINED_COT_INSTRUCTION_MCQ.format(n=n, errors=error_msg)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_msg}]},
        ]
        try:
            response = llm.chat.completions.create(
                model=model_name, temperature=temperature,
                messages=messages, top_p=top_p, logprobs=True,
            )
        except Exception as exc:
            error_msg = str(exc);
            parsing_errors.append(error_msg)
            logger.debug(f"[MCQ_COMBINED] API error: {exc}")
            continue

        raw_message = response.choices[0].message.content
        if debug:
            print(raw_message);
            print("=" * 40)

        try:
            all_tokens = _normalize_logprobs(response.choices[0])
            segments, final_answer_info = _split_combined_response_mcq(
                raw_message, all_tokens, choice_labels,
            )
        except Exception as exc:
            error_msg = f"Section splitting failed: {exc}"
            parsing_errors.append(error_msg)
            logger.debug(f"[MCQ_COMBINED] Split error: {exc}")
            continue

        cot_parser = CoTContreteSyntaxParser()
        per_label = {}
        parse_ok = True
        for seg_label, seg_text, seg_tokens, init_score, header_tokens in segments:
            try:
                # Qwen-only patch: Qwen-2.5-7B at non-zero T sometimes emits
                # `(ref. <source citation>)` instead of step indices. Rewrite
                # those refs to `(ref. none)` so the parser doesn't crash on
                # "Invalid Boolean syntax". See sanitize_refs() docstring.
                if _is_qwen_model(model_name):
                    seg_text = sanitize_refs(seg_text)
                cot = cot_parser.parse(seg_text, query=query)
                cot._init_score = init_score
                err = cot.check_program(strict=strict)
                if err:
                    raise ValueError(f"[{seg_label}] {err}")
                mock = SimpleNamespace(logprobs=SimpleNamespace(content=seg_tokens))
                step_probs = find_logps(
                    mock, cot.lines,
                    remove_meta_tokens=remove_meta_tokens,
                    length_normalization=length_normalization,
                )
                cot.add_weights(step_probs)
                # Save the FULL section's per-token logprobs (header + body)
                # as one unified `cot_logprobs` field — no separate
                # init_score_logprobs. Analysts can identify the init-line
                # span by looking for '(initial confidence: X%)' inside the
                # saved tokens or by reading cot._init_score directly.
                _attach_cot_logprobs(cot, list(header_tokens) + list(seg_tokens))
                _attach_cot_prompt(cot, system=system_prompt, user=user_msg, response=raw_message)
            except Exception as exc:
                error_msg = f"[{seg_label}] {exc}"
                parsing_errors.append(error_msg)
                logger.debug(f"[MCQ_COMBINED] {error_msg}")
                parse_ok = False
                break
            per_label[seg_label.upper()] = cot

        if not parse_ok:
            continue

        # Reorder to match choice_labels
        result = []
        for lbl in choice_labels:
            cot = per_label.get(lbl.upper())
            if cot is None:
                error_msg = f"Missing section for choice '{lbl}'"
                parsing_errors.append(error_msg)
                logger.debug(f"[MCQ_COMBINED] {error_msg}")
                parse_ok = False
                break
            cot.attempts = attempt
            cot.errors = parsing_errors
            cot._raw_combined = raw_message
            result.append(cot)
        if not parse_ok:
            continue

        # Stash the global Final Answer info on result[0] so the runner
        # can surface it at the trace level (with full per-token logprobs).
        if result and final_answer_info:
            result[0]._final_answer_info = final_answer_info
        logger.debug(f"[MCQ_COMBINED] Success on attempt {attempt}")
        return result

    return None


def _compute_answer_prior_mcq(
        cot: CoTOutput,
        candidate_letter: str,
        n_candidates: int,
) -> tuple:
    """MCQ answer-prior slots — thin wrapper over :func:`_answer_prior_slots`.

    Returns the canonical ``(slot1_logit, slot2_verbal)`` pair for an MCQ
    letter; ``answer_aggregation_fn`` (applied by the writer) selects/combines
    them. The old per-mode axis (teacher_forcing / verbalized / uniform) is
    gone — both slots are always cached at generation, and the verbal slot
    falls back to the answer step's ``(confidence: X%)`` inside
    ``_answer_prior_slots``.
    """
    return _answer_prior_slots(
        cot, candidate=candidate_letter, n_candidates=n_candidates,
    )


def write_mcq_program(
        query: str,
        choice_labels: List[str],
        choice_texts: List[str],
        cot_outputs: List[CoTOutput],
        step_dependency_mode: str = "flat",
        pool_mode: str = "logitpool",
        answer_aggregation_fn: str = "c",
        statement_aggregation_fn: str = "mean",
) -> str:
    """Generate a MULTI_AGGR-style probabilistic program for an N-way MCQ.

    Layout::

        ### original query
        query = "..."

        ### step strings (per-candidate, globally re-indexed)
        s1: str = "..."
        ...

        ## answer priors
        A_pr: bool =~ Flip("A", priorA)
        B_pr: bool =~ Flip("B", priorB)
        ...
        A_x, B_x, ... =~ Categorical("answer", Mixture(A_pr.<rule>, ...))

        ## evidence for A / B / ...
        P{gid}: bool =~ Flip(s{gid}, w{gid}, [conf{gid}])
        ...
        A_E: bool =~ Factor(..., LogitPool(P{gid}.<aggr>, ...))

        observe((A_x and A_E) or (B_x and B_E) or ...)

        if A_x: return s_<answer A>
        elif B_x: return s_<answer B>
        ...
    """
    n = len(choice_labels)
    assert len(choice_texts) == n and len(cot_outputs) == n, \
        "choice_labels/texts/cot_outputs must have matching length"

    # Sanitized label-prefixes used for the generated identifiers. We always
    # add a '_x' suffix on Categorical results (so a label like 'A' doesn't
    # collide with a `Flip` named 'A_pr' or 'A_E').
    safe_labels = [_sanitize_name(l) for l in choice_labels]
    pr_names = [f"{s}_pr" for s in safe_labels]
    cat_names = [f"{s}_x" for s in safe_labels]
    ev_names = [f"{s}_E" for s in safe_labels]

    lines: List[str] = []
    lines.append("### original query")
    q_escaped = query.replace('"', '\\"').replace("\n", " ")
    lines.append(f'query: str = "{q_escaped}"')
    lines.append("")

    # ----- Step strings: globally re-indexed across candidates -----
    candidate_steps: List[List[tuple]] = []  # per-cand: [(gid, step, is_ans), ...]
    answer_string_vars: List[str] = []  # s{gid} that holds final answer
    local_to_global_maps: List[Dict[int, int]] = []
    global_counter = 1

    lines.append("### step strings")
    for ci, (lbl, txt, cot) in enumerate(zip(choice_labels, choice_texts, cot_outputs)):
        info: List[tuple] = []
        local_to_global: Dict[int, int] = {}
        ans_svar = None
        for step in cot.lines:
            gid = global_counter
            global_counter += 1
            local_to_global[step.prefix] = gid
            text = step.claim if step.claim else step.answer
            text_escaped = text.replace('"', '\\"').replace("\n", " ")
            lines.append(f's{gid}: str = "{text_escaped}"')
            info.append((gid, step, bool(step.answer)))
            if step.answer:
                ans_svar = f"s{gid}"
        candidate_steps.append(info)
        local_to_global_maps.append(local_to_global)
        answer_string_vars.append(ans_svar or f"s{info[-1][0]}")
    lines.append("")

    # ----- Answer priors -----
    # Canonical (slot1_logit, slot2_verbal) per letter; answer_aggregation_fn
    # selects/combines via the Flip attribute below.
    priors = [
        _compute_answer_prior_mcq(cot, candidate_letter=lbl, n_candidates=n)
        for lbl, cot in zip(choice_labels, cot_outputs)
    ]

    lines.append("## answer priors")
    for pn, lbl, prior in zip(pr_names, choice_labels, priors):
        lines.append(f'{pn}: bool =~ Flip("{lbl}", {prior[0]}, [{prior[1]}])')

    cat_lhs = ", ".join(cat_names)
    # ``c`` selects the verbal (second) slot, exposed by the compiler as .p2.
    _ans_attr = attr_for(answer_aggregation_fn)
    mix_args = ", ".join(f"{p}.{_ans_attr}" for p in pr_names)
    lines.append(f'{cat_lhs} =~ Categorical("answer", Mixture({mix_args}))')
    lines.append("")

    # ----- Evidence per candidate -----
    dep_constraints: List[List[str]] = []
    for ci in range(n):
        lbl = choice_labels[ci]
        ev_name = ev_names[ci]
        l2g = local_to_global_maps[ci]
        evidence_steps = [(gid, st) for gid, st, is_ans in candidate_steps[ci] if not is_ans]
        evidence_gids = [gid for gid, _ in evidence_steps]
        evidence_gid_set = set(evidence_gids)
        constraints: List[str] = []
        lines.append(f"## evidence for {lbl}")

        for gid, step in evidence_steps:
            if step.confidence is not None:
                lines.append(f"P{gid}: bool =~ Flip(s{gid}, {step.weight}, [{step.confidence}])")
            elif statement_aggregation_fn == "c":
                # "c" reads only the verbalized confidence; without the tag
                # there is nothing to read, so fail with a clear message.
                raise ValueError(
                    f"Step {step.prefix} of choice '{lbl}' (P{gid} in the program) has no (confidence: X%) tag, "
                    "but --statement_aggregation_fn c uses only the verbalized "
                    "step confidence. Use another setting or regenerate the trace."
                )
            else:
                lines.append(f"P{gid}: bool =~ Flip(s{gid}, {step.weight})")
        pool_gids = evidence_gids
        # Aggregate evidence
        _attr = "p2" if statement_aggregation_fn == "c" else statement_aggregation_fn
        if len(pool_gids) >= 2:
            if pool_mode == "logitpool":
                refs = ", ".join(f"P{gid}.{_attr}" for gid in pool_gids)
                lines.append(f'{ev_name}: bool =~ Factor("{ev_name}", LogitPool({refs}))')
            else:
                raise ValueError(f"Unknown pool_mode: {pool_mode!r}")
        elif len(pool_gids) == 1:
            lines.append(f'{ev_name}: bool =~ Factor("{ev_name}", P{pool_gids[0]}.{_attr})')
        else:
            base_p = priors[ci] if not isinstance(priors[ci], tuple) else priors[ci][0]
            lines.append(f'{ev_name}: bool =~ Flip("{ev_name}", {base_p})')
        dep_constraints.append(constraints)
        lines.append("")

    # ----- Observe -----
    obs_parts = []
    for ci in range(n):
        bits = [cat_names[ci], ev_names[ci]] + dep_constraints[ci]
        obs_parts.append("(" + " and ".join(bits) + ")")
    lines.append(f"observe({' or '.join(obs_parts)})")
    lines.append("")

    # ----- Return branches -----
    for i, (cn, svar) in enumerate(zip(cat_names, answer_string_vars)):
        kw = "if" if i == 0 else "elif"
        lines.append(f"{kw} {cn}:")
        lines.append(f"    return {svar}")
    lines.append("else:")
    lines.append("    raise ValueError('Unable to find answer')")
    lines.append("")

    return "\n".join(lines)


def run_mcq_external(
        query: str,
        choice_labels: List[str],
        choice_texts: List[str],
        *,
        strategy: str = "combined",
        pool_mode: str = "logitpool",
        answer_aggregation_fn: str = "c",
        statement_aggregation_fn: str = "mean",
        step_dependency_mode: str = "flat",
        model_name: str = "gpt-4o",
        num_retries: int = 5,
        debug: bool = False,
        temperature: float = 0.0,
):
    """MCQ counterpart of :func:`run_multi_external`.

    Returns ``(final_cot, program_str, per_label_cots)``.
    """
    # Unified routing — Together for vendor/model names, OpenRouter for
    # openai/gpt-* (when OPENROUTER_API_KEY is set), vanilla OpenAI otherwise.
    llm = _get_model(model_name=model_name)

    choices_block = format_choices_block(choice_labels, choice_texts)
    per_label_cots: Dict[str, CoTOutput] = {}

    if strategy == "separate":
        # Reset the per-letter failure scratchpad before this example.
        _LAST_PER_ANSWER_FAILURES.clear()
        for lbl, txt in zip(choice_labels, choice_texts):
            cot = generate_per_answer_cot_mcq(
                query=query, choices_block=choices_block,
                choice_label=lbl, choice_text=txt,
                llm=llm, model_name=model_name, num_retries=num_retries,
                temperature=temperature, strict=False, debug=debug,
            )
            if cot is None:
                # Surface the actual failed attempts (raw model output +
                # parser complaints) inside the exception. process_example
                # catches the exception and writes its repr into
                # result["error"] — so the per-example checkpoint will
                # contain enough information to debug the failure post-hoc.
                fdump = _LAST_PER_ANSWER_FAILURES.get(lbl, {})
                attempts = fdump.get("attempts", [])
                # Trim raw responses so the error string isn't gigantic.
                trimmed = []
                for a in attempts:
                    trimmed.append({
                        "attempt": a.get("attempt"),
                        "kind": a.get("kind"),
                        "error": (a.get("error") or "")[:500],
                        "raw": (a.get("raw") or "")[:1500],
                    })
                msg = (
                    f"Failed to generate per-answer CoT for {lbl} "
                    f"after {num_retries} attempts.\n"
                    f"  system_prompt (head): {(fdump.get('system_prompt', '') or '')[:200]}...\n"
                    f"  user_msg     (head): {(fdump.get('user_msg', '') or '')[:200]}...\n"
                    f"  attempts: {json.dumps(trimmed, ensure_ascii=False)[:6000]}"
                )
                raise RuntimeError(msg)
            per_label_cots[lbl] = cot

    elif strategy == "combined":
        cot_list = generate_combined_cot_mcq(
            query=query, choices_block=choices_block,
            choice_labels=choice_labels, choice_texts=choice_texts,
            llm=llm, model_name=model_name, num_retries=num_retries,
            temperature=temperature, strict=False, debug=debug,
        )
        if cot_list is None:
            raise RuntimeError("Failed to generate combined MCQ CoT")
        for lbl, cot in zip(choice_labels, cot_list):
            per_label_cots[lbl] = cot

    else:
        raise ValueError(
            f"Unsupported MCQ strategy: {strategy!r}. "
            f"Supported: 'separate', 'combined'."
        )

    cot_list = [per_label_cots[l] for l in choice_labels]

    # Teacher-forcing (slot-1 / "logit") answer prior — ONE follow-up LLM
    # call, independent of the CoT analysis. The model labels every option
    # correct/incorrect; the prior per option is the renormalised token
    # logprob of 'correct' vs 'incorrect' at that option.
    #
    # ALWAYS compute it at generation so both answer-prior slots (TF-logit and
    # verbal init) are cached in the trace. answer_aggregation_fn (program
    # construction) then selects/combines them.
    tf_result = teacher_forcing_logprobs_parallel(
        query=query,
        candidate_labels=choice_labels,
        candidate_texts=choice_texts,
        llm=llm,
        model_name=model_name,
        num_retries=num_retries,
        debug=debug,
    )
    tf_meta = None
    if tf_result is None:
        uniform_p = 1.0 / max(len(choice_labels), 1)
        tf_priors_dict = {lbl: uniform_p for lbl in choice_labels}
        tf_meta = {
            "prompt": None,
            "response": "(TF call failed; uniform fallback)",
            "logprobs": None,
        }
        print(
            "[run_mcq_external] WARNING: TF parser returned no priors; "
            f"falling back to uniform ({uniform_p:.3f}) per candidate.",
            file=sys.stderr,
        )
    else:
        tf_priors_dict, tf_meta = tf_result
    for lbl in choice_labels:
        per_label_cots[lbl]._tf_priors = tf_priors_dict
        per_label_cots[lbl]._tf_meta = tf_meta
        # Legacy string attribute for back-compat with old readers.
        per_label_cots[lbl]._tf_raw = tf_meta.get("response") if tf_meta else None

    program_str = write_mcq_program(
        query=query,
        choice_labels=choice_labels,
        choice_texts=choice_texts,
        cot_outputs=cot_list,
        step_dependency_mode=step_dependency_mode,
        pool_mode=pool_mode,
        answer_aggregation_fn=answer_aggregation_fn,
        statement_aggregation_fn=statement_aggregation_fn,
    )

    # Build final_cot WITHOUT mutating per_label_cots[choice_labels[0]].
    # cot_list[0] is the same Python object as per_label_cots["A"]; in-place
    # assignment to .raw_cot would overwrite A's per-letter section text with
    # the full combined output. Use a shallow copy so the per-letter trace
    # stays section-scoped.
    import copy as _copy
    final_cot = _copy.copy(cot_list[0])
    combined_raw = getattr(cot_list[0], "_raw_combined", None)
    if combined_raw:
        final_cot.raw_cot = combined_raw
    else:
        parts = [f"## For {lbl}:\n{cot.raw_cot}" for lbl, cot in per_label_cots.items()]
        final_cot.raw_cot = "\n\n".join(parts)

    return final_cot, program_str, per_label_cots


# ============================================================================
# Teacher-forcing — parallel binary judgment (PURE answer prior)
# ============================================================================
# Sends a single follow-up call asking the model to label each candidate as
# 'correct' or 'incorrect'. Reads top_logprobs at each label position and
# renormalises over {'correct','incorrect'} to get a clean P(letter is
# correct). Independent of the CoT analysis — gives a pure-prior secondary
# slot for the answer-prior Flip.

def _build_tf_blocks(
        candidate_labels: List[str],
        candidate_texts: Optional[List[str]] = None,
):
    """Build (choices_block, template_block) for the TF prompt.

    Binary case: ``candidate_texts is None`` (or labels and texts are equal)
    – we render one line per label. MCQ case: render ``<letter>. <text>``.
    """
    if candidate_texts is None or all(l == t for l, t in zip(candidate_labels, candidate_texts)):
        choices_block = "\n".join(f"- {l}" for l in candidate_labels)
        template_block = "\n".join(
            f"{l} — <correct or incorrect>" for l in candidate_labels
        )
    else:
        choices_block = "\n".join(
            f"{l}. {t}" for l, t in zip(candidate_labels, candidate_texts)
        )
        template_block = "\n".join(
            f"{l}. {t} — <correct or incorrect>"
            for l, t in zip(candidate_labels, candidate_texts)
        )
    return choices_block, template_block


def _serialize_logprobs(normalized) -> Optional[list]:
    """JSON-friendly copy of a :func:`_normalize_logprobs` result (for
    ``tf_meta["logprobs"]``). ``None`` when nothing was returned."""
    if not normalized:
        return None
    out = []
    for tok in normalized:
        out.append({
            "token": getattr(tok, "token", None),
            "logprob": getattr(tok, "logprob", None),
            "top_logprobs": [
                {"token": getattr(t, "token", None),
                 "logprob": getattr(t, "logprob", None)}
                for t in (getattr(tok, "top_logprobs", None) or [])
            ],
        })
    return out


def teacher_forcing_logprobs_parallel(
        query: str,
        candidate_labels: List[str],
        llm: LLMType,
        model_name: str,
        *,
        candidate_texts: Optional[List[str]] = None,
        num_retries: int = 3,
        temperature: float = 0.0,
        debug: bool = False,
        tf_alt_floor: float = -3.0,
        tf_clamp: tuple = (0.02, 0.98),
) -> Optional[tuple]:
    """Compute per-candidate ``P(letter is correct)`` via parallel binary TF.

    This fills slot 1 (the "logit" slot, ``.p``) of every answer-prior Flip.

    Sends a *single* follow-up call (independent of any CoT) where the model
    is asked to label each option as ``correct`` or ``incorrect`` on its own
    line.  At each label position the API returns ``top_logprobs``; we pull
    out logprobs for both the ``correct`` and ``incorrect`` tokens and
    renormalise over those two options:

        P(letter correct | binary) = exp(lp_correct) / (exp(lp_correct) + exp(lp_incorrect))

    Args:
        query:               The question (no analysis attached — pure prior).
        candidate_labels:    e.g. ``["yes","no"]`` or ``["A","B","C","D"]``.
        candidate_texts:     Optional MCQ option texts. ``None`` for binary.
        llm, model_name:     LLM client + model name.
        num_retries:         Re-try the LLM call this many times if parsing fails.
        temperature:         Sampling temperature (default 0.0).
        debug:               Print the response and any parse warnings.

        tf_alt_floor:        Logprob assumed for the alternative word when it
                             is missing from top-K (model too confident).
        tf_clamp:            Final clamp on each P(correct).

    Returns:
        ``(tf_priors, tf_meta)`` if successful, else ``None``.
        ``tf_priors`` is ``{label: P(label is correct)}``: one independent
        Bernoulli per candidate, NOT normalised across labels (the program's
        ``Categorical(Mixture(...))`` / ``Combine`` normalises, exactly as it
        does for the verbalized slot). ``tf_meta`` is::

            {"prompt": ..., "response": ..., "logprobs": [...],
             "per_letter": {label: {chosen, lp_correct, lp_incorrect,
                                    path, p_correct}},
             "params": {"tf_alt_floor": ..., "tf_clamp": [...]}}
    """
    from cot_prog.prompt.prompts import TF_PARALLEL_TEMPLATE

    n = len(candidate_labels)
    choices_block, template_block = _build_tf_blocks(candidate_labels, candidate_texts)

    user_msg = TF_PARALLEL_TEMPLATE.format(
        query=query,
        choices_block=choices_block,
        template_block=template_block,
        n=n,
    )

    error_msg = "first attempt"
    for attempt in range(1, num_retries + 1):
        try:
            response = llm.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": user_msg}],
                temperature=temperature,
                logprobs=True,
                top_logprobs=20,
                max_tokens=512,
            )
        except Exception as exc:
            error_msg = str(exc)
            if debug:
                print(f"[TF_PARALLEL] API error (attempt {attempt}): {exc}")
            continue

        try:
            raw = response.choices[0].message.content
            # Use the shared normaliser — handles BOTH OpenAI's
            # logprobs.content shape and Together's legacy
            # tokens/token_logprobs/top_logprobs shape.
            all_tokens = _normalize_logprobs(response.choices[0])
        except (AttributeError, TypeError, IndexError, ValueError) as exc:
            error_msg = f"missing logprobs: {exc}"
            if debug:
                print(f"[TF_PARALLEL] {error_msg}")
            continue

        if not raw or not all_tokens:
            error_msg = "empty response"
            continue

        # ALWAYS print the raw TF response (not just in debug mode) so the
        # user can verify the model is following the requested format.
        logger.debug(f"\n[TF_PARALLEL] === attempt {attempt}/{num_retries} response ===\n"
            f"{raw}\n"
            f"[TF_PARALLEL] === end response ===")

        tf_priors: Dict[str, float] = {}
        per_letter_diag: Dict[str, dict] = {}  # diagnostic info, printed at end
        all_iter_texts = candidate_texts if candidate_texts is not None else candidate_labels
        unmatched_labels: List[str] = []
        for label, _text in zip(candidate_labels, all_iter_texts):
            label_esc = re.escape(label)
            # Try multiple regex patterns, from strict→lax. Stop at the first match.
            patterns = [
                # 1. Strict: label . text — correct/incorrect (em-dash, en-dash,
                #    --, -, =, :, →, |)
                rf"(?im)^\s*\**\s*\(?{label_esc}\)?[.\)]?(?:\s+[^\n—–=:\->|→]*?)?\s*(?:[—–]|--|->|→|[\-=:|])\s*(correct|incorrect)\b",
                # 2. Relaxed: label followed (anywhere on the line) by correct/incorrect
                rf"(?im)^\s*\**\s*\(?{label_esc}\)?[.\)]?[^\n]*?\b(correct|incorrect)\b",
                # 3. Anywhere within a few hundred chars of the label
                rf"(?is)\b{label_esc}\b[^\n]{{0,200}}?\b(correct|incorrect)\b",
            ]
            m = None
            for pat in patterns:
                m = re.search(pat, raw)
                if m is not None:
                    break
            if m is None:
                unmatched_labels.append(label)
                continue

            chosen_word = m.group(1).lower()
            chosen_pos = m.start(1)

            # Walk the token stream to find the token covering chosen_pos
            # and whose text starts with 'cor' or 'inc' (matches chosen_word).
            # Token "covers" chosen_pos when offset <= chosen_pos < tok_end —
            # this handles tokens with leading whitespace (e.g. " correct"
            # starts at chosen_pos-1 but includes the 'c' at chosen_pos).
            offset = 0
            target_idx = None
            for i, tok in enumerate(all_tokens):
                t = tok.token if hasattr(tok, "token") else tok["token"]
                tok_end = offset + len(t)
                if tok_end > chosen_pos and t.strip().lower().startswith(chosen_word[:3]):
                    target_idx = i
                    break
                offset = tok_end

            if target_idx is None:
                if debug:
                    print(f"[TF_PARALLEL] could not align token for '{label}'")
                continue

            target_tok = all_tokens[target_idx]
            top_lps = (
                target_tok.top_logprobs
                if hasattr(target_tok, "top_logprobs")
                else target_tok.get("top_logprobs", None)
                if isinstance(target_tok, dict)
                else None
            )

            lp_correct: Optional[float] = None
            lp_incorrect: Optional[float] = None
            if top_lps:
                for alt in top_lps:
                    a_token = (alt.token if hasattr(alt, "token") else alt["token"]).strip().lower()
                    a_lp = alt.logprob if hasattr(alt, "logprob") else alt["logprob"]
                    # 'correct' and 'incorrect' may be split across tokenisers in
                    # different ways. Match leading prefix and disambiguate so
                    # 'correct' doesn't match 'incorrect'.
                    if a_token == "correct" and lp_correct is None:
                        lp_correct = a_lp
                    elif a_token == "incorrect" and lp_incorrect is None:
                        lp_incorrect = a_lp

            # Compute P(letter is correct) with Z renormalisation. When ONE
            # of the two logprobs is missing from top-K (typical: the model
            # is so confident on the chosen word that the alternative falls
            # outside top-20), inject ``tf_alt_floor`` as a sane assumed
            # logprob for the missing side, then renormalise.
            path = "unknown"
            applied_floor = False
            if lp_correct is None and lp_incorrect is not None:
                lp_correct = float(tf_alt_floor)
                applied_floor = True
            elif lp_incorrect is None and lp_correct is not None:
                lp_incorrect = float(tf_alt_floor)
                applied_floor = True

            if lp_correct is not None and lp_incorrect is not None:
                p_c = math.exp(lp_correct)
                p_i = math.exp(lp_incorrect)
                Z = p_c + p_i
                p_correct = p_c / Z if Z > 0 else 0.5
                path = "Z-renorm+floor" if applied_floor else "Z-renorm"
            else:
                # Both missing — last-ditch use of chosen-token logprob,
                # also smoothed against tf_alt_floor.
                lp = target_tok.logprob if hasattr(target_tok, "logprob") else target_tok["logprob"]
                lp_alt = float(tf_alt_floor)
                if chosen_word == "correct":
                    p_correct = math.exp(lp) / (math.exp(lp) + math.exp(lp_alt))
                else:
                    p_correct = math.exp(lp_alt) / (math.exp(lp) + math.exp(lp_alt))
                path = "chosen-token only+floor"

            # Final clamp into a sane band so cal/dcal can operate without
            # logit-space blow-up.
            lo, hi = tf_clamp
            tf_priors[label] = max(min(p_correct, hi), lo)
            per_letter_diag[label] = {
                "chosen": chosen_word,
                "lp_correct": lp_correct,
                "lp_incorrect": lp_incorrect,
                "path": path,
                "p_correct": tf_priors[label],
            }

        if len(tf_priors) == n:
            # Always print the per-letter logit table + final priors so the
            # user can confirm extraction is correct without re-running with
            # debug=True.
            logger.debug(f"[TF_PARALLEL] === per-letter logits (attempt {attempt}) ===\n"
                f"  {'label':<10}  {'wrote':<10}  {'lp_correct':>11}  "
                f"{'lp_incorrect':>13}  {'path':<22}  {'P(correct)':>11}")
            for lbl in candidate_labels:
                d = per_letter_diag.get(lbl, {})
                lp_c = d.get('lp_correct')
                lp_i = d.get('lp_incorrect')
                lp_c_s = f"{lp_c:>11.4f}" if isinstance(lp_c, (int, float)) else f"{'None':>11}"
                lp_i_s = f"{lp_i:>13.4f}" if isinstance(lp_i, (int, float)) else f"{'None':>13}"
                pc = d.get('p_correct', float('nan'))
                logger.debug(f"  {lbl:<10}  {d.get('chosen', '?'):<10}  {lp_c_s}  {lp_i_s}  "
                    f"{d.get('path', '?'):<22}  {pc:>11.4f}")
            logger.debug(f"[TF_PARALLEL] final tf_priors = {tf_priors}")
            tf_meta = {
                "prompt": user_msg,
                "response": raw,
                "logprobs": _serialize_logprobs(all_tokens),
                "per_letter": per_letter_diag,
                "params": {
                    "tf_alt_floor": float(tf_alt_floor),
                    "tf_clamp": [float(tf_clamp[0]), float(tf_clamp[1])],
                },
            }
            return (tf_priors, tf_meta)
        else:
            # Print failure info unconditionally (not just in debug mode) so
            # the user can see what went wrong without re-running.
            error_msg = (
                f"only parsed {len(tf_priors)}/{n} candidates "
                f"(missing: {unmatched_labels})"
            )
            print(
                f"[TF_PARALLEL] attempt {attempt}/{num_retries} failed: {error_msg}\n"
                f"  Raw model response (truncated to 1000 chars):\n"
                f"  --- 8< ---\n{(raw or '')[:1000]}\n  --- >8 ---",
                file=sys.stderr,
            )

    print(
        f"[TF_PARALLEL] all {num_retries} attempts exhausted. Final error: {error_msg}",
        file=sys.stderr,
    )
    return None

