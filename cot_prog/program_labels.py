"""Readable names for the parts of a generated Copper program.

Used by the console and the HTML reports so both describe a parameter the
same way, e.g. ``P5__2`` -> "Step 1 for C: stated confidence".
"""
import re

_PROG_STR_RE = re.compile(r'^(s\d+)\s*:\s*str\s*=\s*"(.*)"\s*$')


def answer_names(program):
    """Map each string variable of a program (s1, s2, ...) to its text."""
    names = {}
    for line in (program or "").splitlines():
        m = _PROG_STR_RE.match(line.strip())
        if m:
            names[m.group(1)] = m.group(2).replace('\\"', '"')
    return names


def answer_label(names, var):
    """'s5' -> 'yes (s5)' when the program says s5 = "yes"."""
    text = names.get(var)
    if not text:
        return var
    if len(text) > 70:
        text = text[:67] + "..."
    return f"{text} ({var})"


_PROG_ASSIGN_RE = re.compile(
    r'^(?P<name>\w+)\s*(?::\s*bool)?\s*=~\s*(?P<kind>Flip|Bernoulli|Decision|Factor)\(\s*(?P<arg>[^,)]+)')
_PROG_SECTION_RE = re.compile(r'^#+\s*evidence for\s+(?P<label>.+?)\s*$', re.I)


def param_labels(program):
    """Describe the parameters of a program.

    Returns {name in program: (kind, answer, step number)}, e.g.
        "P5"   -> ("step", "C", 1)
        "C_pr" -> ("prior", "C", None)
        "C_E"  -> ("evidence", "C", None)
    A step is numbered within its own answer section, to match the
    chain-of-thought tables. ``answer`` is None for a single-chain program.
    """
    labels, section, step_in_section, steps_outside = {}, None, 0, 0
    for line in (program or "").splitlines():
        line = line.strip()
        if line.startswith("#"):
            sec = _PROG_SECTION_RE.match(line)
            if sec:
                section, step_in_section = sec.group("label"), 0
            elif line.startswith("##"):
                section = None      # any other heading ends an answer section
            continue
        m = _PROG_ASSIGN_RE.match(line)
        if not m:
            continue
        name, kind, arg = m.group("name"), m.group("kind"), m.group("arg").strip()
        if kind == "Factor":
            labels[name] = ("evidence", section or re.sub(r'_E$', '', name), None)
        elif arg[:1] in ("'", '"'):
            labels[name] = ("prior", arg.strip(chr(39) + chr(34)), None)
        elif section:
            # Inside "## evidence for X": number within that answer's argument.
            step_in_section += 1
            labels[name] = ("step", section, step_in_section)
        else:
            # Single-chain program: use the step's own number (P4 -> Step 4).
            steps_outside += 1
            own = re.search(r'(\d+)$', name)
            labels[name] = ("step", None, int(own.group(1)) if own else steps_outside)
    # Two parameters must never share a description; if they would, keep the
    # program's own name for the later one.
    seen = set()
    for name, desc in list(labels.items()):
        if desc in seen:
            labels.pop(name)
        seen.add(desc)
    return labels


def param_label(labels, param_name, short=False):
    """Readable name for one parameter.

    Full:  'P5__2' -> 'Step 1 for C: stated confidence'
    Short: 'P5__2' -> 'C 1: stated'   (fits on a chart axis)
    """
    base, second = (param_name[:-3], True) if param_name.endswith("__2") else (param_name, False)
    if base not in labels:
        return param_name
    kind, answer, num = labels[base]
    if short and answer and len(answer) > 14:
        answer = answer[:13] + "…"
    if kind == "step":
        if short:
            who = f"{answer} {num}" if answer else f"Step {num}"
            return f"{who}: {'stated' if second else 'token'}"
        where = f" for {answer}" if answer else ""
        return f"Step {num}{where}: {'stated confidence' if second else 'token probability'}"
    if kind == "prior":
        if short:
            return f"{answer} prior: {'stated' if second else 'logit'}"
        return f"Prior for {answer}: {'stated confidence' if second else 'logit prior'}"
    return f"{answer} evidence" if short else f"Evidence for {answer}"
