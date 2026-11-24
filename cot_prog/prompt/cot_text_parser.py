import warnings
import ast
import numpy as np
import re
from itertools import chain
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Iterable, Optional, Set, Union
from openai.types.chat.chat_completion import Choice
from together.types.chat_completions import ChatCompletionChoicesData
from .program_writer import write_program

LLMOutput = Union[Choice, ChatCompletionChoicesData]

def py_string_body(text) -> str:
    """Escape text so it can sit between double quotes in a generated program.

    Model output often quotes things ("The Police"); written out raw, that
    ends the string early and the program no longer parses.
    """
    return str(text).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


@dataclass
class Step:
    """Helper class for storing and represing CoT trace steps. 
    
    Attributes:
        id: The id of the step. 
        raw: The raw text string in that step 
        claim: A cleaned version of the step proposition (taking out
            special CoT keywords). 
        premises: The base steps related to this step.
        answer: Indicates whether the step contains an answer. 
        formula: The particular formula involving the steps involved. 
        
    """
    prefix: int
    raw: str
    claim: str
    premises: List[int]
    answer: Optional[str] = None
    formula: Optional[str] = None
    weight: Optional[float] = 1.0
    start_index: Optional[int] = 0
    end_index: Optional[int] = -1
    confidence: Optional[float] = None
    
    def to_dict(self):
        return asdict(self)

    def string_var(self) -> str:
        assert self.claim or self.answer, "Blank step proposition"
        return f's{self.prefix}: str = "{py_string_body(self.claim if self.claim else self.answer)}"'

    @property 
    def prop_var_name(self):
        return f"P{self.prefix}"

    @property
    def is_answer(self):
        return self.answer.strip()

    def prop_var(self, top: bool = False) -> str: 
        return f"{self.prop_var_name}: bool =~ Flip(s{self.prefix}, {self.weight})"

@dataclass
class CoTOutput:
    """Representation of CoT output

    Attributes
        query: The original query of the chain-of-thought.
        line: The lines in the CoT output.
        raw_cot: The original raw chain of thought string. 

    """
    query: str
    raw_cot: str
    lines: List[Step]
    attempts: Optional[int] = 1
    errors: Optional[List[str]] = None 

    @property
    def answers(self) -> List[int]:
        return [l.prefix for l in self.lines if l.is_answer]

    @property
    def has_answer(self) -> bool:
        """Computes where the cot output contains a parseable answer."""
        return True if self.answers else False

    @property
    def multi_answer(self) -> bool:
        """Determines if the CoT contains multiple answers"""
        return len(self.answers) > 1

    @property
    def check_refs(self) -> bool:
        non_answers = set([l.prefix for l in self.lines if not l.is_answer])
        refs = set(chain(*[l.premises for l in self.lines if l.premises]))
        return non_answers == refs

    @property
    def max_refs(self) -> int:
        return max([len(l.premises) for l in self.lines])
    
    @property
    def has_conditions(self) -> bool:
        """Checks that the CoT has structure with conditionals.

        :rtype: bool
        """
        return True if [l for l in self.lines if l.premises] else False

    def check_program(self, strict: Optional[bool] = True):
        """Check the program to ensure that it has the target structure. Returns 
        back a description of the error (if an error is encountered). 

        The following conditions are checked: 
            -- The CoT has an answer. 
            -- Multi answer CoTs have distinct answers. 
            -- At least one or more step conditions are provided. 
            -- All steps are referenced at least once (strict mode)
            -- A maximum of 4 references are provided in each line (strict mode) 
    
        Arguments: 
            strict: Boolean switch to decide if strict checks should be 
                enforced. 

        """
        answers = self.answers
        max_ref = self.max_refs
        
        if not answers:
            return "CoT does not contain an answer text and line"
        if len(answers) > 1 and len(set(answers)) == 1:
            return "Multiple answers are provided but they are not distinct. Each answer should be distinct."
        if not self.has_conditions:
            return "No conditions or step relations provided via (ref.). Please avoid this."
        if strict and not self.check_refs:
            return "Not all (non-answer) steps referenced at least once in a future step via (ref.). Avoid this mistake."
        if strict and max_ref > 5:
            return f"Too many references per line ({max_ref}) in (ref.), must split argument up."
        
        return None
        

    def add_weights(self, weights: List[float]) -> None:
        """Add weights to the CoT lines

        :param weights:
            The provided weights.
        """
        for idx, weight in enumerate(weights): 
            self.lines[idx].weight = weight

        if len(weights) != len(self.lines):
            warnings.warn(f"Not all lines assigned weights")

    def __getitem__(self,key):
        ## get the particular CoT lines 
        return self.lines[key]
    
    def write_program(self) -> str:
        """Writes the programs in the probabilistic programming language."""
        return write_program(py_string_body(self.query), self.lines)

class BoolParser(ast.NodeTransformer):
    """Parses Boolean expression in CoT and ensure that it's correct"""

    def visit_BoolOp(self, node):
        # Only allow 'and'/'or'
        if not isinstance(node.op, (ast.And, ast.Or)):
            raise ValueError("Only 'and' and 'or' are allowed when referencing past steps")
        return self.generic_visit(node)

    def visit_UnaryOp(self, node):
        # Only allow 'not'
        if not isinstance(node.op, ast.Not):
            raise ValueError("Only 'not' unary operator is allowed in Boolean conditions")
        return self.generic_visit(node)

    def visit_Constant(self, node):
        # Replace integer constants with Name("P{int}")
        if isinstance(node.value, bool):
            # Allow True/False unchanged (optional; remove if undesired)
            return node
        if isinstance(node.value, int):
            return ast.copy_location(
                ast.Name(id=f"P{node.value}", ctx=ast.Load()),
                node,
            )
        raise ValueError(f"Only integer literals (and True/False) allowed; got {node.value!r}")

    def visit_Tuple(self, node):
        # Reject starred items like (1, *x)
        if any(isinstance(e, ast.Starred) for e in node.elts):
            raise ValueError("Starred items are not allowed")

        # First, transform each element
        values = [self.visit(e) for e in node.elts]

        if len(values) == 0:
            raise ValueError("Empty tuples are not allowed")
        if len(values) == 1:
            # A single trailing comma like "3," -> just the item
            return values[0]

        # Replace commas with 'and': (a, b, c) -> (a and b and c)
        new_node = ast.BoolOp(op=ast.And(), values=values)
        return ast.copy_location(new_node, node)

    # For very old Python versions (<3.8) that use ast.Num
    def visit_Num(self, node):  # type: ignore[override]
        return ast.copy_location(
            ast.Name(id=f"P{node.n}", ctx=ast.Load()),
            node,
        )

_REF_FIELD_RE = re.compile(r'\(\s*ref\.?\s*(?P<refs>[^)]*?)\s*\)', re.IGNORECASE)
# A "clearly numeric" ref content is one made up of nothing but digits,
# separators (commas / whitespace), and a small set of connecting words
# the parser already understands or that the prompt template suggests.
# Anything else (proper nouns, free-form citations, years buried in
# titles like "Forbes 2023") is treated as non-numeric and reduced to
# ``(ref. none)`` — safer than letting an out-of-range digit slip
# through and being rejected by ``check_program``.
_NUMERIC_REF_RE = re.compile(
    r'^[\s,]*'
    r'(?:(?:step|steps|and|or|\d+)[\s,]*)+'
    r'$',
    re.IGNORECASE,
)


def sanitize_refs(text: str) -> str:
    """Normalize the contents of every ``(ref. ...)`` annotation so the
    downstream Boolean / integer parsing can't be derailed by free-form text.

    The CoT parser expects ``(ref. none)`` or ``(ref. 1, 2, ...)``. Smaller
    models — Qwen-2.5-7B at non-zero temperature is the case in point —
    sometimes misread "ref." as a citation field and emit things like
    ``(ref. historical records)`` or ``(ref. Forbes 2023)``. The current
    parser then crashes inside ``parse_boolean`` with "Invalid Boolean
    syntax".

    This helper rewrites each ``(ref. ...)`` match as follows:

      * ``(ref. none)`` (case-insensitive) → left unchanged.
      * Content that's "clearly numeric" — only digits, separators, and a
        small set of connecting words (``step``, ``steps``, ``and``,
        ``or``) — has its integer tokens extracted into a clean
        comma-separated form
        (so ``(ref. step 1 and 2)`` → ``(ref. 1, 2)``).
      * Everything else (free-form citations, proper nouns, mixed
        text+digits like ``Forbes 2023``) → replaced with ``(ref. none)``
        — i.e. treated as a leaf step with no upstream premises. Safer
        than letting a buried digit (``2023``) slip through as a bogus
        step reference.

    The number, position, and line structure of the input text are
    preserved aside from per-match parenthesised replacements, so
    downstream token / step alignment in ``find_logps`` continues to work.

    This function is currently invoked only by Qwen-targeted paths in
    ``multi_answer_cot.py``; it is safe to apply unconditionally, but is
    gated by callers to keep behaviour on other models bit-exact.
    """
    def _repl(m: 're.Match[str]') -> str:
        content = m.group("refs")
        if content is None:
            return "(ref. none)"
        if content.strip().lower() == "none":
            return m.group(0)
        if _NUMERIC_REF_RE.fullmatch(content):
            nums = re.findall(r'\d+', content)
            if nums:
                return f"(ref. {', '.join(nums)})"
        return "(ref. none)"
    return _REF_FIELD_RE.sub(_repl, text)


def parse_boolean(expr):
    """Parse the Boolean expression using ast

    Arguments:
        The raw Boolean statement provided in the raw CoT trace.
    Raises:
        ValueError when expr is not parseable by Python parser.

    """
    if not expr.strip():
        return ""
    expr = expr.strip()
    if ',' == expr[-1]:
        expr = expr[:-1]
    expr = expr.replace(', and',' and ')
    expr = re.sub(r'\,\s+', ' and ', expr)
    try: 
        tree = ast.parse(expr)
    except:
        raise ValueError(f"Invalid Boolean syntax used, not parseable: {expr}")
    tree = BoolParser().visit(tree)
    return ast.unparse(tree)

class CoTContreteSyntaxParser:
    """Parser for the concrete CoT text syntax."""

    HEADER_RE = re.compile(r'^\s*(\d+)(\s*[\.\)\-:]\s*)')
    REF_RE = re.compile(r'\(\s*ref\.\s*(?P<refs>[^)]*?)\s*\)', re.IGNORECASE)
    FINAL_RE = re.compile(
        r'\(\s*final answer:\s*(?P<answer>[^)]*?)\s*\)',
        re.IGNORECASE
    )
    CONF_BLOCK_RE = re.compile(
        r'\(\s*confidence:\s*(?P<confidence>[^)]*?)\s*\)',
        re.IGNORECASE
    )
    ANY_META_RE = re.compile(
        r'\(\s*(?:'
        r'ref\.?(?:\s+(?:and|or))?\s+[^)]*'
        r'|final answer:\s*[^)]*'
        r'|confidence:\s*[^)]*'
        r')\s*\)',
        re.IGNORECASE
    )
    BASED_ON_RE = re.compile(
        r'(?is)(?:'
        r'\bBased\s+(?:on|upon)\b\s*(?:step[s]?\s*)?'  # Based on/upon [step[s]]?
        r'|\bUsing\s+step\s*'                          # Using step
        r'|\bAssuming\b\s*(?:step[s]?\s*)?'            # Assuming [step[s]]?
        r'|\bGiven\b\s*(?:step[s]?\s*)?'               # Given [step[s]]?
        r')'
        r'((?:\s*(?:\d+|and|or|not|\(|\)|,|/)\s*)+)'   # capture the references expression
    )
    
    def parse(
            self, text: str, query: Optional[str] = "", relax: Optional[bool] = False
    ) -> CoTOutput:
        """Main method for parsing the raw string outputs of prompt. 

        Arguments: 
            query: The (optional) CoT query 
            text: The raw text received by the language model. 
            relax: Switch to allow for relaxing some of the checks 
                in the CoT. 
        Raises: 
            ValueError when an answer step is not included. 

        """
        steps: List[Step] = []
        has_answer = False
        
        for raw in text.splitlines():
            refs = []
            stop_index = -1
            start_index = 0
            meta_spans = []
            final_answer = ""
            claim = ""
            formula = ""

            m = self.HEADER_RE.match(raw)
            if not m or not m.group(1).isalnum():
                raise ValueError(f"Line didn't start with a valid number: {raw}")
            step_id = int(m.groups()[0])
            start_index = m.end()

            ## reference to past steps 
            ref_groups = self.REF_RE.search(raw)
            if ref_groups:
                meta_spans.append(ref_groups.start())
                raw_refs = "" if ref_groups.groups()[0] == "none" else ref_groups.groups()[0]
                formula = parse_boolean(raw_refs)
                refs = [int(n) for n in re.findall(r'\d+', ref_groups.groups()[0])]
                
            ## answer text 
            final_match = self.FINAL_RE.search(raw)
            if final_match:
                meta_spans.append(final_match.start())
                final_answer = final_match.group('answer').strip()

            ## confidence annotation
            confidence_val = None
            cm = self.CONF_BLOCK_RE.search(raw)
            if cm:
                confidence_text = cm.group('confidence').strip()
                meta_spans.append(cm.start())
                try:
                    confidence_val = float(confidence_text.replace('%', '').strip()) / 100.0
                except (ValueError, TypeError):
                    pass

            ## Based on conditions
            based = self.BASED_ON_RE.search(raw)
            if based:
                start_index = based.end()
                
            if meta_spans:
                k = min(meta_spans)
                stop_index = k

            if not final_answer:
                # Bugfix: was `raw[raw_index:]` (undefined NameError) when a
                # line had no `(ref. ...)`, no `(confidence: ...)`, no
                # `(final answer: ...)`, and no `Based on …` prefix — i.e.
                # gpt-4o sometimes emits a numbered line without any of the
                # required meta-annotations. Use start_index (the position
                # right after the "1." header) instead.
                claim = raw[start_index:stop_index] if stop_index != -1 else raw[start_index:]
                
            ### add step to list
            steps.append(
                Step(
                    prefix=step_id,
                    raw=raw,
                    claim=self._normalize_whitespace(claim),
                    answer=self._normalize_whitespace(final_answer),
                    premises=refs,
                    formula=formula,
                    start_index=start_index,
                    end_index=stop_index,
                    confidence=confidence_val,
                )
            )
            if final_answer:
                has_answer = True
                
        if relax is False and has_answer is False:
            raise ValueError(f"Output does not contain answer output: {text}")
                
        return CoTOutput(query=query, lines=steps, raw_cot=text)

    @staticmethod
    def _normalize_whitespace(s: str) -> str:
        """Helper method to remove extra white space"""
        return re.sub(r'\s+', ' ', s).strip()
