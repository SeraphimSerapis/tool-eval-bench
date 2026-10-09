"""GSM8K answer evaluator — extract and compare numeric answers.

Handles multiple answer-extraction strategies:
1. Standard ``#### N`` marker (used by models that follow GSM8K format)
2. "The answer is N" phrase, then the last standalone number (fallbacks)
3. Normalises commas, dollar signs, percent signs, and whitespace
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

# One whole number.  The trailing lookahead stops a backtrack from shortening
# "25" to "2" when a later rule rejects what follows, and the minus sign must
# touch the digits so a markdown bullet ("- 18") is not read as a sign.  A
# trailing "%" or unit is left in place and simply not part of the number.
_NUMBER = r"-?\$?\d[\d,]*(?:\.\d+)?(?![\d.,]*\d)"

_MARKER_RE = re.compile(r"####\s*([^\n]+)")
_ANSWER_RE = re.compile(rf"(?:the\s+answer\s+is|answer\s*[:=])\s*({_NUMBER})", re.IGNORECASE)
# Standalone numbers: not glued to a word, a decimal point, or a fraction slash.
_STANDALONE_RE = re.compile(rf"(?<![\w./])({_NUMBER})(?!/)")


@dataclass(slots=True)
class EvalResult:
    """Result of evaluating a single GSM8K response."""

    correct: bool
    extracted_answer: float | None
    ground_truth: float
    extraction_method: str  # "marker", "answer_pattern", "last_number", "none"


def extract_answer(text: str) -> tuple[float | None, str]:
    """Extract a numeric answer from the model's response.

    Returns ``(number, method)`` where *method* indicates how the
    answer was found.  Returns ``(None, "none")`` if no number could
    be extracted.

    The ``####`` marker takes its first parseable match, as lm-eval's
    strict-match does: a model that keeps generating after its answer often
    invents the next few-shot exemplar, complete with its own ``####`` line.
    The fallbacks take their last match, because a model that corrects itself
    in prose ends with the answer it settled on.
    """
    # Strategy 1: the first ``#### N`` marker that parses.  A markdown heading
    # such as "#### Step 1" is also a marker, but not a number, so it is skipped.
    for match in _MARKER_RE.finditer(text):
        num = _parse_number(match.group(1).strip())
        if num is not None:
            return num, "marker"

    # Strategy 2: the last "the answer is N" phrase
    for match in reversed(list(_ANSWER_RE.finditer(text))):
        num = _parse_number(match.group(1))
        if num is not None:
            return num, "answer_pattern"

    # Strategy 3: the last standalone number in the text
    for match in reversed(list(_STANDALONE_RE.finditer(text))):
        num = _parse_number(match.group(1))
        if num is not None:
            return num, "last_number"

    return None, "none"


def _parse_number(s: str) -> float | None:
    """Parse a numeric string, stripping common formatting."""
    s = s.strip()
    # Remove currency symbols, commas, spaces
    s = re.sub(r"[$€£¥,\s]", "", s)
    # Remove trailing period that isn't decimal (e.g. "42.")
    if s.endswith("."):
        s = s[:-1]
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def evaluate_answer(
    model_response: str,
    ground_truth: float,
    *,
    tolerance: float = 1e-5,
) -> EvalResult:
    """Compare the model's extracted answer against the ground truth.

    Uses absolute tolerance for integers and relative tolerance for
    floating-point answers.
    """
    extracted, method = extract_answer(model_response)

    if extracted is None:
        return EvalResult(
            correct=False,
            extracted_answer=None,
            ground_truth=ground_truth,
            extraction_method=method,
        )

    # For integer ground truths, require exact match (after rounding)
    if ground_truth == int(ground_truth):
        correct = math.isclose(extracted, ground_truth, abs_tol=tolerance)
    else:
        # For floats, use relative tolerance
        correct = math.isclose(extracted, ground_truth, rel_tol=1e-3, abs_tol=tolerance)

    return EvalResult(
        correct=correct,
        extracted_answer=extracted,
        ground_truth=ground_truth,
        extraction_method=method,
    )
