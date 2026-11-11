"""MMLU answer evaluator — extract and compare multiple-choice answers."""

from __future__ import annotations

import re
from dataclasses import dataclass

from tool_eval_bench.domain.plugin import strip_thinking

_VALID_LETTERS = {"A", "B", "C", "D"}

# The cue words are matched case-insensitively, but the letter is not: a bare
# letter must be an upper-case A-D standing alone.  A lower-case letter counts
# only in parentheses or when it ends the response ("the answer is b.").
# Otherwise "the answer is clearly B" reads the C of "clearly", and "the answer
# is a prime" reads the article as A.
_EMPH = r"[*_`]*"
_LETTER = (
    rf"{_EMPH}(?:"
    rf"\({_EMPH}([A-Da-d]){_EMPH}\)"
    rf"|([A-D])\b"
    rf"|([a-d])(?={_EMPH}\s*[.)]?\s*$)"
    rf")"
)

# Patterns ordered by priority
_BOXED_RE = re.compile(r"\\boxed\{\s*(?:\\(?:text|mathrm|mathbf)\{\s*)?\(?([A-Da-d])\)?\s*\}?\s*\}")
_ANSWER_IS_RE = re.compile(rf"(?i:(?:the\s+)?answer\s+is){_EMPH}\s*:?{_EMPH}\s*{_LETTER}")
_ANSWER_COLON_RE = re.compile(rf"(?i:answer){_EMPH}\s*:{_EMPH}\s*{_LETTER}")
_STANDALONE_LETTER_RE = re.compile(r"\b([A-D])\b")
_FINAL_ANSWER_RE = re.compile(
    rf"(?i:final\s+answer|i\s+(?:choose|select|pick)){_EMPH}\s*(?i:is|:)?{_EMPH}\s*{_LETTER}"
)
_SELECTION_RE = re.compile(
    rf"(?i:(?:choose|select|pick)\s+|(?:option|choice|letter){_EMPH}\s*(?:is|:){_EMPH}\s*){_LETTER}"
)


def _cue_letter(match: re.Match[str]) -> str:
    return (match.group(1) or match.group(2) or match.group(3)).upper()


@dataclass(slots=True)
class MMLUEvalResult:
    """Result of evaluating a single MMLU question."""

    correct: bool
    extracted_answer: str | None  # "A", "B", "C", or "D"
    ground_truth_letter: str  # "A", "B", "C", or "D"
    ground_truth_index: int  # 0-3
    extraction_method: str  # "exact", "answer_pattern", "first_letter", "none"


def extract_answer(response: str) -> tuple[str | None, str]:
    """Extract a multiple-choice letter from a model response.

    Returns ``(letter, method)`` where *letter* is A/B/C/D or ``None``,
    and *method* describes how it was found.
    """
    text = (strip_thinking(response, fallback_to_thought=True) or "").strip()
    if not text:
        return None, "none"

    # 1. Exact single letter (possibly with period/parenthesis/markdown emphasis)
    cleaned = text.strip(".*_`()")
    if cleaned.upper() in _VALID_LETTERS and len(cleaned) == 1:
        return cleaned.upper(), "exact"

    # 2. Prefer LaTeX \boxed{A} or explicit final-answer/selection cues.  Use
    # the last match because a response may discuss an earlier candidate before
    # committing to its final choice.
    boxed = list(_BOXED_RE.finditer(text))
    if boxed:
        return boxed[-1].group(1).upper(), "answer_pattern"

    matches = list(_FINAL_ANSWER_RE.finditer(text))
    if matches:
        return _cue_letter(matches[-1]), "answer_pattern"

    # 3. "The answer is B" / "Answer: C".  Again, the final explicit answer
    # wins when the response contains a quoted earlier answer.
    matches = list(_ANSWER_IS_RE.finditer(text))
    if matches:
        return _cue_letter(matches[-1]), "answer_pattern"

    matches = list(_ANSWER_COLON_RE.finditer(text))
    if matches:
        return _cue_letter(matches[-1]), "answer_pattern"

    matches = list(_SELECTION_RE.finditer(text))
    if matches:
        return _cue_letter(matches[-1]), "answer_pattern"

    # 4. Without an explicit cue, use the last standalone answer letter.  A
    # first-letter fallback incorrectly returns A for option lists such as
    # "A, B, C, D ... final choice D".  Preserve the historical method name
    # for the common single-candidate case.
    candidates = list(_STANDALONE_LETTER_RE.finditer(text))
    if candidates:
        method = "first_letter" if len(candidates) == 1 else "last_letter"
        return candidates[-1].group(1).upper(), method

    return None, "none"


def evaluate_answer(response: str, ground_truth: int) -> MMLUEvalResult:
    """Evaluate a model response against the ground truth.

    Parameters
    ----------
    response
        The model's raw text response.
    ground_truth
        The correct answer index (0=A, 1=B, 2=C, 3=D).
    """
    if isinstance(ground_truth, bool) or ground_truth not in (0, 1, 2, 3):
        raise ValueError(f"ground_truth must be 0..3, got {ground_truth!r}")
    gt_letter = "ABCD"[ground_truth]
    extracted, method = extract_answer(response)

    return MMLUEvalResult(
        correct=extracted == gt_letter,
        extracted_answer=extracted,
        ground_truth_letter=gt_letter,
        ground_truth_index=ground_truth,
        extraction_method=method,
    )
