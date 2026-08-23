"""MMLU answer evaluator — extract and compare multiple-choice answers."""

from __future__ import annotations

import re
from dataclasses import dataclass

_VALID_LETTERS = {"A", "B", "C", "D"}

# Patterns ordered by priority
_ANSWER_IS_RE = re.compile(
    r"(?:the\s+)?answer\s+is\s*:?\s*\(?([A-D])\)?",
    re.IGNORECASE,
)
_ANSWER_COLON_RE = re.compile(r"answer\s*:\s*\(?([A-D])\)?", re.IGNORECASE)
_STANDALONE_LETTER_RE = re.compile(r"\b([A-D])\b")
_FINAL_ANSWER_RE = re.compile(
    r"(?:final\s+answer|i\s+(?:choose|select|pick))"
    r"\s*(?:is|:)?\s*\(?([A-D])\)?",
    re.IGNORECASE,
)
_SELECTION_RE = re.compile(
    r"(?:(?:choose|select|pick)\s+|(?:option|choice|letter)\s*(?:is|:)\s*)"
    r"\(?([A-D])\)?",
    re.IGNORECASE,
)


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
    text = response.strip()
    if not text:
        return None, "none"

    # 1. Exact single letter (possibly with period/parenthesis)
    cleaned = text.strip(".()")
    if cleaned.upper() in _VALID_LETTERS and len(cleaned) == 1:
        return cleaned.upper(), "exact"

    # 2. Prefer an explicit final-answer/selection cue.  Use the last match
    # because a response may discuss an earlier candidate before committing
    # to its final choice.
    matches = list(_FINAL_ANSWER_RE.finditer(text))
    if matches:
        return matches[-1].group(1).upper(), "answer_pattern"

    # 3. "The answer is B" / "Answer: C".  Again, the final explicit answer
    # wins when the response contains a quoted earlier answer.
    matches = list(_ANSWER_IS_RE.finditer(text))
    if matches:
        return matches[-1].group(1).upper(), "answer_pattern"

    matches = list(_ANSWER_COLON_RE.finditer(text))
    if matches:
        return matches[-1].group(1).upper(), "answer_pattern"

    matches = list(_SELECTION_RE.finditer(text))
    if matches:
        return matches[-1].group(1).upper(), "answer_pattern"

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
    gt_letter = "ABCD"[ground_truth]
    extracted, method = extract_answer(response)

    return MMLUEvalResult(
        correct=extracted == gt_letter,
        extracted_answer=extracted,
        ground_truth_letter=gt_letter,
        ground_truth_index=ground_truth,
        extraction_method=method,
    )
