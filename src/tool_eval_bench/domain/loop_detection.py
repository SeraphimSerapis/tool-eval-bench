"""Detect degenerate repetition loops in model output and reasoning.

When a model gets trapped in a repetition cycle (repeating a sentence or
emitting a long run of identical characters/whitespace until ``max_tokens``),
reporting ``reasoning_truncated`` misleads operators into raising token limits.
This stdlib-only module identifies those loops deterministically while avoiding
false positives on structured JSON, Markdown tables, code blocks, and long
valid reasoning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

NGRAM_SIZE = 20
NGRAM_REPEAT_THRESHOLD = 3
# Multi-word phrases with at least this many unique tokens cannot match simple
# numeric matrices or repeated table cells, so overlapping starts count.
_DIVERSE_NGRAM_UNIQUE_TOKENS = 3

CHAR_RUN_THRESHOLD = 120
RULE_CHAR_RUN_THRESHOLD = 200
SHORT_PATTERN_REPEAT_THRESHOLD = 40

# Characters commonly used for Markdown horizontal rules, table dividers, and
# ASCII section banners in code blocks.
_RULE_CHARS: frozenset[str] = frozenset("-=_~*")
_TABLE_RULE_CHARS: frozenset[str] = frozenset("-=_~*|: ")

_SINGLE_CHAR_RUN_RE = re.compile(rf"(.)\1{{{CHAR_RUN_THRESHOLD - 1},}}", re.DOTALL)
_SHORT_PATTERN_RE = re.compile(rf"(.{{2,4}})\1{{{SHORT_PATTERN_REPEAT_THRESHOLD - 1},}}", re.DOTALL)


@dataclass(frozen=True)
class LoopDetectionResult:
    """Outcome of checking text for a degenerate repetition loop."""

    detected: bool
    kind: str | None = None
    detail: str = ""


def _detect_ngram_repetition(text: str) -> LoopDetectionResult | None:
    """Detect any 20-word whitespace-tokenized n-gram repeated >= 3 times."""
    words = text.split()
    if len(words) < NGRAM_SIZE + NGRAM_REPEAT_THRESHOLD - 1:
        return None

    has_alnum = [any(ch.isalnum() for ch in word) for word in words]
    window_alnum = sum(has_alnum[:NGRAM_SIZE])

    # Maps n-gram tuple -> (occurrence_count, end_index_of_last_counted_occurrence).
    seen: dict[tuple[str, ...], tuple[int, int]] = {}
    limit = len(words) - NGRAM_SIZE + 1

    for idx in range(limit):
        if idx > 0:
            if has_alnum[idx - 1]:
                window_alnum -= 1
            if has_alnum[idx + NGRAM_SIZE - 1]:
                window_alnum += 1
        if window_alnum == 0:
            # Skip pure-punctuation spans such as wide Markdown table separators.
            continue

        gram = tuple(words[idx : idx + NGRAM_SIZE])
        entry = seen.get(gram)
        if entry is None:
            seen[gram] = (1, idx + NGRAM_SIZE)
            continue

        count, last_end = entry
        # Low-diversity spans (e.g. "0 0 0..." or "| N/A | N/A |...") require
        # non-overlapping occurrences so matrices and tables do not false-positive.
        if len(set(gram)) < _DIVERSE_NGRAM_UNIQUE_TOKENS and idx < last_end:
            continue

        count += 1
        if count >= NGRAM_REPEAT_THRESHOLD:
            excerpt = " ".join(gram[:6])
            return LoopDetectionResult(
                detected=True,
                kind="ngram_repetition",
                detail=(
                    f"{NGRAM_SIZE}-word sequence repeated {count}+ times "
                    f'(starting with "{excerpt}...")'
                ),
            )
        seen[gram] = (count, idx + NGRAM_SIZE)

    return None


def _detect_char_run(text: str) -> LoopDetectionResult | None:
    """Detect degenerate identical-character runs or short repeating char patterns."""
    if len(text) < SHORT_PATTERN_REPEAT_THRESHOLD * 2:
        return None

    for match in _SINGLE_CHAR_RUN_RE.finditer(text):
        ch = match.group(1)
        run_len = match.end() - match.start()
        threshold = RULE_CHAR_RUN_THRESHOLD if ch in _RULE_CHARS else CHAR_RUN_THRESHOLD
        if run_len >= threshold:
            return LoopDetectionResult(
                detected=True,
                kind="char_run",
                detail=f"Character {ch!r} repeated {run_len} times consecutively",
            )

    for match in _SHORT_PATTERN_RE.finditer(text):
        unit = match.group(1)
        if len(set(unit)) == 1:
            # Single-character runs are governed by the 120/200 thresholds above.
            continue
        matched_len = match.end() - match.start()
        if all(ch in _TABLE_RULE_CHARS for ch in unit) and matched_len < RULE_CHAR_RUN_THRESHOLD:
            continue
        repeats = matched_len // len(unit)
        return LoopDetectionResult(
            detected=True,
            kind="char_run",
            detail=f"Pattern {unit!r} repeated {repeats} times consecutively",
        )

    return None


def detect_degenerate_loop(text: str | None) -> LoopDetectionResult:
    """Return whether ``text`` contains a degenerate repetition loop.

    Checks for:
      1. Any 20-word n-gram repeated at least 3 times (``kind="ngram_repetition"``).
      2. A degenerate run of identical characters (>= 120 for ordinary/whitespace
         characters, >= 200 for rule/table border characters) or a 2-4 character
         pattern repeated >= 40 times consecutively (``kind="char_run"``).
    """
    if not text:
        return LoopDetectionResult(detected=False)

    ngram_hit = _detect_ngram_repetition(text)
    if ngram_hit is not None:
        return ngram_hit

    char_hit = _detect_char_run(text)
    if char_hit is not None:
        return char_hit

    return LoopDetectionResult(detected=False)
