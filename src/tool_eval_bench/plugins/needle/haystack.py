"""Haystack construction, needle placement, and grading.

A needle-in-a-haystack case buries one synthetic fact at a known depth inside a
block of unrelated prose, then asks the model to retrieve it.  Accuracy across a
grid of (context length x depth) cells shows where a model's effective context
stops matching its advertised context.

The prose comes from the shared filler corpus in ``domain.filler``, so a haystack is
topically varied and carries per-run noise rather than one repeated paragraph.
A model cannot answer from the surrounding text: every needle states a fact that
exists only in that one sentence.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from tool_eval_bench.domain.filler import build_haystack_text
from tool_eval_bench.domain.models import ChatMessage

# ---------------------------------------------------------------------------
# Needle templates
# ---------------------------------------------------------------------------

# Each template owns its answer generator so the fact is unguessable from the
# surrounding prose and unlikely to collide with it.  Several templates exist
# because a single phrasing measures one retrieval pattern rather than the
# model's retrieval ability; the grid rotates through them.
_NEEDLE_TEMPLATES: tuple[tuple[str, str, str], ...] = (
    (
        "The maintenance passphrase for the Helios relay is {answer}.",
        "What is the maintenance passphrase for the Helios relay?",
        "passphrase",
    ),
    (
        "Warehouse 7 recorded exactly {answer} damaged pallets during the winter audit.",
        "Exactly how many damaged pallets did Warehouse 7 record during the winter audit?",
        "count",
    ),
    (
        "The emergency contact code for the Brightwater substation is {answer}.",
        "What is the emergency contact code for the Brightwater substation?",
        "code",
    ),
    (
        "Sensor array Kestrel-9 was recalibrated on the {answer}th day of the field season.",
        "On which day of the field season was sensor array Kestrel-9 recalibrated?",
        "day",
    ),
)

_SYSTEM_PROMPT = (
    "You are given a long document. Answer the question using only information "
    "stated in the document. Reply with the answer itself and nothing else."
)

_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


def _make_answer(kind: str, rng: random.Random) -> str:
    """Generate the fact a needle asserts, in the shape its template expects."""
    if kind == "passphrase":
        return "-".join("".join(rng.choice(_ALPHABET) for _ in range(4)) for _ in range(3))
    if kind == "count":
        return str(rng.randint(1000, 9999))
    if kind == "code":
        return f"{rng.choice(_ALPHABET)}{rng.choice(_ALPHABET)}-{rng.randint(100000, 999999)}"
    if kind == "day":
        return str(rng.randint(101, 364))
    raise ValueError(f"Unknown needle kind: {kind!r}")


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NeedleCase:
    """One cell of the retrieval grid."""

    context_tokens: int
    """Approximate size of the haystack this needle is buried in."""
    depth_percent: float
    """Where the needle sits, 0.0 = very start, 1.0 = very end."""
    needle: str
    question: str
    answer: str
    """The exact string the response must contain to count as a retrieval."""

    @property
    def cell_id(self) -> str:
        """Stable identifier used in progress output and reports."""
        return f"{self.context_tokens // 1024}K@{self.depth_percent:.0%}"


def build_cases(
    context_lengths: list[int],
    depths: list[float],
    *,
    seed: int | None = None,
) -> list[NeedleCase]:
    """Build the full (length x depth) grid of retrieval cases.

    Cases are ordered length-major so a partially completed run still covers
    every depth at the lengths it reached.
    """
    rng = random.Random(seed)
    cases: list[NeedleCase] = []
    for length in context_lengths:
        for depth in depths:
            template, question, kind = _NEEDLE_TEMPLATES[len(cases) % len(_NEEDLE_TEMPLATES)]
            answer = _make_answer(kind, rng)
            cases.append(
                NeedleCase(
                    context_tokens=length,
                    depth_percent=depth,
                    needle=template.format(answer=answer),
                    question=question,
                    answer=answer,
                )
            )
    return cases


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------

# Split on sentence-ending punctuation followed by whitespace.  Bounded
# quantifiers only: the filler carries injected noise like "[id:3f2a]" and
# "(ref #10293)", and a backtracking pattern over a 100K-token document is a
# denial of service against our own benchmark.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _insert_at_depth(haystack: str, needle: str, depth_percent: float) -> str:
    """Insert *needle* at the sentence boundary nearest *depth_percent*.

    Splitting on sentence boundaries keeps the needle from landing mid-word,
    which would make a failure a tokenization artifact rather than a retrieval
    result.
    """
    sentences = _SENTENCE_END.split(haystack)
    if len(sentences) <= 1:
        return f"{needle} {haystack}" if depth_percent < 0.5 else f"{haystack} {needle}"
    index = round(depth_percent * len(sentences))
    index = max(0, min(len(sentences), index))
    placed = [*sentences[:index], needle, *sentences[index:]]
    return " ".join(placed)


def build_needle_messages(case: NeedleCase, *, seed: int | None = None) -> list[ChatMessage]:
    """Build the chat messages that present one case to the model."""
    haystack = build_haystack_text(case.context_tokens, seed=seed)
    document = _insert_at_depth(haystack, case.needle, case.depth_percent)
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (f"<document>\n{document}\n</document>\n\n{case.question}"),
        },
    ]


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

# A thousands separator inside a number, so "4,821 pallets" grades like "4821".
_THOUSANDS_COMMA = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_SEPARATORS = re.compile(r"[^a-z0-9]+")


def grade_response(case: NeedleCase, response: str) -> bool:
    """Return whether *response* retrieved the needle.

    The answer must appear as a whole value, case-insensitively, anywhere in
    the response: a model that answers "The passphrase is K7QM-2XPD-9WLR."
    retrieved the fact, and penalising the sentence around it would measure
    instruction following instead of retrieval.  Separators inside the answer
    are optional ("K7QM2XPD9WLR" and "k7qm 2xpd 9wlr" pass), and a thousands
    comma is ignored.  The match is bounded, so the answer cannot be found
    inside a longer number or code ("#48210" does not contain 4821) or
    assembled from separate numbers ("48 and 21").  Numeric answers need only
    digit boundaries, so "245th day" still passes.
    """
    if not response:
        return False
    text = _THOUSANDS_COMMA.sub("", response.lower())
    parts = [re.escape(part) for part in _SEPARATORS.split(case.answer.lower()) if part]
    if not parts:
        return False
    body = r"[^a-z0-9]*".join(parts)
    if case.answer.isdigit():
        pattern = rf"(?<!\d){body}(?!\d)"
    else:
        pattern = rf"(?<![a-z0-9]){body}(?![a-z0-9])"
    return re.search(pattern, text) is not None
