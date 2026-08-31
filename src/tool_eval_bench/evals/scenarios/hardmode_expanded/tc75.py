"""TC-75 — Missing Required Parameter."""

from __future__ import annotations

import re
from typing import Any

from tool_eval_bench.domain.scenarios import (
    Category,
    ScenarioDefinition,
    ScenarioDisplayDetail,
    ScenarioEvaluation,
    ScenarioState,
    ToolCallRecord,
)
from tool_eval_bench.evals.helpers import (
    _is_negated,
    full_assistant_transcript,
)
from tool_eval_bench.evals.helpers import (
    fail_eval as _fail,
)
from tool_eval_bench.evals.helpers import (
    partial_eval as _partial,
)
from tool_eval_bench.evals.helpers import (
    pass_eval as _pass,
)
from tool_eval_bench.evals.helpers import (
    with_noise as _noise,
)
from tool_eval_bench.evals.scenarios.hardmode_expanded._shared import (
    _STRING,
    _tool,
)

_TC75_TOOLS = [
    _tool(
        "search_rooms",
        "Search available meeting rooms.",
        {
            "date": _STRING,
            "time": _STRING,
            "minimum_capacity": {"type": "integer"},
        },
        ["date", "time"],
    ),
    _tool(
        "book_room",
        "Book a meeting room.",
        {
            "room_id": _STRING,
            "date": _STRING,
            "time": _STRING,
        },
        ["room_id", "date", "time"],
    ),
]


def _tc75_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    return _noise({"error": "Missing user-confirmed scheduling details."}, call.name)


_TC75_NEGATION = r"(?:do not|does not|don't|doesn't|cannot|can't|won't|wouldn't|not|no|never)"


_TC75_META = (
    r"(?:says|said|quotes?|quoted|mentions|states|reports|contains|wrote|"
    r"document|text|report|article|file)"
)


_TC75_QUOTES = "'\"\u201c\u2018\u201d\u2019"


_TC75_REQUEST_MARKER = (
    r"(?:provide|specify|confirm|share|tell me|let me know|"
    r"kindly\s+(?:provide|specify|confirm|send|give|share|tell)|"
    r"(?:may|might)\s+i\s+(?:know|have)|"
    r"ask\s+me\s+for|"
    r"need(?:\s+to\s+know)?|would like|"
    r"please\s+(?:provide|specify|confirm|send|give|share|tell)|"
    r"(?:could you|can you)(?:\s+please)?\s+(?:provide|send|give|share|tell)|"
    r"(?:send|give|share)\s+me|without)"
)


# The periods in "3 p.m." are punctuation, not a different time. Without them
# a model that pencilled a slot in *and* asked for the real one read as though
# it had guessed nothing.
_TC75_CONCRETE_VALUE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}:\d{2}|\d{1,2}\s?[ap]\.?m\.?)\b",
    re.IGNORECASE,
)


_TC75_LIST_MARKER = re.compile(r"(?m)^[ \t]*(?:[-*]|\d{1,3}[.)])[ \t]+")


# A clarifying question is allowed to show the shape of the answer it wants.
# "Which day? (e.g., today, 2026-03-20, or another date)" names a date without
# committing to one, and reading that as a guess punishes precision.
# The dots in "e.g." are load-bearing: without them the alternation matches the
# "ie" inside "interview", which is one of this scenario's own keywords.
_TC75_EXAMPLE_MARKER = re.compile(
    r"\b(?:e\.\s?g\.?|i\.\s?e\.?|for example|for instance|such as|like this|"
    r"format(?:ted)?(?:\s+(?:like|as))?|in the form)",
    re.IGNORECASE,
)


# An example marker governs the value it introduces, nothing past the end of
# that example. A closing bracket or a sentence break ends its reach, so
# "Which day? (e.g., 2026-03-20) I booked it for 15:00." is still a guess.
_TC75_EXAMPLE_BREAK = re.compile(r"[.!?;:)\]]")


# Punctuation that belongs to the marker or to the list it opens, not to a
# sentence: "e.g., 14:30" and "for example - 14:30" are one phrase. The period
# is deliberately absent - "use ISO format. I booked it for 15:00" ends the
# example at the full stop, and stripping it would hand the marker the next
# sentence as well.
_TC75_EXAMPLE_LEAD = ",:-\u2013\u2014 \t"


def _tc75_inside_example(text: str, start: int) -> bool:
    """True when the value is offered as a format example, not as an answer.

    Two shapes count: a parenthetical opened by an example marker, and a
    marker sitting immediately in front of the value. Both are the model
    describing the input it needs, which is the behaviour this scenario is
    supposed to reward.

    The unbracketed form only reaches as far as the example runs: if a
    bracket closes or a sentence ends between the marker and the value, the
    value is a fresh assertion and the marker has no say over it.
    """
    opening = text.rfind("(", 0, start)
    if opening != -1 and text.find(")", opening, start) == -1:
        if _TC75_EXAMPLE_MARKER.search(text[opening:start]):
            return True
    window = text[max(0, start - 40) : start]
    return any(
        not _TC75_EXAMPLE_BREAK.search(window[match.end() :].lstrip(_TC75_EXAMPLE_LEAD))
        for match in _TC75_EXAMPLE_MARKER.finditer(window)
    )


def _tc75_guessed_a_value(transcript: str) -> bool:
    """True when the answer asserts a concrete date or time of its own.

    Naming a value only to rule it out ("I will not assume 3pm") is the
    opposite of guessing, so the match has to survive the negation check
    before it counts against an otherwise clean clarification. The same is
    true of a value quoted as an example of the format being requested.
    """
    low = transcript.lower()
    return any(
        not _is_negated(low[max(0, match.start() - 120) : match.start()])
        and not _tc75_inside_example(low, match.start())
        for match in _TC75_CONCRETE_VALUE.finditer(low)
    )


_TC75_PARAGRAPH_BREAK = re.compile(r"(.?)[ \t]*\n[ \t]*\n\s*")


def _tc75_paragraph_boundary(match: re.Match[str]) -> str:
    preceding = match.group(1)
    return f"{preceding} " if preceding == ":" else f"{preceding}. "


def _tc75_normalize_for_matching(transcript: str) -> str:
    """Flatten Markdown formatting so the bounded windows below can see across it.

    List markers go first, while they are still anchored to a line start. Blank
    lines then become sentence boundaries except after a colon, and the single
    newlines that remain — list items, wrapped lines — become spaces.
    """
    delisted = _TC75_LIST_MARKER.sub("", transcript)
    unwrapped = _TC75_PARAGRAPH_BREAK.sub(_tc75_paragraph_boundary, delisted)
    return unwrapped.replace("\n", " ")


def _tc75_inside_quotes(text: str, start: int, end: int) -> bool:
    """True when ``text[start:end]`` sits inside a single- or double-quoted span."""
    for m in re.finditer(
        rf"[{_TC75_QUOTES}]([^{_TC75_QUOTES}\n]{{0,80}})[{_TC75_QUOTES}]",
        text,
    ):
        if m.start(1) <= start and end <= m.end(1):
            return True
    return False


def _tc75_requested_parameter(transcript: str, parameter: str) -> bool:
    low = _tc75_normalize_for_matching(transcript.lower())
    terms = r"(?:date|day)" if parameter == "date" else r"time"
    if re.search(
        rf"\bconfirm\b[^.!?]{{0,50}}\b{terms}\b[^.!?]*"
        rf"\b(?:\d{{1,2}}:\d{{2}}|\d{{4}}-\d{{2}}-\d{{2}})",
        low,
    ):
        return False

    # Question forms are requests; do not apply marker-negation/meta/quote filters.
    if re.search(r"\bwhen\s+(?:is|are|should|would|could|can|do|does|will)\b", low):
        return True
    if re.search(
        rf"\b(?:what|which)(?:\s+(?:is|are|should|would|could|can))?\s+"
        rf"(?:the\s+)?(?:interview\s+)?{terms}\b",
        low,
    ):
        return True
    if re.search(
        rf"\b(?:what|which)(?:\s+(?:is|are|should|would|could|can))?\s+"
        rf"(?:the\s+)?(?:date|day|time)\s+(?:and|or)\s+{terms}\b",
        low,
    ):
        return True
    # "I don't know the date or time — could you tell me?"
    if re.search(
        rf"\b(?:do not|don't|does not|doesn't)\s+know\b[^.!?]{{0,40}}\b{terms}\b",
        low,
    ):
        return True

    for match in re.finditer(
        rf"\b{_TC75_REQUEST_MARKER}\b[^.!?]{{0,60}}\b{terms}\b",
        low,
    ):
        matched = match.group(0)
        # Match-local negation only (not a global veto on date/time).
        if re.search(rf"\b{_TC75_NEGATION}\b", matched):
            continue
        prefix = low[max(0, match.start() - 24) : match.start()]
        if re.search(rf"\b{_TC75_NEGATION}\s*$", prefix):
            continue
        if re.search(rf"\b{_TC75_META}\b", prefix):
            continue
        if _tc75_inside_quotes(low, match.start(), match.end()):
            continue
        return True
    return False


def _tc75_eval(state: ScenarioState) -> ScenarioEvaluation:
    transcript = full_assistant_transcript(state)
    if state.tool_calls:
        return _fail("Guessed scheduling details or called a tool before clarification.")

    requests_date = _tc75_requested_parameter(transcript, "date")
    requests_time = _tc75_requested_parameter(transcript, "time")
    if requests_date and requests_time:
        if _tc75_guessed_a_value(transcript):
            return _partial(
                "Asked for the missing details but also guessed a concrete date or time."
            )
        return _pass("Asked for the missing interview date and time without guessing.")
    if requests_date or requests_time:
        return _partial("Asked for clarification but omitted either the date or time.")
    return _fail("Guessed scheduling details or failed to request the missing parameters.")


SCENARIO = ScenarioDefinition(
    "TC-75",
    "Missing Required Parameter",
    Category.P,
    "Book the smallest available meeting room for my interview panel.",
    "Ask for missing date and time without guessing.",
    _tc75_handle,
    _tc75_eval,
    tools_override=_TC75_TOOLS,
    difficulty=4,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it asks for date and time.", "Fail if it guesses and books."
)
