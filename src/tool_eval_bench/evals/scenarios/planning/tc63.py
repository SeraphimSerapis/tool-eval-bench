"""TC-63 — Accumulating Constraints."""

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
    answer_affirms_number as _answer_affirms_number,
)
from tool_eval_bench.evals.helpers import (
    as_str as _as_str,
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
    tool_calls_by_name as _tool_calls_by_name,
)
from tool_eval_bench.evals.helpers import (
    with_noise as _noise,
)
from tool_eval_bench.evals.scenarios.planning._shared import (
    _has_unexpected_tools,
    _result_matches_if_present,
)

_TC63_FOLLOW_UPS = [
    "Actually, it needs to be Italian.",
    "And keep the budget under $30 per person.",
    "Also, it should be near downtown.",
    "One more thing — it has to be open past 10pm.",
]


def _tc63_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "web_search":
        query = _as_str(call.arguments.get("query", "")).lower()
        constraints = []
        if "italian" in query:
            constraints.append("Italian")
        if "downtown" in query:
            constraints.append("downtown")
        if "30" in query or "budget" in query or "cheap" in query or "affordable" in query:
            constraints.append("budget")
        if "10" in query or "late" in query or "open" in query:
            constraints.append("late-night")

        if len(constraints) >= 3:
            return _noise(
                {
                    "results": [
                        {
                            "snippet": "Trattoria Bella — Italian, downtown, $22/person avg, open until 11pm. ★★★★"
                        },
                    ]
                },
                "web_search",
            )
        if len(constraints) >= 2:
            return _noise(
                {
                    "results": [
                        {"snippet": "Luigi's — Italian, downtown, $25/person, closes 9pm."},
                        {
                            "snippet": "Trattoria Bella — Italian, downtown, $22/person, open until 11pm."
                        },
                    ]
                },
                "web_search",
            )
        return _noise(
            {
                "results": [
                    {
                        "snippet": "Top restaurants: Sushi Palace ($45), Luigi's Italian ($25), "
                        "Burger Joint ($15), Trattoria Bella ($22)."
                    },
                ]
            },
            "web_search",
        )
    return _noise({"error": f"Tool {call.name} is not relevant."}, call.name)


_TC63_BUDGET_CEILING = 30


_TC63_LATE_HOUR = 22


_TC63_PAST_CUTOFF = re.compile(r"\b(?:past|after|later than|beyond)\s+10(?!\d)\s*(?:p\.?m\.?)?")


_TC63_PRICE = re.compile(r"\$\s?(\d{1,3})")


# `p.m.` is the same time as `pm`. `_TC63_PAST_CUTOFF` above already accepts
# the periods and so does TC-03's clock reader, so an answer that wrote
# "open until 11 p.m." lost the constraint here and nowhere else.
_TC63_CLOCK = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*([ap])\.?m\.?\b|\b(\d{1,2}):(\d{2})\b")


def _tc63_within_budget(answer: str) -> bool:
    """True when the answer names a per-person price at or under the ceiling."""
    for match in _TC63_PRICE.finditer(answer):
        price = int(match.group(1))
        if price > _TC63_BUDGET_CEILING or not _answer_affirms_number(answer, str(price)):
            continue
        prefix = answer[max(0, match.start() - 32) : match.start()].lower()
        if re.search(r"\b(?:not|never|no)\b(?:\W+\w+){0,4}\W*$", prefix):
            continue
        return True
    return False


def _tc63_affirms_phrase(answer: str, phrase: str) -> bool:
    pattern = re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE)
    for match in pattern.finditer(answer):
        prefix = answer[max(0, match.start() - 32) : match.start()].lower()
        if not re.search(r"\b(?:not|never|no|without)\b(?:\W+\w+){0,3}\W*$", prefix):
            return True
    return False


def _tc63_open_late(answer: str) -> bool:
    """True when the answer names a closing time at or after the late cutoff.

    Accepts "11pm", "11 PM", "11 p.m.", "23:00", and "open until 11:30pm"
    alike, plus the user's own phrasing ("open past 10"). Closing at 22:00
    exactly does not count, since the request was for somewhere open *past*
    10pm.
    """
    cutoff = _TC63_PAST_CUTOFF.search(answer)
    if cutoff:
        prefix = answer[max(0, cutoff.start() - 32) : cutoff.start()].lower()
        if not re.search(r"\b(?:not|never|no|without)\b(?:\W+\w+){0,4}\W*$", prefix):
            return True
    for match in _TC63_CLOCK.finditer(answer):
        hour12, minute12, meridiem, hour24, minute24 = match.groups()
        prefix = answer[max(0, match.start() - 32) : match.start()].lower()
        if re.search(r"\b(?:not|never|no|without)\b(?:\W+\w+){0,4}\W*$", prefix):
            continue
        if meridiem:
            hour, minute = int(hour12) % 12 + (12 if meridiem == "p" else 0), int(minute12 or 0)
        else:
            hour, minute = int(hour24), int(minute24 or 0)
        if (hour, minute) > (_TC63_LATE_HOUR, 0):
            return True
    return False


def _tc63_eval(state: ScenarioState) -> ScenarioEvaluation:
    """User: 'Find me a restaurant for dinner tonight.'
    Then constraints accumulate: Italian → under $30 → downtown → open past 10pm.

    Final answer should satisfy ALL 4 constraints.
    Best match: Trattoria Bella.
    """
    answer = state.final_answer.lower()

    searched_calls = [
        call
        for call in _tool_calls_by_name(state, "web_search")
        if _result_matches_if_present(
            state,
            call,
            lambda payload: "trattoria" in str(payload).lower() or "luigi" in str(payload).lower(),
        )
    ]
    searched = bool(searched_calls)
    has_italian = any(
        _tc63_affirms_phrase(answer, phrase) for phrase in ("trattoria", "luigi", "italian")
    )
    # Read the numbers rather than matching the fixture's exact spelling. A
    # model that paraphrases the price or uses a 24-hour clock has still
    # retained the constraint.
    has_budget = _tc63_within_budget(answer) or any(
        _tc63_affirms_phrase(answer, w) for w in ("budget", "affordable", "cheap", "inexpensive")
    )
    has_downtown = any(
        _tc63_affirms_phrase(answer, phrase)
        for phrase in ("downtown", "city centre", "city center")
    )
    has_late = _tc63_open_late(answer)
    best_pick = _tc63_affirms_phrase(answer, "trattoria") or _tc63_affirms_phrase(answer, "bella")

    constraints_met = sum([has_italian, has_budget, has_downtown, has_late])

    if best_pick and constraints_met == 4 and searched:
        if _has_unexpected_tools(state, {"web_search"}):
            return _partial("Found a matching restaurant but also called an unrelated tool.")
        return _pass("Maintained all accumulated constraints → recommended Trattoria Bella.")
    if constraints_met == 4 and searched:
        if _has_unexpected_tools(state, {"web_search"}):
            return _partial("Found a matching restaurant but also called an unrelated tool.")
        return _pass("Final recommendation satisfies all 4 accumulated constraints.")
    # Both PASS branches above require a qualifying search, and without one a
    # 4/4 answer would otherwise fall past every count branch to the closing
    # _fail — scoring 0 against 1 for a 1/4 answer, under a summary saying it
    # reflected none of the constraints. It kept all four; it just never looked
    # them up.
    if constraints_met == 4:
        return _partial("Satisfies all 4 constraints but never searched for a match.")
    if constraints_met == 3:
        return _partial(f"Met {constraints_met}/4 constraints — close but dropped one.")
    if constraints_met == 2:
        return _partial(f"Met {constraints_met}/4 constraints — lost context on some additions.")
    if constraints_met == 1:
        return _partial("Only retained 1/4 constraints — significant context drift.")
    return _fail("Final answer doesn't reflect any of the accumulated constraints.")


SCENARIO = ScenarioDefinition(
    id="TC-63",
    title="Accumulating Constraints",
    category=Category.I,
    user_message="Find me a restaurant for dinner tonight.",
    description="Maintain 4 constraints accumulated across 5 turns.",
    handle_tool_call=_tc63_handle,
    evaluate=_tc63_eval,
    follow_up_messages=_TC63_FOLLOW_UPS,
    difficulty=4,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if final recommendation satisfies all 4 accumulated constraints.",
    "Fail if it forgets earlier constraints as new ones are added.",
)
