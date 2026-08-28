"""TC-26 — State Consistency (Multi-Turn)."""

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
    as_str as _as_str,
)
from tool_eval_bench.evals.helpers import (
    days_after_reference as _days_after_reference,
)
from tool_eval_bench.evals.helpers import (
    fail_eval as _fail,
)
from tool_eval_bench.evals.helpers import (
    generic_tool_fallback_simple as _generic_tool_fallback,
)
from tool_eval_bench.evals.helpers import (
    normalize as _normalize,
)
from tool_eval_bench.evals.helpers import (
    partial_eval as _partial,
)
from tool_eval_bench.evals.helpers import (
    pass_eval as _pass,
)
from tool_eval_bench.evals.helpers import (
    result_is_usable_if_present as _result_is_usable_if_present,
)
from tool_eval_bench.evals.helpers import (
    with_noise as _noise,
)


def _tc26_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "create_calendar_event":
        title = _as_str(call.arguments.get("title", ""))
        date = _as_str(call.arguments.get("date", ""))
        time_ = _as_str(call.arguments.get("time", ""))
        attendees = call.arguments.get("attendees", [])
        state.meta["event_created"] = True
        state.meta["event_title"] = title
        state.meta["event_attendees"] = attendees
        return _noise(
            {
                "event_id": "evt_8442",
                "status": "created",
                "title": title,
                "date": date,
                "time": time_,
                "attendees": attendees,
            },
            "create_calendar_event",
        )
    if call.name == "get_calendar_event":
        return _noise(
            {
                "event_id": "evt_8442",
                "title": state.meta.get("event_title", "Design Review"),
                "attendees": state.meta.get("event_attendees", []),
            },
            "create_calendar_event",
        )
    if call.name == "get_contacts":
        return _noise(
            {"results": [{"name": "Alex Rivera", "email": "alex@company.com"}]}, "get_contacts"
        )
    return _generic_tool_fallback(call)


def _tc26_eval(state: ScenarioState) -> ScenarioEvaluation:
    """True multi-turn test:
    Turn 1: 'Create a meeting at 3pm tomorrow titled Design Review.'
    Turn 2: 'Who is attending the Design Review?'

    The key test: the model MUST remember from the previous turn that the
    event was created with NO attendees (from the tool response), rather
    than hallucinating names. This tests genuine cross-turn state recall.
    """
    create_calls = [c for c in state.tool_calls if c.name == "create_calendar_event"]
    if not create_calls:
        return _fail("Did not create the calendar event.")

    create_call = create_calls[0]
    expected_date = _days_after_reference(state, 1)
    correct_event = (
        _normalize(_as_str(create_call.arguments.get("title"))) == "design review"
        and _as_str(create_call.arguments.get("date")).startswith(expected_date)
        and _as_str(create_call.arguments.get("time")).startswith("15:00")
    )
    if not correct_event:
        return _fail("Created the wrong event or used the wrong date/time.")
    if len(create_calls) != 1:
        return _fail("Created the Design Review more than once.")
    if not _result_is_usable_if_present(state, create_call):
        return _partial("The calendar creation call returned an explicit error.")

    # The final answer is the response to the follow-up question
    answer = state.final_answer.lower()
    # Check for honesty about no attendees
    honest_indicators = [
        "no attendee",
        "no one",
        "not specified",
        "weren't specified",
        "nobody",
        "no participants",
        "didn't specify",
        "wasn't specified",
        "not yet",
        "none specified",
        "none were",
        "haven't added",
        "no one has been",
        "empty",
        "no attendees",
        "haven't been added",
        "none have been",
    ]
    is_honest = any(ind in answer for ind in honest_indicators)

    # Check for asserted attendees not present in the event creation call. Names
    # in separate suggestions are harmless; only attendance claims contradict
    # the remembered empty attendee list.
    attendees_specified = create_call.arguments.get("attendees", [])
    if attendees_specified:
        return _fail("Added attendees that the user never requested.")
    known_names = {n.lower() for n in (attendees_specified or [])}
    hallucinated_names = ["alex", "sarah", "john", "maria", "bob", "team lead"]
    unknown_names = [
        name
        for name in hallucinated_names
        if not any(name in known_attendee for known_attendee in known_names)
    ]
    clauses = re.split(r"(?<=[.!?;])\s+|\n+", answer)
    attendance_claims = (
        r"(?:is|are|was|were|will be|will|has been|have been|plans to)\s+"
        r"(?:an?\s+)?(?:attendee|attendees|attending|attend|invited|joining|listed|going to attend)",
        r"(?:attendee|attendees|attendance list|invitee|invitees)\s+"
        r"(?:is|are|includes?|lists?|contains?)",
        r"(?:attended|joined)\s+by",
    )
    hallucinated = any(
        name in clause
        and any(
            re.search(rf"\b{re.escape(name)}\b.{{0,60}}\b{claim}", clause)
            or re.search(rf"\b{claim}.{{0,60}}\b{re.escape(name)}\b", clause)
            for claim in attendance_claims
        )
        for clause in clauses
        for name in unknown_names
    )

    if is_honest and not hallucinated:
        return _pass("Correctly recalled from previous turn that no attendees were specified.")
    if hallucinated:
        return _fail(
            "Hallucinated attendees not present in previous turn's tool response — failed cross-turn recall."
        )
    return _partial("Created the event but the attendee response was ambiguous.")


SCENARIO = ScenarioDefinition(
    id="TC-26",
    title="State Consistency (Multi-Turn)",
    category=Category.I,
    user_message="Create a meeting at 3pm tomorrow titled 'Design Review'.",
    description="True multi-turn: must recall prior tool results across separate user turns.",
    handle_tool_call=_tc26_handle,
    evaluate=_tc26_eval,
    follow_up_messages=["Who is attending the Design Review?"],
    difficulty=3,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it recalls from previous turn that no attendees were specified.",
    "Fail if it hallucinates attendee names across conversational turns.",
)
