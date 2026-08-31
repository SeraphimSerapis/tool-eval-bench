"""TC-05 — Date and Time Parsing."""

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
    as_str_list as _as_str_list,
)
from tool_eval_bench.evals.helpers import (
    date_matches as _date_matches,
)
from tool_eval_bench.evals.helpers import (
    fail_eval as _fail,
)
from tool_eval_bench.evals.helpers import (
    generic_tool_fallback as _generic_tool_fallback,
)
from tool_eval_bench.evals.helpers import (
    next_weekday_after_reference as _next_weekday_after_reference,
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
from tool_eval_bench.evals.scenarios.core._shared import (
    _attendee_matches,
    _positive_argument_contains,
    _result_matches_if_present,
    _status_is_success,
)


def _tc05_calendar_result_is_created(payload: Any) -> bool:
    return _status_is_success(payload, statuses={"accepted", "created", "ok", "success"})


def _tc05_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_contacts":
        query = _as_str(call.arguments.get("query")).strip().lower()
        contacts = [
            {"name": "Alex Stone", "email": "alex.stone@company.com"},
            {"name": "Jamie Liu", "email": "jamie.liu@company.com"},
        ]
        results = [
            contact
            for contact in contacts
            if re.search(rf"\b{re.escape(contact['name'].split()[0].lower())}\b", query)
        ]
        return _noise({"results": results}, "get_contacts")
    if call.name == "create_calendar_event":
        return _noise(
            {
                "event_id": "evt_4412",
                "status": "created",
                "title": _as_str(call.arguments.get("title")) or "Team Standup",
                "date": _as_str(call.arguments.get("date")),
            },
            "create_calendar_event",
        )
    return _generic_tool_fallback(call)


def _tc05_eval(state: ScenarioState) -> ScenarioEvaluation:
    event_calls = _tool_calls_by_name(state, "create_calendar_event")
    event = event_calls[0] if event_calls else None
    if not event:
        return _fail("Did not create the calendar event.")
    if not _result_matches_if_present(state, event, _tc05_calendar_result_is_created):
        return _partial("The calendar mutation did not return a successful creation result.")
    attendees = _as_str_list(event.arguments.get("attendees"))
    has_duration = event.arguments.get("duration_minutes") == 30
    has_attendees = any(_attendee_matches(a, "alex") for a in attendees) and any(
        _attendee_matches(a, "jamie") for a in attendees
    )
    # "next Monday" is relative to the run's reference date, not a fixed day.
    expected_date = _next_weekday_after_reference(state, "monday")
    # Use flexible date matching — accept any ISO 8601 date representation
    correct_date = _date_matches(event.arguments.get("date"), expected_date)
    # Time: accept HH:MM with optional seconds or an explicit offset.
    correct_time = bool(
        re.fullmatch(
            r"09:30(?::\d{2})?(?:Z|[+-]\d{2}:\d{2})?", _as_str(event.arguments.get("time")).strip()
        )
    )
    has_title = _positive_argument_contains(event.arguments.get("title"), "standup")
    if correct_date and correct_time and has_duration and has_attendees and has_title:
        if len(event_calls) > 1:
            return _partial("Created the requested event but issued duplicate calendar mutations.")
        return _pass("Parsed next Monday and included the requested meeting details.")
    if correct_date and correct_time:
        return _partial("Got the date and time right, but missed some optional structure.")
    return _fail("Relative date or time parsing was incorrect.")


SCENARIO = ScenarioDefinition(
    id="TC-05",
    title="Date and Time Parsing",
    category=Category.B,
    user_message="Schedule a team standup for next Monday at 9:30am, 30 minutes, with Alex and Jamie.",
    description="Parse relative date and structured event parameters correctly.",
    handle_tool_call=_tc05_handle,
    evaluate=_tc05_eval,
    difficulty=2,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it creates the event for the Monday after the reference date "
    "at 09:30 with 30 minutes and Alex plus Jamie.",
    "Fail if it misparses next Monday or drops core event details.",
)
