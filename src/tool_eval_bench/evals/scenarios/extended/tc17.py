"""TC-17 — Timezone-Aware Scheduling."""

from __future__ import annotations

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
    fail_eval as _fail,
)
from tool_eval_bench.evals.helpers import (
    generic_tool_fallback_simple as _generic_tool_fallback,
)
from tool_eval_bench.evals.helpers import (
    next_weekday_after_reference as _next_weekday_after_reference,
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
    utc_offset_aliases as _utc_offset_aliases,
)
from tool_eval_bench.evals.helpers import (
    with_noise as _noise,
)
from tool_eval_bench.evals.scenarios.extended._shared import (
    _positive_argument_contains,
    _result_matches_if_present,
    _status_is_success,
)


def _tc17_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "create_calendar_event":
        return _noise(
            {
                "event_id": "evt_9917",
                "status": "created",
                "title": _as_str(call.arguments.get("title")),
                "date": _as_str(call.arguments.get("date")),
                "time": _as_str(call.arguments.get("time")),
                "timezone": _as_str(call.arguments.get("timezone")),
            },
            "create_calendar_event",
        )
    if call.name == "get_contacts":
        return _noise(
            {"results": [{"name": "Hans Müller", "email": "hans.mueller@firma.de"}]}, "get_contacts"
        )
    return _generic_tool_fallback(call)


def _tc17_event_result_is_created(payload: Any) -> bool:
    return _status_is_success(payload, statuses={"accepted", "created", "ok", "success"})


def _tc17_eval(state: ScenarioState) -> ScenarioEvaluation:
    """Should create event at 14:00 Europe/Berlin, not UTC."""
    event_calls = [c for c in state.tool_calls if c.name == "create_calendar_event"]
    event = event_calls[0] if event_calls else None
    if not event:
        return _fail("Did not create the calendar event.")
    if not _result_matches_if_present(state, event, _tc17_event_result_is_created):
        return _partial("The calendar mutation did not return a successful creation result.")

    time_val = _as_str(event.arguments.get("time"))
    tz_val = _normalize(_as_str(event.arguments.get("timezone")))
    date_val = _as_str(event.arguments.get("date"))
    title_val = _normalize(_as_str(event.arguments.get("title")))

    # "nächsten Dienstag" is relative to the run's reference date.
    expected_date = _next_weekday_after_reference(state, "tuesday")

    correct_time = time_val == "14:00"
    # Which offset spelling is correct depends on whether the target date falls
    # inside EU summer time, so derive the accepted aliases from the date rather
    # than assuming the March default.
    correct_tz = tz_val == "europe/berlin" or tz_val in _utc_offset_aliases(
        expected_date, "Europe/Berlin"
    )
    correct_date = date_val == expected_date
    has_title = any(
        _positive_argument_contains(title_val, title)
        for title in ("standup", "meeting", "besprechung")
    )

    if correct_time and correct_tz and correct_date and has_title:
        if len(event_calls) > 1:
            return _partial("Created the requested event but issued duplicate calendar mutations.")
        return _pass("Scheduled for 14:00 Europe/Berlin on the correct date.")
    if correct_time and correct_date and not correct_tz:
        return _partial(
            "Got the time and date right, but defaulted to UTC instead of Europe/Berlin."
        )
    if correct_time and correct_tz:
        return _partial("Got the time and timezone right, but the date was wrong.")
    return _fail("Did not respect the Europe/Berlin timezone in the scheduling request.")


SCENARIO = ScenarioDefinition(
    id="TC-17",
    title="Timezone-Aware Scheduling",
    category=Category.F,
    user_message="Erstelle einen Termin für nächsten Dienstag um 14 Uhr Berliner Zeit. Titel: Team Standup.",
    description="Schedule in Europe/Berlin timezone, not UTC.",
    handle_tool_call=_tc17_handle,
    evaluate=_tc17_eval,
    difficulty=3,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it creates the event at 14:00 with timezone Europe/Berlin "
    "for the Tuesday after the reference date.",
    "Fail if it uses UTC or gets the date wrong.",
)
