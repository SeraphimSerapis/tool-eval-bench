"""TC-79 — Dependency-Aware Event Planning."""

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
    as_str,
    days_after_reference,
    positive_argument_contains,
    result_is_usable_if_present,
    tool_calls_by_name,
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
    _result_matches_if_present,
    _tool,
)

_TC79_TOOLS = [
    _tool("get_weather", "Get weather for a location.", {"location": _STRING}, ["location"]),
    _tool("get_contacts", "Look up contacts.", {"query": _STRING}, ["query"]),
    _tool(
        "create_calendar_event",
        "Create a calendar event.",
        {
            "title": _STRING,
            "date": _STRING,
            "time": _STRING,
            "timezone": _STRING,
            "duration_minutes": {"type": "integer"},
            "attendees": {"type": "array", "items": _STRING},
        },
        ["title", "date", "time", "timezone", "duration_minutes", "attendees"],
    ),
]


def _tc79_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_weather":
        return _noise(
            {"location": "Lisbon", "condition": "Dry", "precipitation_probability": 0}, call.name
        )
    if call.name == "get_contacts":
        return _noise(
            {"results": [{"name": "Priya Shah", "email": "priya.shah@company.com"}]}, call.name
        )
    if call.name == "create_calendar_event":
        return _noise({"status": "created", "event_id": "evt_lisbon"}, call.name)
    return _noise({"error": f"Tool {call.name} is not relevant."}, call.name)


def _tc79_eval(state: ScenarioState) -> ScenarioEvaluation:
    weather = tool_calls_by_name(state, "get_weather")
    contacts = tool_calls_by_name(state, "get_contacts")
    events = tool_calls_by_name(state, "create_calendar_event")
    if not events:
        return _fail("Did not create the conditional event.")
    if len(events) > 1:
        return _fail("Created duplicate calendar events for one conditional request.")
    event = events[-1]
    args = event.arguments
    valid_weather_calls = [
        c
        for c in weather
        if positive_argument_contains(c.arguments.get("location"), "lisbon")
        and result_is_usable_if_present(state, c)
        and _result_matches_if_present(
            state,
            c,
            lambda payload: (
                as_str(payload.get("location")).lower() == "lisbon"
                and as_str(payload.get("condition")).lower() == "dry"
                and payload.get("precipitation_probability") == 0
            ),
        )
    ]
    valid_contact_calls = [
        c
        for c in contacts
        if "priya" in as_str(c.arguments.get("query")).lower()
        and result_is_usable_if_present(state, c)
        and _result_matches_if_present(
            state,
            c,
            lambda payload: any(
                isinstance(item, dict)
                and as_str(item.get("email")).lower() == "priya.shah@company.com"
                for item in payload.get("results", [])
            ),
        )
    ]
    dependencies_first = bool(
        valid_weather_calls
        and valid_contact_calls
        and max(valid_weather_calls[0].turn, valid_contact_calls[0].turn) < event.turn
    )
    event_usable = result_is_usable_if_present(state, event) and _result_matches_if_present(
        state, event, lambda payload: payload.get("status") == "created"
    )
    attendee_values = args.get("attendees")
    required = [
        all(term in as_str(args.get("title")).lower() for term in ("outdoor", "review")),
        args.get("date") == days_after_reference(state, 1),
        args.get("time") == "09:00",
        args.get("timezone") == "Europe/Lisbon",
        args.get("duration_minutes") == 30,
        isinstance(attendee_values, list)
        and len(attendee_values) == 1
        and attendee_values[0] == "priya.shah@company.com",
    ]
    if dependencies_first and event_usable and all(required):
        return _pass("Resolved weather and contact dependencies before creating the Lisbon event.")
    if dependencies_first and sum(required) >= 5:
        return _partial(
            "Created the event after dependency checks but omitted one required parameter."
        )
    return _fail("Created the event before dependencies resolved or lost required event details.")


SCENARIO = ScenarioDefinition(
    "TC-79",
    "Dependency-Aware Event Planning",
    Category.P,
    "Check the weather in Lisbon and find Priya Shah's email. If it will be dry, schedule a 30-minute outdoor review with Priya tomorrow at 09:00 Europe/Lisbon.",
    "Resolve independent dependencies before conditionally creating an event.",
    _tc79_handle,
    _tc79_eval,
    tools_override=_TC79_TOOLS,
    difficulty=5,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if dependencies resolve before event creation.", "Fail if it schedules prematurely."
)
