"""TC-47 — Correction Across Turns."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from tool_eval_bench.domain.scenarios import (
    Category,
    ScenarioDefinition,
    ScenarioDisplayDetail,
    ScenarioEvaluation,
    ScenarioState,
    ToolCallRecord,
)
from tool_eval_bench.domain.tools import BENCHMARK_REFERENCE_DATE
from tool_eval_bench.evals.helpers import (
    as_str as _as_str,
)
from tool_eval_bench.evals.helpers import (
    call_at_or_after_user_phase as _call_at_or_after_user_phase,
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


def _tc47_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    """Handle calendar event creation for the correction scenario.

    Note: no update_calendar_event tool exists in the universal toolset.
    The model must work with create_calendar_event only.
    """
    if call.name == "create_calendar_event":
        title = _as_str(call.arguments.get("title", ""))
        time_ = _as_str(call.arguments.get("time", ""))
        state.meta.setdefault("events_created", []).append(
            {
                "title": title,
                "time": time_,
            }
        )
        return _noise(
            {
                "event_id": f"evt_{len(state.meta['events_created'])}",
                "status": "created",
                "title": title,
                "time": time_,
            },
            "create_calendar_event",
        )
    return _generic_tool_fallback(call)


def _tc47_eval(state: ScenarioState) -> ScenarioEvaluation:
    """Multi-turn correction test:
    Turn 1: prepare a 3pm meeting draft without creating it.
    Turn 2: change it to 4pm and authorize creation.

    The first turn explicitly asks for a draft only. The model should retain the
    correction and create exactly one event at 4pm after the user authorizes it.
    """
    create_calls = [c for c in state.tool_calls if c.name == "create_calendar_event"]
    answer = state.final_answer.lower()

    if any(call.name in {"send_email", "set_reminder", "run_code"} for call in state.tool_calls):
        return _fail("Performed an unrelated side effect while correcting the calendar event.")

    if not create_calls:
        if any(value in answer for value in ("4pm", "4:00", "16:00")):
            return _partial("Tracked the corrected time but did not create the authorized event.")
        return _fail("Did not create the authorized calendar event.")

    # Check if any event was created at 4pm (correction applied)
    def _is_time(call: ToolCallRecord, *values: str) -> bool:
        value = _as_str(call.arguments.get("time", "")).strip().lower()
        return any(value == expected or value.startswith(f"{expected}:") for expected in values)

    has_original = any(_is_time(c, "15:00", "15:00:00", "3pm", "3 pm") for c in create_calls)
    has_4pm_event = any(_is_time(c, "16:00", "16:00:00", "4pm", "4 pm") for c in create_calls)
    reference_date = date.fromisoformat(
        _as_str(state.meta.get("reference_date") or BENCHMARK_REFERENCE_DATE)
    )
    expected_date = (reference_date + timedelta(days=1)).isoformat()
    valid_events = [
        call
        for call in create_calls
        if _is_time(call, "16:00", "16:00:00", "4pm", "4 pm")
        and _normalize(_as_str(call.arguments.get("title"))) == "sprint planning"
        and _as_str(call.arguments.get("date")).startswith(expected_date)
        and _call_at_or_after_user_phase(call, 1)
        and _result_is_usable_if_present(state, call)
    ]

    # Check if the model acknowledged the correction textually
    acknowledges_change = any(
        phrase in answer
        for phrase in (
            "4pm",
            "4:00",
            "16:00",
            "updated",
            "changed",
            "rescheduled",
            "moved",
            "new time",
            "changed the time",
        )
    )

    if len(create_calls) == 1 and len(valid_events) == 1 and not has_original:
        return _pass("Preserved the correction and created exactly one event at 4pm.")
    if any(not _call_at_or_after_user_phase(call, 1) for call in create_calls):
        return _fail("Created the meeting before the user authorized it.")
    if has_original:
        return _fail("Created the 3pm draft prematurely instead of waiting for authorization.")
    if has_4pm_event:
        return _partial("Created the corrected event but also made an unnecessary duplicate event.")
    if acknowledges_change:
        return _partial("Acknowledged the change to 4pm but created the wrong event.")
    return _fail("Did not process the correction in turn 2.")


SCENARIO = ScenarioDefinition(
    id="TC-47",
    title="Correction Across Turns",
    category=Category.I,
    user_message="Prepare a Sprint Planning meeting for 3pm tomorrow, but don't create it yet.",
    description="Retain a corrected time across turns, then create exactly one authorized event.",
    handle_tool_call=_tc47_handle,
    evaluate=_tc47_eval,
    follow_up_messages=["Actually, change that to 4pm. Go ahead and create it now."],
    difficulty=4,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it waits for authorization, then creates exactly one event, at 4pm.",
    "Fail if it creates the 3pm draft in turn 1 or ignores the time change in turn 2.",
)
