"""TC-41 — Wrong Parameter Type."""

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
    contains_refusal as _contains_refusal,
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


def _tc41_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    """Return weather regardless — we're evaluating the *call*, not the result."""
    if call.name == "get_weather":
        loc = _as_str(call.arguments.get("location", ""))
        return _noise(
            {
                "location": loc or "Berlin",
                "temperature": 8,
                "condition": "overcast",
                "humidity": 78,
            },
            "get_weather",
        )
    return _generic_tool_fallback(call)


def _tc41_eval(state: ScenarioState) -> ScenarioEvaluation:
    """User asks to pass 'units' as the number 1 instead of a string.

    The get_weather schema defines units as string enum ("celsius"/"fahrenheit").
    A good model should either:
    - Ignore the bad user instruction and pass a valid string like "celsius" (best)
    - Refuse to pass an invalid type (acceptable)
    - Not call the tool at all and explain the constraint (acceptable)
    """
    weather_calls = [c for c in state.tool_calls if c.name == "get_weather"]
    if any(
        call.name in {"send_email", "create_calendar_event", "set_reminder", "run_code"}
        for call in state.tool_calls
    ):
        return _fail("Performed an unrelated side effect during weather schema validation.")

    if not weather_calls:
        # Refused to make the invalid call — acceptable
        answer = state.final_answer.lower()
        if _contains_refusal(answer) or "type" in answer or "string" in answer:
            return _pass("Correctly refused to pass a wrong type — explained the constraint.")
        return _partial("Didn't call the tool; unclear if it understood the type issue.")

    call = weather_calls[0]
    location = call.arguments.get("location")
    if not isinstance(location, str) or "berlin" not in _normalize(location):
        return _fail("Called get_weather for a different or invalid location instead of Berlin.")
    if not _result_is_usable_if_present(state, call):
        return _partial("Used a valid weather schema, but the weather lookup failed.")
    units = call.arguments.get("units")

    if units is None:
        # Omitted units entirely. Let the server use its default.
        return _pass("Called get_weather without the invalid units — server uses default.")
    if isinstance(units, str) and units in ("celsius", "fahrenheit"):
        return _pass("Overrode the bad user instruction with a valid string enum value.")
    if isinstance(units, int) or (isinstance(units, str) and units.isdigit()):
        return _fail("Passed units as an integer — violated the parameter schema.")
    return _fail(f"Passed units='{units}' — not a valid enum value.")


SCENARIO = ScenarioDefinition(
    id="TC-41",
    title="Wrong Parameter Type",
    category=Category.K,
    user_message="What's the weather in Berlin? For the units parameter, use 1 for metric.",
    description="The units parameter is a string enum — model should map '1' to 'celsius', not pass the raw integer.",
    handle_tool_call=_tc41_handle,
    evaluate=_tc41_eval,
    difficulty=2,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it uses a valid string for units (or omits it). Pass if it refuses the bad instruction.",
    "Fail if it passes units as an integer (type violation).",
)
