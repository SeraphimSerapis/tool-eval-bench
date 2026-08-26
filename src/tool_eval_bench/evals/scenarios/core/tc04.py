"""TC-04 — Unit Handling."""

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
    answer_affirms_number as _answer_affirms_number,
)
from tool_eval_bench.evals.helpers import (
    answer_contains_number as _answer_contains_number,
)
from tool_eval_bench.evals.helpers import (
    as_str as _as_str,
)
from tool_eval_bench.evals.helpers import (
    fail_eval as _fail,
)
from tool_eval_bench.evals.helpers import (
    first_call as _first_call,
)
from tool_eval_bench.evals.helpers import (
    generic_tool_fallback as _generic_tool_fallback,
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
    with_noise as _noise,
)
from tool_eval_bench.evals.scenarios.core._shared import (
    _numeric_value,
    _positive_argument_contains,
    _result_matches_if_present,
)


def _tc04_weather_result_is_tokyo_fahrenheit(payload: Any) -> bool:
    if not isinstance(payload, dict) or "error" in payload:
        return False
    return (
        _positive_argument_contains(payload.get("location"), "tokyo")
        and _normalize(_as_str(payload.get("units"))) == "fahrenheit"
        and _numeric_value(payload.get("temperature")) == 64
    )


def _tc04_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_weather":
        units = _normalize(_as_str(call.arguments.get("units"))) or "celsius"
        if units == "fahrenheit":
            return _noise(
                {
                    "location": "Tokyo",
                    "temperature": 64,
                    "units": "fahrenheit",
                    "condition": "Clear",
                },
                "get_weather",
            )
        return _noise(
            {"location": "Tokyo", "temperature": 18, "units": "celsius", "condition": "Clear"},
            "get_weather",
        )
    return _generic_tool_fallback(call)


def _tc04_eval(state: ScenarioState) -> ScenarioEvaluation:
    weather = _first_call(state, "get_weather")
    if (
        weather
        and _positive_argument_contains(weather.arguments.get("location"), "tokyo")
        and _normalize(_as_str(weather.arguments.get("units"))) == "fahrenheit"
    ):
        if not _result_matches_if_present(state, weather, _tc04_weather_result_is_tokyo_fahrenheit):
            return _partial(
                "get_weather returned an error, so the temperature cannot be confirmed."
            )
        # The requested unit alone is not returned weather data.
        has_data = _answer_affirms_number(state.final_answer, "64")
        if has_data:
            if len(state.tool_calls) != 1:
                return _partial(
                    "Used the right weather parameters but added an unnecessary tool call."
                )
            return _pass("Requested Tokyo weather in Fahrenheit explicitly.")
        return _partial(
            "Called get_weather with correct units but did not surface the temperature "
            "in the answer.",
        )
    if (
        weather
        and _positive_argument_contains(weather.arguments.get("location"), "tokyo")
        and not _as_str(weather.arguments.get("units"))
        and (
            "fahrenheit" in state.final_answer.lower()
            or _answer_contains_number(state.final_answer, "64")
        )
    ):
        return _partial("Omitted the units parameter and converted manually.")
    return _fail("Did not preserve the Fahrenheit instruction.")


SCENARIO = ScenarioDefinition(
    id="TC-04",
    title="Unit Handling",
    category=Category.B,
    user_message="What's the temperature in Tokyo in Fahrenheit?",
    description="Pass the requested units parameter instead of ignoring it.",
    handle_tool_call=_tc04_handle,
    evaluate=_tc04_eval,
    difficulty=2,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it requests Tokyo weather with units set to fahrenheit.",
    "Fail if it ignores the Fahrenheit instruction.",
)
