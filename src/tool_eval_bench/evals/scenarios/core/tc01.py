"""TC-01 — Direct Specialist Match."""

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
    answer_affirms_text as _answer_affirms_text,
)
from tool_eval_bench.evals.helpers import (
    as_str as _as_str,
)
from tool_eval_bench.evals.helpers import (
    fail_eval as _fail,
)
from tool_eval_bench.evals.helpers import (
    generic_tool_fallback as _generic_tool_fallback,
)
from tool_eval_bench.evals.helpers import (
    has_tool_call as _has_tool_call,
)
from tool_eval_bench.evals.helpers import (
    is_only_tool as _is_only_tool,
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
from tool_eval_bench.evals.scenarios.core._shared import (
    _answer_affirms_rounded_value,
    _numeric_value,
    _only_error_retries_before_last,
    _positive_argument_contains,
    _result_matches_if_present,
)

_TC01_CELSIUS_MENTION_RE = re.compile(
    r"(?<![A-Za-z0-9])-?\d+(?:\.\d+)?\s*"
    r"(?:°\s*c\b|℃|°(?!\s*f\b)|degrees?(?!\s*(?:f\b|fahrenheit\b))\b|\bcelsius\b)",
    re.IGNORECASE,
)
_TC01_FAHRENHEIT_MENTION_RE = re.compile(
    r"(?<![A-Za-z0-9])-?\d+(?:\.\d+)?\s*"
    r"(?:°\s*f\b|℉|degrees?\s*(?:f\b|fahrenheit\b)|\bfahrenheit\b)",
    re.IGNORECASE,
)


def _tc01_normalized_units(call: ToolCallRecord) -> str | None:
    """Return 'celsius', 'fahrenheit', or None when units is outside the enum."""
    if "units" not in call.arguments:
        return "celsius"
    raw = call.arguments["units"]
    if not isinstance(raw, str):
        return None
    normalized = _normalize(raw)
    if normalized in ("celsius", "fahrenheit"):
        return normalized
    return None


def _tc01_weather_result_is_berlin(payload: Any, expected_units: str = "celsius") -> bool:
    if not isinstance(payload, dict) or "error" in payload:
        return False
    if "location" in payload and not _positive_argument_contains(payload.get("location"), "berlin"):
        return False
    if "units" in payload and _normalize(_as_str(payload.get("units"))) != expected_units:
        return False
    temp = _numeric_value(payload.get("temperature"))
    temp_ok = (
        temp is not None and abs(temp - 46.4) < 0.5 if expected_units == "fahrenheit" else temp == 8
    )
    return temp_ok or _answer_affirms_text(_as_str(payload.get("condition")), "overcast")


def _tc01_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_weather":
        units = _tc01_normalized_units(call)
        if units is None:
            return _noise({"error": "units must be celsius or fahrenheit"}, call.name)
        if units == "fahrenheit":
            return _noise(
                {
                    "location": "Berlin",
                    "temperature": 46.4,
                    "units": "fahrenheit",
                    "condition": "Overcast",
                    "humidity": 72,
                },
                "get_weather",
            )
        return _noise(
            {
                "location": "Berlin",
                "temperature": 8,
                "units": "celsius",
                "condition": "Overcast",
                "humidity": 72,
            },
            "get_weather",
        )
    if call.name == "web_search":
        return _noise(
            {"results": [{"snippet": "Berlin weather right now: 8C and overcast."}]}, "web_search"
        )
    return _generic_tool_fallback(call)


def _tc01_eval(state: ScenarioState) -> ScenarioEvaluation:
    used_weather = _has_tool_call(
        state,
        "get_weather",
        lambda c: _positive_argument_contains(c.arguments.get("location"), "berlin"),
    )
    used_web = _has_tool_call(
        state,
        "web_search",
        lambda c: (
            _positive_argument_contains(c.arguments.get("query"), "berlin")
            and _result_is_usable_if_present(state, c)
        ),
    )
    only_berlin_weather = bool(state.tool_calls) and all(
        c.name == "get_weather"
        and _positive_argument_contains(c.arguments.get("location"), "berlin")
        for c in state.tool_calls
    )
    if only_berlin_weather:
        # A retry after an error (including one injected by --error-rate or a
        # rejected units value) is recovery, so the last call is graded.
        # Repeating the call after a usable result is redundant.
        if not _only_error_retries_before_last(state, state.tool_calls):
            return _partial(
                "Called get_weather for Berlin again after a usable result; "
                "the duplicate call was redundant."
            )
        weather_call = state.tool_calls[-1]
        requested_units = _tc01_normalized_units(weather_call)
        if requested_units is None:
            return _fail("The weather call violated the units enum.")
        if not _result_matches_if_present(
            state,
            weather_call,
            lambda payload: _tc01_weather_result_is_berlin(payload, requested_units),
        ):
            return _fail(
                "get_weather did not return usable Berlin weather data; the answer cannot "
                "fabricate weather data."
            )
        # Verify the model surfaced actual weather data (temp 8°C / 46.4°F or
        # condition), without stating an inaccurate Celsius or Fahrenheit value.
        normalized_answer = _normalize(state.final_answer)
        has_c_temp = _answer_affirms_rounded_value(state.final_answer, "8")
        has_f_temp = _answer_affirms_rounded_value(state.final_answer, "46.4")
        has_condition = _answer_affirms_text(state.final_answer, "overcast")
        wrong_c_temp = not has_c_temp and bool(_TC01_CELSIUS_MENTION_RE.search(normalized_answer))
        wrong_f_temp = not has_f_temp and bool(
            _TC01_FAHRENHEIT_MENTION_RE.search(normalized_answer)
        )
        if wrong_c_temp or wrong_f_temp:
            return _partial(
                "Called get_weather correctly but reported an inaccurate Berlin temperature."
            )
        has_temp = has_f_temp if requested_units == "fahrenheit" else (has_c_temp or has_f_temp)
        if has_temp or has_condition:
            return _pass("Used get_weather with Berlin only.")
        return _partial(
            "Called get_weather correctly but did not surface the weather data in the answer.",
        )
    if any(c.name == "get_weather" and _tc01_normalized_units(c) is None for c in state.tool_calls):
        return _fail("The weather call violated the units enum.")
    if not used_weather and used_web and _is_only_tool(state, "web_search"):
        return _partial("Answered functionally via web_search, but missed the specialist tool.")
    return _fail("Did not cleanly route the request to get_weather.")


SCENARIO = ScenarioDefinition(
    id="TC-01",
    title="Direct Specialist Match",
    category=Category.A,
    user_message="What's the weather like in Berlin right now?",
    description="Use get_weather instead of falling back to web_search.",
    handle_tool_call=_tc01_handle,
    evaluate=_tc01_eval,
    difficulty=1,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it calls get_weather for Berlin and avoids web_search.",
    "Partial for web_search alone; fail for unrelated tools or an answer from memory.",
)
