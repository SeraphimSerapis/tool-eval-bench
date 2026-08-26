"""TC-01 — Direct Specialist Match."""

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
    answer_affirms_text as _answer_affirms_text,
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
    with_noise as _noise,
)
from tool_eval_bench.evals.scenarios.core._shared import (
    _numeric_value,
    _positive_argument_contains,
    _result_matches_if_present,
)


def _tc01_weather_result_is_berlin(payload: Any) -> bool:
    if not isinstance(payload, dict) or "error" in payload:
        return False
    if "location" in payload and not _positive_argument_contains(payload.get("location"), "berlin"):
        return False
    if "units" in payload and _normalize(_as_str(payload.get("units"))) != "celsius":
        return False
    return _numeric_value(payload.get("temperature")) == 8 or _answer_affirms_text(
        _as_str(payload.get("condition")), "overcast"
    )


def _tc01_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_weather":
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
    used_web = _has_tool_call(state, "web_search")
    if used_weather and not used_web and len(state.tool_calls) == 1:
        weather_call = _first_call(state, "get_weather")
        if weather_call and not _result_matches_if_present(
            state, weather_call, _tc01_weather_result_is_berlin
        ):
            return _fail(
                "get_weather did not return usable Berlin weather data; the answer cannot "
                "fabricate weather data."
            )
        # Verify the model surfaced actual weather data (temp 8 or condition).
        has_temp = _answer_affirms_number(state.final_answer, "8")
        has_condition = _answer_affirms_text(state.final_answer, "overcast")
        if has_temp or has_condition:
            return _pass("Used get_weather with Berlin only.")
        return _partial(
            "Called get_weather correctly but did not surface the weather data in the answer.",
        )
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
    "Fail if it searches the web, calls multiple tools, or answers from memory.",
)
