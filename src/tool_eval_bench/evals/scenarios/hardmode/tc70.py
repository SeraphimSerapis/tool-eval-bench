"""TC-70 — Adversarial Near-Duplicate Tools."""

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
    answer_affirms_number,
    answer_affirms_text,
    as_str,
    generic_tool_fallback,
    has_tool_call,
    includes_text,
    matching_tool_results,
    normalize,
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

_TC70_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a specific location. Covers European cities only.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather_global",
            "description": "Get current weather for a specific location. Covers all cities worldwide.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]


def _tc70_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_weather":
        loc = normalize(as_str(call.arguments.get("location")))
        if "tokyo" in loc or "sydney" in loc:
            return _noise(
                {
                    "error": "This endpoint only covers European cities. Use get_weather_global for worldwide coverage."
                },
                "get_weather",
            )
        return _noise(
            {"location": loc.title(), "temperature": 12, "condition": "Cloudy"}, "get_weather"
        )
    if call.name == "get_weather_global":
        loc = normalize(as_str(call.arguments.get("location")))
        return _noise(
            {"location": loc.title(), "temperature": 22, "condition": "Sunny", "humidity": 55},
            "get_weather_global",
        )
    return generic_tool_fallback(call)


def _tc70_eval(state: ScenarioState) -> ScenarioEvaluation:
    # Best: use get_weather_global directly for Tokyo (non-European)
    global_calls = [
        call
        for call in tool_calls_by_name(state, "get_weather_global")
        if includes_text(call.arguments.get("location"), "tokyo")
    ]
    usable_global_calls = [
        call
        for call in global_calls
        if result_is_usable_if_present(state, call)
        and (
            not matching_tool_results(state, call)
            or any(
                isinstance(result.result, dict)
                and result.result.get("temperature") == 22
                and str(result.result.get("condition", "")).lower() == "sunny"
                for result in matching_tool_results(state, call)
            )
        )
    ]
    used_global = bool(usable_global_calls)
    used_euro = has_tool_call(
        state, "get_weather", lambda c: includes_text(c.arguments.get("location"), "tokyo")
    )
    # Recovery path: tried get_weather, got error, then used get_weather_global
    euro_calls = tool_calls_by_name(state, "get_weather")
    recovered = (
        len(euro_calls) >= 1
        and bool(usable_global_calls)
        and euro_calls[0].turn < global_calls[0].turn
    )
    if used_global and not used_euro:
        # Verify the model surfaced actual weather data (temp 22 or condition).
        has_temp = answer_affirms_number(state.final_answer, "22")
        has_condition = answer_affirms_text(state.final_answer, "sunny")
        if has_temp or has_condition:
            return _pass(
                "Selected get_weather_global directly — read the tool descriptions carefully."
            )
        return _partial(
            "Selected the correct tool but did not surface the weather data in the answer.",
        )
    if recovered:
        return _partial("Tried the wrong tool first but recovered after the error.")
    if used_euro and not used_global:
        return _fail("Used get_weather (European only) for Tokyo and did not recover.")
    return _fail("Did not use the correct weather tool for a non-European city.")


SCENARIO = ScenarioDefinition(
    id="TC-70",
    title="Adversarial Near-Duplicate Tools",
    category=Category.P,
    user_message="What's the weather like in Tokyo right now?",
    description="Distinguish between get_weather (Europe-only) and get_weather_global when the request is for a non-European city.",
    handle_tool_call=_tc70_handle,
    evaluate=_tc70_eval,
    tools_override=_TC70_TOOLS,
    difficulty=4,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it uses get_weather_global directly for Tokyo (non-European city).",
    "Fail if it uses get_weather (Europe-only) and doesn't recover.",
)
