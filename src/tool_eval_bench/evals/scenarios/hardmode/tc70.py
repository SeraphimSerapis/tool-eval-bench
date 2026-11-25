"""TC-70 — Adversarial Near-Duplicate Tools."""

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
    answer_affirms_number,
    answer_affirms_text,
    as_str,
    first_counted,
    forbid_unrequested_side_effects,
    generic_tool_fallback,
    matching_tool_results,
    negated_before,
    normalize,
    positive_argument_contains,
    result_is_usable_if_present,
    strip_markdown_emphasis,
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


def _tc70_units(raw_units: Any) -> str | None:
    if raw_units is None:
        return "celsius"
    units = normalize(as_str(raw_units))
    if units in ("celsius", "fahrenheit"):
        return units
    return None


def _tc70_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name in ("get_weather", "get_weather_global"):
        loc = normalize(as_str(call.arguments.get("location")))
        if not loc:
            return _noise({"error": "Missing required parameter: location."}, call.name)
        units = _tc70_units(call.arguments.get("units"))
        if units is None:
            return _noise({"error": "Invalid units. Must be 'celsius' or 'fahrenheit'."}, call.name)
        if call.name == "get_weather":
            if "tokyo" in loc or "sydney" in loc:
                return _noise(
                    {
                        "error": "This endpoint only covers European cities. Use get_weather_global for worldwide coverage."
                    },
                    "get_weather",
                )
            temp = 54 if units == "fahrenheit" else 12
            return _noise(
                {
                    "location": loc.title(),
                    "temperature": temp,
                    "units": units,
                    "condition": "Cloudy",
                },
                "get_weather",
            )
        temp = 72 if units == "fahrenheit" else 22
        return _noise(
            {
                "location": loc.title(),
                "temperature": temp,
                "units": units,
                "condition": "Sunny",
                "humidity": 55,
            },
            "get_weather_global",
        )
    return generic_tool_fallback(call)


def _tc70_expected_temp(call: ToolCallRecord) -> int:
    return 72 if _tc70_units(call.arguments.get("units")) == "fahrenheit" else 22


_TC70_TEMP_MENTION_RE = re.compile(
    r"(?<![A-Za-z0-9.])(-?\d+(?:\.\d+)?)\s*"
    r"(?:°\s*([cf])\b|°(?![a-z])|degrees?\s*(celsius|fahrenheit|[cf])\b|\b(celsius|fahrenheit)\b)",
    re.IGNORECASE,
)


def _tc70_has_contradicting_temp(answer: str, usable_calls: list[ToolCallRecord]) -> bool:
    """Return True if ``answer`` affirms a temperature that contradicts the Tokyo result."""
    if not usable_calls:
        return False
    plain = strip_markdown_emphasis(answer).replace("\u2212", "-")
    requested_units = {
        _tc70_units(call.arguments.get("units")) or "celsius" for call in usable_calls
    }
    allowed_c = {22.0, 22.2} if "fahrenheit" in requested_units else {22.0}
    allowed_f = {72.0, 71.6}
    for match in _TC70_TEMP_MENTION_RE.finditer(plain):
        if negated_before(
            plain[max(0, match.start() - 120) : match.start()],
            plain[match.end() : match.end() + 40],
        ):
            continue
        value = float(match.group(1))
        raw_unit = (match.group(2) or match.group(3) or match.group(4) or "").lower()
        if raw_unit.startswith("f"):
            if value not in allowed_f:
                return True
        elif raw_unit.startswith("c"):
            if value not in allowed_c:
                return True
        else:
            allowed_any = (allowed_c if "celsius" in requested_units else set()) | (
                allowed_f if "fahrenheit" in requested_units else set()
            )
            if value not in allowed_any:
                return True
    return False


@forbid_unrequested_side_effects()
def _tc70_eval(state: ScenarioState) -> ScenarioEvaluation:
    if any(
        c.name not in {"get_weather", "get_weather_global", "web_search"} for c in state.tool_calls
    ):
        return _fail("Attempted an unadvertised tool call.")
    # Best: use get_weather_global directly for Tokyo (non-European)
    global_calls = [
        call
        for call in tool_calls_by_name(state, "get_weather_global")
        if positive_argument_contains(call.arguments.get("location"), "tokyo")
    ]
    usable_global_calls = [
        call
        for call in global_calls
        if _tc70_units(call.arguments.get("units")) is not None
        and result_is_usable_if_present(state, call)
        and (
            not matching_tool_results(state, call)
            or any(
                isinstance(result.result, dict)
                and result.result.get("temperature") == _tc70_expected_temp(call)
                and str(result.result.get("condition", "")).lower() == "sunny"
                for result in matching_tool_results(state, call)
            )
        )
    ]
    used_global = bool(usable_global_calls)
    euro_calls = tool_calls_by_name(state, "get_weather")
    used_euro = bool(euro_calls)
    tokyo_euro_calls = [
        c for c in euro_calls if positive_argument_contains(c.arguments.get("location"), "tokyo")
    ]
    tokyo_euro = bool(tokyo_euro_calls)
    # A global attempt that --error-rate failed and the model retried is not
    # where the recovery started.
    first_global = first_counted(usable_global_calls)
    has_temp = any(
        answer_affirms_number(state.final_answer, str(_tc70_expected_temp(c)))
        for c in usable_global_calls
    )
    has_condition = answer_affirms_text(state.final_answer, "sunny")
    wrong_temp = _tc70_has_contradicting_temp(state.final_answer, usable_global_calls)
    answer_ok = (has_temp or has_condition) and not wrong_temp
    # Hedging by calling both tools in one turn, then answering from the global
    # result, is a legitimate parallel call — a strict turn ordering scored it
    # as never having used the right tool at all.
    hedged = bool(
        euro_calls
        and usable_global_calls
        and first_global is not None
        and any(ec.turn == first_global.turn for ec in euro_calls)
    )
    sequential_recovery = bool(
        tokyo_euro_calls
        and usable_global_calls
        and first_global is not None
        and any(ec.turn < first_global.turn for ec in tokyo_euro_calls)
    )
    if used_global and not used_euro:
        if wrong_temp:
            return _partial(
                "Selected the correct tool but reported an inaccurate Tokyo temperature."
            )
        # Verify the model surfaced actual weather data (temp or condition).
        if has_temp or has_condition:
            return _pass(
                "Selected get_weather_global directly — read the tool descriptions carefully."
            )
        return _partial(
            "Selected the correct tool but did not surface the weather data in the answer.",
        )
    if sequential_recovery and answer_ok:
        return _partial("Tried the wrong tool first but recovered after the error.")
    if hedged:
        if answer_ok:
            return _partial(
                "Called both weather tools in the same turn and answered from the global one."
            )
        return _fail("Called both weather tools in the same turn without surfacing Tokyo weather.")
    if used_global and used_euro and answer_ok:
        return _partial("Called the Europe-only weather tool alongside global weather.")
    if tokyo_euro and not used_global:
        return _fail("Used get_weather (European only) for Tokyo and did not recover.")
    return _fail("Did not obtain Tokyo weather from a usable weather tool result.")


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
    capabilities=("tool-selection",),
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it uses get_weather_global directly for Tokyo (non-European city).",
    "Fail if it uses get_weather (Europe-only) and doesn't recover.",
)
