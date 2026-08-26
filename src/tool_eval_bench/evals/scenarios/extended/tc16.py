"""TC-16 — German Language Tool Call."""

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
    answer_contains_number as _answer_contains_number,
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
    has_explicit_tool_error as _has_explicit_tool_error,
)
from tool_eval_bench.evals.helpers import (
    includes_text as _includes_text,
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
from tool_eval_bench.evals.scenarios.extended._shared import (
    _GERMAN_MARKERS,
    _numeric_value,
    _positive_argument_contains,
    _result_matches_if_present,
)


def _tc16_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_weather":
        return _noise(
            {
                "location": "München",
                "temperature": 14,
                "units": "celsius",
                "condition": "Teilweise bewölkt",
                "humidity": 58,
            },
            "get_weather",
        )
    if call.name == "translate_text":
        target = _normalize(_as_str(call.arguments.get("target_language")))
        if target in ("german", "de", "deutsch"):
            return _noise(
                {
                    "translated": "Das aktuelle Wetter in München: 14°C, teilweise bewölkt, Luftfeuchtigkeit 58%."
                },
                "translate_text",
            )
        return _noise({"translated": _as_str(call.arguments.get("text"))}, "translate_text")
    if call.name == "web_search":
        return _noise({"results": [{"snippet": "München: 14°C, teilweise bewölkt."}]}, "web_search")
    return _generic_tool_fallback(call)


def _tc16_weather_result_is_munich(payload: Any) -> bool:
    if not isinstance(payload, dict) or "error" in payload:
        return False
    if "location" in payload and not (
        _positive_argument_contains(payload.get("location"), "münchen")
        or _positive_argument_contains(payload.get("location"), "munich")
    ):
        return False
    return _numeric_value(payload.get("temperature")) == 14 or _includes_text(
        payload.get("condition"), "bewölkt"
    )


def _tc16_eval(state: ScenarioState) -> ScenarioEvaluation:
    """User asked in German — should use get_weather and respond in German."""
    matching_weather_calls = [
        c
        for c in state.tool_calls
        if c.name == "get_weather"
        and (
            _positive_argument_contains(c.arguments.get("location"), "münchen")
            or _positive_argument_contains(c.arguments.get("location"), "munich")
        )
    ]
    used_weather = bool(matching_weather_calls)
    # Check that the final answer is in German.
    # Two paths: weather-specific vocabulary (when tool succeeds) or
    # generic German (when tool returns an error and model reports failure).
    answer_lower = state.final_answer.lower()
    weather_german = ["wetter", "grad", "bewölkt", "celsius", "münchen", "temperatur", "°c"]
    generic_german = [
        "tut mir leid",
        "bitte",
        "leider",
        "versuchen",
        "nicht",
        "dienst",
        "erneut",
        "verfügbar",
        "gerade",
        "aktuell",
    ]
    weather_score = sum(1 for w in weather_german if w in answer_lower)
    generic_score = sum(1 for w in generic_german if w in answer_lower)
    # "temperatur", "celsius", "münchen" and "°c" all appear verbatim in English
    # answers, so the vocabulary score alone cannot prove the reply is German.
    # Require one marker that has no English reading — but keep the marker set
    # wide enough that a terse, fully German answer ("München hat aktuell 14°C
    # bei bewölktem Himmel.") still qualifies.
    german_markers = len(_GERMAN_MARKERS.findall(answer_lower))
    answer_in_german = (weather_score >= 3 or generic_score >= 2) and german_markers >= 1
    # Check the model surfaced actual weather data (temp 14 or condition).
    has_data = _answer_contains_number(state.final_answer, "14") or "bewölkt" in answer_lower
    weather_success = any(
        _result_matches_if_present(state, call, _tc16_weather_result_is_munich)
        for call in matching_weather_calls
    )
    weather_failed = any(_has_explicit_tool_error(state, call) for call in matching_weather_calls)
    answered_about_error = generic_score >= 2 and weather_failed and not has_data
    if used_weather and answer_in_german and answered_about_error:
        if generic_score >= 4:
            return _pass("Used get_weather and clearly reported the service error in German.")
        return _partial(
            "Reported the weather tool error in German, but did not receive weather data."
        )
    if used_weather and answer_in_german and has_data and weather_success:
        return _pass("Used get_weather for München and responded in German.")
    if used_weather and answer_in_german and not has_data:
        return _partial(
            "Used get_weather and responded in German but did not surface the temperature."
        )
    if used_weather and not answer_in_german:
        return _partial("Found the weather correctly but responded in English instead of German.")
    if not used_weather and answer_in_german:
        return _partial("Responded in German but didn't use the weather tool.")
    return _fail(
        "Did not handle the German-language request as a tool call in the expected language."
    )


SCENARIO = ScenarioDefinition(
    id="TC-16",
    title="German Language Tool Call",
    category=Category.F,
    user_message="Wie ist das Wetter in München gerade?",
    description="Handle a German-language weather request, respond in German.",
    handle_tool_call=_tc16_handle,
    evaluate=_tc16_eval,
    difficulty=2,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it calls get_weather for München and responds in German.",
    "Fail if it responds in English or misses the weather tool.",
)
