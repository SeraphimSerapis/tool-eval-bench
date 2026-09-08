"""TC-16 — German Language Tool Call."""

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
    answer_lower = re.sub(r"[*_`]", "", state.final_answer.lower())
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
    generic_score = sum(1 for w in generic_german if w in answer_lower)
    # Shared place and unit names cannot establish German. A grammatical marker
    # also accepts terse answers without rewarding a minimum word count.
    german_markers = len(_GERMAN_MARKERS.findall(answer_lower))
    answer_in_german = german_markers >= 1
    # Check the model surfaced actual weather data (temp 14 or condition).
    facts = re.compile(r"(?<!\d)14(?!\d)|\bbewölkt\w*", re.IGNORECASE)
    affirmed = []
    for match in facts.finditer(answer_lower):
        clause = re.split(r"[.!?;]", answer_lower[: match.start()])[-1]
        denied = re.search(r"\b(?:nicht|kein\w*|weder|not|never)\b(?:\s+\w+){0,5}\s*$", clause)
        affirmed.append(not bool(denied))
    wrong_units = re.search(r"\b14\s*(?:grad\s*)?(?:°\s*f\b|fahrenheit\b|f\b)", answer_lower)
    has_data = bool(affirmed) and all(affirmed) and not wrong_units
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
        return _partial(
            "Used the weather tool, but the response does not establish a German answer."
        )
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
    "Partial for incomplete data or unestablished German; fail if neither requirement is met.",
)
