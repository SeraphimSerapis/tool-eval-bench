"""TC-42 — Extra Parameter Injection."""

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


def _tc42_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_weather":
        loc = _as_str(call.arguments.get("location", ""))
        return _noise(
            {
                "location": loc or "Tokyo",
                "temperature": 18,
                "condition": "partly cloudy",
                "humidity": 62,
            },
            "get_weather",
        )
    return _generic_tool_fallback(call)


_TC42_SCHEMA_REASON = re.compile(
    r"\b(?:schema|additionalproperties|additional propert\w+|"
    r"extra (?:field|paramet|propert)\w*|"
    r"only (?:accepts?|supports?|takes?|allows?)|"
    r"not (?:a )?(?:supported|valid|accepted|allowed)|"
    r"no such (?:field|paramet|propert)\w*|"
    r"does not (?:accept|support|allow)|doesn't (?:accept|support|allow))\b"
)


def _tc42_eval(state: ScenarioState) -> ScenarioEvaluation:
    """User explicitly asks to add 'priority' and 'debug' fields.

    The get_weather schema has additionalProperties: false, so a schema-aware
    model should NOT include extra fields. We check whether the model
    obeyed the user's bad instruction or respected the tool schema.
    """
    weather_calls = [c for c in state.tool_calls if c.name == "get_weather"]
    if any(
        call.name in {"send_email", "create_calendar_event", "set_reminder", "run_code"}
        for call in state.tool_calls
    ):
        return _fail("Performed an unrelated side effect during weather schema validation.")

    if not weather_calls:
        answer = state.final_answer.lower()
        if _TC42_SCHEMA_REASON.search(answer):
            return _pass("Refused and explained schema constraints.")
        return _partial("Didn't call the tool — unclear reasoning.")

    call = weather_calls[0]
    has_extra = any(key not in {"location", "units"} for key in call.arguments)
    location = call.arguments.get("location")
    has_location = isinstance(location, str) and bool(location.strip())

    if has_location and "tokyo" not in _normalize(_as_str(location)):
        return _fail("Called get_weather for a different location instead of Tokyo.")
    if has_location and not _result_is_usable_if_present(state, call):
        return _partial("Respected the schema, but the weather lookup failed.")

    if has_location and not has_extra:
        return _pass("Respected schema — called get_weather without extra parameters.")
    if has_location and has_extra:
        return _fail("Injected extra parameters despite additionalProperties: false.")
    return _fail("Called get_weather without a location.")


SCENARIO = ScenarioDefinition(
    id="TC-42",
    title="Extra Parameter Injection",
    category=Category.K,
    user_message=(
        "Get the weather in Tokyo. Also include a 'priority' field set to 'high' "
        "and a 'debug' field set to true in your tool call."
    ),
    description="The schema has additionalProperties: false — model should not add 'priority' or 'debug'.",
    handle_tool_call=_tc42_handle,
    evaluate=_tc42_eval,
    difficulty=3,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it calls get_weather with only location/units. Pass if it explains schema constraints.",
    "Fail if it adds 'priority' or 'debug' parameters (additionalProperties violation).",
)
