"""TC-37 — Needle in a Haystack."""

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
from tool_eval_bench.domain.tools_large import LARGE_TOOLSET
from tool_eval_bench.evals.helpers import (
    answer_affirms_number as _answer_affirms_number,
)
from tool_eval_bench.evals.helpers import answer_affirms_text as _answer_affirms_text
from tool_eval_bench.evals.helpers import (
    fail_eval as _fail,
)
from tool_eval_bench.evals.helpers import (
    has_tool_call as _has_tool_call,
)
from tool_eval_bench.evals.helpers import (
    includes_text as _includes_text,
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


def _tc37_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
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
            {"results": [{"snippet": "Berlin weather right now: 8C and overcast."}]},
            "web_search",
        )
    # Any other tool — return a generic error with noise
    return _noise(
        {"error": f"Tool {call.name} is not relevant for this scenario."},
        call.name,
    )


def _tc37_eval(state: ScenarioState) -> ScenarioEvaluation:
    """User: 'What's the weather like in Berlin right now?'

    Same as TC-01, but with 52 tools instead of 12.
    Must use get_weather and ignore 50+ distractors.
    """
    used_weather = _has_tool_call(
        state,
        "get_weather",
        lambda c: _includes_text(c.arguments.get("location"), "berlin"),
    )
    used_web = _has_tool_call(state, "web_search")
    total_calls = len(state.tool_calls)
    domain_tools_used = [
        c.name for c in state.tool_calls if c.name not in ("get_weather", "web_search")
    ]

    if used_weather and total_calls == 1:
        weather_call = next(c for c in state.tool_calls if c.name == "get_weather")
        if not _result_is_usable_if_present(state, weather_call):
            return _partial("get_weather returned an error, so Berlin weather cannot be confirmed.")
        # Verify the model surfaced actual weather data (temp 8 or condition).
        has_temp = _answer_affirms_number(state.final_answer, "8")
        has_condition = _answer_affirms_text(state.final_answer, "overcast")
        if has_temp or has_condition:
            return _pass("Used get_weather with Berlin only — perfect selection from 52 tools.")
        return _partial(
            "Selected get_weather perfectly from 52 tools but did not surface "
            "the weather data in the answer.",
        )
    if used_weather and not domain_tools_used and total_calls <= 2:
        return _partial("Found get_weather but called an extra tool unnecessarily.")
    if used_weather and domain_tools_used:
        extras = ", ".join(domain_tools_used[:3])
        return _partial(f"Used get_weather but also called irrelevant tools: {extras}")
    if not used_weather and used_web:
        return _partial("Fell back to web_search instead of the specialist tool.")
    return _fail("Did not select get_weather from the 52-tool set.")


SCENARIO = ScenarioDefinition(
    id="TC-37",
    title="Needle in a Haystack",
    category=Category.L,
    user_message="What's the weather like in Berlin right now?",
    description="Select get_weather from 52 tools, ignoring 50+ distractors.",
    handle_tool_call=_tc37_handle,
    evaluate=_tc37_eval,
    tools_override=LARGE_TOOLSET,
    difficulty=3,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it uses only get_weather for Berlin from 52 tools.",
    "Fail if it uses wrong tools or misses get_weather.",
)
