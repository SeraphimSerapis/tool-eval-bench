"""TC-77 — Irrelevant Tool Trap."""

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
from tool_eval_bench.evals.scenarios.hardmode_expanded._shared import (
    _STRING,
    _tool,
)

_TC77_TOOLS = [
    _tool("web_search", "Search the web for current information.", {"query": _STRING}, ["query"]),
    _tool(
        "get_weather_global",
        "Get worldwide weather, including Japan.",
        {"location": _STRING},
        ["location"],
    ),
    _tool("find_japan_office", "Find a company office in Japan.", {"city": _STRING}, ["city"]),
]


def _tc77_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    return _noise({"error": "This tool is unnecessary for a stable factual question."}, call.name)


def _tc77_eval(state: ScenarioState) -> ScenarioEvaluation:
    answer = state.final_answer.strip()
    # "Answer with the city only" is about content, not punctuation: a trailing
    # full stop adds no other city. Anything beyond that is still a format
    # violation, which is what this Hard Mode scenario measures.
    if not state.tool_calls and re.fullmatch(r"tokyo[.!]?", answer, re.IGNORECASE):
        return _pass("Answered the stable factual question directly with the requested format.")
    if not state.tool_calls and "tokyo" in answer.lower():
        return _partial("Answered correctly but violated the city-only output format.")
    return _fail("Called an unnecessary tool or did not answer Tokyo.")


SCENARIO = ScenarioDefinition(
    "TC-77",
    "Irrelevant Tool Trap",
    Category.P,
    "What is the capital of Japan? Answer with the city only.",
    "Answer a stable fact directly despite tool distractors.",
    _tc77_handle,
    _tc77_eval,
    tools_override=_TC77_TOOLS,
    difficulty=4,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it answers Tokyo without tools.", "Fail if it calls a distractor tool."
)
