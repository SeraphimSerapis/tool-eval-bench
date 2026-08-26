"""TC-39 — Restraint Under Abundance."""

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
from tool_eval_bench.evals.helpers import (
    fail_eval as _fail,
)
from tool_eval_bench.evals.helpers import (
    has_tool_call as _has_tool_call,
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


def _tc39_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    """Any tool call here is unnecessary — the answer is trivially 30."""
    if call.name == "calculator":
        return _noise({"result": 30}, "calculator")
    if call.name == "run_code":
        return _noise(
            {"stdout": "30.0", "stderr": "", "exit_code": 0},
            "run_code",
        )
    return _noise(
        {"error": f"Tool {call.name} is not relevant for this scenario."},
        call.name,
    )


def _tc39_eval(state: ScenarioState) -> ScenarioEvaluation:
    """User: 'What is 15% of 200?'

    Same as TC-11, but with 52 tools. The question is trivial arithmetic.
    With many more tools available, the temptation to use one is even higher.
    """
    correct = _answer_affirms_number(state.final_answer, "30")
    total_calls = len(state.tool_calls)
    used_calc = _has_tool_call(state, "calculator")
    used_web = _has_tool_call(state, "web_search")
    used_code = _has_tool_call(state, "run_code")

    if total_calls == 0 and correct:
        return _pass("Answered directly without tools — resisted 52-tool temptation.")
    if used_calc and correct and total_calls == 1:
        return _partial("Used calculator correctly, but unnecessarily given trivial math.")
    if used_code:
        return _fail("Used run_code for trivial arithmetic — over-engineering.")
    if used_web:
        return _fail("Used web_search for trivial arithmetic.")
    if total_calls > 1:
        tools = ", ".join(c.name for c in state.tool_calls[:3])
        return _fail(f"Used multiple tools ({tools}) for trivial math.")
    if not correct:
        return _fail("Got the arithmetic wrong.")
    return _fail("Failed the restraint test with 52 tools available.")


SCENARIO = ScenarioDefinition(
    id="TC-39",
    title="Restraint Under Abundance",
    category=Category.L,
    user_message="What is 15% of 200?",
    description="Answer trivial arithmetic directly with 52 tools tempting you.",
    handle_tool_call=_tc39_handle,
    evaluate=_tc39_eval,
    tools_override=LARGE_TOOLSET,
    difficulty=3,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it answers 30 directly despite 52 tools being available.",
    "Fail if it uses web_search, run_code, or multiple tools.",
)
