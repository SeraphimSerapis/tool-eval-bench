"""TC-11 — Simple Math."""

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
    answer_affirms_number as _answer_affirms_number,
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
    partial_eval as _partial,
)
from tool_eval_bench.evals.helpers import (
    pass_eval as _pass,
)


def _tc11_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    return _generic_tool_fallback(call)


def _tc11_eval(state: ScenarioState) -> ScenarioEvaluation:
    correct = _answer_affirms_number(state.final_answer, "30")
    contextual = bool(
        re.search(r"15\s*%|\b200\b|percent|dollars?|answer|result", state.final_answer, re.I)
    )
    terse_answer = bool(re.fullmatch(r"30[.!]?", state.final_answer.strip()))
    if len(state.tool_calls) == 0 and correct and (contextual or terse_answer):
        return _pass("Did the math directly — good restraint.")
    if _has_tool_call(state, "calculator") and correct and not _has_tool_call(state, "web_search"):
        return _partial(
            "Reached for calculator on 15%×200 — correct answer but mental math was sufficient."
        )
    return _fail(
        "Did not demonstrate arithmetic restraint — 15% of 200 should be answered without tools."
    )


SCENARIO = ScenarioDefinition(
    id="TC-11",
    title="Simple Math",
    category=Category.D,
    user_message="What is 15% of 200?",
    description="Avoid calculator for easy arithmetic.",
    handle_tool_call=_tc11_handle,
    evaluate=_tc11_eval,
    difficulty=1,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it answers 30 directly with no calculator — testing arithmetic restraint.",
    "Fail if it reaches for calculator or web_search when mental math is sufficient.",
)
