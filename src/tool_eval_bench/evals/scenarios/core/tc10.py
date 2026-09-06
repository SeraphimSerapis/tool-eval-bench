"""TC-10 — Trivial Knowledge."""

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
    pass_eval as _pass,
)


def _tc10_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    return _generic_tool_fallback(call)


def _tc10_eval(state: ScenarioState) -> ScenarioEvaluation:
    answer = state.final_answer.strip()
    contextual = bool(re.search(r"\b(?:world\s+war|wwii|war|ended|surrender|year)\b", answer, re.I))
    # A terse reply whose whole content is the year: "1945", "1945.",
    # "It was 1945.", "In 1945." An exact-string match on the bare year
    # rejected the ordinary sentence forms of the same reply, while a plain
    # length cap would accept "I have 1945 unread messages."
    terse_year = bool(
        re.fullmatch(
            r"(?:it\s+was\s+|in\s+|the\s+year\s+(?:was\s+)?|answer:\s*)?1945\s*[.!]?",
            answer,
            re.IGNORECASE,
        )
    )
    if (
        len(state.tool_calls) == 0
        and _answer_affirms_number(answer, "1945")
        and (contextual or terse_year)
    ):
        return _pass("Answered directly without tool use.")
    return _fail("Used tools or missed the basic fact.")


SCENARIO = ScenarioDefinition(
    id="TC-10",
    title="Trivial Knowledge",
    category=Category.D,
    user_message="What year did World War II end?",
    description="Avoid tools for basic knowledge.",
    handle_tool_call=_tc10_handle,
    evaluate=_tc10_eval,
    difficulty=1,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it answers 1945 directly with no tool call.",
    "Fail if it uses any tool for basic history.",
)
