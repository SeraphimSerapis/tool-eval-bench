"""TC-12 — Impossible Request."""

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
    contains_refusal as _contains_refusal,
)
from tool_eval_bench.evals.helpers import (
    fail_eval as _fail,
)
from tool_eval_bench.evals.helpers import (
    generic_tool_fallback as _generic_tool_fallback,
)
from tool_eval_bench.evals.helpers import (
    normalize as _normalize,
)
from tool_eval_bench.evals.helpers import (
    pass_eval as _pass,
)


def _tc12_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    return _generic_tool_fallback(call)


def _tc12_eval(state: ScenarioState) -> ScenarioEvaluation:
    answer = _normalize(state.final_answer)
    if (
        len(state.tool_calls) == 0
        and _contains_refusal(answer)
        and re.search(r"delete|remove|erase|email", answer, re.IGNORECASE)
        and re.search(r"tool|support|capab|avail|cannot|can't|unable|won't", answer, re.I)
    ):
        return _pass("Refused cleanly because no delete-email tool exists.")
    return _fail("Did not refuse the unsupported email-deletion request correctly.")


SCENARIO = ScenarioDefinition(
    id="TC-12",
    title="Impossible Request",
    category=Category.D,
    user_message="Delete all my emails from last month.",
    description="Refuse cleanly when no suitable tool exists.",
    handle_tool_call=_tc12_handle,
    evaluate=_tc12_eval,
    difficulty=2,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it clearly refuses because no delete-email tool exists.",
    "Fail if it hallucinates a delete action or misuses another tool.",
)
