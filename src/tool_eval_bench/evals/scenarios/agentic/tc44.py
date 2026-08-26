"""TC-44 — tool_choice=none Compliance."""

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
    _is_negated,
)
from tool_eval_bench.evals.helpers import (
    fail_eval as _fail,
)
from tool_eval_bench.evals.helpers import (
    generic_tool_fallback_simple as _generic_tool_fallback,
)
from tool_eval_bench.evals.helpers import (
    partial_eval as _partial,
)
from tool_eval_bench.evals.helpers import (
    pass_eval as _pass,
)


def _tc44_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    """No tool calls should reach here — tool_choice is 'none'."""
    # If the server honors tool_choice="none", this should never be called.
    # If it IS called, it means the model/server ignored the constraint.
    return _generic_tool_fallback(call)


def _tc44_eval(state: ScenarioState) -> ScenarioEvaluation:
    """With tool_choice="none", the model must answer purely from knowledge.

    Pass:    No tool calls, mentions pi or 3.14
    Partial: No tool calls but vague/wrong answer
    Fail:    Made tool calls despite tool_choice="none"
    """
    if state.tool_calls:
        return _fail(f"Made {len(state.tool_calls)} tool call(s) despite tool_choice='none'.")

    answer = state.final_answer.lower()
    has_pi_value = any(
        3.139 <= float(match.group(0)) <= 3.143
        and not _is_negated(answer[max(0, match.start() - 120) : match.start()])
        for match in re.finditer(r"3\.14\d*", answer)
    )
    exact_four = bool(
        re.search(r"(?:pi|π).{0,20}(?:exactly|equals?|is)\s*4\b", answer)
        and not re.search(r"(?:pi|π).{0,20}\bnot\b.{0,20}\b4\b", answer)
    )
    if has_pi_value and not exact_four:
        return _pass("Answered from knowledge without using tools.")
    if any(kw in answer for kw in ["circumference", "circle", "ratio", "irrational"]):
        return _partial("No tools used but answer is vague — didn't state the value.")
    return _partial("No tool calls (correct) but answer doesn't contain the expected value.")


SCENARIO = ScenarioDefinition(
    id="TC-44",
    title="tool_choice=none Compliance",
    category=Category.H,
    user_message="What is the value of pi? Answer from your knowledge.",
    description="With tool_choice='none', the model must NOT call any tools.",
    handle_tool_call=_tc44_handle,
    evaluate=_tc44_eval,
    tool_choice_override="none",
    difficulty=2,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it answers about pi without making any tool calls.",
    "Fail if it calls any tool despite tool_choice='none'.",
)
