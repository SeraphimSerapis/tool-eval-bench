"""TC-88 — Preserved Reasoning Across Follow-Ups."""

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
from tool_eval_bench.evals.helpers import fail_eval as _fail
from tool_eval_bench.evals.helpers import partial_eval as _partial
from tool_eval_bench.evals.helpers import pass_eval as _pass

_TC88_SUMS = (73, 91, 109)


_TC88_NUMBER = re.compile(r"(?<!\d)\d{20}(?!\d)")


def _tc88_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    return {"error": f"Tool use is forbidden in this scenario: {call.name}"}


def _tc88_eval(state: ScenarioState) -> ScenarioEvaluation:
    if state.tool_calls:
        return _fail("Used a tool despite the no-tools instruction.")
    if len(state.assistant_messages) != 3:
        return _fail("Did not provide one answer for each of the three user turns.")
    numbers = [message.strip() for message in state.assistant_messages]
    if state.final_answer.strip() != numbers[-1]:
        return _fail("The final answer did not preserve the third follow-up value.")
    if not all(re.fullmatch(r"[1-9]\d{19}", number) for number in numbers):
        return _fail("Returned extra text or a value that was not exactly 20 digits.")
    if len(set(numbers)) != 3:
        return _fail("Repeated a value instead of preserving three distinct choices.")
    if tuple(sum(int(digit) for digit in number) for number in numbers) != _TC88_SUMS:
        return _fail("A returned value did not satisfy its required digit sum.")
    if numbers[1][-6:] != numbers[0][:6][::-1] or numbers[2][-6:] != numbers[1][:6][::-1]:
        return _fail("A returned value broke the cross-turn reversal constraints.")

    first_reasoning = state.assistant_reasoning[0] if state.assistant_reasoning else ""
    planned_numbers = set(_TC88_NUMBER.findall(first_reasoning))
    if all(number in planned_numbers for number in numbers):
        return _pass("Preserved all three privately planned values across two user follow-ups.")
    # PASS is only reachable when the provider surfaces a reasoning channel: the
    # proof this scenario grades is the plan itself, and most OpenAI-compatible
    # endpoints emit no ``reasoning_content`` at all. Say so, rather than
    # reporting a stack's transport as something the model failed to do. The
    # constraints themselves stay verifiable from the visible transcript, so a
    # model on such a stack still earns the middle tier on its own merit.
    if not any(reasoning.strip() for reasoning in state.assistant_reasoning):
        return _partial(
            "Returned three valid values. This endpoint exposed no reasoning channel, "
            "so the planned-ahead check could not run and PASS was unreachable here."
        )
    return _partial(
        "Returned three valid values, but the provider exposed no proof that all were planned first."
    )


SCENARIO = ScenarioDefinition(
    "TC-88",
    "Preserved Reasoning Across Follow-Ups",
    Category.P,
    "Privately construct three distinct 20-digit numbers with no leading zero. Their digit sums must be 73, 91, and 109 in that order. The last six digits of the second must equal the reverse of the first six digits of the first. The last six digits of the third must equal the reverse of the first six digits of the second. Verify every length, sum, and reversal in your reasoning. Do not use tools. Reply with only the first number and nothing else.",
    "Preserve provider-exposed private planning across two user follow-ups.",
    _tc88_handle,
    _tc88_eval,
    follow_up_messages=[
        "Now reply with only the second number you constructed and nothing else.",
        "Now reply with only the third number you constructed and nothing else.",
    ],
    tools_override=[],
    difficulty=5,
    max_turns_override=3,
    preserve_reasoning_across_follow_ups=True,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if all three constrained values were present in the first exposed reasoning payload.",
    "Fail if values change, violate constraints, include extra text, or use tools.",
)
