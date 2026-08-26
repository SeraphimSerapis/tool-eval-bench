"""TC-29 — Explain Without Executing."""

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
    generic_tool_fallback_simple as _generic_tool_fallback,
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


def _tc29_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "run_code":
        return _noise({"stdout": "[0, 1, 4, 9, 16]", "stderr": "", "exit_code": 0}, "run_code")
    if call.name == "web_search":
        return _noise(
            {"results": [{"snippet": "List comprehension in Python creates a new list."}]},
            "web_search",
        )
    return _generic_tool_fallback(call)


def _tc29_eval(state: ScenarioState) -> ScenarioEvaluation:
    """User: 'What does this Python code do: [x**2 for x in range(5)]?'

    Should answer directly. Partial if it uses run_code. Fail if web_search.
    """
    used_run_code = any(c.name == "run_code" for c in state.tool_calls)
    used_web = any(c.name == "web_search" for c in state.tool_calls)
    answer = state.final_answer

    correct_output = any(v in answer for v in ("[0, 1, 4, 9, 16]", "0, 1, 4, 9, 16"))
    low_answer = answer.lower()
    explains_comprehension = (
        "list comprehension" in low_answer
        and any(word in low_answer for word in ("list", "create", "generat", "produce"))
        and any(word in low_answer for word in ("squar", "range", "0", "1", "4", "9", "16"))
    ) or bool(re.search(r"(?:each|every)\s+(?:number|integer|value).{0,40}squar", low_answer))

    if used_web:
        return _fail("Used web_search for a basic Python question.")

    if used_run_code:
        if correct_output:
            return _partial("Got the right answer but unnecessarily executed the code.")
        return _fail("Executed the code but still gave wrong output.")

    if state.tool_calls:
        return _fail("Used an unrelated tool for a code explanation that needed no tools.")

    if correct_output or explains_comprehension:
        return _pass("Correctly explained the code without using any tools.")
    return _fail("Did not explain the code correctly.")


SCENARIO = ScenarioDefinition(
    id="TC-29",
    title="Explain Without Executing",
    category=Category.J,
    user_message="What does this Python code do: [x**2 for x in range(5)]?",
    description="Should explain directly without executing the code.",
    handle_tool_call=_tc29_handle,
    evaluate=_tc29_eval,
    difficulty=3,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it explains [0,1,4,9,16] directly without tools.",
    "Fail if it web-searches for a basic Python question.",
)
