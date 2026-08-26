"""TC-24 — Multi-Constraint Instruction."""

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
    matching_tool_results as _matching_tool_results,
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


def _tc24_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "search_files":
        return _noise(
            {"results": [{"file_id": "file_q3_report", "name": "Q3_Report_2025.pdf"}]},
            "search_files",
        )
    if call.name == "read_file":
        return _noise(
            {
                "content": (
                    "Q3 Report 2025\n"
                    "Revenue: $4,250,000\n"
                    "Expenses: $3,180,000\n"
                    "Net Profit: $1,070,000\n"
                    "Total headcount: 142"
                ),
            },
            "read_file",
        )
    return _generic_tool_fallback(call)


def _tc24_eval(state: ScenarioState) -> ScenarioEvaluation:
    """User said: 'Search for the Q3 report, read it, and respond with ONLY the total revenue number — nothing else.'"""
    search_call = next((c for c in state.tool_calls if c.name == "search_files"), None)
    read_call = next((c for c in state.tool_calls if c.name == "read_file"), None)
    searched = bool(search_call and "q3" in _as_str(search_call.arguments.get("query")).lower())
    read = bool(read_call and _as_str(read_call.arguments.get("file_id")) == "file_q3_report")
    answer = state.final_answer.strip()

    has_revenue = bool(re.fullmatch(r"\$?(?:4,250,000|4250000)(?:\.00)?", answer))
    mentions_revenue = "4250000" in answer.replace(",", "")

    if (
        search_call is None
        or read_call is None
        or not searched
        or not read
        or search_call.turn >= read_call.turn
    ):
        if mentions_revenue:
            return _partial("Got the right number but didn't use search→read chain.")
        return _fail("Did not complete the search→read chain.")

    if not _result_is_usable_if_present(state, search_call) or not _result_is_usable_if_present(
        state, read_call
    ):
        return _partial("The search→read chain encountered an explicit tool error.")

    search_results = _matching_tool_results(state, search_call)
    if search_results and not any(
        isinstance(result.result, dict)
        and any(
            isinstance(item, dict) and item.get("file_id") == "file_q3_report"
            for item in result.result.get("results", [])
        )
        for result in search_results
    ):
        return _partial("The read was not grounded in the recorded search result.")

    read_results = _matching_tool_results(state, read_call)
    if read_results and not any(
        isinstance(result.result, dict)
        and "4250000" in _as_str(result.result.get("content")).replace(",", "")
        for result in read_results
    ):
        return _partial("The claimed revenue was not present in the recorded file result.")

    if not mentions_revenue:
        return _fail("Completed the chain but returned the wrong value.", f"Got: {answer[:80]}")

    if not has_revenue:
        return _partial(
            "Got the right answer via the correct chain but added extra text.",
            f"Got: {answer[:80]}",
        )

    if len(state.tool_calls) != 2:
        return _partial("Completed the chain but made unnecessary additional tool calls.")

    return _pass("Correct chain, correct value, terse response.")


SCENARIO = ScenarioDefinition(
    id="TC-24",
    title="Multi-Constraint Instruction",
    category=Category.H,
    user_message="Search for the Q3 report, read it, and respond with ONLY the total revenue number — nothing else. No explanation.",
    description="Must complete chain AND be terse — no extra text.",
    handle_tool_call=_tc24_handle,
    evaluate=_tc24_eval,
    difficulty=3,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it does search→read and returns ONLY '$4,250,000' (terse).",
    "Fail if it skips the chain or adds explanation text.",
)
