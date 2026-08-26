"""TC-83 — Format-Sensitive Chained Summary."""

from __future__ import annotations

import json
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
    as_str,
    result_is_usable_if_present,
    tool_calls_by_name,
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
    _result_matches_if_present,
    _tool,
)

_TC83_TOOLS = [
    _tool("search_files", "Search files.", {"query": _STRING}, ["query"]),
    _tool("read_file", "Read a file.", {"file_id": _STRING}, ["file_id"]),
    _tool("get_stock_price", "Get stock price.", {"ticker": _STRING}, ["ticker"]),
]


_TC83_EXPECTED = {"quarter": "Q2", "revenue_usd": 1_250_000, "ticker": "ACME", "price_usd": 100.0}


def _tc83_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "search_files":
        return _noise(
            {"results": [{"file_id": "q2_revenue", "name": "Q2_Revenue.xlsx"}]}, call.name
        )
    if call.name == "read_file":
        return _noise({"quarter": "Q2", "revenue_usd": 1_250_000, "employee_count": 74}, call.name)
    if call.name == "get_stock_price":
        return _noise({"ticker": "ACME", "price_usd": 100.0, "change_percent": "+1.74%"}, call.name)
    return _noise({"error": f"Tool {call.name} is not relevant."}, call.name)


def _tc83_eval(state: ScenarioState) -> ScenarioEvaluation:
    searches = [
        c
        for c in tool_calls_by_name(state, "search_files")
        if "q2" in as_str(c.arguments.get("query")).lower()
        and "revenue" in as_str(c.arguments.get("query")).lower()
        and result_is_usable_if_present(state, c)
        and _result_matches_if_present(
            state,
            c,
            lambda payload: any(
                isinstance(item, dict) and item.get("file_id") == "q2_revenue"
                for item in payload.get("results", [])
            ),
        )
    ]
    reads = [
        c
        for c in tool_calls_by_name(state, "read_file")
        if c.arguments.get("file_id") == "q2_revenue"
        and result_is_usable_if_present(state, c)
        and _result_matches_if_present(
            state,
            c,
            lambda payload: (
                payload.get("quarter") == "Q2" and payload.get("revenue_usd") == 1_250_000
            ),
        )
    ]
    stocks = [
        c
        for c in tool_calls_by_name(state, "get_stock_price")
        if as_str(c.arguments.get("ticker")).upper() == "ACME"
        and result_is_usable_if_present(state, c)
        and _result_matches_if_present(
            state,
            c,
            lambda payload: payload.get("ticker") == "ACME" and payload.get("price_usd") == 100.0,
        )
    ]
    required_calls = bool(searches and reads and stocks and searches[0].turn < reads[0].turn)
    answer = state.final_answer.strip()
    # A code fence is stripped and not penalised, matching every other JSON
    # evaluator in the suite. This scenario grades the chained extraction, so
    # scoring a markdown habit here would measure chat tuning instead.
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", answer, re.DOTALL)
    if fenced:
        answer = fenced.group(1)
    try:
        data = json.loads(answer)
    except json.JSONDecodeError:
        return _fail("Output is not valid JSON.")
    if not required_calls or not isinstance(data, dict):
        return _fail("Missing required tool calls or JSON object output.")
    values_ok = all(data.get(key) == value for key, value in _TC83_EXPECTED.items())
    if values_ok and set(data) == set(_TC83_EXPECTED):
        return _pass("Returned exact required JSON after the chained lookups.")
    if values_ok:
        return _partial("Returned the correct values with extra keys.")
    return _fail("Mixed noisy metadata into the required JSON values.")


SCENARIO = ScenarioDefinition(
    "TC-83",
    "Format-Sensitive Chained Summary",
    Category.P,
    "Read the Q2 revenue file and current ACME stock price. Return only JSON with keys quarter, revenue_usd, ticker, and price_usd.",
    "Return exact JSON after chained extraction from noisy payloads.",
    _tc83_handle,
    _tc83_eval,
    tools_override=_TC83_TOOLS,
    difficulty=5,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it emits exact JSON after chained lookups.",
    "Fail if noisy metadata leaks into values.",
)
