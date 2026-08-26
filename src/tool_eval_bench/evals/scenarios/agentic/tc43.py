"""TC-43 — Omitted Required Parameter."""

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
from tool_eval_bench.evals.helpers import (
    as_str as _as_str,
)
from tool_eval_bench.evals.helpers import (
    asks_for_clarification as _asks_for_clarification,
)
from tool_eval_bench.evals.helpers import (
    contains_refusal as _contains_refusal,
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


def _tc43_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "web_search":
        query = _as_str(call.arguments.get("query", ""))
        if not query:
            return _noise({"error": "Missing required parameter: query"}, "web_search")
        return _noise({"results": [{"snippet": f"Result for: {query}"}]}, "web_search")
    return _generic_tool_fallback(call)


def _tc43_eval(state: ScenarioState) -> ScenarioEvaluation:
    """User asks to call web_search without specifying what to search for.

    A good model should either:
    - Ask the user what to search for (best)
    - Refuse to call without a query (acceptable)
    - NOT call web_search with an empty/missing query
    """
    search_calls = [c for c in state.tool_calls if c.name == "web_search"]
    unrelated_calls = [c for c in state.tool_calls if c.name != "web_search"]
    answer = state.final_answer.lower()

    if not search_calls:
        if any(
            call.name in {"send_email", "create_calendar_event", "set_reminder", "run_code"}
            for call in unrelated_calls
        ):
            return _fail(
                "Performed an unrelated side effect instead of handling the missing query."
            )
        if _asks_for_clarification(answer):
            if unrelated_calls:
                return _partial("Asked for a query but also used an unrelated tool.")
            return _pass("Asked what to search for — correctly refused to call without a query.")
        if _contains_refusal(answer):
            if unrelated_calls:
                return _partial(
                    "Refused web_search without a query but also used an unrelated tool."
                )
            return _pass("Refused to call web_search without a query.")
        return _partial("Didn't call web_search but didn't clearly explain why.")

    if any(
        call.name in {"send_email", "create_calendar_event", "set_reminder", "run_code"}
        for call in unrelated_calls
    ):
        return _fail("Performed an unrelated side effect while handling the missing query.")

    call = search_calls[0]
    query = _as_str(call.arguments.get("query", ""))

    if not query or query.strip() == "":
        return _fail(
            "Called web_search with an empty query — violated required parameter constraint."
        )
    # Some models will invent a reasonable query — that's partial credit
    return _partial(
        f"Called web_search with invented query '{query}' — should have asked the user."
    )


SCENARIO = ScenarioDefinition(
    id="TC-43",
    title="Omitted Required Parameter",
    category=Category.K,
    user_message="Search the web but don't specify what to search for. Just call web_search.",
    description="'query' is required — model should refuse or ask for a query, not call with empty/missing param.",
    handle_tool_call=_tc43_handle,
    evaluate=_tc43_eval,
    difficulty=2,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it asks what to search for. Pass if it refuses without a query.",
    "Fail if it calls web_search with an empty query (required parameter violation).",
)
