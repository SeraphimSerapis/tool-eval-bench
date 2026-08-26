"""TC-13 — Empty Results."""

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
    fail_eval as _fail,
)
from tool_eval_bench.evals.helpers import (
    generic_tool_fallback as _generic_tool_fallback,
)
from tool_eval_bench.evals.helpers import (
    normalize as _normalize,
)
from tool_eval_bench.evals.helpers import (
    partial_eval as _partial,
)
from tool_eval_bench.evals.helpers import (
    pass_eval as _pass,
)
from tool_eval_bench.evals.helpers import (
    tool_calls_by_name as _tool_calls_by_name,
)
from tool_eval_bench.evals.helpers import (
    with_noise as _noise,
)
from tool_eval_bench.evals.scenarios.core._shared import (
    _positive_argument_contains,
)


def _tc13_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "search_files":
        query = _normalize(_as_str(call.arguments.get("query")))
        attempts = int(state.meta.get("search_attempts", 0)) + 1
        state.meta["search_attempts"] = attempts
        if "johnson proposal" in query and attempts == 1:
            return _noise({"results": []}, "search_files")
        if "johnson" in query:
            return _noise(
                {"results": [{"file_id": "file_117", "name": "Johnson_Project_Proposal_v2.docx"}]},
                "search_files",
            )
    return _generic_tool_fallback(call)


def _tc13_result_has_document(result: Any) -> bool:
    """Return whether a structured search result contains the target document."""
    unsuccessful_statuses = {"error", "failed", "blocked", "cancelled", "canceled"}
    if (
        not isinstance(result, dict)
        or "error" in result
        or _normalize(_as_str(result.get("status"))) in unsuccessful_statuses
        or not isinstance(result.get("results"), list)
    ):
        return False
    for item in result["results"]:
        if not isinstance(item, dict):
            continue
        file_id = _normalize(_as_str(item.get("file_id")))
        name = _normalize(_as_str(item.get("name")))
        if file_id == "file_117" or name == "johnson_project_proposal_v2.docx":
            return True
    return False


def _tc13_eval(state: ScenarioState) -> ScenarioEvaluation:
    searches = _tool_calls_by_name(state, "search_files")
    retried = len(searches) >= 2
    clarified = len(searches) == 1 and _asks_for_clarification(state.final_answer)
    if retried:
        explicit_results = [r for r in state.tool_results if r.name == "search_files"]
        if explicit_results and searches[0].turn >= searches[1].turn:
            return _fail("Retried search before receiving the first result.")
        # Ensure the retry used a meaningfully different query (not just repeating)
        queries = [_normalize(_as_str(s.arguments.get("query", ""))) for s in searches]
        if len(queries) >= 2 and any(
            left == right for left, right in zip(queries, queries[1:], strict=False)
        ):
            return _partial("Retried search, but used the exact same query both times.")
        if not any(_positive_argument_contains(query, "johnson") for query in queries[1:]):
            return _fail("Retried search without preserving the Johnson document request.")
        if explicit_results:
            # The first search is the seeded empty result; the document must
            # appear in at least one of the retry results.  A later failed
            # search must not erase an earlier successful retry.
            retry_call_ids = {call.id for call in searches[1:]}
            retry_results = [r for r in explicit_results if r.call_id in retry_call_ids]
            if not retry_results and len(explicit_results) > 1:
                # Compatibility for synthetic states that omit stable call IDs.
                retry_results = explicit_results[1:]
            first_results = [r for r in explicit_results if r.call_id == searches[0].id]
            if first_results and any(_tc13_result_has_document(r.result) for r in first_results):
                return _fail("The retry returned no recognized Johnson document.")
            if not any(_tc13_result_has_document(r.result) for r in retry_results):
                return _fail("The retry returned no recognized Johnson document.")
        return _pass("Retried after the empty result and recovered.")
    if clarified:
        explicit_results = [r for r in state.tool_results if r.name == "search_files"]
        if explicit_results and any(_tc13_result_has_document(r.result) for r in explicit_results):
            return _partial("Asked for clarification even though the search returned a document.")
        return _pass("Asked for clarification after the empty result.")
    return _fail("Did not adapt after the empty search response.")


SCENARIO = ScenarioDefinition(
    id="TC-13",
    title="Empty Results",
    category=Category.E,
    user_message="Find the Johnson proposal document.",
    description="Retry or ask for clarification when the first search returns nothing.",
    handle_tool_call=_tc13_handle,
    evaluate=_tc13_eval,
    difficulty=3,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it retries the search or asks for clarification after empty results.",
    "Fail if it gives up or invents a file.",
)
