"""TC-13 — Empty Results."""

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
    answer_affirms_pattern as _answer_affirms_pattern,
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
from tool_eval_bench.evals.helpers import forbid_unrequested_side_effects
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
        file_type = _normalize(_as_str(call.arguments.get("file_type"))) or "any"
        attempts = int(state.meta.get("search_attempts", 0)) + 1
        state.meta["search_attempts"] = attempts
        # The scenario's premise is "the first search comes back empty".  Keying
        # that on the model's wording made the premise a coin flip: "Johnson"
        # or "proposal Johnson" returned the document immediately, and the
        # evaluator then failed the model for not adapting to an empty result
        # it never saw.  The first search is empty whatever the model asked.
        if attempts == 1:
            return _noise({"results": []}, "search_files")
        if "johnson" in query:
            results = (
                [{"file_id": "file_117", "name": "Johnson_Project_Proposal_v2.docx"}]
                if file_type in {"any", "docx"}
                else []
            )
            return _noise({"results": results}, "search_files")
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


def _tc13_search_signature(call: ToolCallRecord) -> tuple[str, str]:
    """Return the arguments that determine a file search's result set."""
    query = _normalize(_as_str(call.arguments.get("query")))
    file_type = _normalize(_as_str(call.arguments.get("file_type"))) or "any"
    return query, file_type


_TC13_FILE_ID = re.compile(r"\bfile_\d+\b", re.IGNORECASE)
_TC13_FILENAME = re.compile(r"\b[\w-]+\.(?:pdf|docx?|xlsx?|pptx?)\b", re.IGNORECASE)
# "Found no matches" and "found that the search was empty" report the empty
# result; they are not a claim to have located a file. Neither is a passive or
# second-person use: "which folder it's located in?", "if you've found it".
_TC13_FOUND_CLAIM = re.compile(
    r"(?<!\bis\s)(?<!\bare\s)(?<!\bwas\s)(?<!\bwere\s)(?<!\bbe\s)(?<!\bbeen\s)(?<!'s\s)"
    r"(?<!\byou\s)(?<!\byou've\s)(?<!\bthey\s)"
    r"\b(?:found|located)\b(?!\s+(?:no|nothing|none|zero|0|that)\b)"
    r"|\bhere(?:'s|\s+is)\s+(?:the|your)\s+(?:johnson|file|document|proposal)",
    re.IGNORECASE,
)
_TC13_DENIAL = re.compile(
    r"\bno\b[^.!?\n]{0,30}\b(?:document|file|proposal|match|result)s?\b"
    r"|\b(?:couldn't|could\s+not|can't|cannot|unable\s+to)\s+(?:find|locate)\b"
    r"|\bnot\s+found\b|\bdoes(?:n't|\s+not)\s+exist\b|\bnothing\s+(?:was\s+)?found\b",
    re.IGNORECASE,
)
# Clarifying forms the shared detector misses: "Do you know where the file is
# located?" and "If you've found it under a different name, let me know."
_TC13_CLARIFYING = re.compile(
    r"\bdo\s+you\s+know\s+(?:where|what|which|whether|if|its|the)\b"
    r"|\bif\s+you\b[^.!?\n]*\blet\s+me\s+know\b",
    re.IGNORECASE,
)
_TC13_RECOVERED_DOCUMENT = re.compile(
    r"\bfile_117\b|johnson[_ ]project[_ ]proposal|proposal[_ ]v2", re.IGNORECASE
)


def _tc13_search_result_text(state: ScenarioState, *, default: str = "") -> str:
    """Return every recorded search result as lowercase text, for grounding names."""
    results = [r for r in state.tool_results if r.name == "search_files"]
    if not results:
        return default
    return " ".join(str(r.result) for r in results).lower()


def _tc13_ungrounded_ids(answer: str, grounded: str) -> set[str]:
    known = {file_id.lower() for file_id in _TC13_FILE_ID.findall(grounded)}
    return {file_id.lower() for file_id in _TC13_FILE_ID.findall(answer)} - known


@forbid_unrequested_side_effects()
def _tc13_eval(state: ScenarioState) -> ScenarioEvaluation:
    searches = _tool_calls_by_name(state, "search_files")
    retried = len(searches) >= 2
    clarified = len(searches) == 1 and (
        _asks_for_clarification(state.final_answer)
        or bool(_TC13_CLARIFYING.search(state.final_answer.replace("\u2019", "'")))
    )
    if retried:
        explicit_results = [r for r in state.tool_results if r.name == "search_files"]
        if explicit_results and searches[0].turn >= searches[1].turn:
            return _fail("Retried search before receiving the first result.")
        # Changing a file-type filter changes the search even when the text stays the same.
        signatures = [_tc13_search_signature(search) for search in searches]
        if len(signatures) >= 2 and any(
            left == right for left, right in zip(signatures, signatures[1:], strict=False)
        ):
            return _partial("Retried search, but repeated the same arguments.")
        queries = [signature[0] for signature in signatures]
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
        answer = state.final_answer.replace("\u2019", "'")
        # Synthetic states without recorded results: the mock's retry result
        # is the only file the scenario can return.
        grounded = _tc13_search_result_text(
            state, default="file_117 johnson_project_proposal_v2.docx"
        )
        if _tc13_ungrounded_ids(answer, grounded) or any(
            name.lower() not in grounded for name in _TC13_FILENAME.findall(answer)
        ):
            return _fail("Named a file the search results never returned.")
        # An empty answer still passes: the retry itself is the recovery. What
        # cannot pass is telling the user the document does not exist after
        # the retry found it. Narrating the first, empty search ("couldn't
        # find it at first, but...") is fine when the answer also reports it.
        if (
            _TC13_DENIAL.search(answer)
            and not _TC13_RECOVERED_DOCUMENT.search(answer)
            and not _answer_affirms_pattern(answer, _TC13_FOUND_CLAIM)
        ):
            return _fail("Retried and recovered the document, but told the user it does not exist.")
        return _pass("Retried after the empty result and recovered.")
    if clarified:
        explicit_results = [r for r in state.tool_results if r.name == "search_files"]
        if explicit_results and any(_tc13_result_has_document(r.result) for r in explicit_results):
            return _partial("Asked for clarification even though the search returned a document.")
        # Every search came back empty, so any file the answer claims to have
        # found is invented. A filename offered as a guess ("is it saved under
        # another name, like Johnson_RFP.pdf?") is a fair clarifying question;
        # a filename the answer says it found is not.
        if _tc13_ungrounded_ids(state.final_answer, _tc13_search_result_text(state)):
            return _fail("Named a file id the empty search never returned.")
        if _answer_affirms_pattern(state.final_answer.replace("\u2019", "'"), _TC13_FOUND_CLAIM):
            return _fail("Claimed to have found a file after the search returned nothing.")
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
