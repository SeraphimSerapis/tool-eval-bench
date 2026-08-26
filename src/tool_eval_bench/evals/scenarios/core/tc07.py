"""TC-07 — Search → Read → Act."""

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
    answer_affirms_number as _answer_affirms_number,
)
from tool_eval_bench.evals.helpers import (
    as_str as _as_str,
)
from tool_eval_bench.evals.helpers import (
    fail_eval as _fail,
)
from tool_eval_bench.evals.helpers import (
    generic_tool_fallback as _generic_tool_fallback,
)
from tool_eval_bench.evals.helpers import (
    includes_text as _includes_text,
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
    _result_matches_if_present,
    _tc03_email_result_is_sent,
)


def _tc07_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "search_files":
        return _noise(
            {"results": [{"file_id": "file_091", "name": "Q3_Budget_Report_2025.xlsx"}]},
            "search_files",
        )
    if call.name == "read_file":
        return _noise(
            {
                "content": "Department budgets: Engineering $2.1M, Marketing $800K, Sales $1.5M. Total: $4.4M"
            },
            "read_file",
        )
    if call.name == "get_contacts":
        return _noise(
            {
                "results": [
                    {"name": "Jordan Park", "email": "jordan.park@company.com", "role": "manager"}
                ]
            },
            "get_contacts",
        )
    if call.name == "send_email":
        return _noise({"status": "sent"}, "send_email")
    return _generic_tool_fallback(call)


def _tc07_search_result_has_report(payload: Any) -> bool:
    """Return whether search output identifies the requested budget report."""
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        return False
    return any(
        isinstance(item, dict)
        and (
            _normalize(_as_str(item.get("file_id"))) == "file_091"
            or (
                _includes_text(item.get("name"), "q3")
                and _includes_text(item.get("name"), "budget")
            )
        )
        for item in payload["results"]
    )


def _tc07_read_result_has_total(payload: Any) -> bool:
    """Return whether file output contains the report total used in the email."""
    return isinstance(payload, dict) and _includes_text(payload.get("content"), "4.4m")


def _tc07_contact_result_has_manager(payload: Any) -> bool:
    """Return whether contact output identifies the requested manager."""
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        return False
    return any(
        isinstance(item, dict)
        and _normalize(_as_str(item.get("email"))) == "jordan.park@company.com"
        for item in payload["results"]
    )


def _tc07_eval(state: ScenarioState) -> ScenarioEvaluation:
    search_calls = _tool_calls_by_name(state, "search_files")
    read_calls = [
        c
        for c in _tool_calls_by_name(state, "read_file")
        if _normalize(_as_str(c.arguments.get("file_id"))) == "file_091"
    ]
    contact_calls = [
        c
        for c in _tool_calls_by_name(state, "get_contacts")
        if _positive_argument_contains(c.arguments.get("query"), "manager")
    ]
    email_calls = [
        c
        for c in _tool_calls_by_name(state, "send_email")
        if (
            _normalize(_as_str(c.arguments.get("to"))) == "jordan.park@company.com"
            and (_answer_affirms_number(_as_str(c.arguments.get("body")), "4.4"))
        )
    ]

    semantic_search_calls = [
        c
        for c in search_calls
        if (
            _positive_argument_contains(c.arguments.get("query"), "q3")
            and _positive_argument_contains(c.arguments.get("query"), "budget")
        )
    ]

    # The search handler resolves any query to file_091, so a search followed
    # by a read of that file is handler-resolved evidence. Keep the candidate
    # calls here, then validate the dependency graph using those same calls.
    search_candidates = [
        call
        for call in search_calls
        if call in semantic_search_calls or (search_calls and read_calls)
    ]
    usable_search_calls = [
        call
        for call in search_candidates
        if _result_matches_if_present(state, call, _tc07_search_result_has_report)
    ]
    usable_read_calls = [
        call
        for call in read_calls
        if _result_matches_if_present(state, call, _tc07_read_result_has_total)
    ]
    usable_contact_calls = [
        call
        for call in contact_calls
        if _result_matches_if_present(state, call, _tc07_contact_result_has_manager)
    ]
    usable_email_calls = [
        call
        for call in email_calls
        if _result_matches_if_present(state, call, _tc03_email_result_is_sent)
    ]
    steps = sum(
        (
            bool(search_candidates),
            bool(read_calls),
            bool(contact_calls),
            bool(email_calls),
        )
    )
    if steps == 4:
        if len(_tool_calls_by_name(state, "send_email")) != 1:
            return _partial(
                "Found the chain but issued an unsafe duplicate or incorrect email mutation."
            )
        # Dependency graph instead of one total order: the file read depends
        # on the search result, and both the read and the contact lookup must
        # complete before the email is sent. read and contacts are independent
        # and may be issued in either relative order. Use the calls that
        # satisfied both the argument and result checks above.
        dependency_satisfied = any(
            search.turn < read.turn < email.turn and contacts.turn < email.turn
            for search in usable_search_calls
            for read in usable_read_calls
            for contacts in usable_contact_calls
            for email in usable_email_calls
        )
        if not dependency_satisfied:
            if (
                usable_search_calls
                and usable_read_calls
                and usable_contact_calls
                and usable_email_calls
            ):
                return _partial("Found all chain steps, but used them out of dependency order.")
            return _partial(
                "Found all chain steps, but a required tool result was unusable or missing."
            )
        return _pass("Completed the full four-step chain with the right data.")
    if steps >= 3:
        return _partial("Completed most of the chain, but missed one dependent step.")
    return _fail("Did not carry the file and contact data across the chain correctly.")


SCENARIO = ScenarioDefinition(
    id="TC-07",
    title="Search → Read → Act",
    category=Category.C,
    user_message="Find the Q3 budget report and email the total to my manager.",
    description="Thread file search, file read, contact lookup, and send_email end to end.",
    handle_tool_call=_tc07_handle,
    evaluate=_tc07_eval,
    difficulty=3,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it searches, reads, resolves the manager, and emails the real total.",
    "Fail if it invents the total or the manager address.",
)
