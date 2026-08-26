"""TC-72 — Cascading Error Recovery."""

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
    as_str,
    generic_tool_fallback,
    has_explicit_tool_error,
    includes_text,
    matching_tool_results,
    normalize,
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
from tool_eval_bench.evals.scenarios.hardmode._shared import (
    _UNRELATED_UNIVERSAL_MUTATIONS,
)


def _tc72_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "search_files":
        return _noise(
            {
                "results": [
                    {"file_id": "file_404", "name": "Project_Alpha_Summary.docx"},
                    {"file_id": "file_405", "name": "Project_Alpha_Summary_Backup.docx"},
                ]
            },
            "search_files",
        )
    if call.name == "read_file":
        fid = normalize(as_str(call.arguments.get("file_id")))
        if fid == "file_404":
            return _noise({"error": "File is corrupted. Unable to read."}, "read_file")
        if fid == "file_405":
            return _noise(
                {
                    "content": "Project Alpha summary backup: approved budget totals $1.2M. Engineering: $750K, Marketing: $450K."
                },
                "read_file",
            )
        return _noise({"error": f"File {fid} not found."}, "read_file")
    if call.name == "send_email":
        return _noise({"status": "sent", "message_id": "msg_7733"}, "send_email")
    if call.name == "get_contacts":
        return _noise(
            {"results": [{"name": "Dana Wells", "email": "dana.wells@company.com"}]}, "get_contacts"
        )
    return generic_tool_fallback(call)


def _tc72_eval(state: ScenarioState) -> ScenarioEvaluation:
    if any(
        call.name in {"create_calendar_event", *_UNRELATED_UNIVERSAL_MUTATIONS}
        for call in state.tool_calls
    ):
        return _fail("Performed an unrelated side effect during file recovery.")

    searches = tool_calls_by_name(state, "search_files")
    read_calls = tool_calls_by_name(state, "read_file")
    contact_calls = tool_calls_by_name(state, "get_contacts")
    email_calls = tool_calls_by_name(state, "send_email")
    email_call = email_calls[0] if len(email_calls) == 1 else None

    read_404_calls = [
        c
        for c in read_calls
        if normalize(as_str(c.arguments.get("file_id"))) == "file_404"
        and (not matching_tool_results(state, c) or has_explicit_tool_error(state, c))
    ]
    read_405_calls = [
        c
        for c in read_calls
        if normalize(as_str(c.arguments.get("file_id"))) == "file_405"
        and result_is_usable_if_present(state, c)
        and (
            not matching_tool_results(state, c)
            or any(
                isinstance(result.result, dict)
                and "1.2m" in as_str(result.result.get("content")).lower()
                and "750k" in as_str(result.result.get("content")).lower()
                for result in matching_tool_results(state, c)
            )
        )
    ]
    tried_404 = bool(read_404_calls)
    tried_405 = bool(read_405_calls)
    email_has_budget = email_call and (
        includes_text(email_call.arguments.get("body"), "1.2m")
        or includes_text(email_call.arguments.get("body"), "$1.2")
        or includes_text(email_call.arguments.get("body"), "750")
    )

    valid_searches = [
        c
        for c in searches
        if "alpha" in as_str(c.arguments.get("query")).lower()
        and result_is_usable_if_present(state, c)
        and (
            not matching_tool_results(state, c)
            or any(
                isinstance(result.result, dict)
                and {
                    "file_404",
                    "file_405",
                }.issubset(
                    {
                        as_str(item.get("file_id"))
                        for item in result.result.get("results", []) or []
                        if isinstance(item, dict)
                    }
                )
                for result in matching_tool_results(state, c)
            )
        )
    ]
    valid_contacts = [
        c
        for c in contact_calls
        if "dana" in as_str(c.arguments.get("query")).lower()
        and result_is_usable_if_present(state, c)
        and (
            not matching_tool_results(state, c)
            or any(
                isinstance(result.result, dict)
                and any(
                    normalize(as_str(item.get("email"))) == "dana.wells@company.com"
                    for item in result.result.get("results", []) or []
                    if isinstance(item, dict)
                )
                for result in matching_tool_results(state, c)
            )
        )
    ]
    searched_alpha = bool(valid_searches)
    resolved_dana = bool(valid_contacts)
    recovery_ordered = bool(
        valid_searches
        and any(
            search.turn < corrupted.turn < backup.turn
            for search in valid_searches
            for corrupted in read_404_calls
            for backup in read_405_calls
        )
    )
    safe_emails = bool(email_call) and all(
        normalize(as_str(c.arguments.get("to"))) == "dana.wells@company.com"
        and result_is_usable_if_present(state, c)
        for c in email_calls
    )
    if email_calls and not safe_emails:
        return _fail(
            "Sent duplicate recovery emails or disclosed the budget to the wrong recipient."
        )
    dependencies_ordered = bool(
        valid_searches
        and valid_contacts
        and email_call
        and recovery_ordered
        and max(valid_searches[0].turn, valid_contacts[0].turn) < email_call.turn
        and any(c.turn < email_call.turn for c in read_405_calls)
    )
    if (
        searched_alpha
        and resolved_dana
        and tried_404
        and tried_405
        and email_has_budget
        and safe_emails
        and dependencies_ordered
    ):
        return _pass(
            "Recovered from corrupted file by trying the alternative, then emailed the budget."
        )
    if tried_405 and email_has_budget and safe_emails:
        return _partial("Found the budget and emailed Dana, but skipped a discovery dependency.")
    if tried_404 and tried_405 and not email_has_budget:
        return _partial(
            "Recovered from the error and read the file, but didn't complete the email chain."
        )
    if tried_404 and not tried_405:
        return _fail("Hit the corrupted file error but did not try the alternative file.")
    return _fail("Did not complete the error recovery chain.")


SCENARIO = ScenarioDefinition(
    id="TC-72",
    title="Cascading Error Recovery",
    category=Category.P,
    user_message="Find the Project Alpha summary, read it, and email the key details to Dana.",
    description="Recover from a corrupted file by trying the alternative, then complete the email chain.",
    handle_tool_call=_tc72_handle,
    evaluate=_tc72_eval,
    difficulty=4,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it recovers from the corrupted file and emails the budget data.",
    "Fail if it stops after the first error without trying alternatives.",
)
