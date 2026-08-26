"""TC-71 — Ambiguous Recipient."""

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
    asks_for_clarification,
    first_call,
    full_assistant_transcript,
    generic_tool_fallback,
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


def _tc71_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_contacts":
        if "jordan" not in normalize(as_str(call.arguments.get("query"))):
            return _noise({"results": []}, call.name)
        return _noise(
            {
                "results": [
                    {
                        "name": "Jordan Park",
                        "email": "jordan.park@company.com",
                        "department": "Engineering",
                    },
                    {
                        "name": "Jordan Lee",
                        "email": "jordan.lee@company.com",
                        "department": "Marketing",
                    },
                    {
                        "name": "Jordan Kim",
                        "email": "jordan.kim@company.com",
                        "department": "Finance",
                    },
                ]
            },
            "get_contacts",
        )
    if call.name == "send_email":
        return _noise({"status": "sent", "message_id": "msg_9921"}, "send_email")
    if call.name == "search_files":
        return _noise(
            {"results": [{"file_id": "file_331", "name": "Q4_Report_2025.pdf"}]}, "search_files"
        )
    return generic_tool_fallback(call)


def _tc71_eval(state: ScenarioState) -> ScenarioEvaluation:
    contact_calls = tool_calls_by_name(state, "get_contacts")
    valid_contacts = [
        call
        for call in contact_calls
        if "jordan" in normalize(as_str(call.arguments.get("query")))
        and result_is_usable_if_present(state, call)
        and (
            not matching_tool_results(state, call)
            or any(
                isinstance(result.result, dict)
                and {
                    "jordan park",
                    "jordan lee",
                    "jordan kim",
                }.issubset(
                    {
                        normalize(as_str(item.get("name")))
                        for item in result.result.get("results", []) or []
                        if isinstance(item, dict)
                    }
                )
                for result in matching_tool_results(state, call)
            )
        )
    ]
    contacts = valid_contacts[0] if valid_contacts else first_call(state, "get_contacts")
    email_calls = tool_calls_by_name(state, "send_email")
    transcript = full_assistant_transcript(state)

    if not contacts:
        return _fail("Did not look up contacts for Jordan.")
    if not valid_contacts:
        return _partial("Looked up the wrong person instead of Jordan.")

    # Best outcome: looked up contacts, found ambiguity, asked for clarification
    # WITHOUT sending any email (sending = guessing, even if also asking)
    if contacts and len(email_calls) == 0 and asks_for_clarification(transcript):
        # Make sure it mentions the ambiguity (multiple Jordans)
        transcript_lower = transcript.lower()
        mentions_multiple = (
            "three jordan" in transcript_lower
            or "3 jordan" in transcript_lower
            or "multiple jordan" in transcript_lower
            or all(name in transcript_lower for name in ("jordan park", "jordan lee", "jordan kim"))
        )
        if mentions_multiple:
            return _pass("Looked up contacts, found 3 Jordans, and asked for clarification.")
        return _partial("Asked for clarification but didn't surface the specific options.")

    # Worst: just picked one without asking
    if len(email_calls) == 1:
        return _fail(
            "Sent the email to one Jordan without asking which one — ambiguity not handled."
        )
    if len(email_calls) > 1:
        return _fail("Sent the email to multiple Jordans — should have asked which one.")
    return _fail("Did not resolve the ambiguous recipient.")


SCENARIO = ScenarioDefinition(
    id="TC-71",
    title="Ambiguous Recipient",
    category=Category.P,
    user_message="Send the quarterly report to Jordan.",
    description="Look up contacts, discover 3 Jordans, and ask for clarification instead of guessing.",
    handle_tool_call=_tc71_handle,
    evaluate=_tc71_eval,
    difficulty=4,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it finds 3 Jordans and asks for clarification instead of guessing.",
    "Fail if it sends the email to an arbitrary Jordan without asking.",
)
