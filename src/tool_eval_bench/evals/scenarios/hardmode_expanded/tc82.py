"""TC-82 — Stale Memory Conflict Resolution."""

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
    full_assistant_transcript,
    has_tool_call,
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
    _EMAIL,
    _STRING,
    _result_matches_if_present,
    _tool,
)

_TC82_TOOLS = [
    _tool("get_memory", "Read stored user memory.", {"key": _STRING}, ["key"]),
    _tool(
        "lookup_directory",
        "Look up current organization relationships.",
        {"query": _STRING},
        ["query"],
    ),
    _tool("search_files", "Search files.", {"query": _STRING}, ["query"]),
    _tool(
        "send_email",
        "Send email.",
        {
            "to": _EMAIL,
            "subject": _STRING,
            "body": _STRING,
            "attachments": {"type": "array", "items": _STRING},
        },
        # `attachments` stays optional: the evaluator scores whether the model
        # chose to attach the checklist, so requiring it in the schema would
        # hand over the very behaviour under test.
        ["to", "subject", "body"],
    ),
]


def _tc82_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_memory":
        return _noise(
            {"name": "Morgan Lee", "email": "morgan.lee@company.com", "expired": True}, call.name
        )
    if call.name == "lookup_directory":
        if "manager" not in as_str(call.arguments.get("query")).lower():
            return _noise({"results": []}, call.name)
        return _noise(
            {"name": "Avery Patel", "email": "avery.patel@company.com", "relationship": "manager"},
            call.name,
        )
    if call.name == "search_files":
        return _noise(
            {"results": [{"file_id": "launch_checklist", "name": "Launch_Checklist.pdf"}]},
            call.name,
        )
    if call.name == "send_email":
        return _noise({"status": "sent"}, call.name)
    return _noise({"error": f"Tool {call.name} is not relevant."}, call.name)


def _tc82_eval(state: ScenarioState) -> ScenarioEvaluation:
    directories = [
        c
        for c in tool_calls_by_name(state, "lookup_directory")
        if "manager" in as_str(c.arguments.get("query")).lower()
        and result_is_usable_if_present(state, c)
        and _result_matches_if_present(
            state,
            c,
            lambda payload: (
                payload.get("name") == "Avery Patel"
                and payload.get("email") == "avery.patel@company.com"
                and payload.get("relationship") == "manager"
            ),
        )
    ]
    searches = [
        c
        for c in tool_calls_by_name(state, "search_files")
        if "checklist" in as_str(c.arguments.get("query")).lower()
        and result_is_usable_if_present(state, c)
        and _result_matches_if_present(
            state,
            c,
            lambda payload: any(
                isinstance(item, dict) and item.get("file_id") == "launch_checklist"
                for item in payload.get("results", [])
            ),
        )
    ]
    directory = bool(directories)
    emails = tool_calls_by_name(state, "send_email")
    sent_avery = (
        len(emails) == 1
        and emails[0].arguments.get("to") == "avery.patel@company.com"
        and set(emails[0].arguments.get("attachments", [])) == {"launch_checklist"}
        and as_str(emails[0].arguments.get("subject")).strip()
        and as_str(emails[0].arguments.get("body")).strip()
        and result_is_usable_if_present(state, emails[0])
        and _result_matches_if_present(
            state, emails[0], lambda payload: payload.get("status") == "sent"
        )
    )
    ordered = bool(
        directories
        and searches
        and emails
        and max(directories[0].turn, searches[0].turn) < emails[0].turn
    )
    if directory and sent_avery and ordered:
        return _pass("Verified the current manager and sent the checklist only to Avery.")
    if directory and not emails and asks_for_clarification(full_assistant_transcript(state)):
        return _partial(
            "Verified Avery as current manager but asked for unnecessary clarification."
        )
    if has_tool_call(state, "lookup_directory"):
        return _partial("Looked up the directory, but did not verify the manager relationship.")
    return _fail("Trusted stale manager memory or sent the checklist to the wrong recipient.")


SCENARIO = ScenarioDefinition(
    "TC-82",
    "Stale Memory Conflict Resolution",
    Category.P,
    "Send the launch checklist to my manager.",
    "Prefer authoritative current directory state over stale memory.",
    _tc82_handle,
    _tc82_eval,
    tools_override=_TC82_TOOLS,
    difficulty=5,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it sends only to the current manager.", "Fail if it trusts stale memory."
)
