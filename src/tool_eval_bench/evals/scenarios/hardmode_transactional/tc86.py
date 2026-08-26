"""TC-86 — Optimistic Concurrency Without Lost Updates."""

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
from tool_eval_bench.evals.helpers import fail_eval as _fail
from tool_eval_bench.evals.helpers import partial_eval as _partial
from tool_eval_bench.evals.helpers import pass_eval as _pass
from tool_eval_bench.evals.helpers import (
    tool_calls_by_name,
    with_noise,
)
from tool_eval_bench.evals.scenarios.hardmode_transactional._shared import (
    _EMAIL,
    _STRING,
    _claims_success,
    _has_result,
    _tool,
)

_TC86_OWNER = "incident-owner@company.com"


_TC86_TOOLS = [
    _tool(
        "get_incident",
        "Read an incident and its version.",
        {"incident_id": _STRING},
        ["incident_id"],
    ),
    _tool(
        "update_incident",
        "Update an incident using optimistic concurrency. Preserve current fields and pass expected_version.",
        {
            "incident_id": _STRING,
            "expected_version": {"type": "integer"},
            "severity": _STRING,
            "assignee": _STRING,
            "tags": {"type": "array", "items": _STRING},
        },
        ["incident_id", "expected_version", "severity", "assignee", "tags"],
    ),
    _tool(
        "notify_owner",
        "Notify the incident owner after a confirmed update.",
        {"to": _EMAIL, "subject": _STRING, "body": _STRING},
        ["to", "subject", "body"],
    ),
]


def _tc86_incident(state: ScenarioState) -> dict[str, Any]:
    return state.meta.setdefault(
        "tc86_incident",
        {
            "incident_id": "INC-442",
            "version": 7,
            "severity": "P2",
            "assignee": "Ana",
            "tags": ["customer-impact"],
            "owner_email": _TC86_OWNER,
        },
    )


def _tc86_tags_match(value: Any, expected: tuple[str, ...]) -> bool:
    """Compare set-like tags exactly while rejecting duplicates and omissions."""
    return (
        isinstance(value, list)
        and all(isinstance(tag, str) for tag in value)
        and len(value) == len(expected)
        and len(set(value)) == len(value)
        and set(value) == set(expected)
    )


def _tc86_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    incident = _tc86_incident(state)
    if call.name == "get_incident":
        if call.arguments.get("incident_id") != "INC-442":
            return with_noise({"error": "Incident not found."}, call.name)
        return with_noise(dict(incident), call.name)
    if call.name == "update_incident":
        if call.arguments.get("incident_id") != "INC-442":
            return with_noise({"error": "Incident not found."}, call.name)
        expected = call.arguments.get("expected_version")
        conflict_count = int(state.meta.get("tc86_conflict_count", 0))
        if expected == 7 and conflict_count == 0:
            state.meta["tc86_conflict_count"] = 1
            incident.update(
                {
                    "version": 8,
                    "assignee": "Mika",
                    "tags": ["customer-impact", "database"],
                }
            )
            return with_noise(
                {"status": "conflict", "expected_version": 7, "current_version": 8},
                call.name,
            )
        if expected == 8 and incident["version"] == 8 and conflict_count == 1:
            state.meta["tc86_conflict_count"] = 2
            incident.update(
                {
                    "version": 9,
                    "tags": ["customer-impact", "database", "priority-customer"],
                }
            )
            return with_noise(
                {"status": "conflict", "expected_version": 8, "current_version": 9},
                call.name,
            )
        if expected != incident["version"]:
            return with_noise(
                {
                    "status": "conflict",
                    "expected_version": expected,
                    "current_version": incident["version"],
                },
                call.name,
            )
        incident.update(
            {
                "version": incident["version"] + 1,
                "severity": call.arguments.get("severity"),
                "assignee": call.arguments.get("assignee"),
                "tags": call.arguments.get("tags"),
            }
        )
        state.meta["tc86_successful_updates"] = (
            int(state.meta.get("tc86_successful_updates", 0)) + 1
        )
        return with_noise({"status": "updated", **incident}, call.name)
    if call.name == "notify_owner":
        return with_noise({"status": "sent", "message_id": "msg_tc86"}, call.name)
    return with_noise({"error": f"Unsupported tool {call.name}."}, call.name)


def _tc86_eval(state: ScenarioState) -> ScenarioEvaluation:
    reads = tool_calls_by_name(state, "get_incident")
    updates = tool_calls_by_name(state, "update_incident")
    notifications = tool_calls_by_name(state, "notify_owner")
    initial_reads = [
        call
        for call in reads
        if call.arguments.get("incident_id") == "INC-442"
        and _has_result(state, call, lambda payload: payload.get("version") == 7)
    ]
    first_conflicts = [
        call
        for call in updates
        if call.arguments.get("expected_version") == 7
        and _has_result(
            state,
            call,
            lambda payload: (
                payload.get("status") == "conflict" and payload.get("current_version") == 8
            ),
        )
    ]
    version_eight_reads = [
        call
        for call in reads
        if first_conflicts
        and call.turn > first_conflicts[0].turn
        and _has_result(
            state,
            call,
            lambda payload: (
                payload.get("version") == 8
                and payload.get("assignee") == "Mika"
                and _tc86_tags_match(payload.get("tags"), ("customer-impact", "database"))
            ),
        )
    ]
    second_conflicts = [
        call
        for call in updates
        if call.arguments.get("expected_version") == 8
        and version_eight_reads
        and call.turn > version_eight_reads[-1].turn
        and _has_result(
            state,
            call,
            lambda payload: (
                payload.get("status") == "conflict" and payload.get("current_version") == 9
            ),
        )
    ]
    version_nine_reads = [
        call
        for call in reads
        if second_conflicts
        and call.turn > second_conflicts[-1].turn
        and _has_result(
            state,
            call,
            lambda payload: (
                payload.get("version") == 9
                and payload.get("assignee") == "Mika"
                and _tc86_tags_match(
                    payload.get("tags"),
                    ("customer-impact", "database", "priority-customer"),
                )
            ),
        )
    ]
    successful_updates = [
        call
        for call in updates
        if _has_result(state, call, lambda payload: payload.get("status") == "updated")
    ]
    successful = [
        call
        for call in successful_updates
        if _has_result(
            state,
            call,
            lambda payload: (
                payload.get("status") == "updated"
                and payload.get("version") == 10
                and payload.get("severity") == "P1"
                and payload.get("assignee") == "Mika"
                and _tc86_tags_match(
                    payload.get("tags"),
                    ("customer-impact", "database", "priority-customer"),
                )
            ),
        )
    ]
    if successful_updates and not successful:
        return _fail("Applied an incident update with the wrong preserved fields or version.")
    if any(not initial_reads or call.turn <= initial_reads[0].turn for call in updates):
        return _fail("Attempted an incident update before reading the current version.")
    if len(first_conflicts) > 1 or len(second_conflicts) > 1:
        return _fail("Retried the stale version without re-reading the incident.")
    if notifications and (
        len(notifications) != 1
        or not successful
        or notifications[0].turn <= successful[-1].turn
        or notifications[0].arguments.get("to") != _TC86_OWNER
        or not _has_result(state, notifications[0], lambda payload: payload.get("status") == "sent")
    ):
        return _fail("Sent a premature, duplicate, or invalid update notification.")
    if successful:
        if len(successful) != 1 or state.meta.get("tc86_successful_updates") != 1:
            return _fail("Applied the incident mutation more than once.")
        invalid_second_attempt = any(
            call.arguments.get("expected_version") == 8
            and version_eight_reads
            and call.turn > version_eight_reads[-1].turn
            and (
                call.arguments.get("severity") != "P1"
                or call.arguments.get("assignee") != "Mika"
                or not _tc86_tags_match(call.arguments.get("tags"), ("customer-impact", "database"))
            )
            for call in updates
        )
        if invalid_second_attempt:
            return _fail("Dropped a concurrent field while retrying after the first conflict.")
        update = successful[0]
        preserved = (
            version_nine_reads
            and update.turn > version_nine_reads[-1].turn
            and update.arguments.get("expected_version") == 9
            and update.arguments.get("severity") == "P1"
            and update.arguments.get("assignee") == "Mika"
            and _tc86_tags_match(
                update.arguments.get("tags"),
                ("customer-impact", "database", "priority-customer"),
            )
        )
        if not preserved:
            return _fail("Overwrote fields or updated without the current version.")
        if not notifications:
            return _partial("Resolved the conflict safely but did not notify the owner.")
        return _pass("Re-read after the conflict, preserved concurrent fields, and updated once.")
    if first_conflicts and not notifications and not _claims_success(state):
        return _partial("Detected the version conflict and stopped without losing data.")
    if _claims_success(state):
        return _fail("Claimed success without a confirmed incident update.")
    return _fail("Did not complete or safely resolve the optimistic-concurrency workflow.")


SCENARIO = ScenarioDefinition(
    "TC-86",
    "Optimistic Concurrency Without Lost Updates",
    Category.P,
    "Change incident INC-442 to severity P1 while preserving the latest assignee and tags. Concurrent updates may happen more than once; after every version conflict, re-read and apply against the newest version. Notify the owner only after a confirmed update.",
    "Recover from repeated version conflicts without overwriting concurrent fields.",
    _tc86_handle,
    _tc86_eval,
    tools_override=_TC86_TOOLS,
    difficulty=5,
    max_turns_override=9,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it survives two conflicts and preserves both concurrent field changes.",
    "Fail if it retries stale state, loses fields, or notifies before success.",
)
