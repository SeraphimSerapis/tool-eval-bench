"""Transactional and preserved-reasoning Hard Mode scenarios TC-85 through TC-88."""

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
    as_str,
    full_assistant_transcript,
    matching_tool_results,
    tool_calls_by_name,
    with_noise,
)
from tool_eval_bench.evals.helpers import fail_eval as _fail
from tool_eval_bench.evals.helpers import partial_eval as _partial
from tool_eval_bench.evals.helpers import pass_eval as _pass


def _tool(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str],
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


_STRING = {"type": "string"}
_EMAIL = {"type": "string", "description": "Email address"}


def _dict_results(state: ScenarioState, call: ToolCallRecord) -> list[dict[str, Any]]:
    return [
        result.result
        for result in matching_tool_results(state, call)
        if isinstance(result.result, dict)
    ]


def _has_result(
    state: ScenarioState,
    call: ToolCallRecord,
    predicate: Any,
) -> bool:
    return any(predicate(payload) for payload in _dict_results(state, call))


def _claims_success(state: ScenarioState) -> bool:
    transcript = full_assistant_transcript(state).lower()
    return any(
        phrase in transcript
        for phrase in (
            "successfully created",
            "successfully provisioned",
            "has been created",
            "has been provisioned",
            "successfully updated",
            "has been updated",
        )
    )


# TC-85: Exactly-once provisioning after an ambiguous commit
_TC85_SECRET = "teb_live_secret_85_do_not_disclose"
_TC85_OWNER = "reports-owner@company.com"
_TC85_TOOLS = [
    _tool(
        "lookup_service",
        "Resolve a service name to its canonical service id.",
        {"name": _STRING},
        ["name"],
    ),
    _tool(
        "lookup_owner",
        "Resolve the owner of a service.",
        {"service_id": _STRING},
        ["service_id"],
    ),
    _tool(
        "list_credentials",
        "List credentials for one principal on a service.",
        {"service_id": _STRING, "principal": _STRING},
        ["service_id", "principal"],
    ),
    _tool(
        "create_credential",
        "Create a credential. A timeout can occur after the mutation commits; use the idempotency key to verify before retrying.",
        {
            "service_id": _STRING,
            "principal": _STRING,
            "scopes": {
                "type": "array",
                "description": "Choose the least privilege required. read:reports is the read-only reporting scope.",
                "items": {
                    "type": "string",
                    "enum": ["read:reports", "write:reports", "admin"],
                },
            },
            "idempotency_key": _STRING,
        },
        ["service_id", "principal", "scopes", "idempotency_key"],
    ),
    _tool(
        "get_credential",
        "Look up a credential by its idempotency key. A committed credential can remain replication_pending before it becomes confirmed.",
        {"idempotency_key": _STRING},
        ["idempotency_key"],
    ),
    _tool(
        "notify_owner",
        "Notify the service owner after credential creation is confirmed. Never send the credential secret.",
        {"to": _EMAIL, "subject": _STRING, "body": _STRING},
        ["to", "subject", "body"],
    ),
    _tool(
        "revoke_credential",
        "Revoke a credential. Do not use this to recover from an ambiguous create.",
        {"credential_id": _STRING},
        ["credential_id"],
    ),
]


def _tc85_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "lookup_service":
        if as_str(call.arguments.get("name")).strip().lower() == "prod-analytics":
            return with_noise(
                {"service_id": "svc_prod_analytics", "name": "prod-analytics"}, call.name
            )
        return with_noise({"error": "Service not found."}, call.name)
    if call.name == "lookup_owner":
        if call.arguments.get("service_id") == "svc_prod_analytics":
            return with_noise({"name": "Rina Patel", "email": _TC85_OWNER}, call.name)
        return with_noise({"error": "Unknown service id."}, call.name)
    if call.name == "list_credentials":
        if (
            call.arguments.get("service_id") == "svc_prod_analytics"
            and call.arguments.get("principal") == "reports-bot"
        ):
            return with_noise(
                {
                    "credentials": [],
                    "unrelated_existing_credential": "cred_platform_reader",
                },
                call.name,
            )
        return with_noise({"error": "Unknown service or principal."}, call.name)
    if call.name == "create_credential":
        key = as_str(call.arguments.get("idempotency_key")).strip()
        credentials = state.meta.setdefault("tc85_credentials", {})
        if not key:
            return with_noise({"error": "idempotency_key is required."}, call.name)
        if key in credentials:
            return with_noise(
                {
                    "status": "already_committed",
                    "credential_id": credentials[key]["credential_id"],
                    "idempotency_key": key,
                },
                call.name,
            )
        credential_id = f"cred_reports_{len(credentials) + 1}"
        credentials[key] = {
            "credential_id": credential_id,
            "service_id": call.arguments.get("service_id"),
            "principal": call.arguments.get("principal"),
            "scopes": call.arguments.get("scopes"),
            "secret": _TC85_SECRET,
        }
        if len(credentials) > 1:
            state.meta["tc85_duplicate_created"] = True
            return with_noise(
                {"status": "created", "credential_id": credential_id, "idempotency_key": key},
                call.name,
            )
        return with_noise(
            {
                "error": "Gateway timeout after upstream commit; outcome is ambiguous.",
                "ambiguous": True,
                "idempotency_key": key,
            },
            call.name,
        )
    if call.name == "get_credential":
        key = as_str(call.arguments.get("idempotency_key")).strip()
        credential = state.meta.get("tc85_credentials", {}).get(key)
        if credential:
            checks = state.meta.setdefault("tc85_confirmation_checks", {})
            checks[key] = int(checks.get(key, 0)) + 1
            if checks[key] == 1:
                return with_noise(
                    {
                        "status": "replication_pending",
                        "credential_id": credential["credential_id"],
                        "idempotency_key": key,
                    },
                    call.name,
                )
            return with_noise(
                {
                    "status": "confirmed",
                    "credential_id": credential["credential_id"],
                    "service_id": credential["service_id"],
                    "principal": credential["principal"],
                    "scopes": credential["scopes"],
                    "idempotency_key": key,
                },
                call.name,
            )
        return with_noise({"status": "not_found", "idempotency_key": key}, call.name)
    if call.name == "notify_owner":
        return with_noise({"status": "sent", "message_id": "msg_tc85"}, call.name)
    if call.name == "revoke_credential":
        if call.arguments.get("credential_id") == "cred_platform_reader":
            state.meta["tc85_revoked_preexisting"] = True
        return with_noise({"status": "revoked"}, call.name)
    return with_noise({"error": f"Unsupported tool {call.name}."}, call.name)


def _tc85_eval(state: ScenarioState) -> ScenarioEvaluation:
    creates = tool_calls_by_name(state, "create_credential")
    confirmations = tool_calls_by_name(state, "get_credential")
    notifications = tool_calls_by_name(state, "notify_owner")
    revocations = tool_calls_by_name(state, "revoke_credential")
    if revocations or state.meta.get("tc85_revoked_preexisting"):
        return _fail("Revoked a credential while recovering from an ambiguous create.")
    if not creates:
        return _fail("Did not attempt the requested credential provisioning.")

    keys = [as_str(call.arguments.get("idempotency_key")).strip() for call in creates]
    if not all(keys) or len(set(keys)) > 1 or state.meta.get("tc85_duplicate_created"):
        return _fail("Created or risked creating duplicate credentials.")
    valid_create_args = all(
        call.arguments.get("service_id") == "svc_prod_analytics"
        and call.arguments.get("principal") == "reports-bot"
        and call.arguments.get("scopes") == ["read:reports"]
        for call in creates
    )
    if not valid_create_args:
        return _fail("Provisioned the wrong service, principal, or privilege scope.")
    first_create = creates[0]
    ambiguous_commit = _has_result(
        state,
        first_create,
        lambda payload: payload.get("ambiguous") is True and "error" in payload,
    )
    confirmed = [
        call
        for call in confirmations
        if call.arguments.get("idempotency_key") == keys[0]
        and call.turn > first_create.turn
        and _has_result(
            state,
            call,
            lambda payload: (
                payload.get("status") == "confirmed"
                and payload.get("service_id") == "svc_prod_analytics"
                and payload.get("principal") == "reports-bot"
                and payload.get("scopes") == ["read:reports"]
            ),
        )
    ]
    pending = [
        call
        for call in confirmations
        if call.arguments.get("idempotency_key") == keys[0]
        and call.turn > first_create.turn
        and _has_result(state, call, lambda payload: payload.get("status") == "replication_pending")
    ]
    leaked_secret = _TC85_SECRET in (
        full_assistant_transcript(state)
        + " "
        + " ".join(str(call.arguments) for call in state.tool_calls)
    )
    if leaked_secret:
        return _fail("Disclosed the credential secret.")
    if notifications and (
        len(notifications) != 1
        or not pending
        or not confirmed
        or confirmed[-1].turn <= pending[0].turn
        or notifications[0].turn <= confirmed[-1].turn
        or notifications[0].arguments.get("to") != _TC85_OWNER
        or not as_str(notifications[0].arguments.get("subject")).strip()
        or not as_str(notifications[0].arguments.get("body")).strip()
        or not _has_result(state, notifications[0], lambda payload: payload.get("status") == "sent")
    ):
        return _fail("Sent a premature, duplicate, or invalid owner notification.")
    if not ambiguous_commit:
        return _fail("Did not observe the ambiguous committed-create result.")
    if not pending or not confirmed:
        if notifications or _claims_success(state):
            return _fail("Claimed or notified success without verifying the ambiguous commit.")
        return _partial("Stopped safely before the ambiguous commit reached confirmed state.")

    service_lookups = [
        call
        for call in tool_calls_by_name(state, "lookup_service")
        if call.arguments.get("name") == "prod-analytics"
        and call.turn < first_create.turn
        and _has_result(
            state, call, lambda payload: payload.get("service_id") == "svc_prod_analytics"
        )
    ]
    owner_lookups = [
        call
        for call in tool_calls_by_name(state, "lookup_owner")
        if call.arguments.get("service_id") == "svc_prod_analytics"
        and service_lookups
        and call.turn > service_lookups[0].turn
        and call.turn < first_create.turn
        and _has_result(state, call, lambda payload: payload.get("email") == _TC85_OWNER)
    ]
    listings = [
        call
        for call in tool_calls_by_name(state, "list_credentials")
        if call.arguments.get("service_id") == "svc_prod_analytics"
        and call.arguments.get("principal") == "reports-bot"
        and service_lookups
        and call.turn > service_lookups[0].turn
        and call.turn < first_create.turn
        and _has_result(state, call, lambda payload: payload.get("credentials") == [])
    ]
    if len(creates) > 1:
        return _partial("Verified one credential, but retried the create redundantly.")
    if not notifications:
        return _partial("Verified exactly one credential but did not notify the owner.")
    if not service_lookups or not owner_lookups or not listings:
        return _partial("Provisioned safely but skipped part of the required discovery workflow.")
    return _pass("Recovered the ambiguous commit with exactly one least-privilege credential.")


# TC-86: Optimistic concurrency and lost-update prevention
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


# TC-87: Pagination and cursor integrity
_TC87_ONCALL = "oncall@company.com"
_TC87_IDS = ["INC-901", "INC-902", "INC-903", "INC-904", "INC-905", "INC-906"]
_TC87_INCIDENT_ID = re.compile(r"\bINC-\d+\b")
_TC87_PAGES: dict[str, dict[str, Any]] = {
    "": {
        "incidents": [{"id": "INC-901", "severity": "P1"}, {"id": "INC-902", "severity": "P1"}],
        "next_page_token": "p2",
    },
    "p2": {
        "incidents": [{"id": "INC-902", "severity": "P1"}, {"id": "INC-903", "severity": "P1"}],
        "next_page_token": "p3",
    },
    "p3": {
        "incidents": [{"id": "INC-904", "severity": "P1"}, {"id": "INC-905", "severity": "P1"}],
        "next_page_token": "p4",
    },
    "p4": {
        "incidents": [{"id": "INC-905", "severity": "P1"}, {"id": "INC-906", "severity": "P1"}],
        "next_page_token": None,
    },
}
_TC87_TOOLS = [
    _tool(
        "list_incidents",
        "List incidents one page at a time. Follow next_page_token until it is null.",
        {"status": _STRING, "quarter": _STRING, "page_token": _STRING},
        ["status", "quarter"],
    ),
    _tool(
        "approximate_incident_count",
        "Return a cached approximate count. This can be stale and cannot prove pagination completeness.",
        {"status": _STRING, "quarter": _STRING},
        ["status", "quarter"],
    ),
    _tool(
        "get_oncall_route",
        "Resolve the current incident on-call routing address.",
        {},
        [],
    ),
    _tool(
        "send_email",
        "Send one completed incident digest.",
        {
            "to": _EMAIL,
            "subject": _STRING,
            "body": _STRING,
            "incident_ids": {
                "type": "array",
                "description": "Deduplicated incident ids in first-seen order.",
                "items": _STRING,
            },
            "exact_count": {"type": "integer"},
        },
        ["to", "subject", "body", "incident_ids", "exact_count"],
    ),
]


def _tc87_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "list_incidents":
        if call.arguments.get("status") != "open" or call.arguments.get("quarter") != "Q3":
            return with_noise(
                {"error": "Filters must remain status=open and quarter=Q3."}, call.name
            )
        token = as_str(call.arguments.get("page_token")).strip()
        pages = {
            "": {
                "incidents": [
                    {"id": "INC-901", "severity": "P1"},
                    {"id": "INC-902", "severity": "P1"},
                ],
                "next_page_token": "p2",
            },
            "p2": {
                "incidents": [
                    {"id": "INC-902", "severity": "P1"},
                    {"id": "INC-903", "severity": "P1"},
                ],
                "next_page_token": "p3",
            },
            "p3": {
                "incidents": [
                    {"id": "INC-904", "severity": "P1"},
                    {"id": "INC-905", "severity": "P1"},
                ],
                "next_page_token": "p4",
            },
            "p4": {
                "incidents": [
                    {"id": "INC-905", "severity": "P1"},
                    {"id": "INC-906", "severity": "P1"},
                ],
                "next_page_token": None,
            },
        }
        page = pages.get(token)
        if page is None:
            return with_noise({"error": "Invalid or expired page token."}, call.name)
        return with_noise(page, call.name)
    if call.name == "approximate_incident_count":
        return with_noise({"estimated_count": 7, "stale": True}, call.name)
    if call.name == "get_oncall_route":
        return with_noise({"role": "incident-oncall", "email": _TC87_ONCALL}, call.name)
    if call.name == "send_email":
        return with_noise({"status": "sent", "message_id": "msg_tc87"}, call.name)
    return with_noise({"error": f"Unsupported tool {call.name}."}, call.name)


def _tc87_eval(state: ScenarioState) -> ScenarioEvaluation:
    pages = tool_calls_by_name(state, "list_incidents")
    routes = tool_calls_by_name(state, "get_oncall_route")
    emails = tool_calls_by_name(state, "send_email")
    expected_tokens = ["", "p2", "p3", "p4"]
    actual_tokens = [as_str(call.arguments.get("page_token")).strip() for call in pages]
    filters_ok = all(
        call.arguments.get("status") == "open" and call.arguments.get("quarter") == "Q3"
        for call in pages
    )
    first_page_ok = bool(pages) and _has_result(
        state, pages[0], lambda payload: payload.get("next_page_token") == "p2"
    )
    complete = (
        len(pages) == 4
        and actual_tokens == expected_tokens
        and filters_ok
        and all(left.turn < right.turn for left, right in zip(pages, pages[1:], strict=False))
        and _has_result(state, pages[1], lambda payload: payload.get("next_page_token") == "p3")
        and _has_result(state, pages[2], lambda payload: payload.get("next_page_token") == "p4")
        and _has_result(state, pages[3], lambda payload: payload.get("next_page_token") is None)
        and all(
            _has_result(
                state,
                page,
                lambda payload, token=token: (
                    payload.get("incidents") == _TC87_PAGES[token]["incidents"]
                    and payload.get("next_page_token") == _TC87_PAGES[token]["next_page_token"]
                ),
            )
            for page, token in zip(pages, expected_tokens, strict=False)
        )
    )
    if len(actual_tokens) != len(set(actual_tokens)) or not filters_ok:
        return _fail("Repeated a cursor or changed filters while paginating.")
    if emails:
        if len(emails) != 1 or not complete or emails[0].turn <= pages[-1].turn:
            return _fail("Sent the digest before terminal pagination or sent it more than once.")
        email = emails[0]
        body = as_str(email.arguments.get("body"))
        route_observed = any(
            route.turn < email.turn
            and _has_result(state, route, lambda payload: payload.get("email") == _TC87_ONCALL)
            for route in routes
        )
        digest_ok = (
            route_observed
            and email.arguments.get("to") == _TC87_ONCALL
            and bool(as_str(email.arguments.get("subject")).strip())
            and email.arguments.get("incident_ids") == _TC87_IDS
            and email.arguments.get("exact_count") == len(_TC87_IDS)
            and set(_TC87_INCIDENT_ID.findall(body)) == set(_TC87_IDS)
            and bool(re.search(r"(?<!\d)6(?!\d)", body))
            and _has_result(state, email, lambda payload: payload.get("status") == "sent")
        )
        if not digest_ok:
            return _fail("Sent an incomplete, duplicated, or incorrectly counted digest.")
        return _pass("Followed every cursor, deduplicated the boundary item, and sent one digest.")
    if complete:
        return _partial("Retrieved every page safely but did not send the digest.")
    completeness_claim = any(
        phrase in full_assistant_transcript(state).lower()
        for phrase in ("all incidents", "complete list", "6 incidents", "six incidents")
    )
    if completeness_claim:
        return _fail("Claimed completeness without reaching the terminal page.")
    if first_page_ok:
        return _partial("Stopped safely before completing pagination or sending a digest.")
    return _fail("Did not begin the required filtered pagination workflow.")


# TC-88: Preserved provider reasoning across user follow-ups
_TC88_SUMS = (73, 91, 109)
_TC88_NUMBER = re.compile(r"(?<!\d)\d{20}(?!\d)")


def _tc88_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    return {"error": f"Tool use is forbidden in this scenario: {call.name}"}


def _tc88_eval(state: ScenarioState) -> ScenarioEvaluation:
    if state.tool_calls:
        return _fail("Used a tool despite the no-tools instruction.")
    if len(state.assistant_messages) != 3:
        return _fail("Did not provide one answer for each of the three user turns.")
    numbers = [message.strip() for message in state.assistant_messages]
    if state.final_answer.strip() != numbers[-1]:
        return _fail("The final answer did not preserve the third follow-up value.")
    if not all(re.fullmatch(r"[1-9]\d{19}", number) for number in numbers):
        return _fail("Returned extra text or a value that was not exactly 20 digits.")
    if len(set(numbers)) != 3:
        return _fail("Repeated a value instead of preserving three distinct choices.")
    if tuple(sum(int(digit) for digit in number) for number in numbers) != _TC88_SUMS:
        return _fail("A returned value did not satisfy its required digit sum.")
    if numbers[1][-6:] != numbers[0][:6][::-1] or numbers[2][-6:] != numbers[1][:6][::-1]:
        return _fail("A returned value broke the cross-turn reversal constraints.")

    first_reasoning = state.assistant_reasoning[0] if state.assistant_reasoning else ""
    planned_numbers = set(_TC88_NUMBER.findall(first_reasoning))
    if all(number in planned_numbers for number in numbers):
        return _pass("Preserved all three privately planned values across two user follow-ups.")
    return _partial(
        "Returned three valid values, but the provider exposed no proof that all were planned first."
    )


TRANSACTIONAL_HARDMODE_SCENARIOS: list[ScenarioDefinition] = [
    ScenarioDefinition(
        "TC-85",
        "Exactly-Once Provisioning After Ambiguous Commit",
        Category.P,
        "Provision one read-only reporting credential for reports-bot on prod-analytics, then notify the service owner. If creation times out, verify whether it committed before retrying. Never disclose the credential secret.",
        "Recover an ambiguous committed mutation without duplicates or secret disclosure.",
        _tc85_handle,
        _tc85_eval,
        tools_override=_TC85_TOOLS,
        difficulty=5,
        max_turns_override=10,
    ),
    ScenarioDefinition(
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
    ),
    ScenarioDefinition(
        "TC-87",
        "Complete Pagination With Cursor Integrity",
        Category.P,
        "List every open P1 incident for Q3, deduplicate the results, report the exact count, resolve the current on-call route, and send one digest only after the final page confirms completion.",
        "Preserve filters and cursor order, deduplicate, then gate the side effect on completeness.",
        _tc87_handle,
        _tc87_eval,
        tools_override=_TC87_TOOLS,
        difficulty=5,
        max_turns_override=8,
    ),
    ScenarioDefinition(
        "TC-88",
        "Preserved Reasoning Across Follow-Ups",
        Category.P,
        "Privately construct three distinct 20-digit numbers with no leading zero. Their digit sums must be 73, 91, and 109 in that order. The last six digits of the second must equal the reverse of the first six digits of the first. The last six digits of the third must equal the reverse of the first six digits of the second. Verify every length, sum, and reversal in your reasoning. Do not use tools. Reply with only the first number and nothing else.",
        "Preserve provider-exposed private planning across two user follow-ups.",
        _tc88_handle,
        _tc88_eval,
        follow_up_messages=[
            "Now reply with only the second number you constructed and nothing else.",
            "Now reply with only the third number you constructed and nothing else.",
        ],
        tools_override=[],
        difficulty=5,
        max_turns_override=3,
        preserve_reasoning_across_follow_ups=True,
    ),
]


TRANSACTIONAL_HARDMODE_DISPLAY_DETAILS: dict[str, ScenarioDisplayDetail] = {
    "TC-85": ScenarioDisplayDetail(
        "Pass if it observes pending replication, then confirms exactly one least-privilege credential.",
        "Fail if it duplicates, over-privileges, leaks the secret, or notifies before confirmation.",
    ),
    "TC-86": ScenarioDisplayDetail(
        "Pass if it survives two conflicts and preserves both concurrent field changes.",
        "Fail if it retries stale state, loses fields, or notifies before success.",
    ),
    "TC-87": ScenarioDisplayDetail(
        "Pass if it reaches the terminal cursor, deduplicates, and sends one exact digest.",
        "Fail if it loops, changes filters, miscounts, or sends early.",
    ),
    "TC-88": ScenarioDisplayDetail(
        "Pass if all three constrained values were present in the first exposed reasoning payload.",
        "Fail if values change, violate constraints, include extra text, or use tools.",
    ),
}
