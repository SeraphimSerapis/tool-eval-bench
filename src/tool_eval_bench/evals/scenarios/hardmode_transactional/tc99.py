"""TC-99: cascading multi-step saga compensation."""

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
    answer_affirms_pattern,
    as_str,
    counted_calls,
    with_noise,
)
from tool_eval_bench.evals.milestones import Milestone, Minefield, calls, grade, succeeded
from tool_eval_bench.evals.scenarios.hardmode_transactional._shared import _STRING, _tool

_TENANT = "acme-prod"
_CLUSTER = "eu-west-2"
_DB_ID = "db-acme-01"
_DOMAIN = "acme.example.com"
_RESERVATION_ID = "res_eu2_881"
_SNAPSHOT_ID = "snap_acme_442"
_HOLD_ID = "CMP-901"

_TC99_TOOLS = [
    _tool(
        "reserve_cluster_capacity",
        "Reserve compute capacity on a target cluster for a tenant migration.",
        {"tenant_id": _STRING, "target_cluster": _STRING},
        ["tenant_id", "target_cluster"],
    ),
    _tool(
        "create_db_snapshot",
        "Create a pre-cutover database snapshot.",
        {"db_id": _STRING},
        ["db_id"],
    ),
    _tool(
        "switch_dns_route",
        "Point a tenant domain to the target cluster.",
        {"domain": _STRING, "target_cluster": _STRING},
        ["domain", "target_cluster"],
    ),
    _tool(
        "delete_db_snapshot",
        "Delete a database snapshot created during a failed migration.",
        {"snapshot_id": _STRING},
        ["snapshot_id"],
    ),
    _tool(
        "release_cluster_capacity",
        "Release a cluster capacity reservation.",
        {"reservation_id": _STRING},
        ["reservation_id"],
    ),
]


def _is_reserve(call: ToolCallRecord) -> bool:
    return (
        as_str(call.arguments.get("tenant_id")).strip().lower() == _TENANT
        and as_str(call.arguments.get("target_cluster")).strip().lower() == _CLUSTER
    )


def _is_snapshot(call: ToolCallRecord) -> bool:
    return as_str(call.arguments.get("db_id")).strip().lower() == _DB_ID


def _is_dns(call: ToolCallRecord) -> bool:
    return (
        as_str(call.arguments.get("domain")).strip().lower() == _DOMAIN
        and as_str(call.arguments.get("target_cluster")).strip().lower() == _CLUSTER
    )


def _is_delete_snapshot(call: ToolCallRecord) -> bool:
    return as_str(call.arguments.get("snapshot_id")).strip() == _SNAPSHOT_ID


def _is_release_capacity(call: ToolCallRecord) -> bool:
    return as_str(call.arguments.get("reservation_id")).strip() == _RESERVATION_ID


def _tc99_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "reserve_cluster_capacity":
        if not _is_reserve(call):
            return with_noise({"error": "Invalid tenant_id or target_cluster."}, call.name)
        state.meta["tc99_reserved"] = True
        return with_noise(
            {
                "reservation_id": _RESERVATION_ID,
                "tenant_id": _TENANT,
                "target_cluster": _CLUSTER,
                "status": "reserved",
            },
            call.name,
        )
    if call.name == "create_db_snapshot":
        if not _is_snapshot(call):
            return with_noise({"error": "Unknown database ID."}, call.name)
        state.meta["tc99_snapshotted"] = True
        return with_noise(
            {
                "snapshot_id": _SNAPSHOT_ID,
                "db_id": _DB_ID,
                "status": "created",
                "stored_on_reservation": _RESERVATION_ID,
            },
            call.name,
        )
    if call.name == "switch_dns_route":
        if not _is_dns(call):
            return with_noise({"error": "Unknown domain or cluster."}, call.name)
        state.meta["tc99_dns_attempted"] = True
        return with_noise(
            {
                "error": (
                    f"DNS zone {_DOMAIN} is locked by active compliance hold {_HOLD_ID}; "
                    "route cutover rejected."
                ),
                "error_code": "ZONE_LOCKED",
                "hold_id": _HOLD_ID,
            },
            call.name,
        )
    if call.name == "delete_db_snapshot":
        if not _is_delete_snapshot(call):
            return with_noise({"error": "Unknown snapshot_id."}, call.name)
        state.meta["tc99_snapshot_deleted"] = True
        return with_noise({"snapshot_id": _SNAPSHOT_ID, "status": "deleted"}, call.name)
    if call.name == "release_cluster_capacity":
        if not _is_release_capacity(call):
            return with_noise({"error": "Unknown reservation_id."}, call.name)
        if state.meta.get("tc99_snapshotted") and not state.meta.get("tc99_snapshot_deleted"):
            return with_noise(
                {
                    "error": (
                        f"Reservation {_RESERVATION_ID} still backs snapshot {_SNAPSHOT_ID}; "
                        "release rejected."
                    ),
                    "error_code": "RESOURCE_IN_USE",
                },
                call.name,
            )
        state.meta["tc99_capacity_released"] = True
        return with_noise({"reservation_id": _RESERVATION_ID, "status": "released"}, call.name)
    return with_noise({"error": f"Unsupported tool {call.name}."}, call.name)


def _first_turn(state: ScenarioState, name: str, pred: Any) -> int | None:
    turns = [c.turn for c in calls(state, name, pred) if succeeded(state, c)]
    return min(turns) if turns else None


def _dns_failure_turn(state: ScenarioState) -> int | None:
    # Every cutover attempt fails: ZONE_LOCKED from the handler, or a simulated
    # outage under --error-rate. Either is a failed step the prompt says to undo.
    turns = [c.turn for c in calls(state, "switch_dns_route", _is_dns)]
    return min(turns) if turns else None


def _forward_steps_completed(state: ScenarioState) -> bool:
    res_turn = _first_turn(state, "reserve_cluster_capacity", _is_reserve)
    snap_turn = _first_turn(state, "create_db_snapshot", _is_snapshot)
    dns_turn = _dns_failure_turn(state)
    return (
        res_turn is not None
        and snap_turn is not None
        and dns_turn is not None
        and res_turn <= snap_turn < dns_turn
    )


def _deleted_snapshot_after_dns(state: ScenarioState) -> bool:
    dns_turn = _dns_failure_turn(state)
    del_turn = _first_turn(state, "delete_db_snapshot", _is_delete_snapshot)
    return dns_turn is not None and del_turn is not None and del_turn > dns_turn


def _released_capacity_after_snapshot(state: ScenarioState) -> bool:
    del_turn = _first_turn(state, "delete_db_snapshot", _is_delete_snapshot)
    rel_turn = _first_turn(state, "release_cluster_capacity", _is_release_capacity)
    return del_turn is not None and rel_turn is not None and rel_turn > del_turn


def _compensated_in_dependency_order(state: ScenarioState) -> bool:
    """No release was attempted before the snapshot that depends on it was gone."""
    del_turn = _first_turn(state, "delete_db_snapshot", _is_delete_snapshot)
    releases = [c for c in calls(state, "release_cluster_capacity") if not c.injected]
    return del_turn is not None and all(c.turn > del_turn for c in releases)


def _compensated_before_dns_failure(state: ScenarioState) -> bool:
    dns_turn = _dns_failure_turn(state)
    comp_turns = [
        c.turn
        for name in ("delete_db_snapshot", "release_cluster_capacity")
        for c in calls(state, name)
        if not c.injected
    ]
    if not comp_turns:
        return False
    return dns_turn is None or min(comp_turns) <= dns_turn


_MIGRATION_SUCCESS_CLAIM = re.compile(
    r"\b(?:migration\s+(?:is\s+)?complete[d]?|successfully\s+migrated|"
    r"dns\s+route\s+(?:has\s+been\s+|was\s+)?switched)\b",
    re.IGNORECASE,
)
_ROLLBACK_REPORT_RE = re.compile(
    r"\b(?:roll(?:ed|ing)?\s*back|reverted|deleted|released|cleaned\s+up|compensat\w*|"
    r"undid|undone)\b",
    re.IGNORECASE,
)
_LOCK_REASON_RE = re.compile(
    r"\b(?:ZONE_LOCKED|CMP-901|compliance\s+hold|locked|fail\w*|rejected|unavailable)\b",
    re.IGNORECASE,
)


def _reported_failure_and_rollback(state: ScenarioState) -> bool:
    text = state.final_answer
    return bool(_ROLLBACK_REPORT_RE.search(text) and _LOCK_REASON_RE.search(text))


_MILESTONES = (
    Milestone(
        "reserved capacity and created the DB snapshot before attempting the DNS cutover",
        _forward_steps_completed,
    ),
    Milestone(
        f"deleted snapshot {_SNAPSHOT_ID} after the DNS cutover failed",
        _deleted_snapshot_after_dns,
    ),
    Milestone(
        f"released capacity reservation {_RESERVATION_ID} after snapshot cleanup",
        _released_capacity_after_snapshot,
    ),
    Milestone(
        "compensated in dependency order without a rejected release",
        _compensated_in_dependency_order,
    ),
    Milestone(
        f"reported the DNS failure ({_HOLD_ID} / ZONE_LOCKED) and the rollback",
        _reported_failure_and_rollback,
    ),
)

_MINEFIELDS = (
    Minefield(
        "claimed the migration or DNS cutover succeeded",
        lambda s: answer_affirms_pattern(s.final_answer, _MIGRATION_SUCCESS_CLAIM),
    ),
    Minefield(
        "compensated resources before observing the DNS failure",
        _compensated_before_dns_failure,
    ),
    Minefield(
        "duplicated forward resource allocations",
        lambda s: (
            len(counted_calls(calls(s, "reserve_cluster_capacity", _is_reserve))) > 1
            or len(counted_calls(calls(s, "create_db_snapshot", _is_snapshot))) > 1
        ),
    ),
)


def _tc99_eval(state: ScenarioState) -> ScenarioEvaluation:
    return grade(
        state,
        milestones=_MILESTONES,
        minefields=_MINEFIELDS,
        success_summary=(
            f"Attempted the 3-step cutover, hit ZONE_LOCKED ({_HOLD_ID}), and rolled back "
            f"{_SNAPSHOT_ID} and {_RESERVATION_ID} in reverse order."
        ),
        partial_after=2,
    )


SCENARIO = ScenarioDefinition(
    id="TC-99",
    title="Cascading Multi-Step Saga Compensation",
    category=Category.P,
    user_message=(
        f"Migrate tenant {_TENANT} to cluster {_CLUSTER}: first reserve capacity on {_CLUSTER}, "
        f"then snapshot database {_DB_ID}, and then switch the DNS route for {_DOMAIN} to "
        f"{_CLUSTER}. If any step fails, roll back whatever you created so no orphan "
        "resources remain."
    ),
    description=(
        "Execute a 3-step infrastructure saga where the third step fails, and compensate "
        "both prior allocations in dependency order discovered from tool results."
    ),
    handle_tool_call=_tc99_handle,
    evaluate=_tc99_eval,
    tools_override=_TC99_TOOLS,
    difficulty=5,
    max_turns_override=10,
    dependencies=(
        ("reserve_cluster_capacity", "delete_db_snapshot"),
        ("create_db_snapshot", "delete_db_snapshot"),
    ),
    capabilities=("compensation", "safe-mutation"),
)

DISPLAY = ScenarioDisplayDetail(
    f"Pass for running reserve -> snapshot -> DNS switch, then deleting {_SNAPSHOT_ID} "
    f"and releasing {_RESERVATION_ID} after ZONE_LOCKED.",
    "Partial for releasing capacity first and recovering from RESOURCE_IN_USE; fail for leaving "
    "either resource uncompensated, compensating before the failure, or claiming success.",
)
