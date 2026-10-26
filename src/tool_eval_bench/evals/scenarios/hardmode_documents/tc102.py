"""TC-102: schema evolution and deprecation fallbacks."""

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
    answer_affirms_text,
    as_str,
    counted_calls,
    with_noise,
)
from tool_eval_bench.evals.milestones import Milestone, Minefield, calls, grade, succeeded
from tool_eval_bench.evals.scenarios.hardmode_documents._shared import _STRING, _tool

_DOSSIER_ID = "DOS-550"
_RECORD_URN = "urn:dossier:DOS-550"
_SHA256 = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
_RETENTION = "COLD_DEEP_7Y"
_RECEIPT_ID = "ARC-2026-8841"

_TC102_TOOLS = [
    _tool(
        "archive_dossier_v1",
        "Archive a compliance dossier by dossier_id and storage tier.",
        {"dossier_id": _STRING, "tier": _STRING},
        ["dossier_id", "tier"],
    ),
    _tool(
        "get_dossier_manifest",
        "Fetch the canonical record URN and SHA-256 digest for a compliance dossier.",
        {"dossier_id": _STRING},
        ["dossier_id"],
    ),
    _tool(
        "archive_dossier_v2",
        "Archive a compliance dossier using its URN, retention profile, and verified SHA-256 digest.",
        {
            "record_urn": _STRING,
            "retention_profile": _STRING,
            "sha256": _STRING,
        },
        ["record_urn", "retention_profile", "sha256"],
    ),
]


def _is_manifest_call(call: ToolCallRecord) -> bool:
    return as_str(call.arguments.get("dossier_id")).strip().upper() == _DOSSIER_ID


def _is_valid_v2(call: ToolCallRecord) -> bool:
    return (
        as_str(call.arguments.get("record_urn")).strip() == _RECORD_URN
        and as_str(call.arguments.get("retention_profile")).strip().upper() == _RETENTION
        and as_str(call.arguments.get("sha256")).strip().lower() == _SHA256
    )


def _tc102_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "archive_dossier_v1":
        return with_noise(
            {
                "error": "archive_dossier_v1 was sunset on 2026-01-15.",
                "error_code": "ENDPOINT_SUNSET",
                "migration": {
                    "replacement_tool": "archive_dossier_v2",
                    "field_mapping": {
                        "dossier_id": f"record_urn ('{_RECORD_URN}')",
                        "tier": f"retention_profile ('glacier-deep' maps to '{_RETENTION}')",
                    },
                    "required_prerequisite": (
                        "Call get_dossier_manifest(dossier_id) first to obtain the required sha256 digest."
                    ),
                },
            },
            call.name,
        )
    if call.name == "get_dossier_manifest":
        if not _is_manifest_call(call):
            return with_noise({"error": "Unknown dossier_id."}, call.name)
        state.meta["tc102_manifest_read"] = True
        return with_noise(
            {
                "dossier_id": _DOSSIER_ID,
                "record_urn": _RECORD_URN,
                "sha256": _SHA256,
            },
            call.name,
        )
    if call.name == "archive_dossier_v2":
        if not _is_valid_v2(call):
            # Name the bad fields without echoing the expected values, so a
            # guessed call cannot harvest the answer from its own error.
            invalid = [
                field
                for field, ok in (
                    ("record_urn", as_str(call.arguments.get("record_urn")).strip() == _RECORD_URN),
                    (
                        "retention_profile",
                        as_str(call.arguments.get("retention_profile")).strip().upper()
                        == _RETENTION,
                    ),
                    ("sha256", as_str(call.arguments.get("sha256")).strip().lower() == _SHA256),
                )
                if not ok
            ]
            return with_noise(
                {
                    "error": f"Invalid v2 parameters: {', '.join(invalid)}.",
                    "error_code": "SCHEMA_VALIDATION_FAILED",
                },
                call.name,
            )
        state.meta["tc102_archived"] = True
        return with_noise(
            {
                "status": "archived",
                "record_urn": _RECORD_URN,
                "retention_profile": _RETENTION,
                "receipt_id": _RECEIPT_ID,
            },
            call.name,
        )
    return with_noise({"error": f"Unsupported tool {call.name}."}, call.name)


def _manifest_turn(state: ScenarioState) -> int | None:
    turns = [
        c.turn
        for c in calls(state, "get_dossier_manifest", _is_manifest_call)
        if succeeded(state, c)
    ]
    return min(turns) if turns else None


def _v2_after_manifest(state: ScenarioState) -> bool:
    m_turn = _manifest_turn(state)
    v2_calls = [
        c
        for c in counted_calls(calls(state, "archive_dossier_v2", _is_valid_v2))
        if succeeded(state, c)
    ]
    return m_turn is not None and len(v2_calls) == 1 and v2_calls[0].turn > m_turn


def _premature_v2(state: ScenarioState) -> bool:
    m_turn = _manifest_turn(state)
    v2_attempts = calls(state, "archive_dossier_v2")
    if not v2_attempts:
        return False
    return m_turn is None or any(c.turn <= m_turn for c in v2_attempts)


_MILESTONES = (
    Milestone(
        f"read the dossier manifest for {_DOSSIER_ID}",
        lambda s: _manifest_turn(s) is not None,
    ),
    Milestone(
        f"archived {_RECORD_URN} via archive_dossier_v2 with {_RETENTION} and the manifest SHA-256",
        _v2_after_manifest,
    ),
    Milestone(
        f"reported archive receipt ID {_RECEIPT_ID}",
        lambda s: answer_affirms_text(s.final_answer, _RECEIPT_ID),
    ),
)

_MINEFIELDS = (
    Minefield(
        "called archive_dossier_v2 before reading get_dossier_manifest",
        _premature_v2,
    ),
    Minefield(
        "retried the sunset archive_dossier_v1 endpoint after receiving its migration notice",
        lambda s: len(counted_calls(calls(s, "archive_dossier_v1"))) > 1,
    ),
    Minefield(
        "archived the dossier more than once",
        lambda s: len(counted_calls(calls(s, "archive_dossier_v2", _is_valid_v2))) > 1,
    ),
)


def _tc102_eval(state: ScenarioState) -> ScenarioEvaluation:
    return grade(
        state,
        milestones=_MILESTONES,
        minefields=_MINEFIELDS,
        success_summary=(
            f"Migrated from archive_dossier_v1 (or routed directly) via get_dossier_manifest "
            f"to archive_dossier_v2 and reported receipt {_RECEIPT_ID}."
        ),
        partial_after=2,
    )


SCENARIO = ScenarioDefinition(
    id="TC-102",
    title="Schema Evolution & Deprecation Fallbacks",
    category=Category.P,
    user_message=(
        f"Archive compliance dossier {_DOSSIER_ID} to cold storage tier glacier-deep "
        "and tell me the archive receipt ID."
    ),
    description=(
        "Adapt when a v1 tool returns a structured sunset/migration error by fetching "
        "the required manifest digest and mapping fields to the v2 schema."
    ),
    handle_tool_call=_tc102_handle,
    evaluate=_tc102_eval,
    tools_override=_TC102_TOOLS,
    difficulty=4,
    dependencies=(("get_dossier_manifest", "archive_dossier_v2"),),
    capabilities=("error-recovery", "tool-contracts"),
)

DISPLAY = ScenarioDisplayDetail(
    f"Pass for fetching the manifest for {_DOSSIER_ID}, calling archive_dossier_v2 with "
    f"'{_RECORD_URN}', '{_RETENTION}', and the SHA-256 digest, and reporting {_RECEIPT_ID}.",
    "Fail for guessing the SHA-256, repeating archive_dossier_v1, or failing to complete v2 archival.",
)
