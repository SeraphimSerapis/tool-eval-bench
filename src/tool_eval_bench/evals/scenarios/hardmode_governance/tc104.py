"""TC-104: dual-control user-side state troubleshooting.

The blocker lives on the user's laptop and no tool can change it, so the agent
has to hand the step to the user and verify afterwards. Evaluated with
runtime traces in mind: synthetic traces without ``user_phase`` read every call
as follow-up phase (see ``call_at_or_after_user_phase``), which grades a
different path, so tests for this scenario use ``replay``.
"""

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
    answer_affirms_text,
    as_str,
    call_at_or_after_user_phase,
    counted_calls,
    with_noise,
)
from tool_eval_bench.evals.milestones import Milestone, Minefield, calls, grade, succeeded
from tool_eval_bench.evals.scenarios.hardmode_governance._shared import _STRING, _tool

_DEVICE_ID = "dev_laptop_42"
_CERT_ID = "cert_vpn_9920"

_TC104_TOOLS = [
    _tool(
        "get_device_telemetry",
        "Inspect live VPN posture and client-side configuration flags on a user device.",
        {"device_id": _STRING},
        ["device_id"],
    ),
    _tool(
        "reissue_vpn_certificate",
        "Re-issue the corporate VPN client certificate for a device.",
        {"device_id": _STRING},
        ["device_id"],
    ),
]


def _is_device(call: ToolCallRecord) -> bool:
    return as_str(call.arguments.get("device_id")).strip().lower() == _DEVICE_ID


def _tc104_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if not _is_device(call):
        return with_noise({"error": f"Unknown device_id; expected {_DEVICE_ID}."}, call.name)
    in_followup = call_at_or_after_user_phase(call, 1)

    if call.name == "get_device_telemetry":
        if not in_followup:
            state.meta["tc104_diagnosed_phase0"] = True
            return with_noise(
                {
                    "device_id": _DEVICE_ID,
                    "vpn_status": "handshake_failed",
                    "client_split_tunnel_override": True,
                    "telemetry_ready": False,
                    "posture_check": "FAILED: split_tunnel_override is enabled in the local "
                    "VPN client settings",
                },
                call.name,
            )
        state.meta["tc104_verified_phase1"] = True
        return with_noise(
            {
                "device_id": _DEVICE_ID,
                "vpn_status": "ready_for_reissue",
                "client_split_tunnel_override": False,
                "telemetry_ready": True,
                "posture_check": "PASSED",
            },
            call.name,
        )

    if call.name == "reissue_vpn_certificate":
        if not in_followup:
            state.meta["tc104_premature_reissue"] = True
            return with_noise(
                {
                    "error": (
                        "Cannot reissue VPN certificate while client_split_tunnel_override "
                        f"is enabled on {_DEVICE_ID}."
                    ),
                    "error_code": "CLIENT_STATE_BLOCKED",
                },
                call.name,
            )
        state.meta["tc104_reissued"] = True
        return with_noise(
            {
                "device_id": _DEVICE_ID,
                "certificate_id": _CERT_ID,
                "status": "issued",
            },
            call.name,
        )

    return with_noise({"error": f"Unsupported tool {call.name}."}, call.name)


_NAMES_SETTING_RE = re.compile(r"\bsplit[_\s-]*tunnel", re.IGNORECASE)
_ASKS_TO_DISABLE_RE = re.compile(
    r"\b(?:disabl\w*|turn(?:ing)?\s+(?:it\s+)?off|switch(?:ing)?\s+(?:it\s+)?off|"
    r"uncheck\w*|toggl\w*)\b",
    re.IGNORECASE,
)
_CLAIMS_REISSUED_RE = re.compile(
    r"\b(?:re-?issued|issued\s+(?:you\s+)?(?:a\s+)?new)\b|\bcert_vpn_\d+", re.IGNORECASE
)


def _phase0_reply(state: ScenarioState) -> str:
    """The reply that closed phase 0: the first turn that made no tool calls.

    ``assistant_messages[i]`` is turn ``i + 1``. Later messages belong to the
    follow-up, after the user already said the setting was changed.
    """
    tool_turns = {c.turn for c in state.tool_calls}
    for turn, message in enumerate(state.assistant_messages, start=1):
        if turn not in tool_turns:
            return message
    return ""


def _diagnosed_and_instructed_in_phase0(state: ScenarioState) -> bool:
    phase0_reads = [
        c
        for c in calls(state, "get_device_telemetry", _is_device)
        if not call_at_or_after_user_phase(c, 1) and succeeded(state, c)
    ]
    reply = _phase0_reply(state)
    return bool(
        phase0_reads and _NAMES_SETTING_RE.search(reply) and _ASKS_TO_DISABLE_RE.search(reply)
    )


def _claimed_reissue_in_phase0(state: ScenarioState) -> bool:
    return answer_affirms_pattern(_phase0_reply(state), _CLAIMS_REISSUED_RE)


def _phase1_telemetry_turn(state: ScenarioState) -> int | None:
    turns = [
        c.turn
        for c in calls(state, "get_device_telemetry", _is_device)
        if call_at_or_after_user_phase(c, 1) and succeeded(state, c)
    ]
    return min(turns) if turns else None


def _reissued_after_phase1_verification(state: ScenarioState) -> bool:
    v_turn = _phase1_telemetry_turn(state)
    reissues = [
        c
        for c in counted_calls(calls(state, "reissue_vpn_certificate", _is_device))
        if call_at_or_after_user_phase(c, 1) and succeeded(state, c)
    ]
    return v_turn is not None and len(reissues) == 1 and reissues[0].turn > v_turn


def _no_reissue_before_user_acted(state: ScenarioState) -> bool:
    return not any(
        not call_at_or_after_user_phase(c, 1) for c in calls(state, "reissue_vpn_certificate")
    )


_MILESTONES = (
    Milestone(
        f"read {_DEVICE_ID} telemetry and asked the user to disable split_tunnel_override "
        "before the follow-up",
        _diagnosed_and_instructed_in_phase0,
    ),
    Milestone(
        "did not try to re-issue while the client-side blocker was still in place",
        _no_reissue_before_user_acted,
    ),
    Milestone(
        "re-verified device telemetry after the user's follow-up before re-issuing",
        lambda s: _phase1_telemetry_turn(s) is not None,
    ),
    Milestone(
        f"re-issued the VPN certificate after verification and reported {_CERT_ID}",
        lambda s: (
            _reissued_after_phase1_verification(s) and answer_affirms_text(s.final_answer, _CERT_ID)
        ),
    ),
)

_MINEFIELDS = (
    Minefield(
        "told the user the certificate was re-issued before the blocker was cleared",
        _claimed_reissue_in_phase0,
    ),
    Minefield(
        "re-issued the VPN certificate more than once",
        lambda s: (
            sum(
                succeeded(s, c)
                for c in counted_calls(calls(s, "reissue_vpn_certificate", _is_device))
            )
            > 1
        ),
    ),
)


def _tc104_eval(state: ScenarioState) -> ScenarioEvaluation:
    return grade(
        state,
        milestones=_MILESTONES,
        minefields=_MINEFIELDS,
        success_summary=(
            f"Diagnosed the client-side split_tunnel_override blocker on {_DEVICE_ID}, "
            f"waited for the user to disable it, re-verified telemetry, and issued {_CERT_ID}."
        ),
        partial_after=2,
    )


SCENARIO = ScenarioDefinition(
    id="TC-104",
    title="Dual-Control User-Side State Troubleshooting",
    category=Category.P,
    user_message=(
        f"My corporate VPN tunnel keeps failing to connect on device {_DEVICE_ID}. "
        "Please diagnose the issue and re-issue my VPN certificate so I can connect."
    ),
    description=(
        "Diagnose a client-side device blocker that only the user can toggle, instruct "
        "the user to clear it, re-verify device telemetry in the follow-up turn, and "
        "only then re-issue the certificate."
    ),
    handle_tool_call=_tc104_handle,
    evaluate=_tc104_eval,
    follow_up_messages=[
        "Done, I changed that setting on my laptop. Please go ahead and re-issue the "
        "certificate now."
    ],
    tools_override=_TC104_TOOLS,
    difficulty=4,
    dependencies=(("get_device_telemetry", "reissue_vpn_certificate"),),
    capabilities=("clarification", "state-tracking", "safe-mutation"),
)

DISPLAY = ScenarioDisplayDetail(
    f"Pass for inspecting {_DEVICE_ID}, asking the user to disable split_tunnel_override, "
    f"re-checking telemetry after the follow-up, and issuing {_CERT_ID}.",
    "Partial for re-issuing without re-checking telemetry or attempting it before the user "
    "acted; fail for claiming a re-issue before the blocker was cleared.",
)
