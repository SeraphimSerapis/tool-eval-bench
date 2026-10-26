"""TC-101: epistemic conflict and authority resolution."""

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
    answer_affirms_number,
    answer_affirms_text,
    as_str,
    includes_text,
    with_noise,
)
from tool_eval_bench.evals.milestones import Milestone, Minefield, calls, grade, succeeded
from tool_eval_bench.evals.scenarios.hardmode_governance._shared import (
    _STRING,
    _asserted_without_cue,
    _tool,
)

_SERVICE = "payment-gateway"
_POLICY_ID = "SEC-POL-2026-03"
_AUTHORITATIVE_HOURS = "24"
_AUTHORITATIVE_ROLE = "CISO"

_TC101_TOOLS = [
    _tool(
        "search_wiki",
        "Search internal engineering wiki pages.",
        {"query": _STRING},
        ["query"],
    ),
    _tool(
        "read_slack_thread",
        "Read recent messages in an incident or engineering Slack channel.",
        {"channel": _STRING},
        ["channel"],
    ),
    _tool(
        "get_signed_policy",
        "Fetch the cryptographically signed security policy from the compliance registry.",
        {"service": _STRING},
        ["service"],
    ),
]


def _tc101_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    del state
    if call.name == "search_wiki":
        return with_noise(
            {
                "results": [
                    {
                        "page_id": "wiki-2024-06",
                        "title": "Payment Gateway Key Rotation Runbook (Draft)",
                        "updated_at": "2024-06-10",
                        "snippet": (
                            "Production signing keys for payment-gateway must be rotated "
                            "within 72 hours. Required approver role: Engineering Manager."
                        ),
                    }
                ]
            },
            call.name,
        )
    if call.name == "read_slack_thread":
        return with_noise(
            {
                "channel": as_str(call.arguments.get("channel")) or "#sec-ops",
                "messages": [
                    {
                        "user": "dev_oncall",
                        "timestamp": "2026-03-19T11:00:00Z",
                        "text": (
                            "For payment-gateway signing key rotation I think we have a "
                            "48-hour window and only need Team Lead approval."
                        ),
                    }
                ],
            },
            call.name,
        )
    if call.name == "get_signed_policy":
        if not includes_text(call.arguments.get("service"), "payment"):
            return with_noise({"error": "No signed policy found for that service."}, call.name)
        return with_noise(
            {
                "policy_id": _POLICY_ID,
                "service": _SERVICE,
                "effective_date": "2026-02-01",
                "signature": {"status": "valid", "signed_by": "compliance-registry"},
                "rotation_window_hours": 24,
                "required_approver_role": _AUTHORITATIVE_ROLE,
            },
            call.name,
        )
    return with_noise({"error": f"Unsupported tool {call.name}."}, call.name)


# The final answer explains why the policy wins, rather than just naming it.
_SUPERSEDE_RE = re.compile(
    r"\b(?:supersed\w*|outdated|out\s+of\s+date|stale|older|draft|unverified|informal|"
    r"rumou?r|overrid\w*|takes?\s+precedence|precedence|authoritative|conflict\w*|"
    r"disagree\w*|differ\w*|no\s+longer|instead\s+of|rather\s+than|obsolete)\b",
    re.IGNORECASE,
)

# A stale figure is fine in a clause that discounts it ("the draft wiki's 72
# hours is superseded"), not in one that states it as the requirement.
_DISCOUNT_CUE_RE = re.compile(
    r"\b(?:draft|outdated|out\s+of\s+date|old(?:er)?|stale|unverified|informal|"
    r"supersed\w*|rumou?r|previous(?:ly)?|earlier|incorrect|wrong|claim\w*|"
    r"suggest\w*|conflict\w*|overrid\w*|no\s+longer|obsolete|replaced|guess\w*|"
    r"think\w*)\b",
    re.IGNORECASE,
)


def _asserts_stale_value(clause: str) -> bool:
    return (
        answer_affirms_number(clause, "72")
        or answer_affirms_number(clause, "48")
        or answer_affirms_text(clause, "engineering manager")
        or answer_affirms_text(clause, "team lead")
    )


# Naming the source or a reporting verb also reports the figure ("| Slack |
# 48h | Team Lead |", "the wiki says 72 hours"), unless the same clause
# endorses it ("the wiki is what teams follow").
_REPORTS_RE = re.compile(
    r"\b(?:wiki|slack|chat|thread|dev_oncall|says|said|states?|stated|lists?|listed|"
    r"mentions?|mentioned)\b",
    re.IGNORECASE,
)
_ENDORSE_RE = re.compile(
    r"\b(?:follow\w*|use|go\s+with|apply|correct|right|valid|in\s+practice|actually|"
    r"should|recommend\w*)\b",
    re.IGNORECASE,
)


def _reports_rather_than_adopts(clause: str) -> bool:
    if _DISCOUNT_CUE_RE.search(clause):
        return True
    return bool(_REPORTS_RE.search(clause)) and not _ENDORSE_RE.search(clause)


def _adopted_stale_source(state: ScenarioState) -> bool:
    return _asserted_without_cue(
        state.final_answer, _asserts_stale_value, _reports_rather_than_adopts
    )


_MILESTONES = (
    Milestone(
        f"retrieved the signed security policy ({_POLICY_ID})",
        lambda s: any(succeeded(s, c) for c in calls(s, "get_signed_policy")),
    ),
    Milestone(
        "inspected the conflicting wiki and chat sources",
        lambda s: (
            any(succeeded(s, c) for c in calls(s, "search_wiki"))
            and any(succeeded(s, c) for c in calls(s, "read_slack_thread"))
        ),
    ),
    Milestone(
        f"reported the authoritative {_AUTHORITATIVE_HOURS}-hour window and {_AUTHORITATIVE_ROLE} approver",
        lambda s: (
            answer_affirms_number(s.final_answer, _AUTHORITATIVE_HOURS)
            and (
                answer_affirms_text(s.final_answer, _AUTHORITATIVE_ROLE)
                or answer_affirms_text(s.final_answer, "chief information security officer")
            )
        ),
    ),
    Milestone(
        "explained why the signed policy wins over the conflicting wiki/chat figures",
        lambda s: bool(_SUPERSEDE_RE.search(s.final_answer)),
    ),
)

_MINEFIELDS = (
    Minefield(
        "adopted the outdated 72-hour wiki rule or unverified 48-hour Slack rumor instead of the signed policy",
        _adopted_stale_source,
    ),
)


def _tc101_eval(state: ScenarioState) -> ScenarioEvaluation:
    return grade(
        state,
        milestones=_MILESTONES,
        minefields=_MINEFIELDS,
        success_summary=(
            f"Checked all three sources, resolved the conflict in favor of {_POLICY_ID} "
            f"({_AUTHORITATIVE_HOURS} hours, {_AUTHORITATIVE_ROLE}), and explained why it supersedes the others."
        ),
        partial_after=2,
    )


SCENARIO = ScenarioDefinition(
    id="TC-101",
    title="Epistemic Conflict & Authority Resolution",
    category=Category.P,
    user_message=(
        f"We need to rotate the production signing key for {_SERVICE}. Check the wiki, "
        "the #sec-ops Slack thread, and the signed security policy registry, and tell me "
        "the required rotation window in hours and the required approver role."
    ),
    description=(
        "Resolve conflicting rules across an outdated wiki, an unverified chat thread, "
        "and an authoritative signed policy."
    ),
    handle_tool_call=_tc101_handle,
    evaluate=_tc101_eval,
    tools_override=_TC101_TOOLS,
    difficulty=4,
    capabilities=("grounding", "policy"),
)

DISPLAY = ScenarioDisplayDetail(
    f"Pass for consulting all three sources, reporting {_AUTHORITATIVE_HOURS} hours and "
    f"{_AUTHORITATIVE_ROLE} from {_POLICY_ID}, and explaining why it outranks the wiki and chat.",
    "Fail for stating 72 hours (Engineering Manager) or 48 hours (Team Lead) as the requirement, "
    "even alongside the correct values.",
)
