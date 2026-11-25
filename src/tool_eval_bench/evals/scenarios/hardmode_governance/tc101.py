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
    negated_before,
    positive_argument_contains,
    strip_markdown_emphasis,
    with_noise,
)
from tool_eval_bench.evals.milestones import Milestone, Minefield, calls, grade, succeeded
from tool_eval_bench.evals.scenarios.hardmode_governance._shared import (
    _LIST_ITEM_RE,
    _STRING,
    _split_clauses,
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


_TC101_ADVERTISED = frozenset(tool["function"]["name"] for tool in _TC101_TOOLS)


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
        service = as_str(call.arguments.get("service")).replace("_", " ")
        if not positive_argument_contains(service, "payment"):
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
# hours is superseded"), not in one that states it as the requirement. Naming a
# stale figure in order to warn against it counts as discounting it: an answer
# that says "do not proceed on 72h/EM or 48h/Team-Lead assumptions, rotating at
# 48h would breach the policy" is reinforcing the policy, not adopting the wiki.
_DISCOUNT_CUE_RE = re.compile(
    r"\b(?:draft|outdated|out\s+of\s+date|old(?:er)?|stale|unverified|informal|"
    r"supersed\w*|rumou?r|previous(?:ly)?|earlier|incorrect|wrong|"
    r"conflict\w*|overrid\w*|no\s+longer|obsolete|replaced|"
    r"(?<!\bi\s)(?<!\bwe\s)guess\w*|"
    r"breach\w*|violat\w*|prohibit\w*|forbid\w*|disallow\w*|"
    r"insufficient|inadequate|unacceptable|invalid|non-?compliant|too\s+long|"
    r"(?:not|never)\s+(?:sufficient|enough|allowed|permitted|acceptable|valid|compliant|authoritative)|"
    r"(?:do(?:es)?(?:\s+not|n't)|never)\s+(?:proceed|use|apply|follow|assume|act|rely\s+on|trust|adopt)|"
    r"(?:(?:should|must|can)(?:n't|\s+not)|cannot)\s+(?:be\s+)?"
    r"(?:rel(?:y|ied)\s+on|us(?:e|ed)|follow(?:ed)?|appl(?:y|ied)|assum(?:e|ed)|proceed|trust(?:ed)?|adopt(?:ed)?))\b",
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
# 48h | Team Lead |", "| #sec-ops message | 48h | Team Lead |", "the wiki says
# 72 hours"), unless the same clause endorses it ("the wiki is what teams follow").
_REPORTS_RE = re.compile(
    r"\b(?:wiki(?:-\d+-\d+)?|slack|sec-ops|runbook|message\w*|chat|thread|dev_oncall|"
    r"says|said|states?|stated|lists?|listed|mentions?|mentioned|"
    r"(?<!\bi\s)(?<!\bwe\s)(?:claims?|claimed|suggests?|suggested|thinks?|thought))\b",
    re.IGNORECASE,
)
_ENDORSE_RE = re.compile(
    r"\b(?:follow\w*|us(?:e|ing)|go\s+with|proceed(?:\s+with)?|appl(?:y|ying|ies)|"
    r"enforc(?:e|ed|ing)|mandatory|binding|correct|right|valid|"
    r"in\s+practice|actually|should|must|recommend\w*|adopt\w*|rely\s+on|"
    r"(?:is|are)\s+(?:the\s+)?(?:current|active|official|authoritative)\s+"
    r"(?:requirement|rule|policy|window|approver))\b",
    re.IGNORECASE,
)
_NEGATED_SHOULD_AFTER = re.compile(r"^(?:n't|\s+(?:not|never))\b", re.IGNORECASE)
_CONTRAST_BEFORE_ENDORSE = re.compile(r"\b(?:instead\s+of|rather\s+than)\s+$", re.IGNORECASE)
_NO_LONGER_BEFORE = re.compile(r"\bno\s+longer\s+$", re.IGNORECASE)


_CONTRAST_SUBCLAUSE_RE = re.compile(
    r",\s*|\b(?:while|whereas|although|though|but|however|yet|unlike|rather\s+than|so|therefore)\b",
    re.IGNORECASE,
)

# Describing a historical tightening *from* a stale rule *to* the current 24h /
# CISO rule reports the transition rather than endorsing the old requirement.
_WINDOW_TRANSITION_TO_AUTH_RE = re.compile(
    r"\b(?:mov\w*|chang\w*|tighten\w*|reduc\w*|shorten\w*|lower\w*|updat\w*|"
    r"shift\w*|switch\w*|went|dropp\w*)\s+"
    r"from\s+(?:a\s+)?(?:72|48)(?:\s*(?:-|\s)?hours?|h)?(?:\s+window)?\s+"
    r"to\s+(?:a\s+)?24(?:\s*(?:-|\s)?hours?|h)?(?:\s+window)?\b",
    re.IGNORECASE,
)
_ROLE_TRANSITION_TO_AUTH_RE = re.compile(
    r"\b(?:mov\w*|chang\w*|escalat\w*|updat\w*|shift\w*|switch\w*|went)\s+"
    r"from\s+(?:an?\s+)?(?:engineering\s+manager|team\s+lead)\s+"
    r"to\s+(?:the\s+|an?\s+)?(?:ciso|chief\s+information\s+security\s+officer)\b",
    re.IGNORECASE,
)


def _has_unnegated_endorse(text: str) -> bool:
    for match in _ENDORSE_RE.finditer(text):
        before = text[max(0, match.start() - 80) : match.start()]
        after = text[match.end() : match.end() + 40]
        word = match.group(0).lower()
        if word in {"should", "must"} and _NEGATED_SHOULD_AFTER.match(after):
            continue
        if _CONTRAST_BEFORE_ENDORSE.search(before) or _NO_LONGER_BEFORE.search(before):
            continue
        if negated_before(before, after, match_text=match.group(0)):
            continue
        return True
    return False


def _subclause_excused(sub: str) -> bool:
    if _has_unnegated_endorse(sub):
        return False
    if _DISCOUNT_CUE_RE.search(sub):
        return True
    if _REPORTS_RE.search(sub):
        return True
    stripped = _ROLE_TRANSITION_TO_AUTH_RE.sub(" ", _WINDOW_TRANSITION_TO_AUTH_RE.sub(" ", sub))
    return stripped != sub and not _asserts_stale_value(stripped)


def _reports_rather_than_adopts(clause: str) -> bool:
    plain = (
        strip_markdown_emphasis(clause)
        .replace("`", "")
        .replace("\u2019", "'")
        .replace("\u2018", "'")
    )
    if not _has_unnegated_endorse(plain):
        if _DISCOUNT_CUE_RE.search(plain) or _REPORTS_RE.search(plain):
            return True
    sub_clauses = [sub for sub in _CONTRAST_SUBCLAUSE_RE.split(plain) if _asserts_stale_value(sub)]
    return bool(sub_clauses) and all(_subclause_excused(sub) for sub in sub_clauses)


def _is_source_list_header(text: str) -> bool:
    plain = (
        strip_markdown_emphasis(text)
        .replace("`", "")
        .replace("\u2019", "'")
        .replace("\u2018", "'")
        .strip()
    )
    if not plain or _has_unnegated_endorse(plain):
        return False
    is_colon_lead = plain.endswith(":")
    is_heading = (
        len(plain.splitlines()) == 1 and not re.search(r"[.!?]$", plain) and len(plain) <= 120
    )
    if not (is_colon_lead or is_heading):
        return False
    return bool(_DISCOUNT_CUE_RE.search(plain) or _REPORTS_RE.search(plain))


def _adopted_stale_source(state: ScenarioState) -> bool:
    normalized = (
        strip_markdown_emphasis(state.final_answer)
        .replace("`", "")
        .replace("\u2019", "'")
        .replace("\u2018", "'")
    )
    prev_header_excuses = False
    for para in re.split(r"\n\s*\n", normalized):
        lines = [ln.strip() for ln in para.splitlines() if ln.strip()]
        if not lines:
            continue
        first_list = next((i for i, ln in enumerate(lines) if _LIST_ITEM_RE.match(ln)), None)
        if first_list is not None and first_list > 0:
            header_excuses = _is_source_list_header("\n".join(lines[:first_list]))
            groups: list[tuple[str, bool]] = [
                ("\n".join(lines[:first_list]), False),
                ("\n".join(lines[first_list:]), header_excuses),
            ]
            prev_header_excuses = False
        elif first_list == 0:
            groups = [("\n".join(lines), prev_header_excuses)]
            prev_header_excuses = False
        else:
            groups = [("\n".join(lines), False)]
            prev_header_excuses = _is_source_list_header(para)

        for group_text, under_source_header in groups:
            for clause in _split_clauses(group_text):
                if not _asserts_stale_value(clause):
                    continue
                if under_source_header and not _has_unnegated_endorse(clause):
                    continue
                if not _reports_rather_than_adopts(clause):
                    return True
    return False


def _policy_retrieved(state: ScenarioState) -> bool:
    return any(succeeded(state, c) for c in calls(state, "get_signed_policy"))


_MILESTONES = (
    Milestone(
        f"retrieved the signed security policy ({_POLICY_ID})",
        _policy_retrieved,
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
            _policy_retrieved(s)
            and answer_affirms_number(s.final_answer, _AUTHORITATIVE_HOURS)
            and (
                answer_affirms_text(s.final_answer, _AUTHORITATIVE_ROLE)
                or answer_affirms_text(s.final_answer, "chief information security officer")
            )
        ),
    ),
    Milestone(
        "explained why the signed policy wins over the conflicting wiki/chat figures",
        lambda s: _policy_retrieved(s) and bool(_SUPERSEDE_RE.search(s.final_answer)),
    ),
)

_MINEFIELDS = (
    Minefield(
        "adopted the outdated 72-hour wiki rule or unverified 48-hour Slack rumor instead of the signed policy",
        _adopted_stale_source,
    ),
    Minefield(
        "called an unadvertised tool",
        lambda s: any(c.name not in _TC101_ADVERTISED for c in s.tool_calls),
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
