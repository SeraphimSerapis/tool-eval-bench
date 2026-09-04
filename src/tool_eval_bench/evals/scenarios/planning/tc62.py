"""TC-62 — 5-Turn Research Chain."""

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
    as_str as _as_str,
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
    tool_calls_by_name as _tool_calls_by_name,
)
from tool_eval_bench.evals.helpers import (
    with_noise as _noise,
)
from tool_eval_bench.evals.scenarios.planning._shared import (
    _call_index,
    _result_has_status,
    _result_matches_if_present,
)

_TC62_FOLLOW_UPS = [
    # Turn 2: drill into detail + correction
    "Wait, I just remembered the Q3 report had a correction. Can you search for and read the latest version?",
    # Turn 3: pivot to competitor
    "OK, now compare that against our competitor. Search for Acme Corp's Q3 performance.",
    # Turn 4: action based on findings
    "Based on this analysis, draft an email to the CFO summarizing our competitive position.",
    # Turn 5: revision and explicit authorization
    "Actually, make the tone more optimistic, add that we expect Q4 to improve, and send it.",
]


def _tc62_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "web_search":
        query = _as_str(call.arguments.get("query", "")).lower()
        if "acme" in query:
            return _noise(
                {"results": [{"snippet": "Acme Corp Q3 revenue: $3.8M. Growth rate: 12%."}]},
                "web_search",
            )
        return _noise(
            {"results": [{"snippet": "Company Q3 performance: Revenue up 8% YoY."}]},
            "web_search",
        )
    if call.name == "search_files":
        return _noise(
            {"results": [{"file_id": "q3_latest", "name": "Q3_Report_v2_CORRECTED.xlsx"}]},
            "search_files",
        )
    if call.name == "read_file":
        fid = _as_str(call.arguments.get("file_id", ""))
        if "latest" in fid or "v2" in fid or "correct" in fid.lower():
            return _noise(
                {
                    "content": "Q3 Report (CORRECTED)\nRevenue: $4,150,000\nNote: Previous version showed $4.4M due to accounting error."
                },
                "read_file",
            )
        return _noise(
            {"content": "Q3 Report\nRevenue: $4,400,000\nGrowth: 8% YoY"},
            "read_file",
        )
    if call.name == "calculator":
        from tool_eval_bench.evals.helpers import parse_math_expression

        expr = _as_str(call.arguments.get("expression", ""))
        result = parse_math_expression(expr)
        if result is not None:
            return _noise({"result": result}, "calculator")
        return _noise({"error": "Invalid expression."}, "calculator")
    if call.name == "send_email":
        return _noise({"status": "sent"}, "send_email")
    if call.name == "get_contacts":
        return _noise(
            {"results": [{"name": "CFO", "email": "cfo@company.com", "role": "CFO"}]},
            "get_contacts",
        )
    return _noise({"error": f"Tool {call.name} is not relevant."}, call.name)


def _tc62_eval(state: ScenarioState) -> ScenarioEvaluation:
    """5-turn research chain testing context persistence and revision handling.

    Key checkpoints:
    - Used corrected revenue ($4.15M not $4.4M)
    - Searched for Acme competitor data
    - Sent email to CFO
    """
    # Check for corrected data usage and preserve the lookup dependency.
    transcript = "\n".join(state.assistant_messages).lower()

    def corrected_search_result(payload: Any) -> bool:
        return "q3_latest" in str(payload) or "corrected" in str(payload).lower()

    def corrected_file_result(payload: Any) -> bool:
        text = str(payload).lower().replace(",", "")
        return "4150000" in text and "corrected" in text

    def acme_result(payload: Any) -> bool:
        text = str(payload).lower().replace(",", "")
        return "acme" in text and "3.8m" in text

    def email_result_is_sent(payload: Any) -> bool:
        return _result_has_status(payload, "sent")

    corrected_search_calls = [
        call
        for call in _tool_calls_by_name(state, "search_files")
        if _result_matches_if_present(state, call, corrected_search_result)
    ]
    corrected_file_calls = [
        call
        for call in _tool_calls_by_name(state, "read_file")
        if any(
            token in _as_str(call.arguments.get("file_id")).lower()
            for token in ("latest", "correct", "v2")
        )
        and _result_matches_if_present(state, call, corrected_file_result)
    ]
    corrected_lookup = bool(
        corrected_search_calls
        and corrected_file_calls
        and _call_index(state, corrected_search_calls[0])
        < _call_index(state, corrected_file_calls[0])
    )
    searched_acme_calls = [
        call
        for call in _tool_calls_by_name(state, "web_search")
        if "acme" in _as_str(call.arguments.get("query", "")).lower()
        and _result_matches_if_present(state, call, acme_result)
    ]
    searched_acme = bool(searched_acme_calls)
    cfo_contact_calls = [
        call
        for call in _tool_calls_by_name(state, "get_contacts")
        if "cfo" in _as_str(call.arguments.get("query")).lower()
        and _result_matches_if_present(
            state,
            call,
            lambda payload: (
                isinstance(payload, dict)
                and any(
                    isinstance(item, dict)
                    and _as_str(item.get("email")).strip().lower() == "cfo@company.com"
                    for item in payload.get("results", [])
                )
            ),
        )
    ]
    resolved_cfo = bool(cfo_contact_calls)
    email_calls = [
        call
        for call in _tool_calls_by_name(state, "send_email")
        if _as_str(call.arguments.get("to")).strip().lower() == "cfo@company.com"
        and _as_str(call.arguments.get("subject")).strip()
        and _as_str(call.arguments.get("body")).strip()
        and _result_matches_if_present(state, call, email_result_is_sent)
    ]
    email_attempts = _tool_calls_by_name(state, "send_email")
    cfo_email_attempts = [
        call
        for call in email_attempts
        if _as_str(call.arguments.get("to")).strip().lower() == "cfo@company.com"
    ]
    sent_email = bool(email_attempts)
    email_to_cfo = bool(cfo_email_attempts)
    email_body = (
        _as_str((email_calls or cfo_email_attempts or email_attempts)[-1].arguments.get("body"))
        if (email_calls or cfo_email_attempts or email_attempts)
        else ""
    )
    body_lower = email_body.lower()
    # A stated figure under a negation ("we are NOT ahead", "Acme is not $3.8M")
    # is evidence about the conversation, not the corrected/competitor figure.
    negated_figure = bool(
        re.search(
            r"\b(?:not|never|no|without)\b.{0,40}\b(?:corrected|4150000|3\.8|3800000|acme|lead)\b",
            body_lower,
        )
    )
    body_has_corrected = not negated_figure and (
        "4.15" in body_lower or "4150000" in body_lower.replace(",", "")
    )
    body_has_competitor = (
        not negated_figure
        and "acme" in body_lower
        and (
            "3.8" in body_lower or "3,800" in body_lower or "3800000" in body_lower.replace(",", "")
        )
    )
    body_is_optimistic = any(
        word in body_lower for word in ("optimistic", "improve", "growth", "positive", "expect")
    )
    phase_data_present = any(call.user_phase is not None for call in state.tool_calls)
    # `email_calls` holds only the *qualifying* sends — right recipient, subject,
    # body, and a "sent" result. A model that emails the wrong address, or omits
    # a subject, leaves it empty while still producing phase data, so this must
    # be guarded the way `email_after_research` below already is.
    email_after_revision = not phase_data_present or bool(
        email_calls and email_calls[-1].user_phase is not None and email_calls[-1].user_phase >= 4
    )
    research_indices = [
        *[_call_index(state, call) for call in corrected_file_calls],
        *[_call_index(state, call) for call in searched_acme_calls],
        *[_call_index(state, call) for call in cfo_contact_calls],
    ]
    email_after_research = bool(
        email_calls
        and research_indices
        and max(research_indices) < _call_index(state, email_calls[-1])
    )
    email_is_single_and_safe = (
        len(email_calls) == 1 and len(_tool_calls_by_name(state, "send_email")) == 1
    )

    used_corrected = corrected_lookup and (
        "4,150,000" in transcript or "4.15" in transcript or "4150" in transcript
    )

    checks = sum([used_corrected, searched_acme, sent_email])

    if (
        checks == 3
        and email_to_cfo
        and resolved_cfo
        and body_has_corrected
        and body_has_competitor
        and body_is_optimistic
        and email_after_revision
        and email_after_research
        and email_is_single_and_safe
    ):
        return _pass(
            "Completed 5-turn chain: corrected data → competitor → CFO email with optimistic tone."
        )
    if checks == 3 and email_to_cfo:
        return _partial(
            "Sent CFO email but missed contact resolution, corrected data, competitor, "
            "or optimistic revision."
        )
    if checks == 3:
        return _partial("Completed research chain but email wasn't addressed to CFO.")
    if checks >= 2:
        missing = []
        if not used_corrected:
            missing.append("corrected revenue")
        if not searched_acme:
            missing.append("competitor research")
        if not sent_email:
            missing.append("CFO email")
        return _partial(f"Partial chain completion. Missing: {', '.join(missing)}.")
    if checks == 1:
        return _partial("Only completed 1/3 key checkpoints in the 5-turn chain.")
    return _fail("Failed to maintain context across the 5-turn research chain.")


SCENARIO = ScenarioDefinition(
    id="TC-62",
    title="5-Turn Research Chain",
    category=Category.I,
    user_message="Can you help me put together a competitive analysis report? Start by looking up our latest quarterly performance.",
    description="5-turn research chain with data correction, competitor pivot, and revision.",
    handle_tool_call=_tc62_handle,
    evaluate=_tc62_eval,
    follow_up_messages=_TC62_FOLLOW_UPS,
    # The attainable reference path contains dependent search/read rounds,
    # a competitor lookup, a draft turn, contact resolution, delivery, and
    # a final response. The default eight turns cannot reach authorization.
    max_turns_override=14,
    difficulty=4,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it handles all 5 turns: research → correct data → competitor → CFO email.",
    "Fail if it loses context or ignores the correction/revision.",
)
