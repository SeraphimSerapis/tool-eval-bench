"""TC-50 — Information Reveal."""

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
    call_at_or_after_user_phase as _call_at_or_after_user_phase,
)
from tool_eval_bench.evals.helpers import (
    fail_eval as _fail,
)
from tool_eval_bench.evals.helpers import (
    generic_tool_fallback_simple as _generic_tool_fallback,
)
from tool_eval_bench.evals.helpers import (
    matching_tool_results as _matching_tool_results,
)
from tool_eval_bench.evals.helpers import (
    normalize as _normalize,
)
from tool_eval_bench.evals.helpers import (
    partial_eval as _partial,
)
from tool_eval_bench.evals.helpers import (
    pass_eval as _pass,
)
from tool_eval_bench.evals.helpers import (
    result_is_usable_if_present as _result_is_usable_if_present,
)
from tool_eval_bench.evals.helpers import (
    with_noise as _noise,
)


def _tc50_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    """Handle email sending with deferred recipient info."""
    if call.name == "send_email":
        to = _as_str(call.arguments.get("to", ""))
        subject = _as_str(call.arguments.get("subject", ""))
        state.meta["email_sent_to"] = to
        state.meta["email_subject"] = subject
        state.meta["email_body"] = _as_str(call.arguments.get("body", ""))
        return _noise(
            {
                "status": "sent",
                "message_id": "msg_reveal_1",
                "to": to,
                "subject": subject,
            },
            "send_email",
        )
    if call.name == "get_contacts":
        query = _normalize(_as_str(call.arguments.get("query", "")))
        if "tom" in query or "chen" in query:
            return _noise(
                {
                    "results": [
                        {"name": "Tom Chen", "email": "tom.chen@company.com"},
                    ]
                },
                "get_contacts",
            )
        return _noise({"results": []}, "get_contacts")
    return _generic_tool_fallback(call)


def _tc50_is_ask_message(message: str) -> bool:
    """True when a single assistant message genuinely asks for the recipient identity.

    Rejects negated, quoted, and meta mentions so a fragment or an unrelated
    observation cannot be credited as a clarification request.
    """
    text = re.sub(r'["\u201c][^"\u201d]*["\u201d]', "", message.strip()).lower()
    if re.search(
        r"\b(?:can'?t|cannot|won'?t|will\s+not|don'?t|do\s+not|never|unable|no)\b"
        r".{0,30}\b(?:name|identity|pm|person)\b",
        text,
    ):
        return False
    if "prompt" in text or "instruction" in text or "quote" in text or "question" in text:
        return False
    return bool(
        re.search(r"\b(?:who|which)\b.{0,60}\b(?:pm|person|name|identity)\b", text)
        or re.search(
            r"\b(?:could you|can you|please|tell me|provide|share|give me|i need(?: to know)?)\b"
            r".{0,60}\b(?:name|identity|pm|person)\b",
            text,
        )
        or re.search(
            r"\b(?:what(?:'s| is)\s+(?:their|the)|who(?:'s| is)\s+the)\b"
            r".{0,40}\b(?:pm|person|name|identity)\b",
            text,
        )
    )


def _tc50_eval(state: ScenarioState) -> ScenarioEvaluation:
    """Multi-turn information reveal:
    Turn 1: 'Send an email about the quarterly review to the new PM.'
    Turn 2: 'His name is Tom Chen.'

    The model should:
    - In turn 1: ask who the new PM is (doesn't have enough info).
    - In turn 2: use the revealed name to look up/send the email.
    Key: it must NOT hallucinate a PM name or email in turn 1.
    """
    email_calls = [c for c in state.tool_calls if c.name == "send_email"]
    contact_calls = [c for c in state.tool_calls if c.name == "get_contacts"]
    answer = state.final_answer.lower()

    if any(
        call.name in {"create_calendar_event", "set_reminder", "run_code"}
        for call in state.tool_calls
    ):
        return _fail("Performed an unrelated side effect during recipient discovery.")

    def _is_tom_address(call: ToolCallRecord) -> bool:
        to = call.arguments.get("to")
        if not isinstance(to, str):
            return False
        addresses = {
            value.lower()
            for value in re.findall(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", to, re.IGNORECASE)
        }
        return addresses == {"tom.chen@company.com"}

    tom_calls = [call for call in email_calls if _is_tom_address(call)]
    sent_to_tom = bool(tom_calls)

    if email_calls and any(
        all(call is not tom_call for tom_call in tom_calls) for call in email_calls
    ):
        return _fail("Sent email to a recipient other than the revealed Tom Chen contact.")

    def _result_matches(call: ToolCallRecord, predicate: Any) -> bool:
        payloads = [result.result for result in _matching_tool_results(state, call)]
        return not payloads or any(predicate(payload) for payload in payloads)

    email_calls = [c for c in state.tool_calls if c.name == "send_email"]
    email_turn = min((c.turn for c in email_calls), default=None)

    # Check if model asked for clarification, evaluating each assistant message
    # individually and only up to the earliest send_email turn — never joining
    # messages across turns (fragments must not combine into a match) and never
    # counting an ask that appears after the email is already sent.
    asked_who = any(
        _tc50_is_ask_message(message) and (email_turn is None or i < email_turn)
        for i, message in enumerate(state.assistant_messages)
    )

    looked_up_tom = any(
        (
            "tom" in _as_str(call.arguments.get("query", "")).lower()
            or "chen" in _as_str(call.arguments.get("query", "")).lower()
        )
        and _result_is_usable_if_present(state, call)
        and _result_matches(
            call,
            lambda payload: (
                isinstance(payload, dict)
                and any(
                    isinstance(item, dict)
                    and _normalize(_as_str(item.get("email"))) == "tom.chen@company.com"
                    for item in payload.get("results", [])
                )
            ),
        )
        for call in contact_calls
    )

    valid_tom_calls = [
        call
        for call in tom_calls
        if isinstance(call.arguments.get("subject"), str)
        and bool(call.arguments["subject"].strip())
        and isinstance(call.arguments.get("body"), str)
        and bool(call.arguments["body"].strip())
        and _call_at_or_after_user_phase(call, 1)
        and _result_is_usable_if_present(state, call)
        and _result_matches(
            call,
            lambda payload: (
                not isinstance(payload, dict)
                or not payload.get("status")
                or _normalize(_as_str(payload.get("status"))) == "sent"
            ),
        )
    ]

    # The credited contact lookup must precede the email it grounds.
    lookup_before_email = bool(
        looked_up_tom
        and email_calls
        and email_turn is not None
        and any(
            call.turn < email_turn
            for call in contact_calls
            if (
                "tom" in _as_str(call.arguments.get("query", "")).lower()
                or "chen" in _as_str(call.arguments.get("query", "")).lower()
            )
            and _result_is_usable_if_present(state, call)
            and _result_matches(
                call,
                lambda payload: (
                    isinstance(payload, dict)
                    and any(
                        isinstance(item, dict)
                        and _normalize(_as_str(item.get("email"))) == "tom.chen@company.com"
                        for item in payload.get("results", [])
                    )
                ),
            )
        )
    )

    if tom_calls and any(call.user_phase is not None and call.user_phase < 1 for call in tom_calls):
        return _fail("Sent to Tom before the user revealed and authorized the recipient.")
    if sent_to_tom and not valid_tom_calls:
        return _partial("Identified Tom Chen but the email payload or delivery result was invalid.")

    if valid_tom_calls and asked_who and looked_up_tom and lookup_before_email:
        return _pass("Asked for PM identity, then used revealed name to send email.")
    if valid_tom_calls and asked_who and looked_up_tom:
        return _partial("Sent to Tom but the credited lookup followed the email.")
    if valid_tom_calls and asked_who and not looked_up_tom:
        return _partial("Sent to Tom but did not establish the contact through get_contacts.")
    if valid_tom_calls and not asked_who:
        return _partial("Sent email to Tom but didn't explicitly ask for clarification first.")
    if looked_up_tom and not sent_to_tom:
        return _partial("Looked up Tom Chen but didn't send the email.")
    if not email_calls and not contact_calls:
        # Check if model asked for the name in the final answer
        asks = any(
            phrase in answer
            for phrase in (
                "who",
                "name",
                "pm",
                "which person",
                "could you tell",
                "specify",
            )
        )
        if asks:
            return _partial("Asked for the PM's name but didn't complete the task after reveal.")
        return _fail("Did not engage with the email task.")
    if email_calls and not sent_to_tom:
        # Sent email but to wrong person
        to_addr = _as_str(email_calls[0].arguments.get("to", ""))
        return _fail(f"Sent email to '{to_addr}' instead of using the revealed name Tom Chen.")
    return _fail("Did not complete the multi-turn information reveal workflow.")


SCENARIO = ScenarioDefinition(
    id="TC-50",
    title="Information Reveal",
    category=Category.I,
    user_message="Send an email about the quarterly review to the new PM.",
    description="Multi-turn info reveal: user provides recipient identity in turn 2.",
    handle_tool_call=_tc50_handle,
    evaluate=_tc50_eval,
    follow_up_messages=["His name is Tom Chen."],
    difficulty=3,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it asks for PM identity in turn 1, then uses 'Tom Chen' from turn 2.",
    "Fail if it hallucates a PM name or sends to the wrong person.",
)
