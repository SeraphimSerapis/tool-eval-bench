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


_TC50_STRAIGHT_QUOTES = re.compile(r"(?<!\w)'[^']*'(?!\w)")
_TC50_CURLY_QUOTES = re.compile(r"[“‘][^”’]*[”’]")
# Targeted rejections: only statements *about* asking/providing/knowing the
# identity, rhetorical uses, and meta descriptions fail to be a request. A
# negation that does not touch the ask ("I can't send it yet; who is the PM?")
# must still count.
_TC50_NEGATED_ASK = re.compile(
    r"\b(?:do\s+not|don'?t|doesn'?t|did\s+not|didn'?t|won'?t|will\s+not|never|no|can'?t|cannot)\b"
    r"[^;.?]{0,40}\b(?:ask|need|want|provide|disclose|say|reveal|know|tell)\b"
)
_TC50_DECLARED_KNOWLEDGE = re.compile(r"\bi\s+(?:know|knew|remember|forgot)\b")
_TC50_RHETORICAL = re.compile(r"\b(?:believe|imagine)\b")
_TC50_META = re.compile(
    r"\b(?:prompt|sentence|example|instruction|question|it)\b[^;.?]{0,30}\b(?:contains?|says|read)"
    r"|\b(?:the\s+test|this\s+example)\b"
)


def _tc50_is_ask_message(message: str) -> bool:
    """True when a message contains a genuine request for the recipient identity."""
    text = _TC50_CURLY_QUOTES.sub(" ", message.strip())
    text = _TC50_STRAIGHT_QUOTES.sub(" ", text).lower().strip()
    clauses = re.split(r"[;\n]+|(?<=[.!?])\s+", text)

    for clause in clauses:
        if (
            _TC50_NEGATED_ASK.search(clause)
            or _TC50_DECLARED_KNOWLEDGE.search(clause)
            or _TC50_RHETORICAL.search(clause)
            or _TC50_META.search(clause)
        ):
            continue
        if (
            re.search(r"\bwho(?:'s| is)\b.{0,60}\b(?:pm|person)\b", clause)
            or re.search(
                r"\bwhich(?:\s+(?:person|name|identity|pm))?\s+"
                r"(?:is|was|would be|will be)\b.{0,60}\b(?:pm|person|name|identity)\b",
                clause,
            )
            or re.search(
                r"\b(?:please|tell me|provide|share|give me|i need(?: to know)?|"
                r"(?:could|can) you (?:tell me|provide|share|give me))\b"
                r".{0,60}\b(?:name|identity|pm|person)\b",
                clause,
            )
            or re.search(
                r"\b(?:could|can) you\b.{0,60}\b(?:name|identity)\b",
                clause,
            )
            or re.search(
                r"\bwhat(?:'s| is)\s+(?:their|the)\b"
                r".{0,40}\b(?:identity|name(?!['\u2019]s))\b",
                clause,
            )
        ):
            return True
    return False


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
    # individually and only in a turn strictly before the earliest send_email
    # turn. Message slots are numbered as one-based turns (matching the
    # one-based ToolCallRecord.turn) so an ask in the same turn as the email is
    # not credited, fragments never combine across turns, and an ask after the
    # send cannot count.
    asked_who = any(
        _tc50_is_ask_message(message) and (email_turn is None or turn < email_turn)
        for turn, message in enumerate(state.assistant_messages, start=1)
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
        return _partial("Sent to Tom but no credited lookup preceded the email.")
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
