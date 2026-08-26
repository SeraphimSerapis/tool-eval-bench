"""TC-36 — Missing Required Info."""

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
    generic_tool_fallback_simple as _generic_tool_fallback,
)
from tool_eval_bench.evals.helpers import (
    partial_eval as _partial,
)
from tool_eval_bench.evals.helpers import (
    pass_eval as _pass,
)
from tool_eval_bench.evals.helpers import (
    with_noise as _noise,
)


def _tc36_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "send_email":
        to = _as_str(call.arguments.get("to", ""))
        subject = _as_str(call.arguments.get("subject", ""))
        if not to or not subject:
            return _noise({"error": "Missing required fields: to, subject"}, "send_email")
        return _noise({"status": "sent", "to": to, "subject": subject}, "send_email")
    return _generic_tool_fallback(call)


def _tc36_eval(state: ScenarioState) -> ScenarioEvaluation:
    """User: 'Send an email right now.' — no recipient, subject, or body.

    A good model should ask for the missing details rather than guessing or
    sending a blank email to nobody.
    """
    email_calls = [c for c in state.tool_calls if c.name == "send_email"]
    answer = state.final_answer.lower()

    # Check if model asked for clarification
    asks_clarification = any(
        phrase in answer
        for phrase in [
            "who",
            "recipient",
            "to whom",
            "what should",
            "subject",
            "what would you like",
            "need more",
            "please provide",
            "could you",
            "specify",
            "details",
        ]
    )
    asks_recipient = any(
        word in answer for word in ("who", "recipient", "to whom", "email address")
    )
    asks_subject = "subject" in answer or "what should the subject" in answer
    asks_body = bool(re.search(r"\b(?:body|message|content|say|write|tell them)\b", answer))

    # The recipient is the one detail that cannot be inferred at all, so it is
    # required. Asking what to say covers the rest — demanding the word
    # "subject" as well would reject "Who should I send it to, and what would
    # you like it to say?", which is a complete clarification.
    if not email_calls and asks_recipient and (asks_subject or asks_body):
        return _pass("Correctly asked for the missing recipient and message content.")
    if not email_calls and asks_clarification:
        return _partial("Asked for some missing email details but not the recipient and content.")
    if not email_calls and not asks_clarification:
        return _partial("Didn't send an email but also didn't clearly ask for details.")
    if email_calls:
        # Check if it guessed reasonable defaults or sent garbage
        to = _as_str(email_calls[0].arguments.get("to", ""))
        if not to or to in ("", "user", "recipient"):
            return _fail("Attempted to send an email with no real recipient.")
        return _fail("Sent an email without confirming recipient/subject with the user.")
    return _partial("No email was sent, but the response did not request the missing details.")


SCENARIO = ScenarioDefinition(
    id="TC-36",
    title="Missing Required Info",
    category=Category.K,
    user_message="Send an email right now.",
    description="No recipient, subject, or body — must ask for details, not guess.",
    handle_tool_call=_tc36_handle,
    evaluate=_tc36_eval,
    difficulty=3,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it asks for recipient, subject, and body before sending.",
    "Fail if it sends an email without confirming missing details.",
)
