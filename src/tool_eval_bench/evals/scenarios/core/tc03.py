"""TC-03 — Implicit Tool Need."""

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
    generic_tool_fallback as _generic_tool_fallback,
)
from tool_eval_bench.evals.helpers import (
    includes_text as _includes_text,
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
    tool_calls_by_name as _tool_calls_by_name,
)
from tool_eval_bench.evals.helpers import (
    with_noise as _noise,
)
from tool_eval_bench.evals.scenarios.core._shared import (
    _positive_argument_contains,
    _result_matches_if_present,
    _tc03_email_result_is_sent,
)


def _tc03_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_contacts":
        return _noise(
            {"results": [{"name": "Sarah Chen", "email": "sarah.chen@company.com"}]}, "get_contacts"
        )
    if call.name == "send_email":
        return _noise({"status": "sent", "message_id": "msg_8821"}, "send_email")
    return _generic_tool_fallback(call)


def _tc03_time_3pm(value: str) -> bool:
    """True if ``value`` spells the requested 3 PM in a common way.

    Accepts 12-hour spellings (3pm, 3 PM, 3:00 PM, 3 p.m.) as well as
    24-hour equivalents (15:00, 1500).  Other minutes such as 3:30 PM are
    intentionally not accepted because they are a different time.
    """
    text = value.lower()
    return bool(
        re.search(r"\b15:00\b|\b1500\b", text) or re.search(r"\b3(?::?00)?\s*p\.?m\.?\b", text)
    )


def _tc03_contact_result_has_sarah(payload: Any) -> bool:
    """Return whether a contact result identifies Sarah's company address."""
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        return False
    return any(
        isinstance(item, dict)
        and _includes_text(item.get("name"), "sarah")
        and _normalize(_as_str(item.get("email"))) == "sarah.chen@company.com"
        for item in payload["results"]
    )


def _tc03_eval(state: ScenarioState) -> ScenarioEvaluation:
    contact_calls = [
        call
        for call in _tool_calls_by_name(state, "get_contacts")
        if _positive_argument_contains(call.arguments.get("query"), "sarah")
    ]
    email_calls = _tool_calls_by_name(state, "send_email")
    complete_email_calls = [
        call
        for call in email_calls
        if (
            _normalize(_as_str(call.arguments.get("to"))) == "sarah.chen@company.com"
            and bool(_as_str(call.arguments.get("subject")).strip())
            and bool(_as_str(call.arguments.get("body")).strip())
            and re.search(
                r"\b(?:the\s+)?(?:meeting\s+)?(?:has\s+been\s+|was\s+)?moved\s+to\s+"
                r"(?:3(?::00)?\s*p\.?m\.?|15:00)\b",
                _as_str(call.arguments.get("body")),
                re.IGNORECASE,
            )
            and not re.search(
                r"\b(?:not|never|no)\s+(?:been\s+)?moved\b",
                _as_str(call.arguments.get("body")),
                re.IGNORECASE,
            )
            and _tc03_time_3pm(_as_str(call.arguments.get("body")))
        )
    ]
    ordered_pairs = [
        (contact, email)
        for contact in contact_calls
        for email in complete_email_calls
        if contact.turn < email.turn
    ]
    usable_pairs = [
        (contact, email)
        for contact, email in ordered_pairs
        if _result_matches_if_present(state, contact, _tc03_contact_result_has_sarah)
        and _result_matches_if_present(state, email, _tc03_email_result_is_sent)
    ]
    if usable_pairs:
        return _pass("Looked up Sarah before sending the email.")

    if ordered_pairs:
        contact_usable = any(
            _result_matches_if_present(state, contact, _tc03_contact_result_has_sarah)
            for contact, _ in ordered_pairs
        )
        email_usable = any(
            _result_matches_if_present(state, email, _tc03_email_result_is_sent)
            for _, email in ordered_pairs
        )
        if not contact_usable and not email_usable:
            return _partial(
                "The contact lookup and email send did not return usable results, "
                "so the message could not be confirmed."
            )
        if not contact_usable:
            return _partial(
                "The contact lookup did not return Sarah's address, "
                "so the recipient could not be confirmed."
            )
        return _partial(
            "send_email did not return a successful result, so delivery could not be confirmed."
        )

    if (
        not contact_calls
        and not email_calls
        and re.search(r"email", state.final_answer, re.IGNORECASE)
        and "?" in state.final_answer
    ):
        return _partial("Asked for Sarah's email instead of inferring the tool chain.")
    if (
        contact_calls
        and email_calls
        and any(
            contact.turn <= email.turn
            and _normalize(_as_str(email.arguments.get("to"))) == "sarah.chen@company.com"
            for contact in contact_calls
            for email in email_calls
        )
    ):
        return _partial("Looked up Sarah and attempted the email, but the message was incomplete.")
    return _fail("Did not complete the contact lookup to email chain correctly.")


SCENARIO = ScenarioDefinition(
    id="TC-03",
    title="Implicit Tool Need",
    category=Category.A,
    user_message="I need to let Sarah know the meeting moved to 3pm.",
    description="Infer get_contacts followed by send_email.",
    handle_tool_call=_tc03_handle,
    evaluate=_tc03_eval,
    difficulty=2,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it looks up Sarah first, then sends the message with the resolved address.",
    "Fail if it invents Sarah's email or never completes the contact-to-email chain.",
)
