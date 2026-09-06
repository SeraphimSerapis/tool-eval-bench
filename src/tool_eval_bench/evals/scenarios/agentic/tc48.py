"""TC-48 — Additive Context (CC)."""

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


def _tc48_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    """Handle email drafting with incremental recipients."""
    if call.name == "send_email":
        to = call.arguments.get("to", "")
        cc = call.arguments.get("cc", "")
        subject = _as_str(call.arguments.get("subject", ""))
        body = _as_str(call.arguments.get("body", ""))
        # Store what was sent
        state.meta.setdefault("emails_sent", []).append(
            {
                "to": to,
                "cc": cc,
                "subject": subject,
                "body": body,
            }
        )
        return _noise(
            {
                "status": "sent",
                "message_id": f"msg_{len(state.meta.get('emails_sent', []))}",
            },
            "send_email",
        )
    if call.name == "get_contacts":
        query = _normalize(_as_str(call.arguments.get("query", "")))
        if "bob" in query:
            return _noise(
                {
                    "results": [
                        {"name": "Bob Martinez", "email": "bob.martinez@company.com"},
                    ]
                },
                "get_contacts",
            )
        if "alice" in query:
            return _noise(
                {
                    "results": [
                        {"name": "Alice Kim", "email": "alice.kim@company.com"},
                    ]
                },
                "get_contacts",
            )
        return _noise({"results": []}, "get_contacts")
    return _generic_tool_fallback(call)


def _tc48_eval(state: ScenarioState) -> ScenarioEvaluation:
    """Multi-turn additive context:
    Turn 1: prepare a concrete email to Alice without sending it.
    Turn 2: add Bob as CC and send it.

    The model must send exactly once, to Alice with Bob CC'd, preserving the
    supplied project-update content.

    Quality signals:
    - Models should use get_contacts to resolve proper email addresses.
    - Models that skip contact resolution and use bare names are downgraded.
    - Models that do preparatory work (contact lookups) and ask for
      clarification rather than fabricating get partial credit.
    """
    email_calls = [c for c in state.tool_calls if c.name == "send_email"]
    contact_calls = [c for c in state.tool_calls if c.name == "get_contacts"]
    answer = state.final_answer.lower()

    if any(
        call.name in {"create_calendar_event", "set_reminder", "run_code"}
        for call in state.tool_calls
    ):
        return _fail("Performed an unrelated side effect while updating the email draft.")

    # Did the model resolve contacts via get_contacts?
    used_contacts = len(contact_calls) > 0

    if not email_calls:
        # No email sent — but did the model do responsible prep work?
        if used_contacts:
            # Model looked up contacts and chose to ask for clarification
            # rather than fabricate content — partial credit for responsible
            # behavior (contact resolution + honest clarification).
            asks_for_content = any(
                phrase in answer
                for phrase in (
                    "what would you like",
                    "what should",
                    "what do you want",
                    "could you provide",
                    "can you provide",
                    "please provide",
                    "please share",
                    "share the details",
                    "provide the content",
                    "need the content",
                    "need the actual content",
                    "need the details",
                    "need more detail",
                    "what to include",
                    "what to say",
                    "more information",
                    "more details",
                    "let me know what",
                    "tell me what",
                    "could you tell me",
                    "can you tell me",
                    "tell me the key points",
                    "tell me the details",
                    "before i can send",
                    "send me the",
                    "could you share",
                    "can you share",
                    "what's the content",
                    "what is the content",
                )
            )
            if asks_for_content:
                return _partial(
                    "Resolved contacts but asked for email content instead of sending "
                    "— responsible, but the task asked to send."
                )
        return _fail("Did not send any emails.")

    if any(not _call_at_or_after_user_phase(call, 1) for call in email_calls):
        return _fail("Sent the draft before the user supplied the CC and authorized delivery.")

    for call in email_calls:
        to = call.arguments.get("to")
        subject = call.arguments.get("subject")
        body = call.arguments.get("body")
        cc = call.arguments.get("cc", "")
        if (
            not isinstance(to, str)
            or not to.strip()
            or not isinstance(subject, str)
            or not subject.strip()
            or not isinstance(body, str)
            or not body.strip()
            or not isinstance(cc, str)
        ):
            return _fail("Used an invalid send_email payload instead of a deliverable draft.")

    if any(not _result_is_usable_if_present(state, call) for call in email_calls):
        return _partial("The send_email attempt failed, so delivery cannot receive full credit.")

    bare_alice = any(
        re.search(r"(?<![a-z])alice(?![a-z])", _as_str(c.arguments.get("to", "")).lower())
        for c in email_calls
    )
    bare_bob_merged = any(
        re.search(r"(?<![a-z])bob(?![a-z])", _as_str(c.arguments.get("cc", "")).lower())
        or (
            "alice" in _as_str(c.arguments.get("to", "")).lower()
            and "bob" in _as_str(c.arguments.get("to", "")).lower()
        )
        for c in email_calls
    )
    if (
        bare_alice
        and bare_bob_merged
        and not any(
            (
                "@" in _as_str(c.arguments.get("to", ""))
                and "@" in _as_str(c.arguments.get("cc", ""))
            )
            or _as_str(c.arguments.get("to", "")).count("@") >= 2
            for c in email_calls
        )
    ):
        return _partial("Merged recipients, but used bare names instead of resolved addresses.")

    # Check if any email included Alice
    alice_emails = [
        c
        for c in email_calls
        if re.search(
            r"(?<![a-z])alice(?![a-z])[^,;\s]*@",
            _as_str(c.arguments.get("to", "")).lower(),
        )
    ]
    if not alice_emails:
        return _fail("Sent email but not to Alice.")

    allowed_addresses = {"alice.kim@company.com", "bob.martinez@company.com"}
    used_addresses = {
        address.lower()
        for call in email_calls
        for field in ("to", "cc")
        for address in re.findall(
            r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}",
            _as_str(call.arguments.get(field, "")),
            re.IGNORECASE,
        )
    }
    if not used_addresses <= allowed_addresses:
        return _fail("Sent the project update to an unintended recipient.")

    # Check for Bob being CC'd (ideal) or model acknowledging the limitation
    bob_ccd = any(
        re.search(r"(?<![a-z])bob(?![a-z])[^,;\s]*@", _as_str(c.arguments.get("cc", "")).lower())
        for c in email_calls
    )
    bob_in_to = any(
        re.search(r"(?<![a-z])bob(?![a-z])[^,;\s]*@", _as_str(c.arguments.get("to", "")).lower())
        for c in email_calls
    )
    if len(email_calls) > 1:
        if bob_ccd or bob_in_to:
            return _partial("Sent more than once instead of preserving and updating the draft.")
        return _fail("Sent multiple emails without including Bob in the requested workflow.")
    explains_already_sent = any(
        phrase in answer
        for phrase in (
            "already sent",
            "already been sent",
            "was already",
            "can't add cc",
            "cannot add",
            "already delivered",
        )
    )

    # Helper: did the model use a resolved email address (contains "@")?
    def _used_real_address(*fields: str) -> bool:
        """Check if any email call used a resolved address (with @) for the given fields."""
        for call in email_calls:
            for field in fields:
                val = _as_str(call.arguments.get(field, "")).lower()
                if val and "@" in val:
                    return True
        return False

    resolved_addresses = _used_real_address("to", "cc")
    preserved_content = all(
        phrase in _as_str(email_calls[0].arguments.get("body", "")).lower()
        for phrase in ("launch", "on track")
    )

    if bob_ccd:
        if resolved_addresses and preserved_content:
            return _pass("Sent email to Alice with Bob CC'd — correctly merged additive context.")
        if resolved_addresses:
            return _partial("Merged recipients but lost the supplied project-update content.")
        return _partial(
            "Merged CC correctly but used bare names instead of resolving "
            "contacts — addresses wouldn't work in a real system."
        )
    if bob_in_to:
        return _partial(
            "Included Bob as a direct recipient instead of preserving the requested CC."
        )
    if explains_already_sent:
        return _partial("Acknowledged the email was already sent — honest about the limitation.")
    # Check if the model mentioned Bob at all in the final answer
    bob_mentioned = "bob" in answer
    if len(email_calls) == 1 and bob_mentioned:
        return _partial("Sent to Alice and acknowledged Bob but didn't send a follow-up.")
    if len(email_calls) == 1:
        return _fail("Sent to Alice but ignored the follow-up to CC Bob.")
    return _partial("Handled the email but the CC request wasn't clearly addressed.")


SCENARIO = ScenarioDefinition(
    id="TC-48",
    title="Additive Context (CC)",
    category=Category.I,
    user_message="Prepare an email to Alice saying 'Project update: launch remains on track.' Don't send it yet.",
    description="Retain concrete email content, add a CC recipient, then send once.",
    handle_tool_call=_tc48_handle,
    evaluate=_tc48_eval,
    follow_up_messages=["Also CC Bob on that email, then send it."],
    difficulty=3,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it sends email to Alice with Bob CC'd using resolved addresses.",
    "Fail if it ignores the CC request. Partial if it merges CC but uses bare names, "
    "or if it resolves contacts and asks for content instead of sending.",
)
