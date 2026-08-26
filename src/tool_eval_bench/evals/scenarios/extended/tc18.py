"""TC-18 — Translate & Forward."""

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
    answer_affirms_text as _answer_affirms_text,
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
    with_noise as _noise,
)
from tool_eval_bench.evals.scenarios.extended._shared import (
    _positive_argument_contains,
    _result_matches_if_present,
    _status_is_success,
)


def _tc18_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "translate_text":
        target = _normalize(_as_str(call.arguments.get("target_language")))
        if target in ("german", "de", "deutsch"):
            return _noise(
                {
                    "translated": "Der Termin wurde auf 15 Uhr verschoben. Bitte bestätigen Sie Ihre Teilnahme."
                },
                "translate_text",
            )
        if target in ("english", "en"):
            return _noise(
                {
                    "translated": "The meeting has been moved to 3 PM. Please confirm your attendance."
                },
                "translate_text",
            )
        return _noise({"error": f"Unsupported language: {target}"}, "translate_text")
    if call.name == "send_email":
        return _noise({"status": "sent", "message_id": "msg_1822"}, "send_email")
    if call.name == "get_contacts":
        return _noise(
            {
                "results": [
                    {"name": "Hans Müller", "email": "hans.mueller@firma.de"},
                    {"name": "Sarah Chen", "email": "sarah.chen@company.com"},
                ]
            },
            "get_contacts",
        )
    return _generic_tool_fallback(call)


def _tc18_translation_result_is_german(payload: Any) -> bool:
    if not isinstance(payload, dict) or "error" in payload:
        return False
    translated = _as_str(payload.get("translated"))
    return _includes_text(translated, "verschoben") and _includes_text(translated, "termin")


def _tc18_email_result_is_sent(payload: Any) -> bool:
    return _status_is_success(
        payload,
        statuses={"accepted", "delivered", "ok", "queued", "sent", "success"},
    )


def _tc18_eval(state: ScenarioState) -> ScenarioEvaluation:
    """Should translate the message to German and then email it.

    User says: 'Translate "The meeting has been moved to 3 PM. Please confirm
    your attendance." to German and email it to Hans.'
    """
    translate_calls = [c for c in state.tool_calls if c.name == "translate_text"]
    email_calls = [c for c in state.tool_calls if c.name == "send_email"]

    translations = [
        c
        for c in translate_calls
        if _normalize(_as_str(c.arguments.get("target_language"))) in ("german", "de", "deutsch")
    ]
    valid_translations = [
        c
        for c in translations
        if _normalize(_as_str(c.arguments.get("source_language"))) in ("english", "en")
        and _positive_argument_contains(c.arguments.get("text"), "meeting")
    ]
    emails = [
        c
        for c in email_calls
        if _normalize(_as_str(c.arguments.get("to"))) == "hans.mueller@firma.de"
    ]
    usable_translations = [
        call
        for call in valid_translations
        if _result_matches_if_present(state, call, _tc18_translation_result_is_german)
    ]
    usable_emails = [
        call
        for call in emails
        if _result_matches_if_present(state, call, _tc18_email_result_is_sent)
    ]
    translated_to_german = bool(translations)
    # Keep attempted, correctly addressed mutations visible to the partial
    # outcome.  Passing still requires a usable result and complete arguments.
    sent_email = bool(emails)
    email_has_german = any(
        _answer_affirms_text(_as_str(c.arguments.get("body")), "verschoben")
        or _answer_affirms_text(_as_str(c.arguments.get("body")), "termin")
        for c in usable_emails
    )
    ordered = any(t.turn < e.turn for t in usable_translations for e in usable_emails)
    if translated_to_german and sent_email and email_has_german and ordered:
        if len(translations) != 1 or len(emails) != 1 or len(email_calls) != 1:
            return _partial(
                "Translated and emailed the message but issued duplicate or incorrect email mutations."
            )
        return _pass("Translated to German and emailed the German version to Hans.")
    if translated_to_german and sent_email and not ordered:
        return _partial("Translated and emailed the message, but used the steps out of order.")
    if translated_to_german and sent_email and not email_has_german:
        return _partial(
            "Translated correctly but emailed the English version instead of the German one."
        )
    if translated_to_german and not sent_email:
        return _partial("Translated to German but didn't complete the email step.")
    return _fail("Did not translate and email correctly.")


SCENARIO = ScenarioDefinition(
    id="TC-18",
    title="Translate & Forward",
    category=Category.F,
    user_message='Translate "The meeting has been moved to 3 PM. Please confirm your attendance." to German and email it to Hans.',
    description="Translate to German and email the translated version.",
    handle_tool_call=_tc18_handle,
    evaluate=_tc18_eval,
    difficulty=3,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it translates to German first, then emails the German text to Hans.",
    "Fail if it skips translation or sends the English version.",
)
