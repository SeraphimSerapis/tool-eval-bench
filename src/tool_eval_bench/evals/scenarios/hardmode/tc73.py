"""TC-73 — Multi-Constraint Composition."""

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
    _PROSE_APOSTROPHES,
    addressed_recipients,
    answer_affirms_text,
    as_str,
    counted_calls,
    full_assistant_transcript,
    generic_tool_fallback,
    includes_text,
    matching_tool_results,
    normalize,
    result_is_usable_if_present,
    strip_markdown_emphasis,
    tool_calls_by_name,
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
from tool_eval_bench.evals.helpers import unsafe_eval as _unsafe
from tool_eval_bench.evals.helpers import (
    with_noise as _noise,
)
from tool_eval_bench.evals.scenarios.hardmode._shared import (
    _UNRELATED_UNIVERSAL_MUTATIONS,
)

_TC73_DINING_TERMS = (
    "restaurant",
    "vegan",
    "dining",
    "food",
    "bistro",
    "cafe",
    "eat",
    "dinner",
    "lunch",
)


def _tc73_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "web_search":
        q = normalize(as_str(call.arguments.get("query")))
        if any(term in q for term in _TC73_DINING_TERMS) and (
            "berlin" in q or "alexanderplatz" in q
        ):
            return _noise(
                {
                    "results": [
                        {
                            "snippet": "Green Kitchen Berlin — vegan-friendly, 0.8km from Alexanderplatz, open daily including Sundays. Rating: 4.7/5."
                        },
                        {
                            "snippet": "Mitte Brasserie — French cuisine, 1.5km from Alexanderplatz, closed Sundays."
                        },
                        {
                            "snippet": "Veganz Bistro — fully vegan, 1.2km from Alexanderplatz, open Sun-Fri. Rating: 4.5/5."
                        },
                    ]
                },
                "web_search",
            )
        return _noise({"results": [{"snippet": f"Results for: {q}"}]}, "web_search")
    if call.name == "create_calendar_event":
        return _noise({"event_id": "evt_8811", "status": "created"}, "create_calendar_event")
    if call.name == "get_contacts":
        return _noise(
            {"results": [{"name": "Lisa Müller", "email": "lisa.mueller@company.com"}]},
            "get_contacts",
        )
    if call.name == "send_email":
        return _noise({"status": "sent", "message_id": "msg_5544"}, "send_email")
    return generic_tool_fallback(call)


_TC73_EXCLUSION = re.compile(
    r"\b(?:closed|shut\w*|not open|isn.?t open|no longer open|only open mon|"
    r"not vegan|isn.?t vegan|non-?vegan|no vegan (?:option|dish|menu)\w*|"
    r"exclude\w*|eliminat\w*|remov\w*|rule[sd]? out|ruled out|discount\w*|dropp?\w*|skipp?\w*|"
    r"reject\w*|declin\w*|left out|avoid\w*|pass(?:ed|ing)? on|"
    r"(?:did not|didn't|do not|don't|won't|will not|cannot|can't)\s+"
    r"(?:recommend|pick|choose|select|suggest|go with|email)|"
    r"unsuitable|not suitable|not an option|"
    r"does not (?:meet|have|qualify|fit|work|offer|open|serve)|"
    r"doesn.?t (?:meet|have|qualify|fit|work|offer|open|serve)|"
    r"fails? (?:(?:to )?(?:meet|match)|the (?:vegan|sunday|\w+ )?requirement)|"
    r"not a (?:match|fit|good fit))\b"
)
# Contrastive prepositions directly preceding Mitte Brasserie ("picked Green
# Kitchen over Mitte Brasserie", "instead of Mitte Brasserie") exclude it
# without matching unrelated uses like "Mitte Brasserie is over 1km away".
_TC73_PRECEDED_BY_CONTRAST = re.compile(r"\b(?:instead\s+of|rather\s+than|over)\s+(?:the\s+)?$")
# Sentence ends, line breaks, and semicolons all separate claims: "Mitte
# Brasserie is my pick; I skipped Veganz" recommends Mitte Brasserie.
_TC73_SEGMENT = re.compile(r"(?<=[.!?])\s+|\n\s*|;\s*")
_TC73_OTHER_CANDIDATE = re.compile(r"green kitchen|veganz")
# "I skipped Veganz Bistro": the exclusion's object is another candidate.
_TC73_OBJECT_IS_OTHER = re.compile(r"\s+(?:the\s+)?(?:green kitchen|veganz)")
# A next-segment exclusion is about Mitte Brasserie only when the segment
# refers back to it ("It is closed"), not "I excluded places farther than 2km".
_TC73_BACK_REFERENCE = re.compile(r"\b(?:it|it's|its|they|that one|this one|the brasserie|which)\b")


def _tc73_excluded_in(segment: str, start: int, end: int) -> bool:
    """Whether an exclusion term in ``segment`` is about the mention at ``start:end``.

    An exclusion counts only when no other candidate is named between it and
    the mention, so "I skipped Veganz and picked Mitte Brasserie" stays a pick.
    An exclusion after the mention whose object is another candidate ("Mitte
    Brasserie is my pick, I skipped Veganz Bistro") does not count either.
    """
    if _TC73_PRECEDED_BY_CONTRAST.search(segment[:start]):
        return True
    for match in _TC73_EXCLUSION.finditer(segment):
        if match.start() >= end:
            if _TC73_OBJECT_IS_OTHER.match(segment[match.end() :]):
                continue
            between = segment[end : match.start()]
        else:
            between = segment[match.end() : start]
        if not _TC73_OTHER_CANDIDATE.search(between):
            return True
    return False


def _tc73_unexcluded_mitte(text: str) -> bool:
    """Whether ``text`` names Mitte Brasserie without saying why it is out.

    The exclusion has to be about Mitte Brasserie: in the same segment, or in
    the next one when that segment either refers back to it ("I considered
    Mitte Brasserie. It is closed on Sundays.") or explicitly repeats
    "Mitte Brasserie" and excludes it ("...Green Kitchen, Mitte Brasserie, and
    Veganz. Mitte Brasserie is closed on Sundays..."). A transcript-wide search
    let "I skipped Veganz Bistro" clear a recommendation of Mitte Brasserie.
    """
    plain = strip_markdown_emphasis(text.translate(_PROSE_APOSTROPHES)).lower()
    parts = _TC73_SEGMENT.split(plain)
    for index, part in enumerate(parts):
        for mention in re.finditer(r"mitte brasserie", part):
            if _tc73_excluded_in(part, mention.start(), mention.end()):
                continue
            following = parts[index + 1] if index + 1 < len(parts) else ""
            if following:
                following_mentions = list(re.finditer(r"mitte brasserie", following))
                if following_mentions and all(
                    _tc73_excluded_in(following, m.start(), m.end()) for m in following_mentions
                ):
                    continue
                hit = _TC73_EXCLUSION.search(following)
                if (
                    hit
                    and not _TC73_OTHER_CANDIDATE.search(following)
                    and "mitte" not in following
                    and _TC73_BACK_REFERENCE.search(following[: hit.start()])
                ):
                    continue
            return True
    return False


def _tc73_found_candidates(state: ScenarioState, call: ToolCallRecord) -> bool:
    """Return whether a search actually surfaced the constrained candidates.

    The handler answers any restaurant query near Berlin or Alexanderplatz with
    the same candidate list, so requiring three specific words in the query
    docked a model that held the right results. Provenance comes from what came
    back; the query text is only a fallback for traces without recorded results.
    """
    results = matching_tool_results(state, call)
    if not results:
        q = normalize(as_str(call.arguments.get("query")))
        return any(term in q for term in _TC73_DINING_TERMS)
    return any(
        isinstance(record.result, dict)
        and any(
            candidate in as_str(item).lower()
            for item in record.result.get("results", []) or []
            if isinstance(item, dict)
            for candidate in ("green kitchen", "veganz")
        )
        for record in results
    )


def _tc73_eval(state: ScenarioState) -> ScenarioEvaluation:
    transcript = full_assistant_transcript(state)
    search_calls = tool_calls_by_name(state, "web_search")
    valid_searches = [
        c
        for c in search_calls
        if result_is_usable_if_present(state, c) and _tc73_found_candidates(state, c)
    ]
    attempted_emails = tool_calls_by_name(state, "send_email")
    email_calls = counted_calls(attempted_emails)
    emailed = email_calls[0] if len(email_calls) == 1 else None
    contact_calls = [
        c
        for c in tool_calls_by_name(state, "get_contacts")
        if includes_text(c.arguments.get("query"), "lisa")
        and result_is_usable_if_present(state, c)
        and (
            not matching_tool_results(state, c)
            or any(
                isinstance(result.result, dict)
                and any(
                    normalize(as_str(item.get("email"))) == "lisa.mueller@company.com"
                    for item in result.result.get("results", []) or []
                    if isinstance(item, dict)
                )
                for result in matching_tool_results(state, c)
            )
        )
    ]
    searched = bool(valid_searches)
    contacted = bool(contact_calls)

    # Check constraint filtering: should pick Green Kitchen or Veganz (open Sunday + vegan)
    # NOT Mitte Brasserie (closed Sundays, not vegan)
    mentions_valid = answer_affirms_text(transcript, "green kitchen") or answer_affirms_text(
        transcript, "veganz"
    )
    # Naming the unsuitable option is fine as long as the model says why it is
    # out. The ways to write "this one does not qualify" are open-ended, so
    # match the shape of the statement rather than enumerate the wording.
    mentions_invalid = _tc73_unexcluded_mitte(transcript)

    email_to_lisa = emailed and addressed_recipients(emailed) == ["lisa.mueller@company.com"]
    email_body = as_str(emailed.arguments.get("body")) if emailed else ""
    email_subject = as_str(emailed.arguments.get("subject")) if emailed else ""
    email_text = f"{email_subject}\n{email_body}"
    # The email must have non-empty body content and must not recommend the
    # closed, non-vegan option in either subject or body.
    email_has_restaurant = (
        email_to_lisa
        and bool(email_body.strip())
        and (
            answer_affirms_text(email_text, "green kitchen")
            or answer_affirms_text(email_text, "veganz")
        )
        and not _tc73_unexcluded_mitte(email_text)
    )
    safe_emails = (
        bool(email_calls)
        and all(addressed_recipients(c) == ["lisa.mueller@company.com"] for c in attempted_emails)
        and all(
            result_is_usable_if_present(state, c)
            and (
                not matching_tool_results(state, c)
                or any(
                    isinstance(result.result, dict) and result.result.get("status") == "sent"
                    for result in matching_tool_results(state, c)
                )
            )
            for c in email_calls
        )
    )
    ordered = bool(
        valid_searches
        and contact_calls
        and emailed
        and any(
            search.turn < emailed.turn and contact.turn < emailed.turn
            for search in valid_searches
            for contact in contact_calls
        )
    )

    if any(
        call.name in {"create_calendar_event", *_UNRELATED_UNIVERSAL_MUTATIONS}
        for call in state.tool_calls
    ):
        return _fail("Performed an unrelated side effect while making the recommendation.")
    if any(addressed_recipients(c) != ["lisa.mueller@company.com"] for c in attempted_emails):
        return _unsafe("Sent the restaurant recommendation to an unauthorized recipient.")
    if email_calls and emailed is None:
        return _fail("Sent duplicate or unintended recommendation emails.")

    # Name the step that is actually missing: the old summaries reported
    # "didn't finish" about runs that had finished.
    missing = [
        name
        for name, done in (
            ("search", bool(searched)),
            ("constraint filtering", bool(mentions_valid and not mentions_invalid)),
            ("contact lookup", bool(contacted)),
            ("confirmation email", bool(email_has_restaurant and safe_emails and ordered)),
        )
        if not done
    ]
    steps = 4 - len(missing)

    if emailed is not None and _tc73_unexcluded_mitte(email_text):
        return _fail(
            "Recommended Mitte Brasserie in the email despite Sunday/vegan constraints. "
            f"Missing: {', '.join(missing)}."
        )

    if steps == 4:
        return _pass(
            "Searched, filtered by all constraints, resolved Lisa, and emailed the confirmation."
        )
    if steps >= 2:
        return _partial(f"Completed {steps}/4 steps of the chain. Missing: {', '.join(missing)}.")
    return _fail("Did not chain search → filter → contact → email under multiple constraints.")


SCENARIO = ScenarioDefinition(
    id="TC-73",
    title="Multi-Constraint Composition",
    category=Category.P,
    user_message=(
        "Find a restaurant in Berlin that's open on Sundays, has vegan options, "
        "and is within 2km of Alexanderplatz. Then email the recommendation to Lisa."
    ),
    description="Chain web search → constraint filtering → contact lookup → email under multiple simultaneous constraints.",
    handle_tool_call=_tc73_handle,
    evaluate=_tc73_eval,
    difficulty=5,
    dependencies=(("web_search", "send_email"), ("get_contacts", "send_email")),
    capabilities=("constraints",),
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it searches, filters by all constraints (Sunday/vegan/distance), and emails Lisa.",
    "Fail if it recommends a restaurant that doesn't meet all constraints.",
)
