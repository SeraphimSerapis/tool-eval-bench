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
    as_str,
    full_assistant_transcript,
    generic_tool_fallback,
    includes_text,
    matching_tool_results,
    normalize,
    result_is_usable_if_present,
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
from tool_eval_bench.evals.helpers import (
    with_noise as _noise,
)
from tool_eval_bench.evals.scenarios.hardmode._shared import (
    _UNRELATED_UNIVERSAL_MUTATIONS,
)


def _tc73_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "web_search":
        q = normalize(as_str(call.arguments.get("query")))
        if "restaurant" in q and ("berlin" in q or "alexanderplatz" in q):
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
    r"\b(?:closed|shut|not open|isn.?t open|no longer open|"
    r"not vegan|isn.?t vegan|non-?vegan|no vegan (?:option|dish|menu)\w*|"
    r"exclude\w*|rule[sd]? out|ruled out|discount\w*|dropp?\w*|skipp?\w*|"
    r"unsuitable|not suitable|"
    r"does not (?:meet|have|qualify|fit|work|offer)|"
    r"doesn.?t (?:meet|have|qualify|fit|work|offer)|"
    r"fails? (?:to )?(?:meet|match)|not a (?:match|fit|good fit))\b"
)


def _tc73_eval(state: ScenarioState) -> ScenarioEvaluation:
    transcript = full_assistant_transcript(state)
    search_calls = tool_calls_by_name(state, "web_search")
    valid_searches = [
        c
        for c in search_calls
        if all(
            term in normalize(as_str(c.arguments.get("query"))) for term in ("restaurant", "berlin")
        )
        and "alexanderplatz" in normalize(as_str(c.arguments.get("query")))
        and result_is_usable_if_present(state, c)
        and (
            not matching_tool_results(state, c)
            or any(
                isinstance(result.result, dict)
                and any(
                    candidate in as_str(item).lower()
                    for item in result.result.get("results", []) or []
                    if isinstance(item, dict)
                    for candidate in ("green kitchen", "veganz")
                )
                for result in matching_tool_results(state, c)
            )
        )
    ]
    email_calls = tool_calls_by_name(state, "send_email")
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
    mentions_valid = "green kitchen" in transcript.lower() or "veganz" in transcript.lower()
    # Naming the unsuitable option is fine as long as the model says why it is
    # out. The ways to write "this one does not qualify" are open-ended, so
    # match the shape of the statement rather than enumerate the wording.
    mentions_invalid = "mitte brasserie" in transcript.lower() and not _TC73_EXCLUSION.search(
        transcript.lower()
    )

    email_to_lisa = (
        emailed and normalize(as_str(emailed.arguments.get("to"))) == "lisa.mueller@company.com"
    )
    email_body = emailed.arguments.get("body") if emailed else ""
    email_has_restaurant = email_to_lisa and (
        includes_text(email_body, "green kitchen") or includes_text(email_body, "veganz")
    )
    safe_emails = bool(email_calls) and all(
        normalize(as_str(c.arguments.get("to"))) == "lisa.mueller@company.com"
        and result_is_usable_if_present(state, c)
        and (
            not matching_tool_results(state, c)
            or any(
                isinstance(result.result, dict) and result.result.get("status") == "sent"
                for result in matching_tool_results(state, c)
            )
        )
        for c in email_calls
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
    if email_calls and emailed is None:
        return _fail("Sent duplicate or unintended recommendation emails.")

    steps = sum(
        [
            bool(searched),
            bool(mentions_valid and not mentions_invalid),
            bool(contacted),
            bool(email_has_restaurant and safe_emails and ordered),
        ]
    )

    if steps == 4:
        return _pass(
            "Searched, filtered by all constraints, resolved Lisa, and emailed the confirmation."
        )
    if steps >= 3:
        return _partial("Completed most of the chain but missed one constraint or step.")
    if steps >= 2:
        return _partial("Partially completed — searched and identified options but didn't finish.")
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
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it searches, filters by all constraints (Sunday/vegan/distance), and emails Lisa.",
    "Fail if it recommends a restaurant that doesn't meet all constraints.",
)
