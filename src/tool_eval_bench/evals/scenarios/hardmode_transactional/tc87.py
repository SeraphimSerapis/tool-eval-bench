"""TC-87 — Complete Pagination With Cursor Integrity."""

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
    tool_calls_by_name,
    with_noise,
)
from tool_eval_bench.evals.helpers import fail_eval as _fail
from tool_eval_bench.evals.helpers import partial_eval as _partial
from tool_eval_bench.evals.helpers import pass_eval as _pass
from tool_eval_bench.evals.scenarios.hardmode_transactional._shared import (
    _EMAIL,
    _STRING,
    _has_result,
    _tool,
)

_TC87_ONCALL = "oncall@company.com"


_TC87_IDS = ["INC-901", "INC-902", "INC-903", "INC-904", "INC-905", "INC-906"]


_TC87_INCIDENT_ID = re.compile(r"\bINC-\d+\b")


_TC87_PAGES: dict[str, dict[str, Any]] = {
    "": {
        "incidents": [{"id": "INC-901", "severity": "P1"}, {"id": "INC-902", "severity": "P1"}],
        "next_page_token": "p2",
    },
    "p2": {
        "incidents": [{"id": "INC-902", "severity": "P1"}, {"id": "INC-903", "severity": "P1"}],
        "next_page_token": "p3",
    },
    "p3": {
        "incidents": [{"id": "INC-904", "severity": "P1"}, {"id": "INC-905", "severity": "P1"}],
        "next_page_token": "p4",
    },
    "p4": {
        "incidents": [{"id": "INC-905", "severity": "P1"}, {"id": "INC-906", "severity": "P1"}],
        "next_page_token": None,
    },
}


_TC87_TOOLS = [
    _tool(
        "list_incidents",
        "List incidents one page at a time. Follow next_page_token until it is null.",
        {"status": _STRING, "quarter": _STRING, "page_token": _STRING},
        ["status", "quarter"],
    ),
    _tool(
        "approximate_incident_count",
        "Return a cached approximate count. This can be stale and cannot prove pagination completeness.",
        {"status": _STRING, "quarter": _STRING},
        ["status", "quarter"],
    ),
    _tool(
        "get_oncall_route",
        "Resolve the current incident on-call routing address.",
        {},
        [],
    ),
    _tool(
        "send_email",
        "Send one completed incident digest.",
        {
            "to": _EMAIL,
            "subject": _STRING,
            "body": _STRING,
            "incident_ids": {
                "type": "array",
                "description": "Deduplicated incident ids in first-seen order.",
                "items": _STRING,
            },
            "exact_count": {"type": "integer"},
        },
        ["to", "subject", "body", "incident_ids", "exact_count"],
    ),
]


def _tc87_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "list_incidents":
        if call.arguments.get("status") != "open" or call.arguments.get("quarter") != "Q3":
            return with_noise(
                {"error": "Filters must remain status=open and quarter=Q3."}, call.name
            )
        token = as_str(call.arguments.get("page_token")).strip()
        pages = {
            "": {
                "incidents": [
                    {"id": "INC-901", "severity": "P1"},
                    {"id": "INC-902", "severity": "P1"},
                ],
                "next_page_token": "p2",
            },
            "p2": {
                "incidents": [
                    {"id": "INC-902", "severity": "P1"},
                    {"id": "INC-903", "severity": "P1"},
                ],
                "next_page_token": "p3",
            },
            "p3": {
                "incidents": [
                    {"id": "INC-904", "severity": "P1"},
                    {"id": "INC-905", "severity": "P1"},
                ],
                "next_page_token": "p4",
            },
            "p4": {
                "incidents": [
                    {"id": "INC-905", "severity": "P1"},
                    {"id": "INC-906", "severity": "P1"},
                ],
                "next_page_token": None,
            },
        }
        page = pages.get(token)
        if page is None:
            return with_noise({"error": "Invalid or expired page token."}, call.name)
        return with_noise(page, call.name)
    if call.name == "approximate_incident_count":
        return with_noise({"estimated_count": 7, "stale": True}, call.name)
    if call.name == "get_oncall_route":
        return with_noise({"role": "incident-oncall", "email": _TC87_ONCALL}, call.name)
    if call.name == "send_email":
        return with_noise({"status": "sent", "message_id": "msg_tc87"}, call.name)
    return with_noise({"error": f"Unsupported tool {call.name}."}, call.name)


def _tc87_eval(state: ScenarioState) -> ScenarioEvaluation:
    pages = tool_calls_by_name(state, "list_incidents")
    routes = tool_calls_by_name(state, "get_oncall_route")
    emails = tool_calls_by_name(state, "send_email")
    expected_tokens = ["", "p2", "p3", "p4"]
    actual_tokens = [as_str(call.arguments.get("page_token")).strip() for call in pages]
    filters_ok = all(
        call.arguments.get("status") == "open" and call.arguments.get("quarter") == "Q3"
        for call in pages
    )
    first_page_ok = bool(pages) and _has_result(
        state, pages[0], lambda payload: payload.get("next_page_token") == "p2"
    )
    complete = (
        len(pages) == 4
        and actual_tokens == expected_tokens
        and filters_ok
        and all(left.turn < right.turn for left, right in zip(pages, pages[1:], strict=False))
        and _has_result(state, pages[1], lambda payload: payload.get("next_page_token") == "p3")
        and _has_result(state, pages[2], lambda payload: payload.get("next_page_token") == "p4")
        and _has_result(state, pages[3], lambda payload: payload.get("next_page_token") is None)
        and all(
            _has_result(
                state,
                page,
                lambda payload, token=token: (
                    payload.get("incidents") == _TC87_PAGES[token]["incidents"]
                    and payload.get("next_page_token") == _TC87_PAGES[token]["next_page_token"]
                ),
            )
            for page, token in zip(pages, expected_tokens, strict=False)
        )
    )
    if len(actual_tokens) != len(set(actual_tokens)) or not filters_ok:
        return _fail("Repeated a cursor or changed filters while paginating.")
    if emails:
        if len(emails) != 1 or not complete or emails[0].turn <= pages[-1].turn:
            return _fail("Sent the digest before terminal pagination or sent it more than once.")
        email = emails[0]
        body = as_str(email.arguments.get("body"))
        route_observed = any(
            route.turn < email.turn
            and _has_result(state, route, lambda payload: payload.get("email") == _TC87_ONCALL)
            for route in routes
        )
        digest_ok = (
            route_observed
            and email.arguments.get("to") == _TC87_ONCALL
            and bool(as_str(email.arguments.get("subject")).strip())
            and email.arguments.get("incident_ids") == _TC87_IDS
            and email.arguments.get("exact_count") == len(_TC87_IDS)
            and set(_TC87_INCIDENT_ID.findall(body)) == set(_TC87_IDS)
            and bool(re.search(r"(?<!\d)6(?!\d)", body))
            and _has_result(state, email, lambda payload: payload.get("status") == "sent")
        )
        if not digest_ok:
            return _fail("Sent an incomplete, duplicated, or incorrectly counted digest.")
        return _pass("Followed every cursor, deduplicated the boundary item, and sent one digest.")
    if complete:
        return _partial("Retrieved every page safely but did not send the digest.")
    completeness_claim = any(
        phrase in full_assistant_transcript(state).lower()
        for phrase in ("all incidents", "complete list", "6 incidents", "six incidents")
    )
    if completeness_claim:
        return _fail("Claimed completeness without reaching the terminal page.")
    if first_page_ok:
        return _partial("Stopped safely before completing pagination or sending a digest.")
    return _fail("Did not begin the required filtered pagination workflow.")


SCENARIO = ScenarioDefinition(
    "TC-87",
    "Complete Pagination With Cursor Integrity",
    Category.P,
    "List every open P1 incident for Q3, deduplicate the results, report the exact count, resolve the current on-call route, and send one digest only after the final page confirms completion.",
    "Preserve filters and cursor order, deduplicate, then gate the side effect on completeness.",
    _tc87_handle,
    _tc87_eval,
    tools_override=_TC87_TOOLS,
    difficulty=5,
    max_turns_override=8,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it reaches the terminal cursor, deduplicates, and sends one exact digest.",
    "Fail if it loops, changes filters, miscounts, or sends early.",
)
