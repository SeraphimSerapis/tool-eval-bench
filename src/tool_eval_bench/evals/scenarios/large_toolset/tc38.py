"""TC-38 — Multi-Step Crowded Namespace."""

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
from tool_eval_bench.domain.tools_large import LARGE_TOOLSET
from tool_eval_bench.evals.helpers import (
    address_observed_before as _address_observed_before,
)
from tool_eval_bench.evals.helpers import (
    addressed_recipients as _addressed_recipients,
)
from tool_eval_bench.evals.helpers import (
    as_str as _as_str,
)
from tool_eval_bench.evals.helpers import (
    counted_calls as _counted_calls,
)
from tool_eval_bench.evals.helpers import (
    fail_eval as _fail,
)
from tool_eval_bench.evals.helpers import forbid_unrequested_side_effects
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
    positive_argument_contains as _positive_argument_contains,
)
from tool_eval_bench.evals.helpers import (
    recipient_values as _recipient_values,
)
from tool_eval_bench.evals.helpers import (
    with_noise as _noise,
)
from tool_eval_bench.evals.scenarios.large_toolset._shared import (
    LARGE_SIDE_EFFECT_TOOLS,
    _answer_affirms_exact_value,
    _is_engineering_department,
    _result_matches_if_present,
    _status_is_success,
)

# Either directory can supply the manager address.
_TC38_LOOKUPS = frozenset({"get_contacts", "get_org_chart"})


def _tc38_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "search_files":
        return _noise(
            {"results": [{"file_id": "file_091", "name": "Q3_Budget_Report_2025.xlsx"}]},
            "search_files",
        )
    if call.name == "read_file":
        return _noise(
            {
                "content": "Department budgets: Engineering $2.1M, Marketing $800K, "
                "Sales $1.5M. Total: $4.4M"
            },
            "read_file",
        )
    if call.name == "get_contacts":
        return _noise(
            {
                "results": [
                    {"name": "Jordan Park", "email": "jordan.park@company.com", "role": "manager"}
                ]
            },
            "get_contacts",
        )
    if call.name == "send_email":
        return _noise({"status": "sent"}, "send_email")
    if call.name == "get_org_chart":
        # Engineering org chart — the canonical manager record must agree with
        # the get_contacts fixture so an org-chart verification resolves (not
        # re-introduces) the manager identity.
        if _is_engineering_department(call.arguments.get("department")):
            return _noise(
                {
                    "department": "Engineering",
                    "manager": {
                        "name": "Jordan Park",
                        "email": "jordan.park@company.com",
                        "role": "manager",
                    },
                },
                "get_org_chart",
            )
        return _noise({"results": []}, "get_org_chart")
    # Domain-specific tools that might be tempting but wrong
    if call.name == "get_customer_profile":
        return _noise(
            {"error": "No customer found matching 'manager'."},
            "get_customer_profile",
        )
    if call.name == "query_database":
        return _noise(
            {"error": "Access denied: read-only analytics database."},
            "query_database",
        )
    return _noise(
        {"error": f"Tool {call.name} is not available."},
        call.name,
    )


def _tc38_search_result_has_report(payload: Any) -> bool:
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        return False
    return any(
        isinstance(item, dict)
        and (
            _normalize(_as_str(item.get("file_id"))) == "file_091"
            or (
                _includes_text(item.get("name"), "q3")
                and _includes_text(item.get("name"), "budget")
            )
        )
        for item in payload["results"]
    )


def _tc38_read_result_has_total(payload: Any) -> bool:
    return isinstance(payload, dict) and _includes_text(payload.get("content"), "4.4m")


def _tc38_manager_result_has_jordan(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    manager = payload.get("manager")
    if isinstance(manager, dict) and _normalize(_as_str(manager.get("email"))) == (
        "jordan.park@company.com"
    ):
        return True
    results = payload.get("results")
    if isinstance(results, list):
        return any(
            isinstance(item, dict)
            and _normalize(_as_str(item.get("email"))) == "jordan.park@company.com"
            for item in results
        )
    return False


def _tc38_email_result_is_sent(payload: Any) -> bool:
    return _status_is_success(
        payload,
        statuses={"accepted", "delivered", "ok", "queued", "sent", "success"},
    )


@forbid_unrequested_side_effects({"send_email": 1}, tools=LARGE_SIDE_EFFECT_TOOLS)
def _tc38_eval(state: ScenarioState) -> ScenarioEvaluation:
    """User: 'Find the Q3 budget report and email the total to my manager.'

    Same chain as TC-07, but with 52 tools. Must complete 4 steps
    using the correct tools from a crowded namespace.
    """

    def _is_valid_search(call: ToolCallRecord) -> bool:
        return call.name == "search_files" and _positive_argument_contains(
            call.arguments.get("query"), "budget"
        )

    def _is_valid_read(call: ToolCallRecord) -> bool:
        return (
            call.name == "read_file"
            and _normalize(_as_str(call.arguments.get("file_id"))) == "file_091"
        )

    def _is_manager_verification(c: ToolCallRecord) -> bool:
        return c.name == "get_org_chart" and _is_engineering_department(
            c.arguments.get("department")
        )

    def _is_valid_manager(c: ToolCallRecord) -> bool:
        return (
            c.name == "get_contacts"
            and _positive_argument_contains(c.arguments.get("query"), "manager")
        ) or _is_manager_verification(c)

    def _is_valid_email(c: ToolCallRecord) -> bool:
        body = _as_str(c.arguments.get("body"))
        return (
            c.name == "send_email"
            and _recipient_values(c.arguments.get("to")) == ["jordan.park@company.com"]
            and _addressed_recipients(c) == ["jordan.park@company.com"]
            # TC-38 is TC-07 with 52 tools, so a score difference between them
            # has to mean the toolset size mattered. Spelling the unit out
            # ("$4.4 million", "$4.40M", "$4,400,000") or sending the address as
            # a one-element array is not a toolset effect, so use TC-07's checks.
            and (
                _answer_affirms_exact_value(body, "4.4")
                or _answer_affirms_exact_value(body, "4400000")
            )
        )

    search_calls = [c for c in state.tool_calls if _is_valid_search(c)]
    read_calls = [c for c in state.tool_calls if _is_valid_read(c)]
    manager_calls = [c for c in state.tool_calls if _is_valid_manager(c)]
    email_calls = [c for c in state.tool_calls if _is_valid_email(c)]

    steps = sum(
        (
            bool(search_calls),
            bool(read_calls),
            bool(manager_calls),
            bool(email_calls),
        )
    )

    # Check for domain-tool contamination
    domain_calls = [
        c.name
        for c in state.tool_calls
        if c.name not in ("search_files", "read_file", "get_contacts", "send_email", "web_search")
        and not _is_manager_verification(c)
    ]

    if steps == 4 and not domain_calls:
        email_attempts = [c for c in state.tool_calls if c.name == "send_email"]
        if len(_counted_calls(email_attempts)) != 1 or any(
            _recipient_values(c.arguments.get("to")) != ["jordan.park@company.com"]
            or _addressed_recipients(c) != ["jordan.park@company.com"]
            for c in email_attempts
        ):
            return _partial(
                "Found the chain but issued an unsafe duplicate or incorrect email mutation."
            )
        usable_searches = [
            c
            for c in search_calls
            if _result_matches_if_present(state, c, _tc38_search_result_has_report)
        ]
        usable_reads = [
            c
            for c in read_calls
            if _result_matches_if_present(state, c, _tc38_read_result_has_total)
        ]
        usable_managers = [
            c
            for c in manager_calls
            if _result_matches_if_present(state, c, _tc38_manager_result_has_jordan)
        ]
        usable_emails = [
            c
            for c in email_calls
            if _result_matches_if_present(state, c, _tc38_email_result_is_sent)
        ]
        if not (usable_searches and usable_reads and usable_managers and usable_emails):
            return _partial(
                "Completed the calls, but a required tool result was unusable or missing."
            )
        ordered_emails = [
            email
            for email in usable_emails
            if any(
                search.turn < read.turn < email.turn and manager.turn < email.turn
                for search in usable_searches
                for read in usable_reads
                for manager in usable_managers
            )
        ]
        if not ordered_emails:
            return _partial("Completed the calls, but violated a data dependency before email.")
        if not any(
            _address_observed_before(state, email, "jordan.park@company.com", sources=_TC38_LOOKUPS)
            for email in ordered_emails
        ):
            return _partial("Emailed the manager at an address no earlier lookup returned.")
        return _pass("Completed the full 4-step chain correctly from 52 tools.")
    if steps == 4 and domain_calls:
        extras = ", ".join(domain_calls[:3])
        return _partial(f"Completed the chain but also called irrelevant tools: {extras}")
    if steps >= 3:
        return _partial(f"Completed {steps}/4 steps of the chain in a 52-tool namespace.")
    return _fail(f"Only completed {steps}/4 steps — struggled with the crowded namespace.")


SCENARIO = ScenarioDefinition(
    id="TC-38",
    title="Multi-Step Crowded Namespace",
    category=Category.L,
    user_message="Find the Q3 budget report and email the total to my manager.",
    description="Complete a 4-step chain using correct tools from 52 options.",
    handle_tool_call=_tc38_handle,
    evaluate=_tc38_eval,
    tools_override=LARGE_TOOLSET,
    difficulty=4,
    dependencies=(
        ("search_files", "read_file"),
        ("read_file", "send_email"),
        ("get_contacts", "send_email"),
        ("get_org_chart", "send_email"),
    ),
    variant_literals=("file_091",),
    control_scenario_id="TC-07",
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it resolves file and contact dependencies before email from 52 tools.",
    "Fail if it uses domain-specific tools that don't apply.",
)
