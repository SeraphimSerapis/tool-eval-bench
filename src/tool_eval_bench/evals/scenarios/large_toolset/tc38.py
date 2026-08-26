"""TC-38 — Multi-Step Crowded Namespace."""

from __future__ import annotations

from typing import Any, cast

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
    as_str as _as_str,
)
from tool_eval_bench.evals.helpers import (
    fail_eval as _fail,
)
from tool_eval_bench.evals.helpers import (
    has_tool_call as _has_tool_call,
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
from tool_eval_bench.evals.scenarios.large_toolset._shared import (
    _is_engineering_department,
)


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
        {"error": f"Tool {call.name} is not relevant for this scenario."},
        call.name,
    )


def _tc38_eval(state: ScenarioState) -> ScenarioEvaluation:
    """User: 'Find the Q3 budget report and email the total to my manager.'

    Same chain as TC-07, but with 52 tools. Must complete 4 steps
    using the correct tools from a crowded namespace.
    """
    steps = 0
    if _has_tool_call(
        state,
        "search_files",
        lambda c: (
            _includes_text(c.arguments.get("query"), "q3 budget report")
            or _includes_text(c.arguments.get("query"), "budget")
        ),
    ):
        steps += 1
    if _has_tool_call(
        state,
        "read_file",
        lambda c: _normalize(_as_str(c.arguments.get("file_id"))) == "file_091",
    ):
        steps += 1
    manager_lookup = _has_tool_call(
        state,
        "get_contacts",
        lambda c: _includes_text(c.arguments.get("query"), "manager"),
    )

    # The org chart is an equivalent authoritative source for the manager.
    # Keep the dependency semantic: either lookup must precede delivery.
    manager_lookup = manager_lookup or _has_tool_call(
        state,
        "get_org_chart",
        lambda c: _is_engineering_department(c.arguments.get("department")),
    )
    if manager_lookup:
        steps += 1
    if _has_tool_call(
        state,
        "send_email",
        lambda c: (
            _normalize(_as_str(c.arguments.get("to"))) == "jordan.park@company.com"
            and (
                _includes_text(c.arguments.get("body"), "4.4m")
                or _includes_text(c.arguments.get("body"), "$4.4m")
            )
        ),
    ):
        steps += 1

    # A get_org_chart lookup for the Engineering department is an accepted
    # manager-verification step and is not treated as domain-tool contamination.
    # Unrelated org-chart lookups still count as irrelevant calls.
    def _is_manager_verification(c: ToolCallRecord) -> bool:
        return c.name == "get_org_chart" and _is_engineering_department(
            c.arguments.get("department")
        )

    # Check for domain-tool contamination
    domain_calls = [
        c.name
        for c in state.tool_calls
        if c.name not in ("search_files", "read_file", "get_contacts", "send_email", "web_search")
        and not _is_manager_verification(c)
    ]

    if steps == 4 and not domain_calls:
        search = next((c for c in state.tool_calls if c.name == "search_files"), None)
        read = next((c for c in state.tool_calls if c.name == "read_file"), None)
        manager = next(
            (
                c
                for c in state.tool_calls
                if c.name == "get_contacts" and _includes_text(c.arguments.get("query"), "manager")
            ),
            None,
        )
        manager = manager or next(
            (
                c
                for c in state.tool_calls
                if c.name == "get_org_chart"
                and _is_engineering_department(c.arguments.get("department"))
            ),
            None,
        )
        email = next((c for c in state.tool_calls if c.name == "send_email"), None)
        if any(call is None for call in (search, read, manager, email)):
            return _partial("Completed the calls, but one dependency record was incomplete.")
        search = cast(ToolCallRecord, search)
        read = cast(ToolCallRecord, read)
        manager = cast(ToolCallRecord, manager)
        email = cast(ToolCallRecord, email)
        if not (search.turn < read.turn < email.turn and manager.turn < email.turn):
            return _partial("Completed the calls, but violated a data dependency before email.")
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
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it resolves file and contact dependencies before email from 52 tools.",
    "Fail if it uses domain-specific tools that don't apply.",
)
