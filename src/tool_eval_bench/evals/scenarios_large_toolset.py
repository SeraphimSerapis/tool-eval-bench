"""Large-toolset scenario pack — tests tool selection at scale.

TC-37 to TC-40: 4 scenarios in Category L (Toolset Scale) that evaluate
model ability to navigate a crowded 52-tool namespace.
"""

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
    as_str as _as_str,
    normalize as _normalize,
    includes_text as _includes_text,
    answer_contains_number as _answer_contains_number,
    has_tool_call as _has_tool_call,
    with_noise as _noise,
    pass_eval as _pass,
    partial_eval as _partial,
    fail_eval as _fail,
)


# ===================================================================
# TC-37: Needle in a Haystack — Tool Selection at Scale
# Category L — Toolset Scale
# ===================================================================

def _tc37_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_weather":
        return _noise(
            {"location": "Berlin", "temperature": 8, "units": "celsius",
             "condition": "Overcast", "humidity": 72},
            "get_weather",
        )
    if call.name == "web_search":
        return _noise(
            {"results": [{"snippet": "Berlin weather right now: 8C and overcast."}]},
            "web_search",
        )
    # Any other tool — return a generic error with noise
    return _noise(
        {"error": f"Tool {call.name} is not relevant for this scenario."},
        call.name,
    )


def _tc37_eval(state: ScenarioState) -> ScenarioEvaluation:
    """User: 'What's the weather like in Berlin right now?'

    Same as TC-01, but with 52 tools instead of 12.
    Must use get_weather and ignore 50+ distractors.
    """
    used_weather = _has_tool_call(
        state, "get_weather",
        lambda c: _includes_text(c.arguments.get("location"), "berlin"),
    )
    used_web = _has_tool_call(state, "web_search")
    total_calls = len(state.tool_calls)
    domain_tools_used = [
        c.name for c in state.tool_calls
        if c.name not in ("get_weather", "web_search")
    ]

    if used_weather and total_calls == 1:
        return _pass("Used get_weather with Berlin only — perfect selection from 52 tools.")
    if used_weather and not domain_tools_used and total_calls <= 2:
        return _partial("Found get_weather but called an extra tool unnecessarily.")
    if used_weather and domain_tools_used:
        extras = ", ".join(domain_tools_used[:3])
        return _partial(f"Used get_weather but also called irrelevant tools: {extras}")
    if not used_weather and used_web:
        return _partial("Fell back to web_search instead of the specialist tool.")
    return _fail("Did not select get_weather from the 52-tool set.")


# ===================================================================
# TC-38: Multi-Step in a Crowded Namespace
# Category L — Toolset Scale
# ===================================================================

def _tc38_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "search_files":
        return _noise(
            {"results": [{"file_id": "file_091", "name": "Q3_Budget_Report_2025.xlsx"}]},
            "search_files",
        )
    if call.name == "read_file":
        return _noise(
            {"content": "Department budgets: Engineering $2.1M, Marketing $800K, "
             "Sales $1.5M. Total: $4.4M"},
            "read_file",
        )
    if call.name == "get_contacts":
        return _noise(
            {"results": [{"name": "Jordan Park", "email": "jordan.park@company.com",
                          "role": "manager"}]},
            "get_contacts",
        )
    if call.name == "send_email":
        return _noise({"status": "sent"}, "send_email")
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
        state, "search_files",
        lambda c: _includes_text(c.arguments.get("query"), "q3 budget report")
        or _includes_text(c.arguments.get("query"), "budget"),
    ):
        steps += 1
    if _has_tool_call(
        state, "read_file",
        lambda c: _normalize(_as_str(c.arguments.get("file_id"))) == "file_091",
    ):
        steps += 1
    if _has_tool_call(
        state, "get_contacts",
        lambda c: _includes_text(c.arguments.get("query"), "manager"),
    ):
        steps += 1
    if _has_tool_call(
        state, "send_email",
        lambda c: (
            _normalize(_as_str(c.arguments.get("to"))) == "jordan.park@company.com"
            and (_includes_text(c.arguments.get("body"), "4.4m")
                 or _includes_text(c.arguments.get("body"), "$4.4m"))
        ),
    ):
        steps += 1

    # Check for domain-tool contamination
    domain_calls = [
        c.name for c in state.tool_calls
        if c.name not in ("search_files", "read_file", "get_contacts",
                          "send_email", "web_search")
    ]

    if steps == 4 and not domain_calls:
        return _pass("Completed the full 4-step chain correctly from 52 tools.")
    if steps == 4 and domain_calls:
        extras = ", ".join(domain_calls[:3])
        return _partial(
            f"Completed the chain but also called irrelevant tools: {extras}"
        )
    if steps >= 3:
        return _partial(
            f"Completed {steps}/4 steps of the chain in a 52-tool namespace."
        )
    return _fail(
        f"Only completed {steps}/4 steps — struggled with the crowded namespace."
    )


# ===================================================================
# TC-39: Restraint Under Tool Abundance
# Category L — Toolset Scale
# ===================================================================

def _tc39_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    """Any tool call here is unnecessary — the answer is trivially 30."""
    if call.name == "calculator":
        return _noise({"result": 30}, "calculator")
    if call.name == "run_code":
        return _noise(
            {"stdout": "30.0", "stderr": "", "exit_code": 0},
            "run_code",
        )
    return _noise(
        {"error": f"Tool {call.name} is not relevant for this scenario."},
        call.name,
    )


def _tc39_eval(state: ScenarioState) -> ScenarioEvaluation:
    """User: 'What is 15% of 200?'

    Same as TC-11, but with 52 tools. The question is trivial arithmetic.
    With many more tools available, the temptation to use one is even higher.
    """
    correct = _answer_contains_number(state.final_answer, "30")
    total_calls = len(state.tool_calls)
    used_calc = _has_tool_call(state, "calculator")
    used_web = _has_tool_call(state, "web_search")
    used_code = _has_tool_call(state, "run_code")

    if total_calls == 0 and correct:
        return _pass("Answered directly without tools — resisted 52-tool temptation.")
    if used_calc and correct and total_calls == 1:
        return _partial("Used calculator correctly, but unnecessarily given trivial math.")
    if used_code:
        return _fail("Used run_code for trivial arithmetic — over-engineering.")
    if used_web:
        return _fail("Used web_search for trivial arithmetic.")
    if total_calls > 1:
        tools = ", ".join(c.name for c in state.tool_calls[:3])
        return _fail(f"Used multiple tools ({tools}) for trivial math.")
    if not correct:
        return _fail("Got the arithmetic wrong.")
    return _fail("Failed the restraint test with 52 tools available.")


# ===================================================================
# TC-40: Domain Confusion — Similar Tool Names
# Category L — Toolset Scale
# ===================================================================

def _tc40_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_order_status":
        order_id = _as_str(call.arguments.get("order_id", ""))
        if _includes_text(order_id, "sarah") or _includes_text(order_id, "chen"):
            return _noise({
                "order_id": "ORD-2026-1847",
                "customer": "Sarah Chen",
                "status": "shipped",
                "items": [
                    {"name": "Wireless Keyboard", "qty": 1, "price": 79.99},
                    {"name": "USB-C Hub", "qty": 1, "price": 45.00},
                ],
                "tracking_number": "1Z999AA10123456784",
                "estimated_delivery": "2026-03-25",
            }, "get_order_status")
        return _noise({"error": f"Order not found: {order_id}"}, "get_order_status")
    if call.name == "get_shipping_status":
        return _noise({
            "tracking_number": _as_str(call.arguments.get("tracking_number", "")),
            "status": "in_transit",
            "carrier": "UPS",
            "last_location": "Chicago, IL",
        }, "get_shipping_status")
    if call.name == "get_customer_profile":
        customer_id = _as_str(call.arguments.get("customer_id", ""))
        if _includes_text(customer_id, "sarah") or _includes_text(customer_id, "chen"):
            return _noise({
                "customer_id": "CUST-8821",
                "name": "Sarah Chen",
                "email": "sarah.chen@customer.com",
                "tier": "premium",
                "orders": ["ORD-2026-1847", "ORD-2026-1512"],
            }, "get_customer_profile")
        return _noise(
            {"error": f"Customer not found: {customer_id}"},
            "get_customer_profile",
        )
    if call.name == "get_contacts":
        return _noise(
            {"results": [{"name": "Sarah Chen", "email": "sarah.chen@company.com"}]},
            "get_contacts",
        )
    return _noise(
        {"error": f"Tool {call.name} is not relevant for this scenario."},
        call.name,
    )


def _tc40_eval(state: ScenarioState) -> ScenarioEvaluation:
    """User: 'Get the current status of the order for customer Sarah Chen.'

    Multiple near-match tools: get_order_status (correct), get_shipping_status,
    get_customer_profile, get_contacts. Model must pick the right one.
    """
    used_order = _has_tool_call(
        state, "get_order_status",
        lambda c: (
            _includes_text(c.arguments.get("order_id"), "sarah")
            or _includes_text(c.arguments.get("order_id"), "chen")
        ),
    )
    used_shipping = _has_tool_call(state, "get_shipping_status")
    used_customer = _has_tool_call(state, "get_customer_profile")
    used_contacts = _has_tool_call(state, "get_contacts")
    total_calls = len(state.tool_calls)

    answer = state.final_answer.lower()
    mentions_status = any(
        w in answer for w in ("shipped", "order", "ORD-2026-1847".lower(),
                              "keyboard", "usb-c", "tracking")
    )

    if used_order and total_calls == 1 and mentions_status:
        return _pass(
            "Selected get_order_status precisely from similar-named tools."
        )
    if used_order and total_calls == 1:
        return _pass(
            "Used get_order_status correctly — answered from 52 tools."
        )
    if used_order and used_customer and total_calls == 2:
        return _partial(
            "Used get_order_status + get_customer_profile — "
            "customer lookup was unnecessary but not wrong."
        )
    if used_order and total_calls > 2:
        extras = ", ".join(
            c.name for c in state.tool_calls if c.name != "get_order_status"
        )
        return _partial(f"Found the right tool but also called: {extras}")
    if not used_order and used_shipping:
        return _partial(
            "Used get_shipping_status instead of get_order_status — "
            "close but wrong tool for the request."
        )
    if not used_order and used_customer:
        return _partial(
            "Used get_customer_profile instead of get_order_status — "
            "got customer data but not order status."
        )
    if not used_order and used_contacts:
        return _fail(
            "Used get_contacts (internal directory) instead of "
            "get_order_status — wrong domain entirely."
        )
    return _fail(
        "Did not use get_order_status to check the customer's order."
    )


# ===================================================================
# Large-toolset scenario registry
# ===================================================================

LARGE_TOOLSET_SCENARIOS: list[ScenarioDefinition] = [
    ScenarioDefinition(
        id="TC-37", title="Needle in a Haystack", category=Category.L,
        user_message="What's the weather like in Berlin right now?",
        description="Select get_weather from 52 tools, ignoring 50+ distractors.",
        handle_tool_call=_tc37_handle, evaluate=_tc37_eval,
        tools_override=LARGE_TOOLSET,
    ),
    ScenarioDefinition(
        id="TC-38", title="Multi-Step Crowded Namespace", category=Category.L,
        user_message="Find the Q3 budget report and email the total to my manager.",
        description="Complete a 4-step chain using correct tools from 52 options.",
        handle_tool_call=_tc38_handle, evaluate=_tc38_eval,
        tools_override=LARGE_TOOLSET,
    ),
    ScenarioDefinition(
        id="TC-39", title="Restraint Under Abundance", category=Category.L,
        user_message="What is 15% of 200?",
        description="Answer trivial arithmetic directly with 52 tools tempting you.",
        handle_tool_call=_tc39_handle, evaluate=_tc39_eval,
        tools_override=LARGE_TOOLSET,
    ),
    ScenarioDefinition(
        id="TC-40", title="Domain Confusion", category=Category.L,
        user_message="Get the current status of the order for customer Sarah Chen.",
        description="Select get_order_status from similar-named tools "
                    "(get_shipping_status, get_customer_profile, get_contacts).",
        handle_tool_call=_tc40_handle, evaluate=_tc40_eval,
        tools_override=LARGE_TOOLSET,
    ),
]


LARGE_TOOLSET_DISPLAY_DETAILS: dict[str, ScenarioDisplayDetail] = {
    "TC-37": ScenarioDisplayDetail(
        "Pass if it uses only get_weather for Berlin from 52 tools.",
        "Fail if it uses wrong tools or misses get_weather.",
    ),
    "TC-38": ScenarioDisplayDetail(
        "Pass if it completes search→read→contacts→email from 52 tools.",
        "Fail if it uses domain-specific tools that don't apply.",
    ),
    "TC-39": ScenarioDisplayDetail(
        "Pass if it answers 30 directly despite 52 tools being available.",
        "Fail if it uses web_search, run_code, or multiple tools.",
    ),
    "TC-40": ScenarioDisplayDetail(
        "Pass if it uses get_order_status (not get_shipping_status or get_contacts).",
        "Fail if it confuses similar-named tools or misses the right one.",
    ),
}
