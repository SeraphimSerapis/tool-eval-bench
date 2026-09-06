"""TC-40 — Domain Confusion."""

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
from tool_eval_bench.evals.helpers import answer_affirms_text as _answer_affirms_text
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
    result_is_usable_if_present as _result_is_usable_if_present,
)
from tool_eval_bench.evals.helpers import (
    with_noise as _noise,
)


def _tc40_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_order_status":
        order_id = _as_str(call.arguments.get("order_id", ""))
        # The schema types order_id as "Order ID or customer name", so both the
        # name and the resolved id have to reach the same order.
        if (
            _includes_text(order_id, "sarah")
            or _includes_text(order_id, "chen")
            or _includes_text(order_id, "ORD-2026-1847")
        ):
            return _noise(
                {
                    "order_id": "ORD-2026-1847",
                    "customer": "Sarah Chen",
                    "status": "shipped",
                    "items": [
                        {"name": "Wireless Keyboard", "qty": 1, "price": 79.99},
                        {"name": "USB-C Hub", "qty": 1, "price": 45.00},
                    ],
                    "tracking_number": "1Z999AA10123456784",
                    "estimated_delivery": "2026-03-25",
                },
                "get_order_status",
            )
        return _noise({"error": f"Order not found: {order_id}"}, "get_order_status")
    if call.name == "get_shipping_status":
        return _noise(
            {
                "tracking_number": _as_str(call.arguments.get("tracking_number", "")),
                "status": "in_transit",
                "carrier": "UPS",
                "last_location": "Chicago, IL",
            },
            "get_shipping_status",
        )
    if call.name == "get_customer_profile":
        customer_id = _as_str(call.arguments.get("customer_id", ""))
        if _includes_text(customer_id, "sarah") or _includes_text(customer_id, "chen"):
            return _noise(
                {
                    "customer_id": "CUST-8821",
                    "name": "Sarah Chen",
                    "email": "sarah.chen@customer.com",
                    "tier": "premium",
                    "orders": ["ORD-2026-1847", "ORD-2026-1512"],
                },
                "get_customer_profile",
            )
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


def _tc40_order_id_reaches_sarah(state: ScenarioState, call: ToolCallRecord) -> bool:
    """Return whether a get_order_status call identifies Sarah Chen's order.

    Either by naming her, which the schema explicitly allows, or by an order id
    the trace shows the model resolved from an earlier lookup rather than
    guessed. Crediting only the name rewarded stuffing a customer name into an
    id field and scored the careful two-step path as the wrong tool.
    """
    order_id = _normalize(_as_str(call.arguments.get("order_id")))
    if not order_id:
        return False
    if "sarah" in order_id or "chen" in order_id:
        return True
    # Strictly earlier calls only: the failing result of this very call echoes
    # the id back, which would make any guess look resolved.
    position = next((i for i, c in enumerate(state.tool_calls) if c is call), len(state.tool_calls))
    earlier_ids = {c.id for c in state.tool_calls[:position]}
    return any(
        record.call_id in earlier_ids and order_id in _normalize(_as_str(record.result))
        for record in state.tool_results
    )


def _tc40_eval(state: ScenarioState) -> ScenarioEvaluation:
    """User: 'Get the current status of the order for customer Sarah Chen.'

    Multiple near-match tools: get_order_status (correct), get_shipping_status,
    get_customer_profile, get_contacts. Model must pick the right one.
    """
    order_calls = [c for c in state.tool_calls if c.name == "get_order_status"]
    resolved_order_calls = [c for c in order_calls if _tc40_order_id_reaches_sarah(state, c)]
    used_order = bool(resolved_order_calls)
    # Resolving the id from a customer lookup first is at least as good as
    # posting the customer's name into an order_id field, so the second call is
    # a dependency rather than an unnecessary extra.
    resolved_by_lookup = any(
        not _includes_text(c.arguments.get("order_id"), "sarah")
        and not _includes_text(c.arguments.get("order_id"), "chen")
        for c in resolved_order_calls
    )
    used_shipping = _has_tool_call(state, "get_shipping_status")
    used_customer = _has_tool_call(state, "get_customer_profile")
    used_contacts = _has_tool_call(state, "get_contacts")
    total_calls = len(state.tool_calls)

    answer = state.final_answer.lower()
    mentions_status = (
        _answer_affirms_text(answer, "shipped")
        or "ord-2026-1847" in answer
        or "keyboard" in answer
        or "usb-c" in answer
        or "tracking" in answer
    )

    if used_order and total_calls == 1 and mentions_status:
        order_call = next(c for c in state.tool_calls if c.name == "get_order_status")
        if not _result_is_usable_if_present(state, order_call):
            return _partial(
                "get_order_status returned an error, so order details cannot be confirmed."
            )
        return _pass("Selected get_order_status precisely from similar-named tools.")
    if used_order and total_calls == 1:
        return _partial(
            "Used get_order_status correctly but did not surface the order details in the answer.",
        )
    if used_order and used_customer and total_calls == 2 and resolved_by_lookup and mentions_status:
        return _pass(
            "Resolved the customer's order id, then read its status with get_order_status."
        )
    if used_order and used_customer and total_calls == 2:
        return _partial(
            "Used get_order_status + get_customer_profile — "
            "customer lookup was unnecessary but not wrong."
        )
    if used_order and total_calls >= 2:
        extras = ", ".join(c.name for c in state.tool_calls if c.name != "get_order_status")
        return _partial(f"Found the right tool but also called: {extras}")
    if not used_order and used_shipping:
        return _partial(
            "Used get_shipping_status instead of get_order_status — "
            "close but wrong tool for the request."
        )
    if not used_order and order_calls:
        return _partial("Called get_order_status with an order id that the trace does not support.")
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
    return _fail("Did not use get_order_status to check the customer's order.")


SCENARIO = ScenarioDefinition(
    id="TC-40",
    title="Domain Confusion",
    category=Category.L,
    user_message="Get the current status of the order for customer Sarah Chen.",
    description="Select get_order_status from similar-named tools "
    "(get_shipping_status, get_customer_profile, get_contacts).",
    handle_tool_call=_tc40_handle,
    evaluate=_tc40_eval,
    tools_override=LARGE_TOOLSET,
    difficulty=3,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it uses get_order_status (not get_shipping_status or get_contacts).",
    "Fail if it confuses similar-named tools or misses the right one.",
)
