"""TC-103: multi-constraint cart and coupon set optimization."""

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
    answer_affirms_number,
    answer_affirms_text,
    as_str,
    with_noise,
)
from tool_eval_bench.evals.milestones import Milestone, Minefield, calls, grade, succeeded
from tool_eval_bench.evals.scenarios.hardmode_transactional._shared import _STRING, _tool

_CATALOG: dict[str, dict[str, Any]] = {
    "CHR-101": {
        "sku": "CHR-101",
        "name": "AeroMesh Ergonomic Chair",
        "category": "chairs",
        "price_usd": 280,
        "in_stock": True,
        "express_eligible": True,
    },
    "CHR-102": {
        "sku": "CHR-102",
        "name": "ExecLeather Ergonomic Chair",
        "category": "chairs",
        "price_usd": 350,
        "in_stock": True,
        "express_eligible": False,
    },
    "CHR-103": {
        "sku": "CHR-103",
        "name": "BasicTask Ergonomic Chair",
        "category": "chairs",
        "price_usd": 190,
        "in_stock": False,
        "express_eligible": True,
    },
    "DSK-201": {
        "sku": "DSK-201",
        "name": "LiftPro Standing Desk",
        "category": "desks",
        "price_usd": 560,
        "in_stock": True,
        "express_eligible": True,
    },
    "DSK-202": {
        "sku": "DSK-202",
        "name": "Studio Standing Desk",
        "category": "desks",
        "price_usd": 490,
        "in_stock": True,
        "express_eligible": False,
    },
}

_COUPONS: list[dict[str, Any]] = [
    {
        "code": "WELCOME50",
        "discount_usd": 50,
        "min_subtotal_usd": 500,
        "requires_bundle": False,
        "exclude_express_items": False,
        "description": "$50 off any order of $500 or more.",
    },
    {
        "code": "LAB100",
        "discount_usd": 100,
        "min_subtotal_usd": 1100,
        "requires_bundle": True,
        "exclude_express_items": False,
        "description": "$100 off orders >= $1,100 containing both chairs and a standing desk; valid on express items.",
    },
    {
        "code": "CLEARANCE150",
        "discount_usd": 150,
        "min_subtotal_usd": 1000,
        "requires_bundle": False,
        "exclude_express_items": True,
        "description": "$150 off orders >= $1,000, but invalid if the cart contains any express-eligible items.",
    },
]

_TC103_TOOLS = [
    _tool(
        "search_catalog",
        "Search the office equipment catalog by category ('chairs', 'desks', or 'all').",
        {"category": _STRING},
        ["category"],
    ),
    _tool(
        "list_coupons",
        "List available promotional coupon codes and their eligibility rules.",
        {},
        [],
    ),
    _tool(
        "add_to_cart",
        "Add a quantity of a SKU to the shopping cart (accumulates if already in cart).",
        {"sku": _STRING, "quantity": {"type": "integer"}},
        ["sku", "quantity"],
    ),
    _tool(
        "remove_from_cart",
        "Remove a SKU completely from the shopping cart.",
        {"sku": _STRING},
        ["sku"],
    ),
    _tool(
        "apply_coupon",
        "Apply a single coupon code to the current shopping cart.",
        {"coupon_code": _STRING},
        ["coupon_code"],
    ),
    _tool(
        "get_cart_summary",
        "Inspect the current cart items, subtotal, applied coupon, and final total.",
        {},
        [],
    ),
]


# The shared cart already holds a colleague's non-express chair, so building
# the order is a set difference against existing state, not a fresh fill.
_INITIAL_CART = {"CHR-102": 1}
_COUPON_BY_CODE = {coupon["code"]: coupon for coupon in _COUPONS}


def _cart(state: ScenarioState) -> dict[str, int]:
    return state.meta.setdefault("tc103_cart", dict(_INITIAL_CART))


def _coupon_problem(code: str, cart: dict[str, int]) -> str | None:
    """Why ``code`` cannot apply to ``cart``, or None when it can."""
    coupon = _COUPON_BY_CODE[code]
    subtotal = sum(_CATALOG[s]["price_usd"] * q for s, q in cart.items())
    if subtotal < coupon["min_subtotal_usd"]:
        return f"{code} requires a subtotal of at least ${coupon['min_subtotal_usd']:,}."
    if coupon["requires_bundle"] and {_CATALOG[s]["category"] for s in cart} != {
        "chairs",
        "desks",
    }:
        return f"{code} requires both chairs and a standing desk in the cart."
    if coupon["exclude_express_items"] and any(_CATALOG[s]["express_eligible"] for s in cart):
        return f"{code} cannot be applied when the cart contains express-eligible items."
    return None


def _summary(state: ScenarioState) -> dict[str, Any]:
    cart = _cart(state)
    subtotal = sum(_CATALOG[sku]["price_usd"] * qty for sku, qty in cart.items())
    coupon = state.meta.get("tc103_coupon")
    # A coupon whose conditions stop holding after a cart edit stops discounting.
    discount = (
        _COUPON_BY_CODE[coupon]["discount_usd"]
        if coupon and _coupon_problem(coupon, cart) is None
        else 0
    )
    return {
        "items": [
            {"sku": sku, "quantity": qty, "unit_price_usd": _CATALOG[sku]["price_usd"]}
            for sku, qty in sorted(cart.items())
        ],
        "subtotal_usd": subtotal,
        "applied_coupon": coupon,
        "discount_usd": discount,
        "final_total_usd": subtotal - discount,
    }


def _tc103_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "search_catalog":
        cat = as_str(call.arguments.get("category")).strip().lower()
        items = [
            item
            for item in _CATALOG.values()
            if cat in {"", "all", "office", "equipment"}
            or cat in item["category"]
            or item["category"].rstrip("s") in cat
        ]
        return with_noise({"items": items}, call.name)
    if call.name == "list_coupons":
        return with_noise({"coupons": _COUPONS}, call.name)
    if call.name == "add_to_cart":
        sku = as_str(call.arguments.get("sku")).strip().upper()
        qty_raw = call.arguments.get("quantity")
        qty = qty_raw if isinstance(qty_raw, int) and not isinstance(qty_raw, bool) else 0
        if sku not in _CATALOG or qty <= 0:
            return with_noise({"error": "Invalid SKU or non-positive quantity."}, call.name)
        if not _CATALOG[sku]["in_stock"]:
            return with_noise(
                {"error": f"SKU {sku} is currently out of stock.", "error_code": "OUT_OF_STOCK"},
                call.name,
            )
        cart = _cart(state)
        cart[sku] = cart.get(sku, 0) + qty
        return with_noise(_summary(state), call.name)
    if call.name == "remove_from_cart":
        sku = as_str(call.arguments.get("sku")).strip().upper()
        cart = _cart(state)
        cart.pop(sku, None)
        return with_noise(_summary(state), call.name)
    if call.name == "apply_coupon":
        code = as_str(call.arguments.get("coupon_code")).strip().upper()
        if code not in _COUPON_BY_CODE:
            return with_noise({"error": f"Unknown coupon code {code}."}, call.name)
        problem = _coupon_problem(code, _cart(state))
        if problem is not None:
            return with_noise({"error": problem, "error_code": "COUPON_INELIGIBLE"}, call.name)
        state.meta["tc103_coupon"] = code
        return with_noise(_summary(state), call.name)
    if call.name == "get_cart_summary":
        return with_noise(_summary(state), call.name)
    return with_noise({"error": f"Unsupported tool {call.name}."}, call.name)


def _exact_cart(state: ScenarioState) -> bool:
    return _cart(state) == {"CHR-101": 2, "DSK-201": 1}


def _has_extra_or_non_express_items(state: ScenarioState) -> bool:
    cart = _cart(state)
    return (
        any(sku not in {"CHR-101", "DSK-201"} for sku in cart)
        or cart.get("CHR-101", 0) > 2
        or cart.get("DSK-201", 0) > 1
    )


def _over_budget(state: ScenarioState) -> bool:
    cart = _cart(state)
    if not cart:
        return False
    return int(_summary(state)["final_total_usd"]) > 1050


_MILESTONES = (
    Milestone(
        "searched the catalog and listed available coupons",
        lambda s: (
            any(succeeded(s, c) for c in calls(s, "search_catalog"))
            and any(succeeded(s, c) for c in calls(s, "list_coupons"))
        ),
    ),
    Milestone(
        "left the cart with 2x CHR-101 and 1x DSK-201 (both in stock and express-eligible)",
        _exact_cart,
    ),
    Milestone(
        "applied coupon LAB100 bringing the final cart total to $1,020",
        lambda s: _exact_cart(s) and s.meta.get("tc103_coupon") == "LAB100",
    ),
    Milestone(
        "reported the $1,020 final total and coupon LAB100",
        lambda s: (
            answer_affirms_number(s.final_answer, "1020")
            and answer_affirms_text(s.final_answer, "LAB100")
        ),
    ),
)

_MINEFIELDS = (
    Minefield(
        "left non-express or extra items in the final cart",
        _has_extra_or_non_express_items,
    ),
    Minefield(
        "left the final cart total above the $1,050 budget",
        _over_budget,
    ),
)


def _tc103_eval(state: ScenarioState) -> ScenarioEvaluation:
    return grade(
        state,
        milestones=_MILESTONES,
        minefields=_MINEFIELDS,
        success_summary=(
            "Removed the leftover CHR-102, selected 2x CHR-101 and 1x DSK-201, applied LAB100 for a $1,020 total "
            "within the $1,050 budget, and left no extra items in the cart."
        ),
        partial_after=2,
    )


SCENARIO = ScenarioDefinition(
    id="TC-103",
    title="Multi-Constraint Cart & Coupon Optimization",
    category=Category.P,
    user_message=(
        "Set up our shared cart for the Berlin lab: we need 2 ergonomic chairs and 1 standing "
        "desk, all in stock and eligible for express shipping. Apply the one coupon that brings "
        "the final total after discount to at most $1,050, make sure nothing else is in the "
        "cart, and tell me the final total and the coupon code."
    ),
    description=(
        "Reconcile a pre-filled shared cart against stock and express-shipping constraints, "
        "then pick the only coupon that satisfies its item rules and the budget cap."
    ),
    handle_tool_call=_tc103_handle,
    evaluate=_tc103_eval,
    tools_override=_TC103_TOOLS,
    difficulty=4,
    dependencies=(("search_catalog", "add_to_cart"), ("list_coupons", "apply_coupon")),
    capabilities=("constraints", "planning", "safe-mutation"),
)

DISPLAY = ScenarioDisplayDetail(
    "Pass for removing the leftover CHR-102, adding 2x CHR-101 and 1x DSK-201, applying "
    "LAB100 ($1,020 total), and reporting $1,020 and LAB100.",
    "Fail for leaving non-express or extra items in the cart (including the CLEARANCE150 "
    "route through non-express items), or leaving the total above $1,050.",
)
