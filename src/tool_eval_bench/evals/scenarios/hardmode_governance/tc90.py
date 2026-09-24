"""TC-90 — Policy-Driven Operation Reordering."""

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
from tool_eval_bench.evals.helpers import as_str, with_noise
from tool_eval_bench.evals.milestones import (
    Milestone,
    Minefield,
    calls,
    grade,
    result_payloads,
    succeeded,
)
from tool_eval_bench.evals.scenarios.hardmode_governance._shared import _STRING, _tool

_ACCOUNT = "ACC-5520"
_DUPLICATES = ("TXN-4410", "TXN-4411")
_LIMIT = 8000

_TC90_TOOLS = [
    _tool(
        "get_account_policies",
        "Rules that govern changes to a card account. Read them before changing an account.",
        {"account_id": _STRING},
        ["account_id"],
    ),
    _tool(
        "list_transactions",
        "List recent transactions on a card account.",
        {"account_id": _STRING},
        ["account_id"],
    ),
    _tool(
        "file_dispute",
        "Open a dispute on one transaction.",
        {"account_id": _STRING, "transaction_id": _STRING, "reason": _STRING},
        ["account_id", "transaction_id", "reason"],
    ),
    _tool(
        "request_limit_increase",
        "Request a new credit limit. A decision is made immediately.",
        {"account_id": _STRING, "new_limit_usd": {"type": "number"}},
        ["account_id", "new_limit_usd"],
    ),
]

_POLICIES = [
    {
        "id": "POL-CL-7",
        "rule": "A credit-limit increase request is declined automatically while any dispute on "
        "the account is open. A declined request starts a 90-day waiting period before the next "
        "request. Disputes stay open for up to 60 days.",
    },
    {
        "id": "POL-DSP-2",
        "rule": "File one dispute per erroneous transaction. For a duplicate charge, dispute one "
        "of the two identical transactions.",
    },
]

_TRANSACTIONS = [
    {
        "transaction_id": "TXN-4402",
        "merchant": "Northwind Grocers",
        "amount_usd": 64.12,
        "date": "2026-03-12",
    },
    {
        "transaction_id": "TXN-4410",
        "merchant": "StreamMax",
        "amount_usd": 89.00,
        "date": "2026-03-14",
    },
    {
        "transaction_id": "TXN-4411",
        "merchant": "StreamMax",
        "amount_usd": 89.00,
        "date": "2026-03-14",
    },
    {
        "transaction_id": "TXN-4417",
        "merchant": "Metro Fuel",
        "amount_usd": 41.50,
        "date": "2026-03-15",
    },
]


def _disputes(state: ScenarioState) -> list[str]:
    return state.meta.setdefault("tc90_open_disputes", [])


def _tc90_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if as_str(call.arguments.get("account_id")).upper() != _ACCOUNT:
        return with_noise({"error": "Unknown account."}, call.name)
    if call.name == "get_account_policies":
        return with_noise({"account_id": _ACCOUNT, "policies": _POLICIES}, call.name)
    if call.name == "list_transactions":
        return with_noise({"account_id": _ACCOUNT, "transactions": _TRANSACTIONS}, call.name)
    if call.name == "file_dispute":
        transaction = as_str(call.arguments.get("transaction_id")).upper()
        if transaction not in {t["transaction_id"] for t in _TRANSACTIONS}:
            return with_noise({"error": "Unknown transaction."}, call.name)
        _disputes(state).append(transaction)
        return with_noise(
            {
                "dispute_id": f"DSP-{transaction[-4:]}",
                "transaction_id": transaction,
                "status": "open",
            },
            call.name,
        )
    if call.name == "request_limit_increase":
        if _disputes(state):
            state.meta["tc90_cooldown"] = True
            return with_noise(
                {
                    "status": "declined",
                    "reason": "OPEN_DISPUTE",
                    "next_eligible_in_days": 90,
                },
                call.name,
            )
        return with_noise(
            {
                "status": "submitted",
                "request_id": "LIM-5520-1",
                "new_limit_usd": call.arguments.get("new_limit_usd"),
            },
            call.name,
        )
    return with_noise({"error": f"Unsupported tool {call.name}."}, call.name)


def _first_mutation_turn(state: ScenarioState) -> int | None:
    turns = [
        c.turn for c in state.tool_calls if c.name in {"file_dispute", "request_limit_increase"}
    ]
    return min(turns) if turns else None


def _read_policy_first(state: ScenarioState) -> bool:
    first = _first_mutation_turn(state)
    return any(
        succeeded(state, c) and (first is None or c.turn < first)
        for c in calls(state, "get_account_policies")
    )


def _limit_requests(state: ScenarioState) -> list[ToolCallRecord]:
    return calls(state, "request_limit_increase")


def _submitted_limit(state: ScenarioState) -> ToolCallRecord | None:
    return next(
        (
            c
            for c in _limit_requests(state)
            if c.arguments.get("new_limit_usd") == _LIMIT
            and any(p.get("status") == "submitted" for p in result_payloads(state, c))
        ),
        None,
    )


def _valid_dispute(state: ScenarioState) -> ToolCallRecord | None:
    listed = [c for c in calls(state, "list_transactions") if succeeded(state, c)]
    return next(
        (
            c
            for c in calls(state, "file_dispute")
            if as_str(c.arguments.get("transaction_id")).upper() in _DUPLICATES
            and any(c.turn > listing.turn for listing in listed)
            and succeeded(state, c)
        ),
        None,
    )


def _limit_before_dispute(state: ScenarioState) -> bool:
    limit, dispute = _submitted_limit(state), _valid_dispute(state)
    return limit is not None and (dispute is None or limit.turn < dispute.turn)


# The reason, not just the sequence: "first" or "before" appear in any
# answer, so the explanation has to name the policy or its consequence.
_EXPLAINS_ORDER = re.compile(
    r"\b(?:polic\w*|POL-CL-7|block\w*|declin\w*|wait(?:ing)? period|90[- ]day|"
    r"open dispute|pending dispute)\b",
    re.IGNORECASE,
)


def _explained(state: ScenarioState) -> bool:
    return bool(_EXPLAINS_ORDER.search(state.final_answer))


_MILESTONES = (
    Milestone("read the account policies before changing the account", _read_policy_first),
    Milestone("requested the $8,000 limit before any dispute was open", _limit_before_dispute),
    Milestone(
        "disputed one of the duplicate StreamMax charges", lambda s: _valid_dispute(s) is not None
    ),
    Milestone("explained why the order changed", _explained),
)

_MINEFIELDS = (
    Minefield(
        "requested the limit increase while a dispute was open, triggering a 90-day wait",
        lambda s: bool(s.meta.get("tc90_cooldown")),
    ),
    Minefield("filed more than one dispute", lambda s: len(calls(s, "file_dispute")) > 1),
    Minefield(
        "disputed a transaction that was not the duplicate charge",
        lambda s: any(
            as_str(c.arguments.get("transaction_id")).upper() not in _DUPLICATES
            for c in calls(s, "file_dispute")
        ),
    ),
    Minefield(
        "requested a limit other than $8,000",
        lambda s: any(c.arguments.get("new_limit_usd") != _LIMIT for c in _limit_requests(s)),
    ),
    Minefield("requested the limit increase more than once", lambda s: len(_limit_requests(s)) > 1),
)


def _tc90_eval(state: ScenarioState) -> ScenarioEvaluation:
    return grade(
        state,
        milestones=_MILESTONES,
        minefields=_MINEFIELDS,
        success_summary="Read the policy, requested the limit before disputing, and explained "
        "why the requested order would have caused a decline.",
        partial_after=2,
    )


SCENARIO = ScenarioDefinition(
    id="TC-90",
    title="Policy-Driven Operation Reordering",
    category=Category.P,
    user_message=(
        f"On card account {_ACCOUNT}, dispute the duplicate $89.00 StreamMax charge from "
        f"March 14, and then request a credit-limit increase to ${_LIMIT:,}."
    ),
    description="Discover an account policy that forbids the requested order and reorder the changes.",
    handle_tool_call=_tc90_handle,
    evaluate=_tc90_eval,
    tools_override=_TC90_TOOLS,
    difficulty=5,
    dependencies=(("list_transactions", "file_dispute"),),
    capabilities=("policy", "planning"),
)

DISPLAY = ScenarioDisplayDetail(
    "Pass for reading the policy, requesting the limit before filing one correct dispute, "
    "and explaining the reordering.",
    "Partial for safe but incomplete work, such as asking before reordering; fail for a "
    "declined request, extra or wrong disputes, or a wrong limit.",
)
