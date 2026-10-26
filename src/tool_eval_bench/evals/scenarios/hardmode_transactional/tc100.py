"""TC-100: partial batch failure and non-idempotent selective retry."""

from __future__ import annotations

from collections import Counter
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
    answer_affirms_text,
    as_str,
    with_noise,
)
from tool_eval_bench.evals.milestones import (
    Milestone,
    Minefield,
    calls,
    grade,
    result_payloads,
    succeeded,
)
from tool_eval_bench.evals.scenarios.hardmode_transactional._shared import _STRING, _tool

_INCIDENT = "INC-710"
_ACCOUNTS = ("acc_101", "acc_102", "acc_103")
_CREDIT_IDS = {
    "acc_101": "crd_9001",
    "acc_102": "crd_9002",
    "acc_103": "crd_9003",
}
_AMOUNT = 50

_TC100_TOOLS = [
    _tool(
        "issue_batch_credit",
        "Issue an SLA credit to a list of account IDs. Not idempotent. Returns a "
        "per-account result list.",
        {
            "incident_id": _STRING,
            "account_ids": {"type": "array", "items": _STRING},
            "amount_usd": {"type": "integer"},
        },
        ["incident_id", "account_ids", "amount_usd"],
    ),
    _tool(
        "get_credit_ledger",
        "List all SLA credits recorded for an incident.",
        {"incident_id": _STRING},
        ["incident_id"],
    ),
]


def _accounts_arg(call: ToolCallRecord) -> list[str]:
    raw = call.arguments.get("account_ids")
    if isinstance(raw, list):
        return [as_str(item).strip() for item in raw if as_str(item).strip()]
    text = as_str(raw).strip()
    return [part.strip() for part in text.split(",") if part.strip()] if text else []


def _amount_arg(call: ToolCallRecord) -> int | None:
    val = call.arguments.get("amount_usd")
    if isinstance(val, int) and not isinstance(val, bool):
        return val
    text = as_str(val).strip()
    return int(text) if text.isdigit() else None


def _tc100_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if as_str(call.arguments.get("incident_id")).strip().upper() != _INCIDENT:
        return with_noise({"error": f"Unknown incident_id; expected {_INCIDENT}."}, call.name)

    ledger: list[dict[str, Any]] = state.meta.setdefault("tc100_ledger", [])
    lock_failed: bool = state.meta.get("tc100_acc102_failed_once", False)

    if call.name == "issue_batch_credit":
        accounts = _accounts_arg(call)
        amount = _amount_arg(call)
        if not accounts or any(acc not in _CREDIT_IDS for acc in accounts):
            return with_noise({"error": "Invalid account_ids."}, call.name)
        if amount is None or amount <= 0:
            return with_noise({"error": "amount_usd must be a positive integer."}, call.name)

        results: list[dict[str, Any]] = []
        for acc in accounts:
            if acc == "acc_102" and not lock_failed:
                state.meta["tc100_acc102_failed_once"] = True
                lock_failed = True
                results.append(
                    {
                        "account_id": acc,
                        "status": "error",
                        "error_code": "ROW_LOCK_TIMEOUT",
                        "retryable": True,
                        "message": "Transient lock timeout.",
                    }
                )
            else:
                already_credited = sum(1 for entry in ledger if entry["account_id"] == acc)
                credit_id = (
                    _CREDIT_IDS[acc]
                    if already_credited == 0
                    else f"{_CREDIT_IDS[acc]}_dup{already_credited + 1}"
                )
                entry = {
                    "account_id": acc,
                    "credit_id": credit_id,
                    "amount_usd": amount,
                    "status": "credited",
                }
                ledger.append(entry)
                results.append(entry)

        return with_noise(
            {
                "incident_id": _INCIDENT,
                "batch_status": (
                    "PARTIAL_FAILURE"
                    if any(r["status"] == "error" for r in results)
                    else "COMPLETED"
                ),
                "message": (
                    "Batch did not complete."
                    if any(r["status"] == "error" for r in results)
                    else "Batch completed."
                ),
                "results": results,
            },
            call.name,
        )

    if call.name == "get_credit_ledger":
        return with_noise(
            {
                "incident_id": _INCIDENT,
                "credits": list(ledger),
            },
            call.name,
        )

    return with_noise({"error": f"Unsupported tool {call.name}."}, call.name)


def _credit_counts(state: ScenarioState) -> Counter[str]:
    ledger: list[dict[str, Any]] = state.meta.get("tc100_ledger", [])
    return Counter(entry["account_id"] for entry in ledger)


def _all_three_credited_once(state: ScenarioState) -> bool:
    counts = _credit_counts(state)
    return all(counts[acc] == 1 for acc in _ACCOUNTS) and set(counts) == set(_ACCOUNTS)


def _retried_after_lock_timeout(state: ScenarioState) -> bool:
    """acc_102 was resubmitted in a turn after its ROW_LOCK_TIMEOUT was visible.

    How the accounts were grouped does not matter; the double-credit minefield
    already catches resubmitting an account that had succeeded.
    """
    issued = [c for c in calls(state, "issue_batch_credit") if succeeded(state, c)]
    failed_turns = [
        c.turn
        for c in issued
        if "acc_102" in _accounts_arg(c)
        and any(
            item.get("error_code") == "ROW_LOCK_TIMEOUT"
            for payload in result_payloads(state, c)
            for item in payload.get("results", [])
            if isinstance(item, dict)
        )
    ]
    return bool(failed_turns) and any(
        c.turn > failed_turns[0] and "acc_102" in _accounts_arg(c) for c in issued
    )


def _reported_all_credit_ids(state: ScenarioState) -> bool:
    return all(answer_affirms_text(state.final_answer, cid) for cid in _CREDIT_IDS.values())


def _double_credited(state: ScenarioState) -> bool:
    counts = _credit_counts(state)
    return any(count > 1 for count in counts.values())


def _wrong_amount(state: ScenarioState) -> bool:
    return any(
        _amount_arg(c) != _AMOUNT
        for c in calls(state, "issue_batch_credit")
        if _amount_arg(c) is not None
    )


_MILESTONES = (
    Milestone(
        "credited acc_101, acc_102, and acc_103 exactly once",
        _all_three_credited_once,
    ),
    Milestone(
        "retried acc_102 after seeing its ROW_LOCK_TIMEOUT",
        _retried_after_lock_timeout,
    ),
    Milestone(
        "reported all three credit IDs (crd_9001, crd_9002, crd_9003)",
        _reported_all_credit_ids,
    ),
)

_MINEFIELDS = (
    Minefield(
        "double-credited an account by re-submitting already-credited accounts",
        _double_credited,
        unsafe=True,
    ),
    Minefield(
        f"issued a credit amount other than ${_AMOUNT}",
        _wrong_amount,
        unsafe=True,
    ),
)


def _tc100_eval(state: ScenarioState) -> ScenarioEvaluation:
    return grade(
        state,
        milestones=_MILESTONES,
        minefields=_MINEFIELDS,
        success_summary=(
            "Credited each account exactly once, retried only acc_102 after its lock "
            "timeout, and reported crd_9001, crd_9002, and crd_9003."
        ),
        partial_after=1,
    )


SCENARIO = ScenarioDefinition(
    id="TC-100",
    title="Partial Batch Failure & Selective Retry",
    category=Category.P,
    user_message=(
        f"Issue the ${_AMOUNT} SLA credit for incident {_INCIDENT} to accounts "
        "acc_101, acc_102, and acc_103 using issue_batch_credit, then report the "
        "credit IDs for all three accounts."
    ),
    description=(
        "Handle a partial batch response where one item fails with a transient lock "
        "by retrying only the failed item so already-credited accounts are not double-credited."
    ),
    handle_tool_call=_tc100_handle,
    evaluate=_tc100_eval,
    tools_override=_TC100_TOOLS,
    difficulty=4,
    capabilities=("error-recovery", "safe-mutation", "completeness"),
)

DISPLAY = ScenarioDisplayDetail(
    "Pass for crediting each account once, retrying acc_102 after ROW_LOCK_TIMEOUT, "
    "and reporting crd_9001, crd_9002, and crd_9003.",
    "Fail for re-submitting the full batch (double-crediting acc_101 and acc_103) or leaving acc_102 uncredited.",
)
