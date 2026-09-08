"""TC-61 — Async Polling."""

from __future__ import annotations

import ast
from dataclasses import replace
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
    answer_affirms_number as _answer_affirms_number,
)
from tool_eval_bench.evals.helpers import (
    as_str as _as_str,
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
    tool_calls_by_name as _tool_calls_by_name,
)
from tool_eval_bench.evals.helpers import (
    with_noise as _noise,
)
from tool_eval_bench.evals.scenarios.planning._shared import (
    _result_matches_if_present,
)

_JOB_ID = "job_tc61_9f3a"


def _operation(call: ToolCallRecord) -> tuple[str, dict[str, Any], list[Any]] | None:
    """Recognize the documented mock API without executing submitted code."""
    try:
        tree = ast.parse(_as_str(call.arguments.get("code")))
        if len(tree.body) not in (1, 2):
            return None
        statement = tree.body[0]
        if len(tree.body) == 2:
            output = tree.body[1]
            if not (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and isinstance(output, ast.Expr)
                and isinstance(output.value, ast.Call)
                and isinstance(output.value.func, ast.Name)
                and output.value.func.id == "print"
                and len(output.value.args) == 1
                and not output.value.keywords
                and isinstance(output.value.args[0], ast.Name)
                and output.value.args[0].id == statement.targets[0].id
            ):
                return None
        if not isinstance(statement, (ast.Expr, ast.Assign)):
            return None
        expression = statement.value
        if (
            isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Name)
            and expression.func.id == "print"
            and len(expression.args) == 1
            and not expression.keywords
        ):
            expression = expression.args[0]
        if not isinstance(expression, ast.Call) or not isinstance(expression.func, ast.Name):
            return None
        if any(keyword.arg is None for keyword in expression.keywords):
            return None
        keywords = {
            str(keyword.arg): ast.literal_eval(keyword.value) for keyword in expression.keywords
        }
        if len(keywords) != len(expression.keywords):
            return None
        return expression.func.id, keywords, [ast.literal_eval(arg) for arg in expression.args]
    except (SyntaxError, ValueError, TypeError, RecursionError):
        return None


def _is_submission(call: ToolCallRecord) -> bool:
    return _operation(call) == ("analyze_data", {"source": "transactions_2026"}, [])


def _is_genuine_poll(call: ToolCallRecord) -> bool:
    operation = _operation(call)
    return bool(
        operation
        and operation[0] == "check_job_status"
        and (operation[1:] == ({}, [_JOB_ID]) or operation[1:] == ({"job_id": _JOB_ID}, []))
    )


def _tc61_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name != "run_code":
        return _noise({"error": f"Tool {call.name} is not relevant."}, call.name)
    previous = [
        c
        for c in state.tool_calls
        if c is not call and c.turn < call.turn and c.name == "run_code" and _is_submission(c)
    ]
    if _is_submission(call):
        if previous:
            return _noise({"error": "Analysis already submitted; poll its job instead."}, call.name)
        return _noise(
            {
                "status": "pending",
                "job_id": _JOB_ID,
                "message": 'Poll with run_code: check_job_status("job_tc61_9f3a"). Repeat while status is pending; report complete results or a terminal failed status.',
            },
            call.name,
        )
    if not previous or not _is_genuine_poll(call):
        return _noise(
            {
                "error": 'Submit analyze_data(source="transactions_2026") first, then call check_job_status with its job ID.'
            },
            call.name,
        )
    return _noise(
        {
            "status": "complete",
            "job_id": _JOB_ID,
            "result": {
                "total_records": 15420,
                "anomalies_found": 3,
                "summary": "3 anomalies detected in 15,420 records.",
            },
        },
        call.name,
    )


def _tc61_eval(state: ScenarioState) -> ScenarioEvaluation:
    """User: 'Run this data analysis script and give me the results:
    analyze_data(source="transactions_2026")'

    First run_code returns pending. Model must retry/poll to get the result.
    """
    run_calls = _tool_calls_by_name(state, "run_code")

    answer = state.final_answer.lower()
    has_result = (
        _answer_affirms_number(answer, "15420")
        or _answer_affirms_number(answer, "15,420")
        or "3 anomal" in answer
    )

    def pending_result(payload: Any) -> bool:
        return (
            isinstance(payload, dict)
            and payload.get("status") == "pending"
            and payload.get("job_id") == "job_tc61_9f3a"
        )

    def complete_result(payload: Any) -> bool:
        return (
            isinstance(payload, dict)
            and payload.get("status") == "complete"
            and payload.get("job_id") == "job_tc61_9f3a"
            and ("15420" in str(payload).replace(",", "") or "15,420" in str(payload))
            and (
                "anomalies_found" in payload
                or "3 anomalies" in str(payload).lower()
                or "records" in payload
            )
        )

    first_call = run_calls[0] if run_calls else None
    poll_calls = run_calls[1:]
    first_result_ok = bool(
        first_call
        and _is_submission(first_call)
        and _result_matches_if_present(state, first_call, pending_result)
    )
    poll_result_ok = bool(first_call) and any(
        first_call is not None
        and poll.turn > first_call.turn
        and _is_genuine_poll(poll)
        and _result_matches_if_present(state, poll, complete_result)
        for poll in poll_calls
    )
    valid_poll = first_result_ok and poll_result_ok
    if len(run_calls) >= 2 and has_result and valid_poll:
        return _pass("Polled async operation: submitted → detected pending → retried → got result.")
    if len(run_calls) >= 2:
        return _partial(
            "Retried the operation, but the trace does not prove a completed poll with surfaced results."
        )
    if len(run_calls) == 1 and "pending" in answer:
        return _partial("Ran once, reported pending status, but didn't poll for completion.")
    if len(run_calls) == 1:
        return _fail("Ran once but didn't handle the 'pending' response by retrying.")
    return _fail("Did not attempt to run the analysis script.")


def _tc61_variant(scenario: ScenarioDefinition, seed: int) -> ScenarioDefinition:
    mode = seed % 3

    def handle(state: ScenarioState, call: ToolCallRecord) -> Any:
        result = _tc61_handle(state, call)
        if result.get("status") == "complete":
            polls = sum(_is_genuine_poll(c) for c in state.tool_calls)
            if mode == 1 and polls < 3:
                return _noise(
                    {
                        "status": "pending",
                        "job_id": _JOB_ID,
                        "message": "Still running; poll again.",
                    },
                    call.name,
                )
            if mode == 2:
                return _noise(
                    {"status": "failed", "job_id": _JOB_ID, "error": "Dataset read failed."},
                    call.name,
                )
        return result

    def evaluate(state: ScenarioState) -> ScenarioEvaluation:
        if mode != 2:
            return _tc61_eval(state)
        calls = _tool_calls_by_name(state, "run_code")
        if (
            len(calls) == 2
            and _is_submission(calls[0])
            and _is_genuine_poll(calls[1])
            and calls[0].turn < calls[1].turn
        ):
            failed = any(
                r.call_id == calls[1].id
                and isinstance(r.result, dict)
                and r.result.get("status") == "failed"
                for r in state.tool_results
            )
            if (
                failed
                and "failed" in state.final_answer.lower()
                and not any(n in state.final_answer for n in ("15420", "15,420"))
            ):
                return _pass("Stopped polling and reported the terminal analysis failure.")
        return _fail("Did not stop and report the terminal analysis failure faithfully.")

    return replace(
        scenario,
        handle_tool_call=handle,
        evaluate=evaluate,
        variant_metadata={
            "version": 1,
            "seed": seed,
            "kind": ("complete", "repeated-pending", "failed")[mode],
        },
    )


SCENARIO = ScenarioDefinition(
    id="TC-61",
    title="Async Polling",
    category=Category.C,
    user_message=(
        'Run this data analysis script and give me the results: analyze_data(source="transactions_2026"). '
        "Use run_code with Python. This mock accepts one API operation per tool call, optionally assigned to a variable and printed. It does not execute loops or preserve Python variables between calls. Read the returned job ID before polling in a later tool call. The mock API provides analyze_data(source=...) and "
        "check_job_status(job_id). Poll the returned job ID until complete or failed."
    ),
    description="Handle async tool response: submit → detect pending → poll → surface result.",
    handle_tool_call=_tc61_handle,
    evaluate=_tc61_eval,
    difficulty=3,
    variant_factory=_tc61_variant,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it submits → detects 'pending' → polls again → surfaces the result.",
    "Fail if it doesn't retry after receiving the pending status.",
)
