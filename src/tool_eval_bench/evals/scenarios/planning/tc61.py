"""TC-61 — Async Polling."""

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

_POLL_ACTION_RE = re.compile(
    r"\b(?:check(?:ing)?|poll(?:ing)?|quer(?:y|ying)|get(?:ting)?)\b"
    r"(?:\s+\w+){0,4}\s+\b(?:job|status|result|completion)\b"
    r"|\b(?:job|status|result|completion)\b"
    r"(?:\s+\w+){0,4}\s+\b(?:check(?:ing)?|poll(?:ing)?|quer(?:y|ying)|get(?:ting)?)\b"
)
_NEGATED_POLL_RE = re.compile(r"\b(?:do\s+not|don'?t|never|not|refus\w*|won'?t|will\s+not)\b")
_DESTRUCTIVE_CODE_RE = re.compile(r"\b(?:delete|drop|erase|remove|rmtree|truncate|unlink|wipe)\b")


def _is_genuine_poll(call: ToolCallRecord) -> bool:
    """Recognize an attempted poll without requiring one exact code spelling."""
    code = _as_str(call.arguments.get("code")).strip().lower()
    if not code:
        return False
    normalized = re.sub(r"[^a-z0-9]+", " ", code).strip()
    if _DESTRUCTIVE_CODE_RE.search(normalized) or "analyze data" in normalized:
        return False
    if "job_tc61_9f3a" in code:
        return True
    if _NEGATED_POLL_RE.search(normalized):
        return False
    return bool(_POLL_ACTION_RE.search(normalized))


def _tc61_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    """Stateful handler: first run_code returns pending, second returns complete."""
    if call.name == "run_code":
        # Count how many run_code calls have been made (including this one)
        run_code_count = sum(1 for c in state.tool_calls if c.name == "run_code")
        if run_code_count <= 1:
            # First call: return pending
            return _noise(
                {
                    "status": "pending",
                    "job_id": "job_tc61_9f3a",
                    "message": "Analysis job is still running. Please call run_code again to check if the job has completed.",
                },
                "run_code",
            )
        if not _is_genuine_poll(call):
            return _noise(
                {"error": "The submitted code did not poll the pending analysis job."},
                "run_code",
            )
        return _noise(
            {
                "status": "complete",
                "job_id": "job_tc61_9f3a",
                "result": {
                    "total_records": 15420,
                    "anomalies_found": 3,
                    "summary": "3 anomalies detected in 15,420 records.",
                },
            },
            "run_code",
        )
    if call.name == "web_search":
        return _noise(
            {"results": [{"snippet": "Polling pattern: retry with backoff."}]},
            "web_search",
        )
    return _noise({"error": f"Tool {call.name} is not relevant."}, call.name)


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
    poll_call = run_calls[1] if len(run_calls) >= 2 else None
    first_result_ok = bool(
        first_call
        and "analyze_data" in _as_str(first_call.arguments.get("code")).lower()
        and _result_matches_if_present(state, first_call, pending_result)
    )
    poll_result_ok = bool(
        poll_call
        and _is_genuine_poll(poll_call)
        and _result_matches_if_present(state, poll_call, complete_result)
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


SCENARIO = ScenarioDefinition(
    id="TC-61",
    title="Async Polling",
    category=Category.C,
    user_message='Run this data analysis script and give me the results: analyze_data(source="transactions_2026")',
    description="Handle async tool response: submit → detect pending → poll → surface result.",
    handle_tool_call=_tc61_handle,
    evaluate=_tc61_eval,
    difficulty=3,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it submits → detects 'pending' → polls again → surfaces the result.",
    "Fail if it doesn't retry after receiving the pending status.",
)
