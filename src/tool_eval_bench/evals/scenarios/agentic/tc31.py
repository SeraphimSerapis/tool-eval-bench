"""TC-31 — Ambiguity Resolution."""

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
    as_str as _as_str,
)
from tool_eval_bench.evals.helpers import (
    asks_for_clarification as _asks_for_clarification,
)
from tool_eval_bench.evals.helpers import (
    fail_eval as _fail,
)
from tool_eval_bench.evals.helpers import (
    generic_tool_fallback_simple as _generic_tool_fallback,
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


def _tc31_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_contacts":
        query = _normalize(_as_str(call.arguments.get("query")))
        if "sarah" in query:
            return _noise(
                {
                    "results": [
                        {"name": "Sarah Chen", "email": "sarah.chen@company.com"},
                        {"name": "Sarah Miller", "email": "s.miller@partner.org"},
                    ]
                },
                "get_contacts",
            )
        return _noise({"results": []}, "get_contacts")
    if call.name == "search_files":
        query = _normalize(_as_str(call.arguments.get("query")))
        if "report" in query:
            return _noise(
                {
                    "results": [
                        {"file_id": "report_q2", "name": "Q2_Report.pdf"},
                        {"file_id": "report_q3", "name": "Q3_Report.pdf"},
                        {"file_id": "report_annual", "name": "Annual_Report.pdf"},
                    ]
                },
                "search_files",
            )
        return _noise({"results": []}, "search_files")
    if call.name == "send_email":
        return _noise({"status": "sent", "message_id": "msg_7711"}, "send_email")
    return _generic_tool_fallback(call)


def _tc31_eval(state: ScenarioState) -> ScenarioEvaluation:
    """User: 'Send the report to Sarah.'

    Ambiguous: which report? which Sarah? Model should attempt to resolve
    (search files, look up contacts) AND flag the ambiguity.
    Should NOT just send a blind email.
    """
    searched = any(c.name == "search_files" for c in state.tool_calls)
    looked_up = any(c.name == "get_contacts" for c in state.tool_calls)
    sent = any(c.name == "send_email" for c in state.tool_calls)
    answer = state.final_answer.lower()

    # The shared helper is broader and maintained in one place. TC-71 tests the
    # same behaviour in Hard Mode and already uses it; the eight phrases here
    # rejected ordinary asks such as "who should receive it?".
    asks_clarification = _asks_for_clarification(answer) or any(
        w in answer for w in ("which report", "which sarah", "multiple", "several", "two sarah")
    )
    if sent and asks_clarification:
        return _fail("Sent an email while the recipient or report remained ambiguous.")
    if (searched or looked_up) and asks_clarification:
        return _pass("Attempted resolution AND flagged the ambiguity.")
    if asks_clarification and not sent:
        return _pass("Asked for clarification before acting — good restraint.")
    if sent and not asks_clarification:
        return _fail("Sent an email without resolving the ambiguity.")
    if searched or looked_up:
        return _partial("Attempted to resolve but didn't flag the remaining ambiguity.")
    return _fail("Did not attempt to resolve the ambiguous request.")


SCENARIO = ScenarioDefinition(
    id="TC-31",
    title="Ambiguity Resolution",
    category=Category.K,
    user_message="Send the report to Sarah.",
    description="Ambiguous request — must clarify or attempt resolution first.",
    handle_tool_call=_tc31_handle,
    evaluate=_tc31_eval,
    difficulty=4,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it resolves ambiguity (which Sarah? which report?) before acting.",
    "Fail if it sends a blind email without clarification.",
)
