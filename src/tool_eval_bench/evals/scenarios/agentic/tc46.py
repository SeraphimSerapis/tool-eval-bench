"""TC-46 — Deep Multi-Turn Research (5 turns)."""

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
    as_str as _as_str,
)
from tool_eval_bench.evals.helpers import (
    fail_eval as _fail,
)
from tool_eval_bench.evals.helpers import (
    generic_tool_fallback_simple as _generic_tool_fallback,
)
from tool_eval_bench.evals.helpers import (
    matching_tool_results as _matching_tool_results,
)
from tool_eval_bench.evals.helpers import (
    normalize as _normalize,
)
from tool_eval_bench.evals.helpers import (
    parse_math_expression as _parse_math_expression,
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


def _tc46_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    """Mock handler for a 5-turn research workflow."""
    if call.name == "search_files":
        query = _normalize(_as_str(call.arguments.get("query", "")))
        if "competitor" in query or "analysis" in query:
            return _noise(
                {
                    "results": [
                        {"file_id": "comp_report_2025", "name": "Competitor_Analysis_2025.pdf"},
                        {"file_id": "comp_report_2024", "name": "Competitor_Analysis_2024.pdf"},
                    ]
                },
                "search_files",
            )
        return _noise({"results": []}, "search_files")
    if call.name == "read_file":
        fid = _as_str(call.arguments.get("file_id", ""))
        if "2025" in fid:
            return _noise(
                {
                    "content": (
                        "Competitor Analysis 2025\n"
                        "Market Share: Acme 35%, BetaCorp 28%, Gamma Inc 22%, Others 15%\n"
                        "Key Trend: AI-driven automation growing 40% YoY\n"
                        "Risk: BetaCorp launching new platform Q4 2025"
                    ),
                },
                "read_file",
            )
        if "2024" in fid:
            return _noise(
                {
                    "content": (
                        "Competitor Analysis 2024\n"
                        "Market Share: Acme 32%, BetaCorp 25%, Gamma Inc 24%, Others 19%\n"
                        "Key Trend: Cloud migration accelerating\n"
                        "Risk: Gamma Inc acquired CloudFirst"
                    ),
                },
                "read_file",
            )
        return _noise({"error": "File not found"}, "read_file")
    if call.name == "calculator":
        expr = _as_str(call.arguments.get("expression", ""))
        result = _parse_math_expression(expr)
        if result is not None:
            return _noise({"result": result}, "calculator")
        return _noise({"error": "Invalid expression"}, "calculator")
    if call.name == "send_email":
        return _noise(
            {
                "status": "sent",
                "to": _as_str(call.arguments.get("to", "")),
                "subject": _as_str(call.arguments.get("subject", "")),
            },
            "send_email",
        )
    if call.name == "get_contacts":
        query = _normalize(_as_str(call.arguments.get("query", "")))
        if "manager" in query or "boss" in query or "jordan" in query:
            return _noise(
                {
                    "results": [
                        {"name": "Jordan Park", "email": "jordan.park@company.com"},
                    ]
                },
                "get_contacts",
            )
        return _noise({"results": []}, "get_contacts")
    return _generic_tool_fallback(call)


def _tc46_eval(state: ScenarioState) -> ScenarioEvaluation:
    """5-turn research workflow:
    1. 'Find the competitor analysis report' → search_files
    2. 'Read the 2025 one' → read_file (must recall file_id from turn 1)
    3. 'What's our market share growth vs last year?' → read_file 2024 + calculator
    4. 'Summarize the key risks' → answer from context (no tools needed)
    5. 'Email the summary to my manager' → get_contacts + send_email

    A model must track state across all 5 turns.
    """
    calls = state.tool_calls

    def _after(candidate: ToolCallRecord, earlier: ToolCallRecord | None) -> bool:
        """Check trace order while retaining compatibility with old records."""
        if earlier is None:
            return False
        candidate_index = next((index for index, call in enumerate(calls) if call is candidate), -1)
        earlier_index = next((index for index, call in enumerate(calls) if call is earlier), -1)
        return candidate_index >= 0 and earlier_index >= 0 and candidate_index > earlier_index

    def _result_matches(call: ToolCallRecord, predicate: Any) -> bool:
        """Validate a concrete result when the trace includes one.

        Synthetic evaluator states often omit tool results.  Runtime traces do
        include them, and an empty/error result must not be treated as proof
        that a dependency was satisfied.
        """
        payloads = [result.result for result in _matching_tool_results(state, call)]
        return not payloads or any(predicate(payload) for payload in payloads)

    search_call = next(
        (
            call
            for call in calls
            if call.name == "search_files"
            and any(
                term in _normalize(_as_str(call.arguments.get("query", "")))
                for term in ("competitor", "analysis")
            )
            and _result_is_usable_if_present(state, call)
            and _result_matches(
                call,
                lambda payload: (
                    isinstance(payload, dict)
                    and any(
                        isinstance(item, dict)
                        and _normalize(_as_str(item.get("file_id")))
                        in {"comp_report_2025", "comp_report_2024"}
                        for item in payload.get("results", [])
                    )
                ),
            )
        ),
        None,
    )
    searched = search_call is not None

    read_2025_call = next(
        (
            call
            for call in calls
            if call.name == "read_file"
            and _normalize(_as_str(call.arguments.get("file_id", ""))) == "comp_report_2025"
            and _after(call, search_call)
            and _result_is_usable_if_present(state, call)
            and _result_matches(
                call,
                lambda payload: (
                    isinstance(payload, dict)
                    and "2025" in _as_str(payload.get("content"))
                    and "market share" in _normalize(_as_str(payload.get("content")))
                ),
            )
        ),
        None,
    )
    read_2025 = read_2025_call is not None
    read_2024_call = next(
        (
            call
            for call in calls
            if call.name == "read_file"
            and _normalize(_as_str(call.arguments.get("file_id", ""))) == "comp_report_2024"
            and _after(call, read_2025_call)
            and _result_is_usable_if_present(state, call)
            and _result_matches(
                call,
                lambda payload: (
                    isinstance(payload, dict)
                    and "2024" in _as_str(payload.get("content"))
                    and "market share" in _normalize(_as_str(payload.get("content")))
                ),
            )
        ),
        None,
    )
    read_2024 = read_2024_call is not None
    manager_call = next(
        (
            call
            for call in calls
            if call.name == "get_contacts"
            and any(
                term in _normalize(_as_str(call.arguments.get("query", "")))
                for term in ("manager", "boss", "jordan")
            )
            and _after(call, read_2024_call)
            and _result_is_usable_if_present(state, call)
            and _result_matches(
                call,
                lambda payload: (
                    isinstance(payload, dict)
                    and any(
                        isinstance(item, dict)
                        and _normalize(_as_str(item.get("email"))) == "jordan.park@company.com"
                        for item in payload.get("results", [])
                    )
                ),
            )
        ),
        None,
    )
    emailed = any(
        call.name == "send_email"
        and _as_str(call.arguments.get("to", "")).strip().lower() == "jordan.park@company.com"
        and isinstance(call.arguments.get("subject"), str)
        and bool(call.arguments["subject"].strip())
        and isinstance(call.arguments.get("body"), str)
        and bool(call.arguments["body"].strip())
        and _after(call, manager_call)
        and _result_is_usable_if_present(state, call)
        and _result_matches(
            call,
            lambda payload: (
                not isinstance(payload, dict)
                or not payload.get("status")
                or _normalize(_as_str(payload.get("status"))) == "sent"
            ),
        )
        for call in calls
    )
    answer = f"{state.final_answer} {' '.join(state.assistant_messages)}".lower()

    # Score based on how many phases the model completed
    phases_done = sum(
        [
            searched,  # Phase 1: searched for the report
            read_2025,  # Phase 2: read the 2025 report
            read_2024,  # Phase 3: read 2024 for comparison
            emailed,  # Phase 5: sent email
        ]
    )

    # Check for key content recall
    mentions_market_share = (
        bool(re.search(r"\b35\s*(?:%|percent)", answer))
        and bool(re.search(r"\b32\s*(?:%|percent)", answer))
    ) or (
        "market share" in answer
        and any(
            phrase in answer for phrase in ("3 percentage points", "grew by 3", "increased by 3")
        )
    )
    mentions_risk = any(w in answer for w in ("betacorp", "platform", "q4", "risk"))

    if phases_done >= 4 and (mentions_market_share or mentions_risk):
        return _pass(f"Completed all {phases_done} tool phases and recalled prior context.")
    if phases_done >= 3:
        return _partial(f"Completed {phases_done}/4 tool phases — good state tracking.")
    if phases_done >= 2:
        return _partial(f"Completed {phases_done}/4 tool phases — partial state tracking.")
    if searched or read_2025:
        return _fail(f"Only completed {phases_done}/4 tool phases — lost context across turns.")
    return _fail("Did not engage with the multi-turn research workflow.")


SCENARIO = ScenarioDefinition(
    id="TC-46",
    title="Deep Multi-Turn Research (5 turns)",
    category=Category.I,
    user_message="Find the competitor analysis report.",
    description="5-turn research workflow: search→read→compare→summarize→email. Tests deep state tracking.",
    handle_tool_call=_tc46_handle,
    evaluate=_tc46_eval,
    # 5 user turns + tool-call rounds need up to 11 assistant turns for the
    # canonical reference path; 12 keeps it finite with one-turn headroom.
    max_turns_override=12,
    follow_up_messages=[
        "Read the 2025 one.",
        "What's our market share growth compared to last year? Check the 2024 report too.",
        "Summarize the key risks from both reports.",
        "Email that summary to my manager.",
    ],
    difficulty=4,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it completes all 5 turns: search→read 2025→read 2024→summarize risks→email manager.",
    "Fail if it loses context or skips phases across the 5-turn conversation.",
)
