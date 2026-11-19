"""Helpers shared by more than one scenario in this group.

Scoped to the group rather than global: several groups define helpers under the same name with deliberately different behaviour.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal
from typing import Any

from tool_eval_bench.domain.scenarios import (
    ScenarioState,
    ToolCallRecord,
)
from tool_eval_bench.evals.helpers import (
    UNIVERSAL_SIDE_EFFECT_TOOLS,
    _is_negated,
)
from tool_eval_bench.evals.helpers import (
    as_str as _as_str,
)
from tool_eval_bench.evals.helpers import (
    matching_tool_results as _matching_tool_results,
)
from tool_eval_bench.evals.helpers import (
    normalize as _normalize,
)
from tool_eval_bench.evals.helpers import (
    result_is_usable_if_present as _result_is_usable_if_present,
)

LARGE_SIDE_EFFECT_TOOLS: frozenset[str] = UNIVERSAL_SIDE_EFFECT_TOOLS | frozenset(
    {
        "update_customer",
        "create_ticket",
        "resolve_ticket",
        "create_return",
        "create_invoice",
        "process_payment",
        "transfer_funds",
        "deploy_service",
        "rollback_deploy",
        "scale_service",
        "restart_service",
        "submit_timesheet",
        "request_leave",
        "export_csv",
        "create_dashboard",
        "send_slack_message",
        "create_channel",
        "schedule_meeting",
        "post_announcement",
        "publish_page",
        "draft_blog_post",
        "upload_asset",
    }
)


def _is_engineering_department(value: Any) -> bool:
    """Return whether a department value requests Engineering itself.

    The scenario accepts natural variants such as "Engineering team", but a
    negated value like "not engineering" must not resolve to the canonical
    Engineering org chart merely because it contains the word "engineering".
    """
    department = _as_str(value)
    return bool(
        re.search(r"\bengineering\b", department, re.IGNORECASE)
        and not re.search(
            r"\b(?:not|no|without)\s+(?:the\s+)?engineering\b",
            department,
            re.IGNORECASE,
        )
    )


def _numeric_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = _as_str(value).strip().replace(",", "").replace("$", "")
    try:
        return float(text)
    except ValueError:
        return None


def _status_is_success(payload: Any, *, statuses: set[str] | None = None) -> bool:
    if not isinstance(payload, dict) or "error" in payload:
        return False
    accepted = statuses or {
        "accepted",
        "created",
        "delivered",
        "ok",
        "queued",
        "scheduled",
        "sent",
        "set",
        "success",
    }
    return _normalize(_as_str(payload.get("status"))) in accepted


def _result_matches_if_present(
    state: ScenarioState,
    call: ToolCallRecord,
    predicate: Callable[[Any], bool],
) -> bool:
    """Validate an explicit result when present, keeping synthetic traces valid."""
    exact_results = [result for result in state.tool_results if result.call_id == call.id]
    known_call_ids = {candidate.id for candidate in state.tool_calls}
    if exact_results:
        results = exact_results
    elif any(result.call_id in known_call_ids for result in state.tool_results):
        return False
    else:
        results = _matching_tool_results(state, call)
    if not results:
        return state.meta.get("trace_policy") != "complete"
    return _result_is_usable_if_present(state, call) and any(
        predicate(result.result) for result in results
    )


_STATED_NUMBER = re.compile(r"(?<![\d.])(?:([-\u2212])[$€£¥]?)?(\d+)(?:\.(\d+))?(?!\d|\.\d)")


def _parsed_stated_number(match: re.Match[str]) -> Decimal:
    sign = "-" if match.group(1) else ""
    fractional = f".{match.group(3)}" if match.group(3) is not None else ""
    return Decimal(f"{sign}{match.group(2)}{fractional}")


def _answer_affirms_rounded_value(answer: str, value: str) -> bool:
    """Whether ``answer`` asserts ``value`` at the precision it chose."""
    collapsed = re.sub(r"(?<=\d),(?=\d)", "", answer)
    expected = Decimal(value)
    for match in _STATED_NUMBER.finditer(collapsed):
        stated = _parsed_stated_number(match)
        quantum = Decimal(1).scaleb(-len(match.group(3) or ""))
        accepted = {
            expected.quantize(quantum, rounding=ROUND_HALF_UP),
            expected.quantize(quantum, rounding=ROUND_DOWN),
        }
        if stated in accepted and not _is_negated(
            collapsed[max(0, match.start() - 120) : match.start()]
        ):
            return True
    return False


def _answer_affirms_exact_value(answer: str, value: str) -> bool:
    """Whether ``answer`` asserts ``value`` with exact numeric equality."""
    collapsed = re.sub(r"(?<=\d),(?=\d)", "", answer)
    expected = Decimal(value)
    for match in _STATED_NUMBER.finditer(collapsed):
        if _parsed_stated_number(match) == expected and not _is_negated(
            collapsed[max(0, match.start() - 120) : match.start()]
        ):
            return True
    return False
