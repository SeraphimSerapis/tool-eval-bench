"""Optional answer audits, isolated from benchmark scoring and credentials."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from collections.abc import Awaitable, Callable, Iterable
from typing import Any, get_args
from urllib.parse import urlsplit

from tool_eval_bench.domain.decision import ChoiceAnswer, ChoiceQuestion, DecisionBackend
from tool_eval_bench.domain.scenarios import (
    AnswerAuditDefinition,
    AuditPhase,
    AuditTier,
    ScenarioDefinition,
    ScenarioResult,
    ScenarioState,
)
from tool_eval_bench.utils.urls import endpoint_identity

logger = logging.getLogger(__name__)

AUDIT_TIMEOUT_SECONDS = 10.0
# ponytail: conservative byte cap for an 8k context; use tokenizer-aware sizing
# if longer inputs are needed. Server context rejections remain unavailable audits.
MAX_AUDIT_REQUEST_BYTES = 6000

#: A judge set names the highest tier it includes: ``all`` also runs every
#: ``recommended`` audit.
DecisionJudgeSet = AuditTier
DECISION_JUDGE_SETS: tuple[DecisionJudgeSet, ...] = get_args(AuditTier)
DEFAULT_DECISION_JUDGE_SET: DecisionJudgeSet = "recommended"


def decision_judge_config(
    base_url: str | None,
    model: str | None,
    api_key: str | None = None,
    judge_set: str | None = None,
) -> dict[str, Any] | None:
    """Validate an independent connection without persisting its API key.

    Without ``judge_set`` a configured judge runs the ``recommended`` set.
    """
    if base_url is None:
        if model is not None or api_key is not None or judge_set is not None:
            raise ValueError("A decision judge requires a base URL and model")
        return None
    if not model or not model.strip():
        raise ValueError("A decision judge requires a model")
    if judge_set is None:
        judge_set = DEFAULT_DECISION_JUDGE_SET
    elif judge_set not in DECISION_JUDGE_SETS:
        raise ValueError(f"Decision judge set must be one of: {', '.join(DECISION_JUDGE_SETS)}")
    try:
        parsed = urlsplit(base_url)
        port = parsed.port
        valid = (
            parsed.scheme in {"http", "https"}
            and bool(parsed.hostname)
            and parsed.username is None
            and parsed.password is None
            and not parsed.query
            and not parsed.fragment
            and not any(c.isspace() for c in base_url)
            and (port is None or port > 0)
        )
    except ValueError:
        valid = False
    if not valid:
        raise ValueError(
            "Decision judge URL must be HTTP(S), without credentials, query, or fragment"
        )
    return {
        "mode": "audit",
        "base_url": base_url,
        "endpoint_id": endpoint_identity(base_url),
        "model": model.strip(),
        "set": judge_set,
    }


def selected_audit(
    scenario: ScenarioDefinition, judge_set: DecisionJudgeSet
) -> AnswerAuditDefinition | None:
    """The scenario's audit if ``judge_set`` includes its tier; never for held-out packs."""
    definition = scenario.answer_audit
    if definition is None or scenario.held_out:
        return None
    if judge_set == "recommended" and definition.tier != "recommended":
        return None
    return definition


def with_selected_checks(
    config: dict[str, Any], scenarios: Iterable[ScenarioDefinition]
) -> dict[str, Any]:
    """Record which checks the run's scenarios select, so resume can compare them.

    A question version bump changes a ``check_id`` and therefore this list,
    which the resume compatibility check refuses like a URL or model change.
    """
    checks = sorted(
        definition.check_id
        for scenario in scenarios
        if (definition := selected_audit(scenario, config["set"])) is not None
    )
    return {**config, "checks": checks}


def audit_metadata(definition: AnswerAuditDefinition) -> dict[str, Any]:
    """Everything a later resume needs to judge saved evidence without current code."""
    return {
        "check_id": definition.check_id,
        "question": definition.question.to_wire(),
        "question_sha256": definition.question_sha256,
        "evidence": definition.evidence,
        "label": definition.label,
    }


def _has_text(messages: Iterable[str]) -> bool:
    return any(message.strip() for message in messages)


def capture_decision_audit(
    scenario: ScenarioDefinition,
    state: ScenarioState,
    result: ScenarioResult,
    *,
    judge_set: DecisionJudgeSet,
) -> None:
    """Keep public assistant text, never reasoning or unrelated pack evidence."""
    definition = selected_audit(scenario, judge_set)
    if definition is None:
        return
    if definition.evidence == "final_answer":
        messages = [state.final_answer] if state.final_answer else []
    else:
        messages = list(state.assistant_messages)
        if state.final_answer and (not messages or messages[-1] != state.final_answer):
            messages.append(state.final_answer)
    audit: dict[str, Any] = {
        "status": "pending",
        **audit_metadata(definition),
        "input": json.dumps({"assistant_messages": messages}, ensure_ascii=False),
    }
    result.decision_audit = audit
    try:
        audit["deterministic_choice"] = definition.deterministic_choice(state)
        if not _has_text(messages):
            audit.update(status="unavailable", error_type="NoAssistantMessages")
    except Exception as exc:  # noqa: BLE001 — an audit cannot fail the benchmark
        audit.update(status="unavailable", error_type=type(exc).__name__)


def refresh_saved_audit(audit: dict[str, Any], definition: AnswerAuditDefinition) -> dict[str, Any]:
    """Reuse a saved audit only if the current question produced it.

    This is defence in depth. Resume already refuses a changed set or
    ``check_id``, and the pinned hashes keep wording from changing without a
    version bump, so a mismatch here means one of those guards was bypassed.
    The saved evidence is then judged again with the current question,
    without rerunning the benchmark model. Evidence captured for a different
    scope, or a deterministic value the current options cannot express, is
    reported unavailable rather than compared.
    """
    if (audit.get("check_id"), audit.get("question_sha256")) == (
        definition.check_id,
        definition.question_sha256,
    ):
        return audit
    refreshed: dict[str, Any] = {"status": "pending", **audit_metadata(definition)}
    refreshed.update({key: audit[key] for key in ("input", "deterministic_choice") if key in audit})
    if "input" not in audit:
        refreshed.update(status="unavailable", error_type="NoEvaluationEvidence")
    elif "deterministic_choice" not in audit:
        # The deterministic check raised at capture; keep what it raised.
        refreshed.update(
            status="unavailable", error_type=audit.get("error_type", "StaleAuditEvidence")
        )
    elif (
        audit.get("evidence", "messages") != definition.evidence
        or audit["deterministic_choice"] not in definition.question.options
    ):
        refreshed.update(status="unavailable", error_type="StaleAuditEvidence")
    elif not _has_text(json.loads(audit["input"]).get("assistant_messages", [])):
        refreshed.update(status="unavailable", error_type="NoAssistantMessages")
    return refreshed


async def run_decision_audit(
    adapter: DecisionBackend,
    audit: dict[str, Any],
    *,
    config: dict[str, Any],
    api_key: str | None = None,
    on_progress: Callable[[AuditPhase], Awaitable[None]] | None = None,
) -> None:
    """Complete captured evidence in place; errors never become scenario failures."""

    async def notify(phase: AuditPhase) -> None:
        if on_progress is not None:
            try:
                await on_progress(phase)
            except Exception as exc:  # noqa: BLE001 — progress cannot fail an audit
                logger.warning("Decision audit progress callback failed: %s", type(exc).__name__)

    audit.update(model=config["model"], base_url=config["base_url"])
    audit.setdefault("elapsed_ms", 0.0)
    if audit["status"] != "pending":
        if audit.get("request_started") or audit["status"] in {"completed", "abstained"}:
            await notify("reused")
        return
    request_started = False
    started = time.perf_counter()
    try:
        wire = audit["question"]
        question = ChoiceQuestion(instructions=wire["instructions"], options=wire["criteria"])
        payload = {
            "model": config["model"],
            "state": audit["input"],
            "questions": {audit["check_id"]: wire},
        }
        request_bytes = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        audit["request_bytes"] = request_bytes
        if request_bytes > MAX_AUDIT_REQUEST_BYTES:
            audit.update(status="unavailable", error_type="InputTooLarge")
            return
        request_started = True
        audit["request_started"] = True
        await notify("started")
        decision = await asyncio.wait_for(
            adapter.decide(
                model=config["model"],
                state=audit["input"],
                questions={audit["check_id"]: question},
                timeout_seconds=AUDIT_TIMEOUT_SECONDS,
                api_key=api_key,
                base_url=config["base_url"],
            ),
            timeout=AUDIT_TIMEOUT_SECONDS,
        )
        answer = decision.answers[audit["check_id"]]
        if not isinstance(answer, ChoiceAnswer):
            raise ValueError("Expected a choice answer")
        probabilities = dict(answer.probabilities)
        if (
            set(probabilities) != set(question.options)
            or answer.choice not in probabilities
            or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values())
            or not math.isclose(sum(probabilities.values()), 1.0, abs_tol=0.001)
            or not math.isclose(
                probabilities[answer.choice], max(probabilities.values()), abs_tol=1e-9
            )
        ):
            raise ValueError("Invalid choice probabilities")
        top = max(probabilities.values())
        tied = sum(math.isclose(p, top, abs_tol=1e-9) for p in probabilities.values()) > 1
        abstained = answer.choice == "unclear" or tied
        audit.update(
            status="abstained" if abstained else "completed",
            choice=answer.choice,
            probabilities=probabilities,
            input_tokens=decision.input_tokens,
            disagreement=None if abstained else answer.choice != audit["deterministic_choice"],
        )
    except Exception as exc:  # noqa: BLE001 — unavailable is not a model failure
        # Exception text may contain credentials or server-echoed input.
        audit.update(status="unavailable", error_type=type(exc).__name__)
    finally:
        audit["elapsed_ms"] = (time.perf_counter() - started) * 1000
        if request_started and audit["status"] != "pending":
            await notify("completed")
