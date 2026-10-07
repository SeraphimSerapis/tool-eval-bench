"""Optional answer audits, isolated from benchmark scoring and credentials."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urlsplit

from tool_eval_bench.domain.decision import ChoiceAnswer, ChoiceQuestion, DecisionBackend
from tool_eval_bench.domain.scenarios import (
    AuditPhase,
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


def decision_judge_config(
    base_url: str | None, model: str | None, api_key: str | None = None
) -> dict[str, Any] | None:
    """Validate an independent connection without persisting its API key."""
    if base_url is None:
        if model is not None or api_key is not None:
            raise ValueError("A decision judge requires a base URL and model")
        return None
    if not model or not model.strip():
        raise ValueError("A decision judge requires a model")
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
    }


def capture_decision_audit(
    scenario: ScenarioDefinition, state: ScenarioState, result: ScenarioResult
) -> None:
    """Keep public assistant text, never reasoning or unrelated pack evidence."""
    definition = scenario.answer_audit
    if definition is None or scenario.held_out:
        return
    messages = list(state.assistant_messages)
    if state.final_answer and (not messages or messages[-1] != state.final_answer):
        messages.append(state.final_answer)
    audit: dict[str, Any] = {
        "status": "pending",
        "check_id": definition.check_id,
        "question": definition.question.to_wire(),
        "input": json.dumps({"assistant_messages": messages}, ensure_ascii=False),
    }
    result.decision_audit = audit
    try:
        audit["deterministic_choice"] = definition.deterministic_choice(state)
        if not any(message.strip() for message in messages):
            audit.update(status="unavailable", error_type="NoAssistantMessages")
    except Exception as exc:  # noqa: BLE001 — an audit cannot fail the benchmark
        audit.update(status="unavailable", error_type=type(exc).__name__)


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
