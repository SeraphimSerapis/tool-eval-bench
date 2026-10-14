"""Live run SSE hub and cross-process SQLite checkpoint tailing."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from tool_eval_bench.application.run_queries import run_progress
from tool_eval_bench.cli.held_out import HELD_OUT
from tool_eval_bench.domain.redaction import redact_urls
from tool_eval_bench.domain.scenarios import (
    AuditPhase,
    ScenarioDefinition,
    ScenarioResult,
)
from tool_eval_bench.web.sanitize import (
    sanitize_result,
    sanitize_run_for_web,
    stored_held_out_ids,
)

TERMINAL_EVENTS = frozenset({"run_complete", "run_interrupted", "stream_error", "stream_idle"})

#: Finished sessions kept for late subscribers; older ones are dropped.
MAX_FINISHED_SESSIONS = 20
#: Seconds between SSE comment frames, so proxies keep the stream open and a
#: dropped client is noticed even when the run is quiet.
HEARTBEAT_SECONDS = 15.0
#: A CLI run that writes no checkpoint for this long probably died without
#: marking itself interrupted; the tail stops instead of polling forever.
IDLE_TIMEOUT_SECONDS = 15 * 60.0

_HEARTBEAT_FRAME = ": keep-alive\n\n"
_PER_SCENARIO_EVENTS = frozenset({"scenario_start", "scenario_turn"})


def format_sse(event_type: str, data: Mapping[str, Any]) -> str:
    """Format a single Server-Sent Events frame."""
    payload = json.dumps(dict(data), default=str)
    return f"event: {event_type}\ndata: {payload}\n\n"


@dataclass
class LiveRunSession:
    """In-memory state and subscriber queues for a web-managed benchmark run."""

    run_id: str
    model: str
    backend: str
    total: int
    status: str = "running"
    held_out: frozenset[str] = field(default_factory=frozenset)
    history: list[dict[str, Any]] = field(default_factory=list)
    subscribers: set[asyncio.Queue[dict[str, Any]]] = field(default_factory=set)
    task: asyncio.Task[Any] | None = None

    @property
    def is_live(self) -> bool:
        """Whether this server is still executing the run."""
        return self.task is not None and not self.task.done()

    def publish(self, event: dict[str, Any]) -> None:
        # Once a scenario has a result, its start and turn events only matter
        # to a subscriber that was watching at the time. Dropping them keeps
        # the replay history to one result per scenario plus the active one.
        if event.get("event") == "scenario_result":
            sid = event.get("scenario_id")
            self.history = [
                past
                for past in self.history
                if not (
                    past.get("event") in _PER_SCENARIO_EVENTS and past.get("scenario_id") == sid
                )
            ]
        self.history.append(event)
        for queue in list(self.subscribers):
            queue.put_nowait(event)


class LiveRunHub:
    """Registry of active and recently finished web-launched benchmark runs."""

    def __init__(self, *, max_finished: int = MAX_FINISHED_SESSIONS) -> None:
        self._sessions: dict[str, LiveRunSession] = {}
        self._max_finished = max_finished

    def get(self, run_id: str) -> LiveRunSession | None:
        return self._sessions.get(run_id)

    def active_sessions(self) -> list[LiveRunSession]:
        return [s for s in self._sessions.values() if s.status == "running"]

    def register(
        self,
        run_id: str,
        *,
        model: str,
        backend: str,
        total: int,
        held_out: frozenset[str] = frozenset(),
        preserved: list[dict[str, Any]] | None = None,
    ) -> LiveRunSession:
        """Start a session; ``preserved`` holds sanitized results a resume keeps.

        Replaying preserved results lets a viewer of a resumed run see its true
        progress rather than only the scenarios still to run.
        """
        session = LiveRunSession(
            run_id=run_id,
            model=model,
            backend=backend,
            total=total,
            held_out=held_out,
        )
        # Re-registering (a resume) moves the run to the newest position.
        self._sessions.pop(run_id, None)
        self._sessions[run_id] = session
        self._evict_finished()
        session.publish(
            {
                "event": "run_started",
                "run_id": run_id,
                "model": model,
                "backend": backend,
                "total": total,
                "status": "running",
                "managed": True,
            }
        )
        for index, result in enumerate(preserved or []):
            session.publish(
                {
                    "event": "scenario_result",
                    "run_id": run_id,
                    "scenario_id": result.get("scenario_id"),
                    "index": index,
                    "total": total,
                    "preserved": True,
                    "result": result,
                }
            )
        return session

    def _evict_finished(self) -> None:
        finished = [rid for rid, s in self._sessions.items() if s.status != "running"]
        for rid in finished[: max(0, len(finished) - self._max_finished)]:
            del self._sessions[rid]

    def make_callbacks(self, session: LiveRunSession) -> dict[str, Any]:
        """Build orchestrator callbacks that publish sanitized live events."""

        async def on_scenario_start(scenario: ScenarioDefinition, idx: int, total: int) -> None:
            is_held_out = scenario.held_out or scenario.id in session.held_out
            session.publish(
                {
                    "event": "scenario_start",
                    "run_id": session.run_id,
                    "scenario_id": scenario.id,
                    "title": HELD_OUT if is_held_out else scenario.title,
                    "category": "held_out" if is_held_out else scenario.category.value,
                    "held_out": is_held_out,
                    "index": idx,
                    # The session's total, not the orchestrator's: a resume
                    # executes only the remaining subset.
                    "total": session.total,
                }
            )

        async def on_scenario_turn(
            scenario: ScenarioDefinition, turn: int, turn_payload: dict[str, Any]
        ) -> None:
            is_held_out = scenario.held_out or scenario.id in session.held_out
            session.publish(
                {
                    "event": "scenario_turn",
                    "run_id": session.run_id,
                    "scenario_id": scenario.id,
                    "held_out": is_held_out,
                    "turn": turn,
                    "max_turns": turn_payload.get("max_turns"),
                    "latency_ms": turn_payload.get("latency_ms"),
                    "ttft_ms": turn_payload.get("ttft_ms"),
                    "content": "" if is_held_out else (turn_payload.get("content") or ""),
                    "tool_calls": [] if is_held_out else list(turn_payload.get("tool_calls") or []),
                    "tool_results": []
                    if is_held_out
                    else list(turn_payload.get("tool_results") or []),
                }
            )

        async def on_scenario_result(
            scenario: ScenarioDefinition, result: ScenarioResult, idx: int, total: int
        ) -> None:
            held = session.held_out | ({scenario.id} if scenario.held_out else frozenset())
            session.publish(
                {
                    "event": "scenario_result",
                    "run_id": session.run_id,
                    "scenario_id": scenario.id,
                    "index": idx,
                    "total": session.total,
                    "result": sanitize_result(result.to_dict(), held),
                }
            )

        async def on_scenario_audit(
            scenario: ScenarioDefinition, result: ScenarioResult, phase: AuditPhase
        ) -> None:
            if not result.was_decision_audited:
                return
            is_held_out = scenario.held_out or scenario.id in session.held_out
            audit = result.decision_audit or {}
            session.publish(
                {
                    "event": "decision_audit",
                    "run_id": session.run_id,
                    "scenario_id": scenario.id,
                    "phase": phase,
                    "model": audit.get("model"),
                    "status": audit.get("status"),
                    "choice": None if is_held_out else audit.get("choice"),
                    "disagreement": audit.get("disagreement"),
                    "probabilities": None if is_held_out else audit.get("probabilities"),
                }
            )

        return {
            "on_scenario_start": on_scenario_start,
            "on_scenario_turn": on_scenario_turn,
            "on_scenario_result": on_scenario_result,
            "on_scenario_audit": on_scenario_audit,
        }

    def mark_complete(self, run_id: str, run_data: Mapping[str, Any]) -> None:
        session = self._sessions.get(run_id)
        if session is None:
            return
        session.status = "completed"
        sanitized = sanitize_run_for_web(run_data, include_traces=False)
        session.publish(
            {
                "event": "run_complete",
                "run_id": run_id,
                "final_score": sanitized.get("final_score"),
                "rating": sanitized.get("rating"),
                "passes": sanitized.get("passes"),
                "partials": sanitized.get("partials"),
                "fails": sanitized.get("fails"),
            }
        )
        self._evict_finished()

    def mark_interrupted(self, run_id: str, reason: str = "interrupted") -> None:
        session = self._sessions.get(run_id)
        if session is None:
            return
        session.status = "interrupted"
        session.publish(
            {
                "event": "run_interrupted",
                "run_id": run_id,
                # Exception text from httpx quotes the full request URL.
                "reason": redact_urls(reason),
            }
        )
        self._evict_finished()


async def stream_run_events(
    hub: LiveRunHub,
    run_id: str,
    *,
    poll_interval: float = 1.0,
    heartbeat_seconds: float = HEARTBEAT_SECONDS,
    idle_timeout: float = IDLE_TIMEOUT_SECONDS,
    max_polls: int | None = None,
) -> AsyncIterator[str]:
    """Yield SSE frames for *run_id* from the in-memory hub or SQLite checkpoints."""
    session = hub.get(run_id)
    if session is not None:
        async for frame in _stream_session(session, heartbeat_seconds):
            yield frame
        return
    async for frame in _tail_stored_run(
        run_id,
        poll_interval=poll_interval,
        heartbeat_seconds=heartbeat_seconds,
        idle_timeout=idle_timeout,
        max_polls=max_polls,
    ):
        yield frame


async def _stream_session(session: LiveRunSession, heartbeat_seconds: float) -> AsyncIterator[str]:
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    session.subscribers.add(queue)
    try:
        for past in list(session.history):
            ev_type = str(past.get("event", "message"))
            yield format_sse(ev_type, past)
            if ev_type in TERMINAL_EVENTS:
                return
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=heartbeat_seconds)
            except asyncio.TimeoutError:
                yield _HEARTBEAT_FRAME
                continue
            ev_type = str(event.get("event", "message"))
            yield format_sse(ev_type, event)
            if ev_type in TERMINAL_EVENTS:
                return
    finally:
        session.subscribers.discard(queue)


async def _tail_stored_run(
    run_id: str,
    *,
    poll_interval: float,
    heartbeat_seconds: float,
    idle_timeout: float,
    max_polls: int | None,
) -> AsyncIterator[str]:
    """Follow a run another process is writing, through its SQLite checkpoints."""
    seen_scenarios: set[str] = set()
    sent_header = False
    polls = 0
    last_progress = last_frame = time.monotonic()
    while True:
        # SQLite reads block; keep them off the loop that runs web benchmarks.
        progress = await asyncio.to_thread(run_progress, run_id)
        if progress is None:
            yield format_sse(
                "stream_error",
                {"event": "stream_error", "error": f"Run '{run_id}' not found"},
            )
            return

        run, checkpoints = progress
        held_out = stored_held_out_ids(run)
        config = run.get("config") or {}
        total = int(config.get("scenario_count") or 0)
        status = str(run.get("status") or "completed")

        if not sent_header:
            sent_header = True
            last_frame = time.monotonic()
            yield format_sse(
                "run_started",
                {
                    "event": "run_started",
                    "run_id": run_id,
                    "model": run.get("model") or config.get("model", "unknown"),
                    "backend": config.get("backend", "unknown"),
                    "total": total,
                    "status": status,
                    # Owned by another process: this server cannot stop it.
                    "managed": False,
                },
            )

        for cp in checkpoints:
            sid = str(cp.get("scenario_id") or "")
            if not sid or sid in seen_scenarios:
                continue
            seen_scenarios.add(sid)
            last_progress = last_frame = time.monotonic()
            yield format_sse(
                "scenario_result",
                {
                    "event": "scenario_result",
                    "run_id": run_id,
                    "scenario_id": sid,
                    "index": len(seen_scenarios) - 1,
                    "total": total,
                    "result": sanitize_result(cp, held_out),
                },
            )

        if status == "completed":
            sanitized_run = sanitize_run_for_web(run, include_traces=False)
            results = sanitized_run.get("scores", {}).get("scenario_results", [])
            for sr in results:
                sid = str(sr.get("scenario_id") or "")
                if sid and sid not in seen_scenarios:
                    seen_scenarios.add(sid)
                    yield format_sse(
                        "scenario_result",
                        {
                            "event": "scenario_result",
                            "run_id": run_id,
                            "scenario_id": sid,
                            "index": len(seen_scenarios) - 1,
                            "total": total or len(results),
                            "result": sr,
                        },
                    )
            yield format_sse(
                "run_complete",
                {
                    "event": "run_complete",
                    "run_id": run_id,
                    "final_score": sanitized_run.get("final_score"),
                    "rating": sanitized_run.get("rating"),
                    "passes": sanitized_run.get("passes"),
                    "partials": sanitized_run.get("partials"),
                    "fails": sanitized_run.get("fails"),
                },
            )
            return

        if status in {"interrupted", "failed"}:
            yield format_sse(
                "run_interrupted",
                {"event": "run_interrupted", "run_id": run_id, "reason": status},
            )
            return

        now = time.monotonic()
        if now - last_progress >= idle_timeout:
            minutes = round(idle_timeout / 60)
            yield format_sse(
                "stream_idle",
                {
                    "event": "stream_idle",
                    "run_id": run_id,
                    "message": (
                        f"No scenario finished in {minutes} min. The run's process may have "
                        "exited without marking it interrupted. Reconnect to keep watching."
                    ),
                },
            )
            return
        if now - last_frame >= heartbeat_seconds:
            last_frame = now
            yield _HEARTBEAT_FRAME

        polls += 1
        if max_polls is not None and polls >= max_polls:
            return
        await asyncio.sleep(poll_interval)
