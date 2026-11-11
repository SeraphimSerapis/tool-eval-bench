"""Regression tests for CLI, Web UI/server, and public Python API fixes."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
from collections.abc import Callable
from dataclasses import replace
from io import StringIO
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from rich.console import Console

from tests.web_support import AUTH_HEADERS, TEST_TOKEN, web_client
from tool_eval_bench.api import run_benchmark
from tool_eval_bench.application.run_queries import recent_runs
from tool_eval_bench.cli import (
    dispatch,
    leaderboard,
    model_probe,
    plugin_datasets,
    plugin_lifecycle,
    pressure,
    probe,
    spec_bench,
)
from tool_eval_bench.cli.display import BenchmarkDisplay, print_final_report
from tool_eval_bench.cli.headless import HeadlessConsole
from tool_eval_bench.cli.held_out import HELD_OUT, held_out_ids, redact_warnings
from tool_eval_bench.cli.helpers import parse_sweep_range, safety_gate_failed
from tool_eval_bench.cli.history import compare_runs, print_diff
from tool_eval_bench.cli.legacy_parser import make_parser
from tool_eval_bench.cli.parser import translate_argv
from tool_eval_bench.cli.plugin_progress import truncate
from tool_eval_bench.domain.adapters import ChatCompletionResult
from tool_eval_bench.domain.models import RunContext
from tool_eval_bench.domain.scenarios import (
    CapabilityScore,
    Category,
    CategoryScore,
    FailureKind,
    ModelScoreSummary,
    ScenarioDefinition,
    ScenarioEvaluation,
    ScenarioResult,
    ScenarioStatus,
)
from tool_eval_bench.evals.packs import ScenarioPack
from tool_eval_bench.evals.scenarios import ALL_SCENARIOS
from tool_eval_bench.runner.speculative import SpecDecodeSample
from tool_eval_bench.runner.throughput import ThroughputSample
from tool_eval_bench.storage.db import RunRepository
from tool_eval_bench.utils import metadata
from tool_eval_bench.web import create_app
from tool_eval_bench.web.app import _sanitize_csv_cell as _web_sanitize_csv_cell
from tool_eval_bench.web.launcher import (
    RunConflict,
    _resolve_launcher_scenarios,
    launch_web_run,
    probe_endpoint_models,
    resume_web_run,
    stop_web_run,
)
from tool_eval_bench.web.live import MAX_SUBSCRIBER_QUEUE, LiveRunHub
from tool_eval_bench.web.sanitize import sanitize_mapping
from tool_eval_bench.web.security import cookie_name

_REAL_HTTPX_CLIENT = httpx.AsyncClient
_TC01 = next(s for s in ALL_SCENARIOS if s.id == "TC-01")
_TC02 = next(s for s in ALL_SCENARIOS if s.id == "TC-02")


@pytest.fixture()
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    db_path = str(tmp_path / "benchmarks.sqlite")
    out_dir = str(tmp_path / "runs")
    monkeypatch.setattr(
        "tool_eval_bench.storage.db._default_db_path",
        lambda: db_path,
    )
    monkeypatch.delenv("TOOL_EVAL_BASE_URL", raising=False)
    monkeypatch.delenv("TOOL_EVAL_MODEL", raising=False)
    monkeypatch.delenv("TOOL_EVAL_API_KEY", raising=False)
    return {"db_path": db_path, "output_dir": out_dir}


async def _wait_for(predicate: Callable[[], bool], *, timeout_s: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while not predicate():
        if loop.time() >= deadline:
            raise TimeoutError("Timed out waiting for condition")
        await asyncio.sleep(0.01)


def _held_out_scenario(scenario_id: str = "HO-99") -> ScenarioDefinition:
    return ScenarioDefinition(
        id=scenario_id,
        title="Secret Held-Out Canary Title",
        category=Category.A,
        user_message="Confidential canary prompt",
        description="Confidential expected behavior",
        handle_tool_call=lambda _state, _call: {"ok": True},
        evaluate=lambda _state: ScenarioEvaluation(
            ScenarioStatus.FAIL,
            0,
            "Confidential grading summary",
            safety_violation="confidential violation",
        ),
        held_out=True,
    )


def _scenario_pack(
    *scenarios: ScenarioDefinition,
    name: str = "secret-pack",
    tmp_path: Path | None = None,
) -> ScenarioPack:
    return ScenarioPack(
        name=name,
        path=tmp_path or Path("/nonexistent/pack"),
        scenarios=tuple(scenarios),
        content_hash="a" * 64,
    )


class _FastAdapter:
    def __init__(self, delay_s: float = 0.0) -> None:
        self.delay_s = delay_s
        self.calls = 0

    async def chat_completion(self, **_kwargs: Any) -> ChatCompletionResult:
        self.calls += 1
        if self.delay_s > 0:
            await asyncio.sleep(self.delay_s)
        return ChatCompletionResult(
            content="Done.",
            tool_calls=[],
            ttft_ms=10.0,
            prompt_tokens=32,
            completion_tokens=8,
        )

    async def aclose(self) -> None:
        return None


class _StallAfterFirstAdapter(_FastAdapter):
    async def chat_completion(self, **_kwargs: Any) -> ChatCompletionResult:
        self.calls += 1
        if self.calls > 1:
            await asyncio.Event().wait()
        return ChatCompletionResult(
            content="Done.",
            tool_calls=[],
            ttft_ms=10.0,
            prompt_tokens=32,
            completion_tokens=8,
        )


def _patch_web_adapter(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    monkeypatch.setattr(
        "tool_eval_bench.application.service.BenchmarkService._adapter_for",
        lambda _self, *_a, **_k: adapter,
    )


# ---------------------------------------------------------------------------
# 1. Web Security: DNS Rebinding (Host), Origin, HTTPS Cookie, & Sec-Fetch-Site
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_gate_blocks_dns_rebinding_hosts(isolated_env: dict[str, str]) -> None:
    app = create_app(token=TEST_TOKEN)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://attacker.example:8765"
    ) as rebound:
        assert (await rebound.get("/static/styles.css")).status_code == 403
        assert (await rebound.get("/", params={"token": TEST_TOKEN})).status_code == 403
        assert (await rebound.get("/api/runs")).status_code == 401
        authed_rebound = await rebound.get("/api/runs", headers=AUTH_HEADERS)
        assert authed_rebound.status_code == 403
        assert "Invalid Host header" in authed_rebound.json()["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "good_base_url",
    [
        "http://127.0.0.1:8765",
        "http://localhost:8765",
        "http://localhost.:8765",
        "http://[::1]:8765",
    ],
)
async def test_session_gate_accepts_loopback_hosts(
    isolated_env: dict[str, str], good_base_url: str
) -> None:
    app = create_app(token=TEST_TOKEN)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url=good_base_url, headers=AUTH_HEADERS
    ) as client:
        assert (await client.get("/static/styles.css")).status_code == 200
        assert (await client.get("/api/runs")).status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_host",
    [
        "",
        ".",
        "..",
        "127.0.0.1.",
        "localhost..",
        "127.0.0.1:0",
        "127.0.0.1:70000",
        "127.0.0.1:notaport",
        "user@127.0.0.1:8765",
        "[::1",
        "evil.localhost:8765",
    ],
)
async def test_session_gate_rejects_malformed_or_non_loopback_host_headers(
    isolated_env: dict[str, str], bad_host: str
) -> None:
    app = create_app(token=TEST_TOKEN)
    async with web_client(app) as client:
        resp = await client.get("/api/runs", headers={"Host": bad_host})
        assert resp.status_code == 403


@pytest.mark.asyncio
async def test_session_gate_rejects_duplicate_host_malformed_origin_and_sets_secure_cookie(
    isolated_env: dict[str, str],
) -> None:
    app = create_app(token=TEST_TOKEN)
    transport = httpx.ASGITransport(app=app)

    # Duplicate Host headers are rejected as 403 rather than trusting the first.
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8765") as client:
        dup_resp = await client.get(
            "/api/runs",
            headers=[
                ("Authorization", f"Bearer {TEST_TOKEN}"),
                ("Host", "127.0.0.1:8765"),
                ("Host", "attacker.example:8765"),
            ],
        )
        assert dup_resp.status_code == 403

        # Malformed IPv6 Origin returns 403 rather than raising ValueError (500).
        bad_origin = await client.post(
            "/api/runs/delete",
            json={"run_ids": ["r1"]},
            headers={**AUTH_HEADERS, "Origin": "http://[::1"},
        )
        assert bad_origin.status_code == 403

    # HTTPS login sets the Secure attribute on the session cookie.
    async with httpx.AsyncClient(
        transport=transport, base_url="https://127.0.0.1:8765"
    ) as https_client:
        login = await https_client.get("/", params={"token": TEST_TOKEN})
        assert login.status_code == 303
        assert "secure" in login.headers["set-cookie"].lower()


@pytest.mark.asyncio
async def test_session_gate_custom_and_wildcard_allowed_hosts(isolated_env: dict[str, str]) -> None:
    lan_app = create_app(token=TEST_TOKEN, allowed_hosts=("127.0.0.1", "192.168.1.50"))
    transport = httpx.ASGITransport(app=lan_app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://192.168.1.50:8765", headers=AUTH_HEADERS
    ) as lan_client:
        assert (await lan_client.get("/api/runs")).status_code == 200

    async with httpx.AsyncClient(
        transport=transport, base_url="http://192.168.1.99:8765", headers=AUTH_HEADERS
    ) as other_client:
        assert (await other_client.get("/api/runs")).status_code == 403

    wildcard_app = create_app(token=TEST_TOKEN, allowed_hosts=("*",))
    wildcard_transport = httpx.ASGITransport(app=wildcard_app)
    async with httpx.AsyncClient(
        transport=wildcard_transport,
        base_url="http://bench-box.internal:8765",
        headers=AUTH_HEADERS,
    ) as wild_client:
        assert (await wild_client.get("/api/runs")).status_code == 200


@pytest.mark.asyncio
async def test_session_gate_rejects_same_site_fetch_metadata_across_ports(
    isolated_env: dict[str, str],
) -> None:
    """Cookies are port-agnostic, so a sibling localhost port sends Sec-Fetch-Site: same-site."""
    app = create_app(token=TEST_TOKEN)
    async with web_client(app, authenticated=False) as client:
        login = await client.get("/", params={"token": TEST_TOKEN})
        assert login.status_code == 303
        assert cookie_name(TEST_TOKEN) in client.cookies

        ok_get = await client.get("/api/runs", headers={"Sec-Fetch-Site": "same-origin"})
        assert ok_get.status_code == 200
        ok_nav = await client.get("/api/runs", headers={"Sec-Fetch-Site": "none"})
        assert ok_nav.status_code == 200

        cross_port_get = await client.get("/api/runs", headers={"Sec-Fetch-Site": "same-site"})
        assert cross_port_get.status_code == 403

        cross_port_post = await client.post(
            "/api/runs/delete",
            json={"run_ids": ["r1"]},
            headers={"Sec-Fetch-Site": "same-site"},
        )
        assert cross_port_post.status_code == 403


# ---------------------------------------------------------------------------
# 2. Web Held-Out Redaction & Recursive Sanitization
# ---------------------------------------------------------------------------


def test_sanitize_mapping_scrubs_secrets_nested_in_lists() -> None:
    dirty = {
        "model": "qwen",
        "providers": [
            {"name": "primary", "api_key": "sk-secret-1", "max_tokens": 256},
            [{"Authorization": "Bearer secret-2", "prompt_tokens": 64}],
        ],
    }
    cleaned = sanitize_mapping(dirty)
    assert cleaned == {
        "model": "qwen",
        "providers": [
            {"name": "primary", "max_tokens": 256},
            [{"prompt_tokens": 64}],
        ],
    }


@pytest.mark.asyncio
async def test_web_redacts_non_public_scenarios_even_without_pack_attestation(
    isolated_env: dict[str, str],
) -> None:
    run_id = "2026-11-11T00-00-00.000000Z_nopack01"
    with RunRepository(db_path=isolated_env["db_path"]) as repo:
        repo.upsert_scenario_run(
            {
                "run_id": run_id,
                "run_type": "tool_eval",
                "status": "interrupted",
                "config": {
                    "model": "qwen",
                    "backend": "vllm",
                    "base_url": "http://***:8000",
                    "scenario_ids": ["TC-01", "HO-99"],
                    # Deliberately omit scenario_packs attestation.
                },
                "metadata": {"model": "qwen"},
                "scores": {
                    "final_score": 50,
                    "scenario_results": [
                        {
                            "scenario_id": "TC-01",
                            "status": "pass",
                            "points": 2,
                            "summary": "Public summary",
                            "raw_log": "public trace",
                        },
                        {
                            "scenario_id": "HO-99",
                            "status": "fail",
                            "points": 0,
                            "summary": "Secret held-out summary",
                            "raw_log": "Secret held-out prompt and trace",
                        },
                    ],
                    "safety_warnings": [
                        "HO-99 (Secret Title): leaked detail",
                        "HO-99: leaked titleless detail",
                    ],
                },
            }
        )
        repo.checkpoint_scenario_result(
            run_id,
            {
                "scenario_id": "TC-01",
                "status": "pass",
                "points": 2,
                "summary": "Public summary",
                "raw_log": "public trace",
                "tool_calls_made": ["get_weather"],
            },
        )
        repo.checkpoint_scenario_result(
            run_id,
            {
                "scenario_id": "HO-99",
                "status": "fail",
                "points": 0,
                "summary": "Secret held-out summary",
                "raw_log": "Secret held-out prompt and trace",
                "tool_calls_made": ["secret_tool"],
                "expected_behavior": "Secret behavior",
                "note": "Secret note",
                "decision_audit": {
                    "status": "completed",
                    "request_started": True,
                    "choice": "Secret choice",
                },
            },
        )

    app = create_app(output_dir=isolated_env["output_dir"], token=TEST_TOKEN)
    async with web_client(app) as client:
        progress = (await client.get(f"/api/runs/{run_id}/progress")).json()
        by_id = {cp["scenario_id"]: cp for cp in progress["checkpoints"]}
        assert by_id["TC-01"]["summary"] == "Public summary"
        assert by_id["HO-99"]["summary"] == HELD_OUT
        assert "Secret" not in json.dumps(progress)

        detail = (await client.get(f"/api/runs/{run_id}")).json()
        assert "Secret" not in json.dumps(detail)

        events_text = (await client.get(f"/api/runs/{run_id}/events")).text
        assert "Secret" not in events_text
        assert HELD_OUT in events_text


@pytest.mark.asyncio
async def test_live_hub_callbacks_redact_unregistered_held_out_scenario() -> None:
    hub = LiveRunHub()
    session = hub.register("run-ho", model="qwen", backend="vllm", total=1)
    callbacks = hub.make_callbacks(session)
    secret_sc = _held_out_scenario("HO-77")
    unmarked_nonpublic = replace(secret_sc, id="CUSTOM-77", held_out=False)

    await callbacks["on_scenario_start"](unmarked_nonpublic, 0, 1)
    start_ev = next(ev for ev in session.history if ev["event"] == "scenario_start")
    assert start_ev["title"] == HELD_OUT
    assert start_ev["held_out"] is True

    await callbacks["on_scenario_turn"](
        unmarked_nonpublic,
        1,
        {"content": "secret turn content", "tool_calls": [{"name": "secret_fn"}]},
    )
    turn_ev = next(ev for ev in session.history if ev["event"] == "scenario_turn")
    assert turn_ev["content"] == ""
    assert turn_ev["tool_calls"] == []

    await callbacks["on_scenario_result"](
        unmarked_nonpublic,
        ScenarioResult(
            scenario_id="CUSTOM-77",
            status=ScenarioStatus.FAIL,
            points=0,
            summary="secret failure summary",
            raw_log="secret raw log",
        ),
        0,
        1,
    )
    await callbacks["on_scenario_audit"](
        unmarked_nonpublic,
        ScenarioResult(
            scenario_id="CUSTOM-77",
            status=ScenarioStatus.FAIL,
            points=0,
            summary="secret",
            decision_audit={
                "status": "completed",
                "request_started": True,
                "choice": "secret_choice",
                "probabilities": {"secret_choice": 0.95},
            },
        ),
        "completed",
    )

    serialized = json.dumps(session.history)
    assert "secret" not in serialized.lower()
    audit_ev = next(ev for ev in session.history if ev["event"] == "decision_audit")
    assert audit_ev["choice"] is None
    assert audit_ev["probabilities"] is None


# ---------------------------------------------------------------------------
# 3. Web Live Hub Queue Bounds, Delete Cleanup, Stop/Resume Races, & Launcher
# ---------------------------------------------------------------------------


def test_live_session_publish_evicts_full_subscriber_without_raising() -> None:
    hub = LiveRunHub()
    session = hub.register("run-bounded", model="qwen", backend="vllm", total=2000)
    slow_q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=MAX_SUBSCRIBER_QUEUE)
    fast_q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=MAX_SUBSCRIBER_QUEUE)
    session.subscribers.add(slow_q)
    session.subscribers.add(fast_q)

    for idx in range(MAX_SUBSCRIBER_QUEUE + 10):
        session.publish({"event": "scenario_turn", "seq": idx})
        while not fast_q.empty():
            fast_q.get_nowait()

    assert slow_q not in session.subscribers
    assert fast_q in session.subscribers
    drained = []
    while not slow_q.empty():
        drained.append(slow_q.get_nowait())
    assert drained[-1]["event"] == "stream_error"
    assert "reconnect" in drained[-1]["error"]


@pytest.mark.asyncio
async def test_delete_runs_forgets_cached_live_session_but_keeps_running_one(
    isolated_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_web_adapter(monkeypatch, _FastAdapter())
    app = create_app(output_dir=isolated_env["output_dir"], token=TEST_TOKEN)
    async with web_client(app) as client:
        started = (
            await client.post(
                "/api/runs",
                json={
                    "base_url": "http://localhost:8000",
                    "model": "qwen",
                    "backend": "vllm",
                    "preset": "custom",
                    "scenarios": ["TC-01"],
                },
            )
        ).json()
        run_id = started["run_id"]
        session = app.state.live_hub.get(run_id)
        assert session is not None and session.task is not None
        await session.task

        assert (await client.get(f"/api/runs/{run_id}/events")).status_code == 200

        deleted = await client.post("/api/runs/delete", json={"run_ids": [run_id]})
        assert deleted.status_code == 200
        assert deleted.json()["deleted"] == [run_id]

        assert app.state.live_hub.get(run_id) is None
        assert (await client.get(f"/api/runs/{run_id}/events")).status_code == 404

    hub = LiveRunHub()
    live_session = hub.register("still-running", model="qwen", backend="vllm", total=1)
    live_session.task = asyncio.create_task(asyncio.sleep(10))
    try:
        hub.forget("still-running")
        assert hub.get("still-running") is live_session
    finally:
        live_session.task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await live_session.task


@pytest.mark.asyncio
async def test_stop_web_run_immediately_after_launch_marks_session_interrupted(
    isolated_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_web_adapter(monkeypatch, _StallAfterFirstAdapter())
    hub = LiveRunHub()
    started = await launch_web_run(
        hub,
        {
            "base_url": "http://localhost:8000",
            "model": "qwen",
            "backend": "vllm",
            "preset": "custom",
            "scenarios": ["TC-01", "TC-02"],
        },
        output_dir=isolated_env["output_dir"],
    )
    run_id = started["run_id"]
    stopped = await stop_web_run(hub, run_id)
    assert stopped == {"run_id": run_id, "status": "interrupted"}
    session = hub.get(run_id)
    assert session is not None and session.status == "interrupted"

    with pytest.raises(RunConflict, match="interrupted"):
        await stop_web_run(hub, run_id)


@pytest.mark.asyncio
async def test_stop_web_run_raises_conflict_when_task_completes_during_cancel(
    isolated_env: dict[str, str],
) -> None:
    hub = LiveRunHub()
    session = hub.register("race-done", model="qwen", backend="vllm", total=1)

    async def finish_cleanly() -> None:
        session.status = "completed"

    session.task = asyncio.create_task(finish_cleanly())
    await session.task
    with pytest.raises(RunConflict, match="completed"):
        await stop_web_run(hub, "race-done")


@pytest.mark.asyncio
async def test_concurrent_resume_requests_allow_only_one_and_preserve_label_and_session(
    isolated_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    stalling = _StallAfterFirstAdapter()
    _patch_web_adapter(monkeypatch, stalling)
    hub = LiveRunHub()
    started = await launch_web_run(
        hub,
        {
            "base_url": "http://localhost:8000",
            "model": "qwen",
            "backend": "vllm",
            "preset": "custom",
            "scenarios": ["TC-01", "TC-02"],
            "label": "canary-label",
        },
        output_dir=isolated_env["output_dir"],
    )
    run_id = started["run_id"]
    session = hub.get(run_id)
    assert session is not None
    await _wait_for(lambda: stalling.calls >= 2)

    await stop_web_run(hub, run_id)
    with RunRepository(db_path=isolated_env["db_path"]) as repo:
        row = repo.get(run_id)
        assert row is not None
        assert row["metadata"]["label"] == "canary-label"

    _patch_web_adapter(monkeypatch, _FastAdapter(delay_s=0.02))
    outcomes = await asyncio.gather(
        resume_web_run(
            hub,
            run_id,
            {"base_url": "http://localhost:8000"},
            output_dir=isolated_env["output_dir"],
        ),
        resume_web_run(
            hub,
            run_id,
            {"base_url": "http://localhost:8000"},
            output_dir=isolated_env["output_dir"],
        ),
        return_exceptions=True,
    )
    successes = [o for o in outcomes if isinstance(o, dict)]
    conflicts = [o for o in outcomes if isinstance(o, RunConflict)]
    assert len(successes) == 1
    assert len(conflicts) == 1

    resumed_session = hub.get(run_id)
    assert resumed_session is not None and resumed_session.task is not None
    await resumed_session.task

    with RunRepository(db_path=isolated_env["db_path"]) as repo:
        completed = repo.get(run_id)
        assert completed is not None
        assert completed["status"] == "completed"
        assert completed["metadata"]["label"] == "canary-label"


@pytest.mark.asyncio
async def test_web_launcher_rejects_non_http_urls_and_unknown_backends(
    isolated_env: dict[str, str],
) -> None:
    hub = LiveRunHub()
    probe_res = await probe_endpoint_models(base_url="file:///etc/passwd")
    assert probe_res["reachable"] is False
    assert "must use http or https" in probe_res["error"]

    with pytest.raises(ValueError, match="base_url must use http or https"):
        await launch_web_run(
            hub,
            {"base_url": "ftp://localhost:8000", "model": "qwen", "scenarios": ["TC-01"]},
            output_dir=isolated_env["output_dir"],
        )

    with pytest.raises(ValueError, match="Unsupported backend"):
        await launch_web_run(
            hub,
            {
                "base_url": "http://localhost:8000",
                "model": "qwen",
                "backend": "not-a-backend",
                "scenarios": ["TC-01"],
            },
            output_dir=isolated_env["output_dir"],
        )

    with RunRepository(db_path=isolated_env["db_path"]) as repo:
        repo.upsert_scenario_run(
            {
                "run_id": "interrupted-url-test",
                "run_type": "tool_eval",
                "status": "interrupted",
                "config": {
                    "model": "qwen",
                    "backend": "vllm",
                    "base_url": "http://***:8000",
                    "scenario_ids": ["TC-01"],
                },
                "metadata": {"model": "qwen"},
                "scores": {},
            }
        )
    with pytest.raises(ValueError, match="base_url must use http or https"):
        await resume_web_run(
            hub,
            "interrupted-url-test",
            {"base_url": "file:///tmp/socket"},
            output_dir=isolated_env["output_dir"],
        )


@pytest.mark.asyncio
async def test_web_compare_handles_null_points_without_500(isolated_env: dict[str, str]) -> None:
    with RunRepository(db_path=isolated_env["db_path"]) as repo:
        for idx, pts in (("run-a", None), ("run-b", 2)):
            repo.upsert_scenario_run(
                {
                    "run_id": idx,
                    "run_type": "tool_eval",
                    "status": "completed",
                    "config": {"model": idx, "backend": "vllm", "base_url": "http://***:8000"},
                    "metadata": {"model": idx},
                    "scores": {
                        "final_score": 0 if pts is None else 100,
                        "scenario_results": [
                            {
                                "scenario_id": "TC-01",
                                "status": "fail" if pts is None else "pass",
                                "points": pts,
                            }
                        ],
                    },
                }
            )

    app = create_app(output_dir=isolated_env["output_dir"], token=TEST_TOKEN)
    async with web_client(app) as client:
        resp = await client.get("/api/compare", params={"left": "run-a", "right": "run-b"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["summary"]["total_points_a"] == 0
        assert body["summary"]["total_points_b"] == 2
        assert body["summary"]["total_delta"] == 2


def test_app_js_numeric_validation_and_live_state_cleanup_guards() -> None:
    app_js = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "tool_eval_bench"
        / "web"
        / "static"
        / "app.js"
    ).read_text(encoding="utf-8")

    assert "function readNumericField(" in app_js
    assert "el.validity?.badInput" in app_js
    assert "Number.isFinite(num)" in app_js
    assert "gone.has(state.liveRunId)" in app_js
    assert "Could not resume the run:" in app_js
    submit_block = app_js.split("async function submitLaunch(", maxsplit=1)[1].split(
        "function turnRow(", maxsplit=1
    )[0]
    assert "runId" not in submit_block


# ---------------------------------------------------------------------------
# 4. Public Python API Pre-Probe Input Validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("bad_kwargs", "match"),
    [
        ({"model": ""}, "model must be a non-empty string"),
        ({"model": "   "}, "model must be a non-empty string"),
        ({"base_url": ""}, "base_url must be a non-empty string"),
        ({"base_url": "ftp://localhost:8000"}, "base_url must use http or https"),
        ({"base_url": "http:///v1"}, "base_url is missing a host"),
        ({"backend": "not-a-real-backend"}, "Unsupported backend"),
        ({"concurrency": 0}, "concurrency must be at least 1"),
        ({"concurrency": -2}, "concurrency must be at least 1"),
        ({"concurrency": False}, "concurrency must be an integer"),
        ({"max_turns": True}, "max_turns must be an integer"),
        ({"seed": True}, "seed must be an integer"),
        ({"variant_seed": False}, "variant_seed must be an integer"),
        ({"timeout_seconds": 0}, "timeout_seconds must be a finite number greater than 0"),
        ({"timeout_seconds": -5.0}, "timeout_seconds must be a finite number greater than 0"),
        ({"timeout_seconds": math.nan}, "timeout_seconds must be a finite number greater than 0"),
        ({"timeout_seconds": math.inf}, "timeout_seconds must be a finite number greater than 0"),
        ({"temperature": -0.1}, "temperature must be a finite number >= 0"),
        ({"temperature": math.nan}, "temperature must be a finite number >= 0"),
        ({"temperature": math.inf}, "temperature must be a finite number >= 0"),
        ({"reference_date": "2026-3-5"}, "Invalid --reference-date"),
        ({"decision_judge_model": "judge-m"}, "requires a base URL"),
        ({"decision_judge": "recommended"}, "requires a base URL"),
        (
            {"decision_judge_base_url": "http://localhost:8090", "decision_judge_model": " "},
            "requires a model",
        ),
        ({"locale": "klingon"}, "Unsupported locale"),
        (
            {"scenarios": [_TC01], "locale": "de"},
            "None of the selected scenarios support --locale de",
        ),
    ],
)
async def test_api_run_benchmark_rejects_invalid_inputs_before_any_probe(
    monkeypatch: pytest.MonkeyPatch, bad_kwargs: dict[str, Any], match: str
) -> None:
    seen_requests: list[httpx.Request] = []

    def fail_on_network(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        return httpx.Response(500)

    monkeypatch.setattr(
        metadata.httpx,
        "AsyncClient",
        lambda **kw: _REAL_HTTPX_CLIENT(transport=httpx.MockTransport(fail_on_network), **kw),
    )

    defaults: dict[str, Any] = {
        "model": "qwen",
        "base_url": "http://localhost:8000",
        "scenarios": [_TC02],
        "persist": False,
    }
    with pytest.raises(ValueError, match=match):
        await run_benchmark(**{**defaults, **bad_kwargs})

    assert seen_requests == []


# ---------------------------------------------------------------------------
# 5. CLI Range Validation, Held-Out Safety Gate, Null-Points History, & Markup
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_sweep", ["0.2-nan", "nan-0.8", "0.2-inf", "-inf-0.8", "nan-nan"])
def test_parse_sweep_range_rejects_non_finite_floats(bad_sweep: str) -> None:
    with pytest.raises(ValueError, match="finite"):
        parse_sweep_range(bad_sweep)


def test_parse_sweep_range_accepts_valid_scientific_and_clamps_bounds() -> None:
    assert parse_sweep_range("0.25-0.75") == (0.25, 0.75)
    assert parse_sweep_range("1e-2-5e-1") == (0.01, 0.5)
    assert parse_sweep_range("-0.5-1.5") == (0.0, 1.0)


def test_plugin_needle_rejects_shots_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        translate_argv(["plugin", "needle", "--shots", "3"])
    assert exc.value.code == 2
    assert "--shots is not supported by the needle plugin" in capsys.readouterr().err


@pytest.mark.parametrize(
    "bad_argv",
    [
        ["--trials", "0"],
        ["--trials", "-1"],
        ["--timeout", "0"],
        ["--timeout", "-1"],
        ["--timeout", "nan"],
        ["--timeout", "inf"],
        ["--temperature", "-0.1"],
        ["--temperature", "nan"],
        ["--temperature", "inf"],
        ["--top-p", "0"],
        ["--top-p", "1.2"],
        ["--top-p", "nan"],
        ["--min-p", "-0.1"],
        ["--min-p", "1.1"],
        ["--min-p", "nan"],
        ["--repeat-penalty", "0"],
        ["--repeat-penalty", "nan"],
        ["--baseline-tgs", "0"],
        ["--baseline-tgs", "nan"],
        ["--spec-live-interval", "0"],
        ["--decision-live-interval", "0"],
        ["--sweep-steps", "1"],
        ["--spec-runs", "0"],
        ["--benchy-runs", "0"],
        ["--pp", "0"],
        ["--tg", "0"],
        ["--context-size", "0"],
        ["--needle-depths", "0"],
        ["--needle-lengths", "0"],
        ["--gsm8k-shots", "-1"],
        ["--gsm8k-shots", "9"],
        ["--gsm8k-limit", "-1"],
        ["--mmlu-shots", "-1"],
        ["--mmlu-shots", "6"],
        ["--mmlu-limit", "-1"],
        ["--ifeval-limit", "-1"],
        ["--toolscale-limit", "-1"],
        ["--reference-date", "2026-3-5"],
    ],
)
def test_cli_validate_run_ranges_rejects_invalid_values_in_plain_and_json_modes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], bad_argv: list[str]
) -> None:
    parser = make_parser()
    args = parser.parse_args(bad_argv)
    with pytest.raises(SystemExit) as plain_exc:
        dispatch._prepare_args(args, parser, Console(file=StringIO()))
    assert plain_exc.value.code == 2
    capsys.readouterr()

    monkeypatch.setattr("sys.argv", ["tool-eval-bench", *bad_argv, "--json"])
    with pytest.raises(SystemExit) as json_exc:
        dispatch.main()
    assert json_exc.value.code == 2
    err_line = capsys.readouterr().err.strip().splitlines()[-1]
    payload = json.loads(err_line)
    assert payload["event"] == "error"
    assert payload["error"] == "invalid_arguments"


def test_safety_gate_failed_redacts_held_out_warnings_in_non_json_stderr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    ho = _held_out_scenario("HO-42")
    resumed_ho = _held_out_scenario("HO-43")
    args = argparse.Namespace(
        fail_on_safety=True,
        json=False,
        include_held_out=False,
        _resolved_scenario_packs=[_scenario_pack(ho)],
        _resume_scenarios=[_TC01, resumed_ho],
    )
    assert held_out_ids(args) == frozenset({"HO-42", "HO-43"})
    assert redact_warnings(["HO-43: titleless secret violation"], held_out_ids(args)) == [
        f"HO-43: {HELD_OUT}"
    ]

    result = {
        "scores": {
            "safety_warnings": [
                "HO-42 (Confidential Pack Title): leaked sensitive details",
                "HO-43: titleless leaked details",
                "TC-01 (Weather): public warning text",
            ]
        }
    }
    assert safety_gate_failed(args, result) is True
    stderr = capsys.readouterr().err
    assert "Confidential Pack Title" not in stderr
    assert "leaked" not in stderr
    assert f"SAFETY GATE: HO-42: {HELD_OUT}" in stderr
    assert f"SAFETY GATE: HO-43: {HELD_OUT}" in stderr
    assert "SAFETY GATE: TC-01 (Weather): public warning text" in stderr

    args.include_held_out = True
    assert safety_gate_failed(args, result) is True
    unredacted = capsys.readouterr().err
    assert "Confidential Pack Title" in unredacted


def test_cli_history_diff_and_compare_tolerate_null_points(
    isolated_env: dict[str, str],
) -> None:
    with RunRepository(db_path=isolated_env["db_path"]) as repo:
        for rid, pts in (("hist-a", None), ("hist-b", 2)):
            repo.upsert_scenario_run(
                {
                    "run_id": rid,
                    "run_type": "tool_eval",
                    "status": "completed",
                    "config": {"model": rid, "backend": "vllm", "base_url": "http://***:8000"},
                    "metadata": {"model": rid},
                    "scores": {
                        "final_score": 0 if pts is None else 100,
                        "scenario_results": [
                            {
                                "scenario_id": "TC-01",
                                "status": "fail" if pts is None else "pass",
                                "points": pts,
                                "duration_seconds": None,
                            }
                        ],
                    },
                }
            )

    buf = StringIO()
    console = Console(file=buf, width=120, no_color=True)
    compare_runs(console, "hist-a", "hist-b")
    print_diff(
        console,
        [ScenarioResult("TC-01", ScenarioStatus.PASS, 2, "ok", duration_seconds=1.2)],
        "hist-a",
    )
    rendered = buf.getvalue()
    assert "0" in rendered and "2" in rendered


def test_rich_markup_brackets_and_url_redaction_do_not_crash_or_leak(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # 1. plugin_progress.truncate escapes square-bracket markup tokens.
    buf = StringIO()
    console = Console(file=buf, width=120, no_color=True)
    escaped = truncate("Wrap answer in [/] and [bold red]brackets[/]")
    console.print(f"[dim italic]{escaped}[/]")
    assert "[/]" in buf.getvalue()

    # 2. plugin_lifecycle.execute_plugin survives unclosed [/] in both normal and HeadlessConsole.
    def boom() -> Any:
        async def _coro() -> None:
            raise RuntimeError("bad bracket [/] from http://secret.internal:8000/v1")

        return _coro()

    with pytest.raises(SystemExit) as exc_plain:
        plugin_lifecycle.execute_plugin(console, "GSM8K[/]", boom, [])
    assert exc_plain.value.code == 1

    with pytest.raises(SystemExit) as exc_json:
        plugin_lifecycle.execute_plugin(HeadlessConsole(), "GSM8K[/]", boom, [])
    assert exc_json.value.code == 1
    headless_err = json.loads(capsys.readouterr().err.strip())
    assert headless_err["error"] == "run_failed"
    assert "secret.internal" not in headless_err["message"]

    # 3. plugin_datasets.load_dataset_with_progress escapes [hf], name, noun, and exception text.
    ds_buf = StringIO()
    ds_console = Console(file=ds_buf, width=120, no_color=True)
    items = plugin_datasets.load_dataset_with_progress(
        ds_console,
        name="IFEval[/]",
        noun="prompts[/]",
        cache_path=tmp_path / "missing.jsonl",
        load=lambda **_kw: (_ for _ in ()).throw(RuntimeError("rate limit [/] [bad")),
        cache_note="data/ifeval/test[/].jsonl",
    )
    assert items is None
    assert "tool-eval-bench[hf]" in ds_buf.getvalue()
    assert "rate limit [/]" in ds_buf.getvalue()

    # 4. probe.preflight_model_check redacts secret URL in 500 body when display_url is masked.
    def mock_500(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500, text="upstream http://secret-upstream.internal:8000/v1 failed [/] [bad"
        )

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: _REAL_HTTPX_CLIENT(transport=httpx.MockTransport(mock_500), **kw),
    )
    probe_buf = StringIO()
    probe_console = Console(file=probe_buf, width=120, no_color=True)
    with pytest.raises(SystemExit) as probe_exc:
        probe.preflight_model_check(
            probe_console,
            "http://secret.internal:8000",
            "model[/]",
            None,
            display_url="http://***:8000",
        )
    assert probe_exc.value.code == 3
    assert "[/]" in probe_buf.getvalue()
    assert "secret-upstream.internal" not in probe_buf.getvalue()

    # 5. model_probe._detect_model redacts URL in non-JSON body when display_url is masked.
    def mock_html(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "text/html"},
            text="login at http://sso.secret.internal:8443/auth [/]",
        )

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: _REAL_HTTPX_CLIENT(transport=httpx.MockTransport(mock_html), **kw),
    )
    mp_buf = StringIO()
    mp_console = Console(file=mp_buf, width=160, no_color=True)
    with pytest.raises(SystemExit) as mp_exc:
        model_probe._detect_model(
            "http://secret.internal:8000",
            None,
            mp_console,
            display_url="http://***:8000",
        )
    assert mp_exc.value.code == 2
    assert "sso.secret.internal" not in mp_buf.getvalue()


def test_multi_trial_summary_reports_receive_throughput_samples_and_resumed_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ho = _held_out_scenario("HO-55")
    samples = [
        ThroughputSample(
            pp_tokens=512,
            tg_tokens=128,
            depth=0,
            concurrency=1,
            pp_tps=1200.0,
            tg_tps=85.0,
        )
    ]

    trial_payload = {
        "run_id": "trial-run-1",
        "report_path": str(tmp_path / "trial1.md"),
        "scores": {
            "final_score": 50,
            "rating": "★★★ Adequate",
            "scenario_results": [
                ScenarioResult("TC-01", ScenarioStatus.PASS, 2, "ok").to_dict(),
                ScenarioResult("HO-55", ScenarioStatus.FAIL, 0, "secret").to_dict(),
            ],
            "safety_warnings": [],
        },
    }
    service = MagicMock()
    service.run_benchmark = AsyncMock(return_value=trial_payload)

    captured_calls: list[dict[str, Any]] = []

    def fake_write_summary_report(self: Any, **kwargs: Any) -> Path:
        captured_calls.append(kwargs)
        out = tmp_path / f"summary-{len(captured_calls)}.md"
        out.write_text("# Summary\n", encoding="utf-8")
        return out

    monkeypatch.setattr(
        "tool_eval_bench.cli.dispatch.MarkdownReporter.write_summary_report",
        fake_write_summary_report,
    )

    args = make_parser().parse_args(
        ["--trials", "2", "--scenarios", "TC-01", "--output-dir", str(tmp_path)]
    )
    args._resume_prior_results = [ScenarioResult("HO-55", ScenarioStatus.FAIL, 0, "secret")]
    args._resume_remaining_scenarios = [_TC01]
    args._resume_scenarios = [_TC01, ho]
    args._resolved_scenario_packs = [_scenario_pack(ho, name="ho-pack", tmp_path=tmp_path)]

    console = Console(file=StringIO(), width=120, no_color=True)
    dispatch._run_plain(
        service,
        console,
        "qwen",
        "Qwen",
        "vllm",
        "http://localhost:8000",
        None,
        args,
        throughput_samples=samples,
    )
    dispatch._run_with_live_display(
        service,
        console,
        "qwen",
        "Qwen",
        "vllm",
        "http://localhost:8000",
        None,
        args,
        throughput_samples=samples,
    )

    assert len(captured_calls) == 2
    for call_kwargs in captured_calls:
        assert call_kwargs["throughput_samples"] == samples
        assert "HO-55" in call_kwargs["scenario_metadata"]
        assert call_kwargs["scenario_metadata"]["HO-55"].held_out is True
        assert call_kwargs["scenario_packs"] == [
            {
                "name": "ho-pack",
                "scenario_count": 1,
                "scenario_ids": ["HO-55"],
                "content_hash": "a" * 64,
            }
        ]
        summaries: list[ModelScoreSummary] = call_kwargs["summaries"]
        assert len(summaries) == 2


def test_pressure_sweep_records_pack_attestations_and_redacts_held_out_report_end_to_end(
    isolated_env: dict[str, str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ho = _held_out_scenario("HO-88")
    monkeypatch.setattr(
        "tool_eval_bench.runner.context_pressure.calibrate_pressure_messages",
        AsyncMock(return_value=([], 1024)),
    )
    monkeypatch.setattr(
        "tool_eval_bench.adapters.factory.build_adapter",
        lambda *_a, **_k: _FastAdapter(),
    )
    monkeypatch.setattr(
        "tool_eval_bench.runner.orchestrator.run_all_scenarios",
        AsyncMock(
            return_value=ModelScoreSummary(
                scenario_results=[
                    ScenarioResult(
                        "TC-01",
                        ScenarioStatus.PASS,
                        2,
                        "Public weather summary",
                        expected_behavior="Public expected behavior",
                        tool_calls_made=["get_weather"],
                        raw_log="public weather trace",
                    ),
                    ScenarioResult(
                        "HO-88",
                        ScenarioStatus.PASS,
                        2,
                        "secret-ho-summary",
                        expected_behavior="secret-ho-expected",
                        tool_calls_made=["secret_canary_tool"],
                        raw_log="secret-ho-trace",
                    ),
                ],
                category_scores=[],
                final_score=100,
                total_points=4,
                max_points=4,
                rating="★★★★★ Excellent",
            )
        ),
    )

    real_pressure_report = pressure.pressure_sweep_report
    captured_report_kwargs: list[dict[str, Any]] = []

    def spy_pressure_report(**kwargs: Any) -> Any:
        captured_report_kwargs.append(kwargs)
        return real_pressure_report(**kwargs)

    monkeypatch.setattr(pressure, "pressure_sweep_report", spy_pressure_report)

    runs_out = tmp_path / "runs[brackets]"
    args = make_parser().parse_args(
        [
            "--context-pressure-sweep",
            "0.5-0.6",
            "--sweep-steps",
            "2",
            "--context-size",
            "32768",
            "--scenarios",
            "TC-01",
            "HO-88",
            "--output-dir",
            str(runs_out),
        ]
    )
    args._resolved_scenario_packs = [_scenario_pack(ho, name="sweep-pack", tmp_path=tmp_path)]

    buf = StringIO()
    console = Console(file=buf, width=120, no_color=True)
    pressure.run_pressure_sweep(
        console,
        "qwen",
        "Qwen[/]",
        "vllm",
        "http://localhost:8000",
        None,
        args,
    )

    assert len(captured_report_kwargs) == 1
    assert captured_report_kwargs[0]["scenario_metadata"]["TC-01"].held_out is False
    assert captured_report_kwargs[0]["scenario_metadata"]["HO-88"].held_out is True
    assert captured_report_kwargs[0]["scenario_packs"][0]["name"] == "sweep-pack"

    report_files = list(runs_out.rglob("*.md"))
    assert len(report_files) == 1
    report_md = report_files[0].read_text(encoding="utf-8")
    assert "Public weather summary" in report_md
    assert "public weather trace" in report_md
    assert "### HO-88" in report_md
    assert "_Held-out scenario: title, summary, and trace withheld._" in report_md
    assert "sweep-pack" in report_md
    assert "a" * 64 in report_md
    for secret_token in (
        "Secret Held-Out Canary Title",
        "Confidential",
        "secret-ho-summary",
        "secret-ho-expected",
        "secret_canary_tool",
        "secret-ho-trace",
    ):
        assert secret_token not in report_md

    stored = recent_runs(limit=1)[0]
    assert stored["config"]["scenario_packs"][0]["name"] == "sweep-pack"
    stored_ho = next(
        sr
        for sr in stored["scores"]["level_results"][0]["scenario_results"]
        if sr["scenario_id"] == "HO-88"
    )
    assert stored_ho["raw_log"] == "secret-ho-trace"


# ---------------------------------------------------------------------------
# 6. Follow-Up Hardening: Display Markup, Interactive Redaction, Presets,
#    Resumed Live Indices, and CSV Formula Injection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_display_and_final_report_escape_dynamic_markup_and_preserve_title_padding() -> None:
    sc_bracket = replace(_TC01, title="Short [/] Title")
    sc_plain = replace(_TC02, title="Plain Title")

    res_pass = ScenarioResult(
        scenario_id=sc_bracket.id,
        status=ScenarioStatus.PASS,
        points=2,
        summary="Passed with [/] and [bad token",
        duration_seconds=1.1,
    )
    res_plain = ScenarioResult(
        scenario_id=sc_plain.id,
        status=ScenarioStatus.PASS,
        points=2,
        summary="Plain summary",
        duration_seconds=1.1,
    )
    res_fail = ScenarioResult(
        scenario_id="TC-03[/]",
        status=ScenarioStatus.FAIL,
        points=0,
        summary="Failed [/] [bad output",
        tool_calls_made=["lookup([/])", "run([bad)"],
        expected_behavior="Call safe_tool([/]) [bad",
        duration_seconds=2.3,
    )
    res_infra = ScenarioResult(
        scenario_id="TC-04",
        status=ScenarioStatus.FAIL,
        points=0,
        summary="",
        note="Request timed out [/] [bad",
        failure_kind=FailureKind.TIMEOUT,
        duration_seconds=120.0,
    )

    summary = ModelScoreSummary(
        scenario_results=[res_pass, res_fail, res_infra],
        category_scores=[
            CategoryScore(
                category=Category.A,
                label="Direct [/] [bad",
                earned=2,
                max_points=4,
                percent=50.0,
                pass_count=1,
                fail_count=1,
            )
        ],
        capability_scores=[
            CapabilityScore(
                capability="policy",
                label="Policy [/] [bad",
                earned=2,
                max_points=4,
                percent=50.0,
                scenario_ids=["TC-01[/]", "TC-03[bad"],
            )
        ],
        final_score=50,
        total_points=2,
        max_points=4,
        rating="★★★ Adequate [/]",
        excluded_scenarios=["TC-04[/]"],
        completion_rate=66.7,
        worst_category="Direct [/] [bad",
        worst_category_percent=50,
        safety_warnings=["TC-03: leaked [/] [bad"],
    )
    run_ctx = RunContext(
        tool_version="0.1.0",
        git_sha=None,
        hostname="host",
        platform_info="linux",
        python_version="3.13",
        model="qwen[/][bad",
        backend="vllm[/]",
        base_url="http://***:8000/[/]",
        timeout_seconds=120.0,
        server_model_root="/models/[bad]/qwen[/]",
        engine_name="vLLM[/]",
        engine_version="0.8[bad",
        quantization="AWQ[/][bad",
    )

    buf = StringIO()
    console = Console(file=buf, width=160, no_color=True)
    disp = BenchmarkDisplay(
        model="qwen[/][bad",
        backend="vllm[/]",
        base_url="http://***:8000/[/]",
        scenarios=[sc_bracket, sc_plain],
        run_context=run_ctx,
    )
    disp.console = console

    # Alignment check: padding before escape keeps identical status column offset.
    line_bracket_buf = StringIO()
    line_plain_buf = StringIO()
    Console(file=line_bracket_buf, width=200, no_color=True).print(
        disp._format_result_line(sc_bracket, res_pass)
    )
    Console(file=line_plain_buf, width=200, no_color=True).print(
        disp._format_result_line(sc_plain, res_plain)
    )
    assert line_bracket_buf.getvalue().index("PASS") == line_plain_buf.getvalue().index("PASS")

    disp.start()
    await disp.on_scenario_start(sc_bracket, 0, 2)
    footer = disp._build_footer()
    assert "Short [/] Title" in footer.plain
    await disp.on_scenario_result(sc_bracket, res_pass, 0, 2)
    await disp.on_scenario_result(sc_plain, res_infra, 1, 2)
    disp.set_finished(summary)

    print_final_report(
        console,
        "qwen[/][bad",
        summary,
        3.4,
        run_context=run_ctx,
    )

    rendered = buf.getvalue()
    assert "qwen[/][bad" in rendered
    assert "vLLM[/] 0.8[bad" in rendered
    assert "AWQ[/][bad" in rendered
    assert "/models/[bad]/qwen[/]" in rendered
    assert "TC-03: leaked [/] [bad" in rendered
    assert "lookup([/])" in rendered
    assert "Call safe_tool([/]) [bad" in rendered


def test_plugin_lifecycle_and_spec_bench_redact_urls_on_interactive_console_and_escape_paths(
    isolated_env: dict[str, str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # 1. Interactive Console in plugin_lifecycle redacts URLs and escapes benchmark_name / paths.
    plain_buf = StringIO()
    plain_console = Console(file=plain_buf, width=140, no_color=True)

    def fail_coro() -> Any:
        async def _inner() -> None:
            raise httpx.ConnectError(
                "failed connecting to http://secret-plugin.internal:8000/v1?key=sk-plugin [/] [bad"
            )

        return _inner()

    with pytest.raises(SystemExit) as exc_info:
        plugin_lifecycle.execute_plugin(plain_console, "GSM8K[/]", fail_coro, [])
    assert exc_info.value.code == 1
    out = plain_buf.getvalue()
    assert "GSM8K[/] error:" in out
    assert "secret-plugin.internal" not in out
    assert "sk-plugin" not in out
    assert "[/]" in out

    empty_buf = StringIO()
    empty_console = Console(file=empty_buf, width=140, no_color=True)
    assert plugin_lifecycle.execute_plugin(empty_console, "MMLU[/][bad", AsyncMock(), []) is None
    assert "No MMLU[/][bad results." in empty_buf.getvalue()

    fake_res = MagicMock(score=87.5, rating="★★★★ Good", details={"correct": 7, "total": 8})
    out_dir = tmp_path / "plugin[bold]out"
    fin_buf = StringIO()
    fin_console = Console(file=fin_buf, width=160, no_color=True)
    plugin_lifecycle.finalize_plugin_run(
        fin_console,
        mode="gsm8k",
        title="GSM8K",
        display_name="Qwen[/]",
        result=fake_res,
        config={"model": "qwen", "backend": "vllm", "base_url": "http://***:8000"},
        report_metrics=["- **Accuracy**: 87.5%"],
        report_lines=["ok"],
        output_dir=str(out_dir),
        label=None,
        run_context=None,
    )
    assert "plugin[bold]out" in fin_buf.getvalue().replace("\n", "")

    # 2. Interactive Console in spec_bench redacts URLs on exception and escapes prompt_type / paths.
    async def fail_spec(*_a: Any, **_kw: Any) -> list[SpecDecodeSample]:
        raise httpx.ConnectError(
            "spec failed at http://secret-spec.internal:8090/v1?api_key=sk-spec-secret [/] [bad"
        )

    monkeypatch.setattr("tool_eval_bench.runner.speculative.run_spec_bench", fail_spec)
    spec_err_buf = StringIO()
    spec_err_console = Console(file=spec_err_buf, width=160, no_color=True)
    with pytest.raises(SystemExit) as spec_exc:
        spec_bench.run_spec_bench(
            spec_err_console,
            "qwen",
            "Qwen[/]",
            "http://secret-spec.internal:8090",
            None,
            pp=128,
            tg=64,
            depths=[0],
            output_dir=str(tmp_path / "spec_err"),
        )
    assert spec_exc.value.code == 1
    spec_err_text = spec_err_buf.getvalue()
    assert "secret-spec.internal" not in spec_err_text
    assert "sk-spec-secret" not in spec_err_text
    assert "[/]" in spec_err_text

    async def ok_spec(*_a: Any, on_sample: Any = None, **_kw: Any) -> list[SpecDecodeSample]:
        s_ok = SpecDecodeSample(
            pp_tokens=128,
            tg_tokens=64,
            depth=0,
            ttft_ms=12.0,
            total_ms=500.0,
            pp_tps=900.0,
            tg_tps=120.0,
            acceptance_rate=0.75,
            acceptance_length=3.2,
            draft_tokens_delta=80,
            accepted_tokens_delta=60,
            num_drafts_delta=20,
            acceptance_source="response",
            prompt_type="code[/]",
        )
        s_err = SpecDecodeSample(
            depth=0,
            prompt_type="bad[p]",
            error="cell error from http://cell-secret.internal:8000/v1 [/]",
        )
        if on_sample is not None:
            await on_sample(s_ok, 0, 2)
            await on_sample(s_err, 1, 2)
        return [s_ok, s_err]

    monkeypatch.setattr("tool_eval_bench.runner.speculative.run_spec_bench", ok_spec)
    spec_ok_dir = tmp_path / "spec[bold]dir"
    spec_ok_buf = StringIO()
    spec_ok_console = Console(file=spec_ok_buf, width=160, no_color=True)
    samples = spec_bench.run_spec_bench(
        spec_ok_console,
        "qwen",
        "Qwen[/]",
        "http://localhost:8000",
        None,
        pp=128,
        tg=64,
        depths=[0],
        prompt_types=["code[/]", "bad[p]"],
        output_dir=str(spec_ok_dir),
    )
    assert len(samples) == 2
    spec_ok_text = spec_ok_buf.getvalue()
    assert "code[/]" in spec_ok_text
    assert "spec[bold]dir" in spec_ok_text.replace("\n", "")
    assert "cell-secret.internal" not in spec_ok_text


@pytest.mark.asyncio
async def test_web_launcher_rejects_scenario_packs_and_unknown_presets(
    isolated_env: dict[str, str],
) -> None:
    # Positive preset coverage.
    short_default, packs, held = _resolve_launcher_scenarios({})
    assert len(short_default) == 15 and packs is None and held == frozenset()
    short_empty, _, _ = _resolve_launcher_scenarios({"preset": ""})
    assert len(short_empty) == 15
    standard, _, _ = _resolve_launcher_scenarios({"preset": "standard"})
    assert len(standard) == 69
    hardmode, _, _ = _resolve_launcher_scenarios({"preset": "hardmode"})
    assert len(hardmode) == 104
    hardmode_only, _, _ = _resolve_launcher_scenarios({"preset": "hardmode_only"})
    assert len(hardmode_only) == 35
    custom_one, _, _ = _resolve_launcher_scenarios({"preset": "custom", "scenarios": ["TC-01"]})
    assert [s.id for s in custom_one] == ["TC-01"]

    # Negative coverage via both direct launcher call and HTTP POST /api/runs.
    hub = LiveRunHub()
    app = create_app(token=TEST_TOKEN, output_dir=isolated_env["output_dir"])
    bad_payloads: list[tuple[dict[str, Any], str]] = [
        ({"scenario_pack": "/etc"}, "Held-out scenario packs are only supported from the CLI"),
        ({"scenario_pack": None}, "Held-out scenario packs are only supported from the CLI"),
        ({"pack_only": True}, "Held-out scenario packs are only supported from the CLI"),
        ({"pack_only": False}, "Held-out scenario packs are only supported from the CLI"),
        ({"preset": "everything"}, "Unsupported preset"),
        ({"preset": "   "}, "Unsupported preset"),
        ({"preset": 123}, "Unsupported preset"),
    ]
    async with web_client(app) as client:
        for extra, expected_msg in bad_payloads:
            full_payload: dict[str, Any] = {
                "base_url": "http://localhost:8000",
                "model": "qwen",
                "backend": "vllm",
                **extra,
            }
            with pytest.raises(ValueError, match=expected_msg):
                await launch_web_run(hub, full_payload, output_dir=isolated_env["output_dir"])

            resp = await client.post("/api/runs", headers=AUTH_HEADERS, json=full_payload)
            assert resp.status_code == 400
            assert expected_msg in resp.json()["error"]


@pytest.mark.asyncio
async def test_live_run_hub_offsets_resumed_scenario_indices_and_keeps_zero_base_when_fresh() -> (
    None
):
    hub = LiveRunHub()

    # Fresh run: indices start at 0.
    fresh = hub.register(
        run_id="fresh-idx-run",
        model="qwen",
        backend="vllm",
        total=2,
    )
    fresh_sub: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    fresh.subscribers.add(fresh_sub)
    fresh_cb = hub.make_callbacks(fresh)
    await fresh_cb["on_scenario_start"](_TC01, 0, 2)
    await fresh_cb["on_scenario_result"](
        _TC01, ScenarioResult("TC-01", ScenarioStatus.PASS, 2, "ok"), 0, 2
    )
    fresh_stream: list[tuple[str, int, int]] = []
    while not fresh_sub.empty():
        ev = fresh_sub.get_nowait()
        if ev["event"] in {"scenario_start", "scenario_result"}:
            fresh_stream.append((ev["event"], ev["index"], ev["total"]))
    assert fresh_stream == [
        ("scenario_start", 0, 2),
        ("scenario_result", 0, 2),
    ]

    # Resumed run with k=2 preserved outcomes and 2 remaining scenarios (total=4):
    # preserved replay emits 0..1, and orchestrator callbacks (idx=0..1) emit 2..3.
    preserved = [
        ScenarioResult("TC-01", ScenarioStatus.PASS, 2, "ok").to_dict(),
        ScenarioResult("TC-02", ScenarioStatus.PARTIAL, 1, "half").to_dict(),
    ]
    resumed = hub.register(
        run_id="resumed-idx-run",
        model="qwen",
        backend="vllm",
        total=4,
        preserved=preserved,
    )
    assert resumed.index_offset == 2
    resumed_sub: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    resumed.subscribers.add(resumed_sub)
    resumed_cb = hub.make_callbacks(resumed)
    tc03 = ALL_SCENARIOS[2]
    tc04 = ALL_SCENARIOS[3]
    await resumed_cb["on_scenario_start"](tc03, 0, 2)
    await resumed_cb["on_scenario_result"](
        tc03, ScenarioResult(tc03.id, ScenarioStatus.PASS, 2, "ok"), 0, 2
    )
    await resumed_cb["on_scenario_start"](tc04, 1, 2)
    await resumed_cb["on_scenario_result"](
        tc04, ScenarioResult(tc04.id, ScenarioStatus.FAIL, 0, "no"), 1, 2
    )

    live_stream: list[tuple[str, str, int, int]] = []
    while not resumed_sub.empty():
        ev = resumed_sub.get_nowait()
        if ev["event"] in {"scenario_start", "scenario_result"}:
            live_stream.append((ev["event"], ev["scenario_id"], ev["index"], ev["total"]))
    assert live_stream == [
        ("scenario_start", tc03.id, 2, 4),
        ("scenario_result", tc03.id, 2, 4),
        ("scenario_start", tc04.id, 3, 4),
        ("scenario_result", tc04.id, 3, 4),
    ]

    history_results = [
        (ev["event"], ev["scenario_id"], ev["index"], ev["total"])
        for ev in resumed.history
        if ev["event"] == "scenario_result"
    ]
    assert history_results == [
        ("scenario_result", "TC-01", 0, 4),
        ("scenario_result", "TC-02", 1, 4),
        ("scenario_result", tc03.id, 2, 4),
        ("scenario_result", tc04.id, 3, 4),
    ]


@pytest.mark.asyncio
async def test_csv_export_neutralizes_formula_injection_in_web_and_cli_without_touching_numbers(
    isolated_env: dict[str, str], tmp_path: Path
) -> None:
    # Unit-level boundary check on helper behavior for numbers, booleans, empty, and prefixes.
    for fn in (_web_sanitize_csv_cell, leaderboard._sanitize_csv_cell):
        assert fn("=1+1") == "'=1+1"
        assert fn("+SUM(A1:A2)") == "'+SUM(A1:A2)"
        assert fn("-2+3") == "'-2+3"
        assert fn("@IMP") == "'@IMP"
        assert fn("\ttabbed") == "'\ttabbed"
        assert fn("\rcr") == "'\rcr"
        assert fn("safe-model") == "safe-model"
        assert fn("") == ""
        assert fn(-42) == -42
        assert fn(-0.25) == -0.25

    with RunRepository(db_path=isolated_env["db_path"]) as repo:
        repo.upsert_scenario_run(
            {
                "run_id": "csv-formula-run",
                "run_type": "tool_eval",
                "status": "completed",
                "config": {
                    "model": "=CMD|' /C calc'!A0",
                    "backend": "@gateway",
                    "base_url": "http://***:8000",
                    "scenario_count": 1,
                    "temperature": 0.0,
                },
                "metadata": {
                    "model": "=CMD|' /C calc'!A0",
                    "engine_name": "+vLLM",
                    "engine_version": "-0.8.5",
                    "quantization": "\tAWQ",
                    "server_model_root": "\r/models/qwen",
                    "tool_version": "0.1.0",
                    "max_model_len": 32768,
                    "temperature": 0.0,
                },
                "scores": {
                    "final_score": 85,
                    "rating": "★★★★ Good",
                    "total_points": 17,
                    "max_points": 20,
                    "total_tokens": 1024,
                    "scenario_results": [{"scenario_id": "TC-01", "status": "pass", "points": 2}],
                    "category_scores": [{"category": "A", "percent": 85.0}],
                },
            }
        )
        repo.upsert_scenario_run(
            {
                "run_id": "csv-clean-run",
                "run_type": "tool_eval",
                "status": "completed",
                "config": {
                    "model": "qwen2.5-7b",
                    "backend": "vllm",
                    "base_url": "http://***:8000",
                    "scenario_count": 1,
                    "temperature": 0.2,
                },
                "metadata": {
                    "model": "qwen2.5-7b",
                    "engine_name": "vLLM",
                    "engine_version": "0.8.5",
                    "quantization": "FP8",
                    "server_model_root": "/models/qwen2.5-7b",
                    "temperature": 0.2,
                },
                "scores": {
                    "final_score": 90,
                    "rating": "★★★★★ Excellent",
                    "total_points": 18,
                    "max_points": 20,
                    "total_tokens": 900,
                    "scenario_results": [{"scenario_id": "TC-01", "status": "pass", "points": 2}],
                    "category_scores": [{"category": "A", "percent": 90.0}],
                },
            }
        )

    # 1. Web CSV and JSON export.
    app = create_app(token=TEST_TOKEN, output_dir=isolated_env["output_dir"])
    async with web_client(app) as client:
        csv_resp = await client.get("/api/export?format=csv", headers=AUTH_HEADERS)
        assert csv_resp.status_code == 200
        web_rows = {r["run_id"]: r for r in csv.DictReader(StringIO(csv_resp.text))}

        json_resp = await client.get("/api/export?format=json", headers=AUTH_HEADERS)
        assert json_resp.status_code == 200
        json_rows = {r["run_id"]: r for r in json_resp.json()}

    formula_row = web_rows["csv-formula-run"]
    assert formula_row["model"].startswith("'=CMD")
    assert formula_row["backend"] == "'@gateway"
    assert formula_row["cohort"].startswith("'@gateway/1")
    assert formula_row["engine_name"] == "'+vLLM"
    assert formula_row["engine_version"] == "'-0.8.5"
    assert formula_row["quantization"] == "'\tAWQ"
    assert formula_row["server_model_root"] == "'\r/models/qwen"
    assert formula_row["final_score"] == "85"
    assert formula_row["temperature"] == "0.0"

    clean_row = web_rows["csv-clean-run"]
    assert clean_row["model"] == "qwen2.5-7b"
    assert clean_row["engine_name"] == "vLLM"
    assert clean_row["final_score"] == "90"
    assert clean_row["temperature"] == "0.2"

    # JSON export preserves raw strings without spreadsheet quoting.
    assert json_rows["csv-formula-run"]["model"] == "=CMD|' /C calc'!A0"

    # 2. CLI CSV export + bracketed output path escaping.
    out_dir = tmp_path / "dir[bold]"
    out_dir.mkdir(parents=True)
    out_csv = out_dir / "export[1].csv"
    cli_buf = StringIO()
    cli_console = Console(file=cli_buf, width=140, no_color=True)
    leaderboard.export_runs(cli_console, fmt="csv", output=str(out_csv))
    cli_out = cli_buf.getvalue().replace("\n", "")
    assert "dir[bold]" in cli_out
    assert "export[1].csv" in cli_out

    with out_csv.open("r", encoding="utf-8", newline="") as fh:
        cli_rows = {r["run_id"]: r for r in csv.DictReader(fh)}
    assert cli_rows["csv-formula-run"]["model"].startswith("'=CMD")
    assert cli_rows["csv-formula-run"]["engine_name"] == "'+vLLM"
    assert cli_rows["csv-formula-run"]["engine_version"] == "'-0.8.5"
    assert cli_rows["csv-formula-run"]["quantization"] == "'\tAWQ"
    assert cli_rows["csv-clean-run"]["model"] == "qwen2.5-7b"
    assert cli_rows["csv-clean-run"]["final_score"] == "90"
