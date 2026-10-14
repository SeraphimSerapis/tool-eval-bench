"""Tests for the web UI schema, endpoint probe, run launcher, stop, and resume endpoints."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from tests.web_support import TEST_TOKEN, web_client
from tool_eval_bench.application.run_config import RunSettings, build_run_config
from tool_eval_bench.domain.adapters import ChatCompletionResult
from tool_eval_bench.evals.scenarios import ALL_SCENARIOS
from tool_eval_bench.storage.db import RunRepository
from tool_eval_bench.web import create_app

_BASE_URL = "http://localhost:8000"
_LAUNCH = {
    "base_url": _BASE_URL,
    "model": "Qwen/Qwen3-8B",
    "backend": "vllm",
    "preset": "custom",
    "scenarios": ["TC-01", "TC-02"],
}


def _answer() -> ChatCompletionResult:
    return ChatCompletionResult(
        content="The weather in Berlin is 18C.",
        tool_calls=[],
        ttft_ms=15.0,
        prompt_tokens=100,
        completion_tokens=20,
    )


class _FakeAdapter:
    def __init__(self, delay_s: float = 0.0) -> None:
        self.delay_s = delay_s
        self.calls = 0

    async def chat_completion(self, **kwargs: Any) -> ChatCompletionResult:
        self.calls += 1
        if self.delay_s > 0:
            await asyncio.sleep(self.delay_s)
        return _answer()

    async def aclose(self) -> None:
        pass


class _StallAfterFirst(_FakeAdapter):
    """Answers the first request, then hangs until the run is cancelled."""

    async def chat_completion(self, **kwargs: Any) -> ChatCompletionResult:
        self.calls += 1
        if self.calls > 1:
            await asyncio.Event().wait()
        return _answer()


def _use_adapter(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    monkeypatch.setattr(
        "tool_eval_bench.application.service.BenchmarkService._adapter_for",
        lambda self, *args, **kwargs: adapter,
    )


@pytest.fixture()
def launcher_env(tmp_path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    db_path = str(tmp_path / "benchmarks.sqlite")
    out_dir = str(tmp_path / "runs")
    monkeypatch.setattr(
        "tool_eval_bench.application.run_queries._repository",
        lambda: RunRepository(db_path=db_path),
    )

    # Web-launched runs open their own repository through open_benchmark_service.
    # A subclass rather than a lambda: the service also uses the name in type
    # expressions evaluated at runtime.
    class _TempRepository(RunRepository):
        def __init__(self, db_path: str | None = None) -> None:
            super().__init__(db_path=db_path or temp_db)

    temp_db = db_path
    monkeypatch.setattr("tool_eval_bench.application.service.RunRepository", _TempRepository)
    monkeypatch.delenv("TOOL_EVAL_BASE_URL", raising=False)
    monkeypatch.delenv("TOOL_EVAL_API_KEY", raising=False)
    return {"db_path": db_path, "output_dir": out_dir}


def _store_run(db_path: str, run_id: str, *, status: str, **overrides: Any) -> None:
    """Persist a run whose config the current schema built, as a CLI run would."""
    scenarios = [s for s in ALL_SCENARIOS if s.id in {"TC-01", "TC-02"}]
    values: dict[str, Any] = {
        "model": "Qwen/Qwen3-8B",
        "backend": "vllm",
        "base_url": _BASE_URL,
        "temperature": 0.0,
        "timeout_seconds": 120.0,
        "max_turns": 8,
        "seed": None,
        "reference_date": None,
        "concurrency": 1,
        "error_rate": 0.0,
        "alpha": 0.7,
        "extra_params": None,
        "context_pressure_config": None,
        "weight_by_difficulty": False,
    }
    values.update(overrides)
    config = build_run_config(RunSettings(**values), scenarios=scenarios, metadata={})
    with RunRepository(db_path=db_path) as repo:
        repo.upsert_scenario_run(
            {
                "run_id": run_id,
                "status": status,
                "run_type": "tool_eval",
                "model": values["model"],
                "config": config,
                "metadata": {},
                "scores": {"scenario_results": []},
            }
        )


async def _wait_for(predicate: Any, *, timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_schema_and_probe_endpoints(
    launcher_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    app = create_app(output_dir=launcher_env["output_dir"], token=TEST_TOKEN)

    async def _fake_fetch_models(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "data": [
                {"id": "qwen3-8b", "root": "Qwen/Qwen3-8B-FP8"},
                {"id": "llama3-70b"},
            ]
        }

    monkeypatch.setattr("tool_eval_bench.web.launcher._fetch_models_json", _fake_fetch_models)

    async with web_client(app) as client:
        schema_resp = await client.get("/api/schema")
        assert schema_resp.status_code == 200
        schema_data = schema_resp.json()
        assert any(p["id"] == "short" and p["count"] == 15 for p in schema_data["presets"])
        assert any(p["id"] == "standard" and p["count"] == 69 for p in schema_data["presets"])
        assert any(p["id"] == "hardmode" and p["count"] == 97 for p in schema_data["presets"])
        assert len(schema_data["categories"]) == 16

        probe_resp = await client.post("/api/probe", json={"base_url": _BASE_URL})
        assert probe_resp.status_code == 200
        probe_data = probe_resp.json()
        assert probe_data["reachable"] is True
        assert probe_data["resolved_base_url"] == _BASE_URL
        assert probe_data["models"] == [
            {"id": "qwen3-8b", "display_name": "Qwen/Qwen3-8B-FP8"},
            {"id": "llama3-70b", "display_name": "llama3-70b"},
        ]

        # The probe used to be a GET, which a cross-site <img> could trigger.
        assert (await client.get("/api/probe")).status_code == 405


@pytest.mark.asyncio
async def test_probe_redacts_urls_in_errors_and_withholds_them_when_redacting(
    launcher_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    secret_url = "http://gpu-7.internal.example:8000/v1?key=sk-leak"

    async def _failing_fetch(*args: Any, **kwargs: Any) -> Any:
        raise httpx.ConnectError(f"Connection refused for url '{secret_url}/models'")

    monkeypatch.setattr("tool_eval_bench.web.launcher._fetch_models_json", _failing_fetch)
    app = create_app(token=TEST_TOKEN, redact_url=True)
    async with web_client(app) as client:
        resp = await client.post("/api/probe", json={"base_url": secret_url})
    body = resp.json()
    assert body["reachable"] is False
    assert "gpu-7.internal.example" not in resp.text
    assert "sk-leak" not in resp.text
    assert "resolved_base_url" not in body


@pytest.mark.asyncio
async def test_launch_run_executes_and_persists_results(
    launcher_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_adapter(monkeypatch, _FakeAdapter())
    app = create_app(output_dir=launcher_env["output_dir"], token=TEST_TOKEN)
    async with web_client(app) as client:
        bad_resp = await client.post("/api/runs", json={"model": "m"})
        assert bad_resp.status_code == 400

        launch_resp = await client.post("/api/runs", json={**_LAUNCH, "label": "web-smoke"})
        assert launch_resp.status_code == 202
        started = launch_resp.json()
        run_id = started["run_id"]
        assert started["total_scenarios"] == 2

        session = app.state.live_hub.get(run_id)
        assert session is not None and session.task is not None
        await session.task

        run_resp = await client.get(f"/api/runs/{run_id}")
        assert run_resp.status_code == 200
        run_data = run_resp.json()
        assert run_data["status"] == "completed"
        assert run_data["label"] == "web-smoke"
        assert len(run_data["scores"]["scenario_results"]) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("temperature", None, None),  # null falls back to the default
        ("temperature", "warm", "temperature must be a number"),
        ("temperature", -0.5, "temperature must be at least 0"),
        ("timeout_seconds", 0, "timeout_seconds must be greater than 0"),
        ("max_turns", 2.5, "max_turns must be a whole number"),
        ("concurrency", 0, "concurrency must be at least 1"),
        ("seed", [1], "seed must be a number"),
        ("temperature", "nan", "temperature must be finite"),
    ],
)
async def test_launch_rejects_malformed_numbers_with_400(
    launcher_env: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: Any,
    message: str | None,
) -> None:
    _use_adapter(monkeypatch, _FakeAdapter())
    app = create_app(output_dir=launcher_env["output_dir"], token=TEST_TOKEN)
    async with web_client(app) as client:
        resp = await client.post("/api/runs", json={**_LAUNCH, field: value})
        if message is None:
            assert resp.status_code == 202
            await app.state.live_hub.get(resp.json()["run_id"]).task
        else:
            assert resp.status_code == 400
            assert message in resp.json()["error"]


@pytest.mark.asyncio
async def test_run_failure_reason_is_redacted(
    launcher_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("upstream refused http://gpu-7.internal.example:8000/v1?key=sk-leak")

    monkeypatch.setattr("tool_eval_bench.application.service.BenchmarkService._adapter_for", _boom)
    app = create_app(output_dir=launcher_env["output_dir"], token=TEST_TOKEN)
    async with web_client(app) as client:
        resp = await client.post("/api/runs", json=_LAUNCH)
        session = app.state.live_hub.get(resp.json()["run_id"])
        await session.task
    final = session.history[-1]
    assert final["event"] == "run_interrupted"
    assert "upstream refused" in final["reason"]
    assert "gpu-7.internal.example" not in final["reason"]
    assert "sk-leak" not in final["reason"]


@pytest.mark.asyncio
async def test_stop_keeps_real_checkpoints_and_resume_replays_them(
    launcher_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    stalling = _StallAfterFirst()
    _use_adapter(monkeypatch, stalling)
    app = create_app(output_dir=launcher_env["output_dir"], token=TEST_TOKEN)
    async with web_client(app) as client:
        run_id = (await client.post("/api/runs", json=_LAUNCH)).json()["run_id"]
        session = app.state.live_hub.get(run_id)

        # TC-01 finishes and is checkpointed; TC-02 hangs mid-request.
        await _wait_for(
            lambda: (
                any(e["event"] == "scenario_result" for e in session.history)
                and stalling.calls >= 2
            )
        )
        stop_resp = await client.post(f"/api/runs/{run_id}/stop", json={})
        assert stop_resp.status_code == 200
        assert stop_resp.json()["status"] == "interrupted"

        with RunRepository(db_path=launcher_env["db_path"]) as repo:
            assert repo.get(run_id)["status"] == "interrupted"
            assert [cp["scenario_id"] for cp in repo.get_checkpoints(run_id)] == ["TC-01"]

        # Stopping twice is a conflict, not a silent rewrite.
        assert (await client.post(f"/api/runs/{run_id}/stop", json={})).status_code == 409

        # The stored URL is redacted, so resume needs the endpoint again.
        assert (await client.post(f"/api/runs/{run_id}/resume", json={})).status_code == 400

        fast = _FakeAdapter()
        _use_adapter(monkeypatch, fast)
        resume_resp = await client.post(f"/api/runs/{run_id}/resume", json={"base_url": _BASE_URL})
        assert resume_resp.status_code == 202
        resumed = resume_resp.json()
        assert resumed["preserved_scenarios"] == 1
        assert resumed["remaining_scenarios"] == 1

        resumed_session = app.state.live_hub.get(run_id)
        preserved = [e for e in resumed_session.history if e.get("preserved")]
        assert [e["scenario_id"] for e in preserved] == ["TC-01"]
        assert "raw_log" not in preserved[0]["result"]

        # A second resume while the first is running is refused.
        again = await client.post(f"/api/runs/{run_id}/resume", json={"base_url": _BASE_URL})
        assert again.status_code == 409

        await resumed_session.task
        assert fast.calls == 1
        final_run = (await client.get(f"/api/runs/{run_id}")).json()
        assert final_run["status"] == "completed"
        assert [r["scenario_id"] for r in final_run["scores"]["scenario_results"]] == [
            "TC-01",
            "TC-02",
        ]

        # Completed runs can neither be stopped nor resumed.
        assert (await client.post(f"/api/runs/{run_id}/stop", json={})).status_code == 409
        completed_resume = await client.post(
            f"/api/runs/{run_id}/resume", json={"base_url": _BASE_URL}
        )
        assert completed_resume.status_code == 409
        with RunRepository(db_path=launcher_env["db_path"]) as repo:
            assert repo.get(run_id)["status"] == "completed"


@pytest.mark.asyncio
async def test_stop_refuses_runs_owned_by_another_process(
    launcher_env: dict[str, str],
) -> None:
    _store_run(launcher_env["db_path"], "cli-running", status="running")
    app = create_app(token=TEST_TOKEN)
    async with web_client(app) as client:
        resp = await client.post("/api/runs/cli-running/stop", json={})
        assert resp.status_code == 409
        assert "another process" in resp.json()["error"]
        assert (await client.post("/api/runs/nope/stop", json={})).status_code == 404
    with RunRepository(db_path=launcher_env["db_path"]) as repo:
        assert repo.get("cli-running")["status"] == "running"


@pytest.mark.asyncio
async def test_resume_refuses_running_rows_and_mismatched_endpoints(
    launcher_env: dict[str, str],
) -> None:
    _store_run(launcher_env["db_path"], "cli-running", status="running")
    _store_run(launcher_env["db_path"], "cli-stopped", status="interrupted")
    app = create_app(token=TEST_TOKEN)
    async with web_client(app) as client:
        running = await client.post("/api/runs/cli-running/resume", json={"base_url": _BASE_URL})
        assert running.status_code == 409
        assert "tool-eval-bench resume cli-running" in running.json()["error"]

        moved = await client.post(
            "/api/runs/cli-stopped/resume", json={"base_url": "http://other-host:9000"}
        )
        assert moved.status_code == 400
        assert "endpoint_id" in moved.json()["error"]
        assert app.state.live_hub.get("cli-stopped") is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("overrides", "named"),
    [
        (
            {"context_pressure_config": {"ratio": 0.5, "fill_tokens_target": 4000}},
            "context_pressure",
        ),
        ({"decision_judge": {"set": "core", "checks": ["TC-89"], "model": "j"}}, "decision_judge"),
    ],
)
async def test_resume_refuses_conditions_only_the_cli_can_rebuild(
    launcher_env: dict[str, str], overrides: dict[str, Any], named: str
) -> None:
    _store_run(launcher_env["db_path"], "cli-special", status="interrupted", **overrides)
    app = create_app(token=TEST_TOKEN)
    async with web_client(app) as client:
        resp = await client.post("/api/runs/cli-special/resume", json={"base_url": _BASE_URL})
    assert resp.status_code == 400
    assert named in resp.json()["error"]
    assert "CLI" in resp.json()["error"]


def test_numeric_defaults_have_the_declared_types() -> None:
    """Omitted fields fall back to defaults of the declared type.

    _number once returned an int default as-is, and _integer then called
    int.is_integer(), which only exists from Python 3.12. Checking the types
    catches that on every interpreter, not only on 3.11.
    """
    from tool_eval_bench.web.launcher import _integer, _number

    assert type(_number({}, "temperature", 0)) is float
    assert type(_integer({}, "max_turns", 8, minimum=1)) is int
    assert _integer({"max_turns": ""}, "max_turns", 8) == 8
