"""Tests for the web UI schema, endpoint probe, run launcher, stop, and resume endpoints."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from tests.web_support import TEST_TOKEN, web_client
from tool_eval_bench.application.run_config import RunSettings, build_run_config
from tool_eval_bench.application.run_queries import get_run
from tool_eval_bench.domain.adapters import ChatCompletionResult
from tool_eval_bench.evals.scenarios import ALL_SCENARIOS
from tool_eval_bench.storage.db import RunRepository
from tool_eval_bench.web import create_app
from tool_eval_bench.web.launcher import _fetch_models_json

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


@pytest.mark.asyncio
async def test_web_model_fallback_preserves_query_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested: list[httpx.URL] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requested.append(request.url)
        status = 404 if request.url.path == "/proxy/v1/models" else 200
        return httpx.Response(status, request=request, json={"data": []})

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    monkeypatch.setattr("tool_eval_bench.web.launcher.httpx.AsyncClient", lambda **kwargs: client)
    try:
        body = await _fetch_models_json(
            "https://proxy.example/proxy/v1?tenant=research",
            "https://proxy.example/proxy/v1/models?tenant=research",
            {},
            "openai",
        )
    finally:
        await client.aclose()

    assert body == {"data": []}
    assert [(url.path, url.query) for url in requested] == [
        ("/proxy/v1/models", b"tenant=research"),
        ("/proxy/models", b"tenant=research"),
    ]


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
                {"id": "qwen3-8b:fast", "root": "Qwen/Qwen3-8B-FP8"},
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
        assert any(p["id"] == "hardmode" and p["count"] == 104 for p in schema_data["presets"])
        assert len(schema_data["categories"]) == 16

        probe_resp = await client.post("/api/probe", json={"base_url": _BASE_URL})
        assert probe_resp.status_code == 200
        probe_data = probe_resp.json()
        assert probe_data["reachable"] is True
        assert probe_data["resolved_base_url"] == _BASE_URL
        assert probe_data["models"] == [
            {"id": "qwen3-8b", "display_name": "qwen3-8b"},
            {"id": "qwen3-8b:fast", "display_name": "qwen3-8b:fast"},
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
async def test_model_picker_uses_a_select_instead_of_a_datalist(
    launcher_env: dict[str, str],
) -> None:
    """A datalist filters its options by the input's value, hiding probed models.

    The launcher bound the model input to a <datalist> and auto-filled the
    first probed model, so browsers filtered every other entry out of the list
    and the models looked unselectable. The picker must stay a <select> fed by
    /api/probe, with the free-text field only behind the custom entry.
    """
    app = create_app(output_dir=launcher_env["output_dir"], token=TEST_TOKEN)
    async with web_client(app) as client:
        script = (await client.get("/static/live.js")).text
    assert "launch-model-select" in script
    assert "probed-models-datalist" not in script


@pytest.mark.asyncio
async def test_launcher_form_exposes_the_seed_field(launcher_env: dict[str, str]) -> None:
    """The Launch & Live form omitted the sampling seed the launcher accepts."""
    app = create_app(output_dir=launcher_env["output_dir"], token=TEST_TOKEN)
    async with web_client(app) as client:
        script = (await client.get("/static/live.js")).text
    assert 'input("launch-seed"' in script
    assert "function readSeedField(" in script
    submit_block = script.split("async function submitLaunch(", maxsplit=1)[1].split(
        "function turnRow(", maxsplit=1
    )[0]
    assert "seed: seed.value" in submit_block


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("seed", "expected"),
    [
        (None, None),  # no seed key at all
        ("", None),  # cleared field
        (0, 0),  # 0 is a seed, not "unset"
        (1234, 1234),
    ],
)
async def test_launch_stores_the_requested_seed(
    launcher_env: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    seed: Any,
    expected: int | None,
) -> None:
    _use_adapter(monkeypatch, _FakeAdapter())
    app = create_app(output_dir=launcher_env["output_dir"], token=TEST_TOKEN)
    payload = dict(_LAUNCH) if seed is None else {**_LAUNCH, "seed": seed}
    async with web_client(app) as client:
        resp = await client.post("/api/runs", json=payload)
        assert resp.status_code == 202
        run_id = resp.json()["run_id"]
        session = app.state.live_hub.get(run_id)
        assert session is not None and session.task is not None
        await session.task
    stored = await asyncio.to_thread(get_run, run_id, include_traces=False)
    assert stored is not None
    assert stored["config"]["seed"] == expected


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
        ("concurrency", 33, "concurrency must be at most 32"),
        ("concurrency", 32, None),  # upper bound is inclusive
        ("seed", [1], "seed must be a number"),
        ("seed", 2.5, "seed must be a whole number"),
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


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["system_prompt", "label", "api_key"])
async def test_launch_rejects_non_string_optional_text_fields(
    launcher_env: dict[str, str], field: str
) -> None:
    app = create_app(output_dir=launcher_env["output_dir"], token=TEST_TOKEN)
    async with web_client(app) as client:
        resp = await client.post("/api/runs", json={**_LAUNCH, field: 123})
    assert resp.status_code == 400
    assert f"{field} must be a string" in resp.json()["error"]


# ---------------------------------------------------------------------------
# Reviewer
# ---------------------------------------------------------------------------


class _ApprovingReviewer:
    """A reviewer endpoint that approves everything and records its requests."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    async def chat_completion(self, **kwargs: Any) -> ChatCompletionResult:
        self.requests.append(kwargs)
        return ChatCompletionResult(content='{"verdict": "approve", "feedback": ""}')

    async def aclose(self) -> None:
        pass


def _use_reviewer(monkeypatch: pytest.MonkeyPatch) -> _ApprovingReviewer:
    reviewer = _ApprovingReviewer()
    monkeypatch.setattr(
        "tool_eval_bench.application.service.build_adapter", lambda *a, **k: reviewer
    )
    return reviewer


_PAIRED = {
    **_LAUNCH,
    "api_key": "main-key",
    "reviewer_base_url": "http://critic.test:8001",
    "reviewer_model": "critic",
    "reviewer_scope": "final",
    "reviewer_max_rounds": 2,
    "reviewer_temperature": 0.1,
}


@pytest.mark.asyncio
async def test_launch_with_a_reviewer_runs_and_stores_the_pair(
    launcher_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_adapter(monkeypatch, _FakeAdapter())
    reviewer = _use_reviewer(monkeypatch)
    app = create_app(output_dir=launcher_env["output_dir"], token=TEST_TOKEN)
    async with web_client(app) as client:
        resp = await client.post("/api/runs", json={**_PAIRED, "reviewer_api_key": "critic-key"})
        assert resp.status_code == 202, resp.text
        body = resp.json()
        session = app.state.live_hub.get(body["run_id"])
        assert session is not None and session.task is not None
        assert session.reviewer_model == "critic"
        await session.task
    assert body["reviewer_model"] == "critic"
    stored = await asyncio.to_thread(get_run, body["run_id"], include_traces=False)
    assert stored is not None and stored["status"] == "completed"
    block = stored["config"]["reviewer"]
    assert (block["model"], block["scope"], block["max_rounds"], block["temperature"]) == (
        "critic",
        "final",
        2,
        0.1,
    )
    # The startup review, then one review per final reply.
    assert len(reviewer.requests) >= 2
    assert {request["api_key"] for request in reviewer.requests} == {"critic-key"}
    assert stored["scores"]["scenario_results"][0]["review"]["reviews"] >= 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reviewer_url", "expected_key"),
    [("http://critic.test:8001", None), (f"{_BASE_URL}/v1", "main-key"), (None, "main-key")],
    ids=["other-origin", "same-origin", "same-server"],
)
async def test_the_run_key_reaches_a_web_reviewer_on_the_same_server_only(
    launcher_env: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    reviewer_url: str | None,
    expected_key: str | None,
) -> None:
    _use_adapter(monkeypatch, _FakeAdapter())
    reviewer = _use_reviewer(monkeypatch)
    payload = {**_PAIRED, "reviewer_base_url": reviewer_url}
    app = create_app(output_dir=launcher_env["output_dir"], token=TEST_TOKEN)
    async with web_client(app) as client:
        resp = await client.post("/api/runs", json=payload)
        assert resp.status_code == 202, resp.text
        session = app.state.live_hub.get(resp.json()["run_id"])
        assert session is not None and session.task is not None
        await session.task
    assert {request["api_key"] for request in reviewer.requests} == {expected_key}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"reviewer_modle": "critic"}, "Unknown reviewer field"),
        ({"reviewer_model": None, "reviewer_scope": "tools"}, "reviewer_model is required"),
        ({"reviewer_max_rounds": 0}, "reviewer_max_rounds"),
        ({"reviewer_scope": "everything"}, "scope"),
        ({"reviewer_base_url": "ftp://critic"}, "reviewer_base_url"),
    ],
)
async def test_launch_rejects_bad_reviewer_fields_with_400(
    launcher_env: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    extra: dict[str, Any],
    message: str,
) -> None:
    _use_adapter(monkeypatch, _FakeAdapter())
    app = create_app(output_dir=launcher_env["output_dir"], token=TEST_TOKEN)
    payload = {key: value for key, value in {**_PAIRED, **extra}.items() if value is not None}
    async with web_client(app) as client:
        resp = await client.post("/api/runs", json=payload)
    assert resp.status_code == 400
    assert message in resp.text
    assert not app.state.live_hub._sessions  # nothing was registered


@pytest.mark.asyncio
async def test_a_reviewer_probe_never_sends_the_run_key(
    launcher_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from tool_eval_bench.web.launcher import probe_endpoint_models

    monkeypatch.setenv("TOOL_EVAL_API_KEY", "main-env-key")
    seen: list[dict[str, str]] = []

    async def fake_fetch(url: str, endpoint: str, headers: dict[str, str], fmt: str) -> Any:
        seen.append(dict(headers))
        return {"data": [{"id": "critic"}]}

    monkeypatch.setattr("tool_eval_bench.web.launcher._fetch_models_json", fake_fetch)

    found = await probe_endpoint_models(base_url="http://critic.test:8001", reviewer=True)
    missing = await probe_endpoint_models(base_url=None, reviewer=True)

    assert [m["id"] for m in found["models"]] == ["critic"]
    assert "main-env-key" not in str(seen)
    assert missing["reachable"] is False and "reviewer" in missing["error"]


@pytest.mark.asyncio
async def test_resume_a_reviewed_run_from_the_web(
    launcher_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from tool_eval_bench.application.reviewer import resolve_reviewer, reviewer_config
    from tool_eval_bench.domain.review import ReviewerSettings

    block = reviewer_config(
        resolve_reviewer(
            ReviewerSettings(model="critic", base_url="http://critic.test:8001"),
            base_url=_BASE_URL,
            api_key=None,
            wire_format=None,
            extra_headers=None,
            session_header=None,
            temperature=0.0,
            timeout_seconds=120.0,
            seed=None,
            extra_params=None,
        )
    )
    _store_run(launcher_env["db_path"], "run-paired", status="interrupted", reviewer=block)
    _use_adapter(monkeypatch, _FakeAdapter())
    reviewer = _use_reviewer(monkeypatch)
    app = create_app(output_dir=launcher_env["output_dir"], token=TEST_TOKEN)
    async with web_client(app) as client:
        refused = await client.post("/api/runs/run-paired/resume", json={"base_url": _BASE_URL})
        resumed = await client.post(
            "/api/runs/run-paired/resume",
            json={
                "base_url": _BASE_URL,
                "reviewer_base_url": "http://critic.test:8001",
                "reviewer_api_key": "critic-key",
            },
        )
        assert resumed.status_code == 202, resumed.text
        session = app.state.live_hub.get("run-paired")
        assert session is not None and session.task is not None
        await session.task
    assert refused.status_code == 400 and "reviewer" in refused.text
    assert resumed.json()["reviewer_model"] == "critic"
    assert {request["api_key"] for request in reviewer.requests} == {"critic-key"}
    stored = await asyncio.to_thread(get_run, "run-paired", include_traces=False)
    assert stored is not None and stored["status"] == "completed"
    assert stored["config"]["reviewer"]["model"] == "critic"


@pytest.mark.asyncio
async def test_the_stored_run_stream_names_the_reviewer(
    launcher_env: dict[str, str],
) -> None:
    from tool_eval_bench.web.live import _tail_stored_run

    _store_run(
        launcher_env["db_path"],
        "run-stream",
        status="completed",
        reviewer={"model": "critic", "endpoint_id": "e", "scope": "turns"},
    )
    first = None
    async for frame in _tail_stored_run(
        "run-stream", poll_interval=0.01, heartbeat_seconds=10.0, idle_timeout=1.0, max_polls=1
    ):
        first = frame
        break
    assert first is not None and '"reviewer_model": "critic"' in first


def test_the_launcher_page_never_carries_a_reviewer_key_to_another_server() -> None:
    """Static guard: the key is cleared when the reviewer URL changes origin.

    The browser behaviour was checked live; this pins the wiring so a refactor
    cannot drop the check from probe, launch, or resume.
    """
    from pathlib import Path

    static_dir = Path(__file__).resolve().parents[1] / "src" / "tool_eval_bench" / "web" / "static"
    live_js = (static_dir / "live.js").read_text(encoding="utf-8")
    leaderboard_js = (static_dir / "leaderboard.js").read_text(encoding="utf-8")

    assert "onchange: forgetReviewerKeyOnNewOrigin" in live_js
    assert "oninput: bindReviewerKeyToOrigin" in live_js
    # A blank reviewer URL means the benchmarked server, so the binding uses
    # the effective URL, falling back to the benchmark Base URL.
    effective = live_js.split("function effectiveReviewerOrigin(", maxsplit=1)[1].split(
        "\n}\n", maxsplit=1
    )[0]
    assert 'byId("launch-base-url")' in effective
    # The reviewer probe sends the run's key only to the run's own origin.
    probe = live_js.split("async function probeReviewer(", maxsplit=1)[1].split(
        "\n}\n", maxsplit=1
    )[0]
    assert "ownKey || (sameOrigin ? runKey" in probe
    # A blank reviewer URL probes the benchmark server with its wire format.
    assert 'wire_format: ownUrl ? undefined : byId("launch-wire-format")' in probe
    # A model the probed server does not list belonged to the previous server:
    # the first listed model replaces it, and the status names the swap.
    assert "!models.some((m) => m.id === current)" in probe
    assert "modelInput.value = models[0].id;" in probe
    assert "is not listed here, now" in probe
    # Leaderboard rows name the reviewer of a paired score.
    assert "row.reviewer_model ? h(" in leaderboard_js
    for function in ("function bindReviewerKeyToOrigin(", "function forgetReviewerKeyOnNewOrigin("):
        body = live_js.split(function, maxsplit=1)[1].split("\n}\n", maxsplit=1)[0]
        assert "effectiveReviewerOrigin()" in body, function
    for function in (
        "async function probeReviewer(",
        "function readReviewerFields(",
        "async function resumeInterruptedRun(",
    ):
        body = live_js.split(function, maxsplit=1)[1].split("\n}\n", maxsplit=1)[0]
        assert "forgetReviewerKeyOnNewOrigin()" in body, function


@pytest.mark.asyncio
async def test_a_reviewer_probe_keeps_url_credentials_out_of_logs_and_answers(
    launcher_env: dict[str, str], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    from tool_eval_bench.web.launcher import probe_endpoint_models

    url = "https://critic.example/route/PATHTOKEN88/v1?key=QUERYTOKEN88"

    async def failing_fetch(base: str, endpoint: str, headers: dict[str, str], fmt: str) -> Any:
        logging.getLogger("httpx").info("HTTP Request: GET %s", endpoint)
        raise httpx.ConnectError(f"cannot reach {endpoint}")

    monkeypatch.setattr("tool_eval_bench.web.launcher._fetch_models_json", failing_fetch)

    with caplog.at_level("INFO"):
        result = await probe_endpoint_models(base_url=url, reviewer=True)

    for token in ("PATHTOKEN88", "QUERYTOKEN88"):
        assert token not in caplog.text
        assert token not in str(result)
    assert result["reachable"] is False
    assert result["base_url"] == "https://***"
    # The filter is gone once the probe returns.
    assert not any(logging.getLogger("httpx").filters)


@pytest.mark.asyncio
async def test_a_reviewer_probe_answers_a_malformed_url_without_raising(
    launcher_env: dict[str, str],
) -> None:
    from tool_eval_bench.web.launcher import probe_endpoint_models

    result = await probe_endpoint_models(base_url="http://[::1", reviewer=True)

    assert result == {
        "reachable": False,
        "error": "The reviewer Base URL is not a valid http(s) URL.",
        "models": [],
    }


@pytest.mark.asyncio
async def test_probe_endpoint_models_env_and_auto_discovery_fallbacks(
    launcher_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from tool_eval_bench.web.launcher import probe_endpoint_models

    probed_urls: list[str] = []

    async def fake_fetch(url: str, endpoint: str, headers: dict[str, str], fmt: str) -> Any:
        probed_urls.append(url)
        return {"data": ["skip-me", {"id": ""}, {"id": "discovered-model"}]}

    monkeypatch.setattr("tool_eval_bench.web.launcher._fetch_models_json", fake_fetch)

    # 1. TOOL_EVAL_BASE_URL fallback when base_url is omitted.
    monkeypatch.setenv("TOOL_EVAL_BASE_URL", "http://127.0.0.1:8123")
    env_res = await probe_endpoint_models(base_url=None)
    assert env_res["reachable"] is True
    assert env_res["resolved_base_url"] == "http://127.0.0.1:8123"
    assert env_res["models"] == [{"id": "discovered-model", "display_name": "discovered-model"}]

    # 2. Local auto-discovery fallback when neither base_url nor env var is set.
    monkeypatch.delenv("TOOL_EVAL_BASE_URL", raising=False)

    async def fake_discover_found() -> tuple[str, str, str]:
        return ("http://127.0.0.1:8000", "vllm", "vLLM")

    monkeypatch.setattr("tool_eval_bench.web.launcher._discover_async", fake_discover_found)
    auto_res = await probe_endpoint_models(base_url=None)
    assert auto_res["reachable"] is True
    assert auto_res["resolved_base_url"] == "http://127.0.0.1:8000"

    # 3. Empty JSON body on POST /api/probe falls through to discovery; None -> unreachable.
    async def fake_discover_none() -> None:
        return None

    monkeypatch.setattr("tool_eval_bench.web.launcher._discover_async", fake_discover_none)
    app = create_app(output_dir=launcher_env["output_dir"], token=TEST_TOKEN)
    async with web_client(app) as client:
        empty_probe = await client.post(
            "/api/probe", content=b"", headers={"Content-Type": "application/json"}
        )
    assert empty_probe.status_code == 200
    assert empty_probe.json()["reachable"] is False
    assert "No inference server found" in empty_probe.json()["error"]


@pytest.mark.asyncio
async def test_stop_web_run_handles_completed_race_running_after_cancel_and_taskless_session(
    launcher_env: dict[str, str],
) -> None:
    from tool_eval_bench.web.launcher import RunConflict, stop_web_run
    from tool_eval_bench.web.live import LiveRunHub

    hub = LiveRunHub()

    # Taskless session absent from storage raises RunConflict with session status.
    taskless = hub.register("taskless-done", model="m", backend="vllm", total=1)
    taskless.status = "completed"
    with pytest.raises(RunConflict, match="is completed, not running"):
        await stop_web_run(hub, "taskless-done")

    # Live task that ignores cancel and finishes with status="completed" raises RunConflict.
    raced = hub.register("race-complete", model="m", backend="vllm", total=1)
    started = asyncio.Event()

    async def _finish_on_cancel() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raced.status = "completed"
            return

    raced.task = asyncio.create_task(_finish_on_cancel())
    await started.wait()
    with pytest.raises(RunConflict, match="is completed, not running"):
        await stop_web_run(hub, "race-complete")

    # Live task that exits on cancel without updating session.status gets marked interrupted.
    still_running = hub.register("cancel-no-hook", model="m", backend="vllm", total=1)
    started2 = asyncio.Event()

    async def _swallow_cancel() -> None:
        started2.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return

    still_running.task = asyncio.create_task(_swallow_cancel())
    await started2.wait()
    stopped = await stop_web_run(hub, "cancel-no-hook")
    assert stopped == {"run_id": "cancel-no-hook", "status": "interrupted"}
    assert still_running.status == "interrupted"


@pytest.mark.asyncio
async def test_resume_web_run_guards_and_prior_session_id_inheritance(
    launcher_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from tool_eval_bench.web.launcher import (
        RunConflict,
        _resolve_launcher_scenarios,
        launch_web_run,
        resume_web_run,
    )
    from tool_eval_bench.web.live import LiveRunHub

    app = create_app(output_dir=launcher_env["output_dir"], token=TEST_TOKEN)

    # Missing model rejected cleanly; comma-separated scenarios and empty filters validated.
    with pytest.raises(ValueError, match="model is required"):
        await launch_web_run(
            LiveRunHub(), {"base_url": _BASE_URL}, output_dir=launcher_env["output_dir"]
        )
    resolved, _, _ = _resolve_launcher_scenarios({"preset": "custom", "scenarios": "TC-01, TC-02"})
    assert [sc.id for sc in resolved] == ["TC-01", "TC-02"]
    with pytest.raises(ValueError, match="No scenarios matched"):
        _resolve_launcher_scenarios({"preset": "custom", "categories": ["ZZ"]})

    # Seed non-interrupted, non-tool_eval, and session-tagged interrupted runs.
    _store_run(launcher_env["db_path"], "run-completed", status="completed")
    with RunRepository(db_path=launcher_env["db_path"]) as repo:
        repo.upsert_scenario_run(
            {
                "run_id": "run-perf-interrupted",
                "status": "interrupted",
                "run_type": "perf",
                "model": "Qwen/Qwen3-8B",
                "config": {"model": "Qwen/Qwen3-8B", "backend": "vllm"},
                "metadata": {},
                "scores": {},
            }
        )
    _store_run(launcher_env["db_path"], "run-with-session", status="interrupted")
    with RunRepository(db_path=launcher_env["db_path"]) as repo:
        row = repo.get("run-with-session")
        assert row is not None
        row["metadata"]["session_id"] = "sess-inherited-42"
        repo.upsert_scenario_run(row)

    async with web_client(app) as client:
        # Missing run -> 404
        assert (
            await client.post("/api/runs/does-not-exist/resume", json={"base_url": _BASE_URL})
        ).status_code == 404
        # Non-interrupted run in DB (no live session in hub) -> 409
        non_int = await client.post("/api/runs/run-completed/resume", json={"base_url": _BASE_URL})
        assert non_int.status_code == 409
        assert "Only interrupted runs can be resumed" in non_int.json()["error"]
        # Non-tool_eval interrupted run -> 400
        non_tool = await client.post(
            "/api/runs/run-perf-interrupted/resume", json={"base_url": _BASE_URL}
        )
        assert non_tool.status_code == 400
        assert "Only tool_eval runs can be resumed" in non_tool.json()["error"]

    # Already-live session raises RunConflict before touching storage.
    hub = LiveRunHub()
    live = hub.register("already-live", model="m", backend="vllm", total=1)
    live.task = asyncio.create_task(asyncio.sleep(10))
    try:
        with pytest.raises(RunConflict, match="already running"):
            await resume_web_run(hub, "already-live", {"base_url": _BASE_URL})
    finally:
        live.task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await live.task

    # Resuming a run with a prior session_id preserves it on the stored row.
    _use_adapter(monkeypatch, _FakeAdapter())
    async with web_client(app) as client:
        resumed = await client.post(
            "/api/runs/run-with-session/resume", json={"base_url": _BASE_URL}
        )
        assert resumed.status_code == 202, resumed.text
        session = app.state.live_hub.get("run-with-session")
        assert session is not None and session.task is not None
        await session.task

    stored = await asyncio.to_thread(get_run, "run-with-session", include_traces=False)
    assert stored is not None
    assert stored["status"] == "completed"
    assert stored["session_id"] == "sess-inherited-42"


@pytest.mark.asyncio
async def test_open_service_async_closes_connection_when_cancelled_during_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import contextlib
    import threading

    from tool_eval_bench.web.launcher import _open_service_async

    entered = threading.Event()
    release = threading.Event()
    closed = False

    @contextlib.contextmanager
    def _slow_open(output_dir: str | None = None) -> Any:
        nonlocal closed
        entered.set()
        release.wait(timeout=5.0)
        try:
            yield object()
        finally:
            closed = True

    monkeypatch.setattr("tool_eval_bench.web.launcher.open_benchmark_service", _slow_open)

    async def _runner() -> None:
        async with _open_service_async(None):
            pass

    task = asyncio.create_task(_runner())
    await asyncio.to_thread(entered.wait, 5.0)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed is True
