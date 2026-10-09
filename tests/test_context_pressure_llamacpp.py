"""Context pressure sizes itself from the window llama.cpp or Strata reports.

The run-context probe already reads ``/props`` ``default_generation_settings.n_ctx``
(and Strata's ``/health`` ``max_context``); these tests drive that probe against a
mocked server and feed its result to the pressure detection, so they cover the
path a real run takes.
"""

from __future__ import annotations

import io
from typing import Any

import httpx
import pytest
from rich.console import Console

from tool_eval_bench.domain.models import RunContext
from tool_eval_bench.runner.context_pressure import (
    compute_fill_budget,
    prepare_context_pressure,
    reported_context_window,
)
from tool_eval_bench.utils import metadata

_MISSING = object()

# Trimmed from llama-server b11478 serving gemma4. The listing carries the
# window only inside "meta", where detect_context_size does not look, so before
# this change llama.cpp always needed --context-size.
LLAMACPP_MODELS: dict[str, Any] = {
    "object": "list",
    "data": [
        {
            "id": "gemma4",
            "object": "model",
            "owned_by": "llamacpp",
            "meta": {"n_ctx": 81920, "n_ctx_train": 131072},
        }
    ],
}


def _props(n_ctx: object, total_slots: int) -> dict[str, Any]:
    settings: dict[str, Any] = {"params": {"temperature": 1.0}}
    if n_ctx is not _MISSING:
        settings["n_ctx"] = n_ctx
    return {
        "default_generation_settings": settings,
        "total_slots": total_slots,
        "build_info": "b11478-18b5f8b18",
    }


async def _probed_run_context(
    monkeypatch: pytest.MonkeyPatch,
    routes: dict[str, Any],
    *,
    backend: str = "unknown",
) -> RunContext:
    """Run the real metadata probe against a mocked server."""
    real_client = httpx.AsyncClient

    def respond(request: httpx.Request) -> httpx.Response:
        body = routes.get(request.url.path)
        if body is None:
            return httpx.Response(404)
        if isinstance(body, httpx.Response):
            return body
        return httpx.Response(200, json=body)

    monkeypatch.setattr(
        metadata.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(respond), **kwargs),
    )
    return await metadata.collect_run_context(
        model="gemma4", backend=backend, base_url="http://test/v1"
    )


def _run_context(engine_name: str | None, window: int | None) -> RunContext:
    return RunContext(
        tool_version="test",
        git_sha=None,
        hostname="host",
        platform_info="linux",
        python_version="3.13",
        model="gemma4",
        backend="unknown",
        base_url="http://test/v1",
        engine_name=engine_name,
        max_model_len=window,
    )


class _FakeMeasurementClient:
    """The measurement endpoints context pressure reads, served from fixtures."""

    def __init__(
        self,
        models_body: dict[str, Any] | None = None,
        *,
        models_error: Exception | None = None,
        metrics_text: str | None = None,
    ) -> None:
        self.models_body = models_body if models_body is not None else LLAMACPP_MODELS
        self.models_error = models_error
        self.metrics_text = metrics_text
        self.models_calls = 0

    async def __aenter__(self) -> _FakeMeasurementClient:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def models(self) -> httpx.Response:
        self.models_calls += 1
        if self.models_error is not None:
            raise self.models_error
        request = httpx.Request("GET", "http://test/v1/models")
        return httpx.Response(200, json=self.models_body, request=request)

    async def metrics(self, *, metrics_url: str | None = None) -> httpx.Response:
        if self.metrics_text is None:
            return httpx.Response(404)
        return httpx.Response(200, text=self.metrics_text)

    def factory(self, **kwargs: object) -> _FakeMeasurementClient:
        return self


async def _prepare(
    client: _FakeMeasurementClient,
    reported: int | None,
    *,
    override: int | None = None,
    ratio: float = 0.5,
) -> Any:
    return await prepare_context_pressure(
        "http://test/v1",
        "gemma4",
        None,
        ratio=ratio,
        context_size_override=override,
        client_factory=client.factory,
        reported_context=reported,
    )


# ---------------------------------------------------------------------------
# Detection from llama.cpp /props
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("total_slots", "n_ctx"),
    [
        pytest.param(1, 32768, id="single-slot"),
        # n_ctx is already per slot upstream. Multiplying by total_slots would
        # overfill a request by 4x, and dividing would under-pressure it.
        pytest.param(4, 20480, id="multi-slot"),
    ],
)
async def test_props_window_sizes_the_pressure_run(
    monkeypatch: pytest.MonkeyPatch, total_slots: int, n_ctx: int
) -> None:
    run_context = await _probed_run_context(
        monkeypatch, {"/v1/models": LLAMACPP_MODELS, "/props": _props(n_ctx, total_slots)}
    )
    assert run_context.engine_name == "llama.cpp"
    assert run_context.slot_count == total_slots

    cfg = await _prepare(_FakeMeasurementClient(), reported_context_window(run_context))

    assert cfg.detected_context == n_ctx
    assert cfg.fill_tokens == compute_fill_budget(n_ctx, 0.5)


@pytest.mark.parametrize(
    "n_ctx",
    [
        pytest.param(_MISSING, id="missing"),
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative"),
        pytest.param(True, id="bool"),
        pytest.param("81920", id="string"),
    ],
)
async def test_unusable_props_window_still_asks_for_context_size(
    monkeypatch: pytest.MonkeyPatch, n_ctx: object
) -> None:
    run_context = await _probed_run_context(
        monkeypatch, {"/v1/models": LLAMACPP_MODELS, "/props": _props(n_ctx, 2)}
    )
    assert run_context.engine_name == "llama.cpp"
    reported = reported_context_window(run_context)
    assert reported is None

    # The listing's meta.n_ctx and n_ctx_train are not a fallback either.
    with pytest.raises(ValueError, match="--context-size"):
        await _prepare(_FakeMeasurementClient(), reported)


async def test_explicit_context_size_wins_without_probing() -> None:
    client = _FakeMeasurementClient()

    cfg = await _prepare(client, 81920, override=8192)

    assert cfg.detected_context == 8192
    assert client.models_calls == 0


async def test_model_listing_window_keeps_precedence() -> None:
    listing = {"data": [{"id": "gemma4", "max_model_len": 4096}]}

    cfg = await _prepare(_FakeMeasurementClient(listing), 81920)

    assert cfg.detected_context == 4096


async def test_unreachable_listing_falls_back_to_reported_window() -> None:
    client = _FakeMeasurementClient(models_error=httpx.ConnectError("refused"))

    cfg = await _prepare(client, 81920)

    assert cfg.detected_context == 81920


async def test_kv_capacity_still_caps_the_reported_window() -> None:
    metrics = 'vllm:cache_config_info{block_size="16",num_gpu_blocks="1000"} 1.0\n'

    cfg = await _prepare(_FakeMeasurementClient(metrics_text=metrics), 81920)

    assert cfg.detected_context == 16000


# ---------------------------------------------------------------------------
# Other engines
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("engine_name", ["vLLM", "TabbyAPI", "TensorFold", None])
def test_other_engines_contribute_no_reported_window(engine_name: str | None) -> None:
    assert reported_context_window(_run_context(engine_name, 81920)) is None


def test_missing_run_context_contributes_no_reported_window() -> None:
    assert reported_context_window(None) is None


# ---------------------------------------------------------------------------
# Strata
# ---------------------------------------------------------------------------

# Shapes from Strata fb58e0d serve/server.py. The listing names no owner and
# keeps the window in "meta", and /props and /health report the same
# Service.reported_ctx() value that /metrics exports as strata:engine_max_context.
STRATA_MODELS: dict[str, Any] = {
    "object": "list",
    "data": [
        {
            "id": "gemma4",
            "object": "model",
            "status": {"value": "loaded"},
            "meta": {"n_ctx": 65536},
            "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
        }
    ],
}
STRATA_HEALTH: dict[str, Any] = {
    "status": "ok",
    "max_context": 65536,
    "model": "gemma4",
    "images": False,
    "api_key": False,
    "loaded": True,
    "service": "strata",
}
# serve/prometheus.py: vLLM's names without vllm:cache_config_info, so no KV cap.
STRATA_METRICS = (
    "# HELP vllm:kv_cache_usage_perc The running request's share of the context (1 = full).\n"
    "# TYPE vllm:kv_cache_usage_perc gauge\n"
    'vllm:kv_cache_usage_perc{model_name="gemma4"} 0\n'
    "# HELP strata:engine_max_context The engine's context (engine.max_context).\n"
    "# TYPE strata:engine_max_context gauge\n"
    'strata:engine_max_context{model_name="gemma4"} 65536\n'
)


def _strata_props(n_ctx: int, total_slots: int) -> dict[str, Any]:
    return {
        "default_generation_settings": {"n_ctx": n_ctx, "params": {"temperature": 1.0}},
        "total_slots": total_slots,
        "model_alias": "gemma4",
        "build_info": "Strata 0.1.40.4",
    }


@pytest.mark.parametrize("total_slots", [1, 4])
async def test_strata_props_window_sizes_the_pressure_run(
    monkeypatch: pytest.MonkeyPatch, total_slots: int
) -> None:
    run_context = await _probed_run_context(
        monkeypatch,
        {
            "/v1/models": STRATA_MODELS,
            "/props": _strata_props(65536, total_slots),
            "/health": STRATA_HEALTH,
        },
    )
    assert run_context.engine_name == "Strata"
    assert run_context.max_model_len == 65536

    client = _FakeMeasurementClient(STRATA_MODELS, metrics_text=STRATA_METRICS)
    cfg = await _prepare(client, reported_context_window(run_context))

    # Every --batch slot reports the full window, which is the per-request limit.
    assert cfg.detected_context == 65536
    assert cfg.fill_tokens == compute_fill_budget(65536, 0.5)


async def test_strata_health_window_sizes_the_pressure_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stopped = httpx.Response(503, json={"error": {"message": "the engine is not running"}})
    run_context = await _probed_run_context(
        monkeypatch,
        {
            "/v1/models": STRATA_MODELS,
            "/props": stopped,
            "/health": {**STRATA_HEALTH, "max_context": 32768},
        },
        backend="strata",
    )
    assert run_context.engine_name == "Strata"

    cfg = await _prepare(
        _FakeMeasurementClient(STRATA_MODELS), reported_context_window(run_context)
    )

    assert cfg.detected_context == 32768


async def test_strata_without_a_window_still_asks_for_context_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Strata reports 0 until the engine is READY.
    run_context = await _probed_run_context(
        monkeypatch,
        {"/v1/models": STRATA_MODELS, "/health": {**STRATA_HEALTH, "max_context": 0}},
        backend="strata",
    )
    assert run_context.engine_name == "Strata"
    reported = reported_context_window(run_context)
    assert reported is None

    with pytest.raises(ValueError, match="--context-size"):
        await _prepare(_FakeMeasurementClient(STRATA_MODELS), reported)


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def test_single_run_passes_the_llamacpp_window(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    from tool_eval_bench.cli import dispatch, plugin_runners
    from tool_eval_bench.runner import context_pressure

    seen: dict[str, Any] = {}

    class Pressure:
        ratio = 0.1
        fill_tokens = 0
        detected_context = 81920

        def summary(self) -> str:
            return "10%"

        def budget_breakdown(self, **kwargs: Any) -> dict[str, int]:
            return {"remaining_headroom_tokens": 100}

    async def prepare(*args: Any, **kwargs: Any) -> Pressure:
        seen.update(kwargs)
        return Pressure()

    async def calibrate(messages: Any, *args: Any, **kwargs: Any) -> tuple[Any, int]:
        return messages, 0

    async def context(**kwargs: Any) -> RunContext:
        return _run_context("llama.cpp", 81920)

    monkeypatch.setattr(dispatch, "_load_dotenv", lambda: None)
    monkeypatch.setattr(dispatch, "_preflight_model_check", lambda *a, **k: None)
    monkeypatch.setattr(metadata, "collect_run_context", context)
    monkeypatch.setattr(plugin_runners, "run_selected_plugins", lambda *a, **k: False)
    monkeypatch.setattr(context_pressure, "prepare_context_pressure", prepare)
    monkeypatch.setattr(context_pressure, "build_pressure_messages", lambda *a, **k: [])
    monkeypatch.setattr(context_pressure, "calibrate_pressure_messages", calibrate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tool-eval-bench",
            "--model",
            "gemma4",
            "--base-url",
            "http://test/v1",
            "--context-pressure",
            "0.1",
            "--skip-tool-eval",
            "--no-warmup",
        ],
    )

    dispatch.main()

    assert seen["context_size_override"] is None
    assert seen["reported_context"] == 81920


def test_sweep_sizes_levels_from_the_llamacpp_window(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import argparse
    from unittest.mock import AsyncMock, patch

    from tool_eval_bench.adapters import factory, measurement
    from tool_eval_bench.cli.pressure import run_pressure_sweep
    from tool_eval_bench.domain.scenarios import (
        Category,
        ModelScoreSummary,
        ScenarioDefinition,
        ScenarioResult,
        ScenarioStatus,
    )
    from tool_eval_bench.runner import orchestrator

    class _ImmediateEventLoop:
        def run_until_complete(self, coroutine: Any) -> tuple[list[Any], int]:
            coroutine.close()
            return [], 0

        def close(self) -> None:
            pass

    scenario = ScenarioDefinition(
        id="TC-01",
        title="Test",
        category=Category.A,
        user_message="test",
        description="test",
        handle_tool_call=lambda s, c: {},
        evaluate=lambda s: None,
    )
    summary = ModelScoreSummary(
        scenario_results=[
            ScenarioResult(scenario_id="TC-01", status=ScenarioStatus.PASS, points=2, summary="ok")
        ],
        total_points=2,
        max_points=2,
        final_score=100.0,
        rating="test",
        category_scores=[],
        safety_warnings=[],
        total_tokens=0,
    )
    client = _FakeMeasurementClient()
    monkeypatch.setattr(measurement, "HTTPMeasurementClient", client.factory)
    monkeypatch.setattr(orchestrator, "run_all_scenarios", AsyncMock(return_value=summary))
    monkeypatch.setattr(factory, "build_adapter", lambda *a, **k: AsyncMock())

    args = argparse.Namespace(
        context_pressure_sweep="0.5-1.0",
        sweep_steps=2,
        scenarios=["TC-01"],
        short=False,
        categories=None,
        context_size=None,
        redact_url=False,
        seed=None,
        system_prompt=None,
        output_dir=str(tmp_path),
        timeout=60.0,
    )
    persisted: list[dict[str, Any]] = []
    out = io.StringIO()

    with patch("asyncio.new_event_loop", return_value=_ImmediateEventLoop()):
        monkeypatch.setattr(
            "tool_eval_bench.cli.pressure.resolve_scenarios", lambda _args: [scenario]
        )
        monkeypatch.setattr("tool_eval_bench.application.run_queries.persist_run", persisted.append)
        run_pressure_sweep(
            Console(file=out, force_terminal=False, width=200),
            "gemma4",
            "gemma4",
            "llamacpp",
            "http://test/v1",
            None,
            args,
            run_context=_run_context("llama.cpp", 81920),
        )

    assert "Context window: 81,920 tokens" in out.getvalue()
    assert [item["run_type"] for item in persisted] == ["context-pressure"]
