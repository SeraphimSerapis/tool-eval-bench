"""The Python API identifies the backend and records engine metadata like the CLI."""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from tool_eval_bench.api import run_benchmark
from tool_eval_bench.application import service as service_module
from tool_eval_bench.application.service import BenchmarkService
from tool_eval_bench.domain.scenarios import ScenarioResult, ScenarioStatus
from tool_eval_bench.evals.scenarios import ALL_SCENARIOS
from tool_eval_bench.runner.orchestrator import score_results
from tool_eval_bench.utils import metadata

# Captured once, so a test can call serve() twice without wrapping its own mock.
_REAL_CLIENT = httpx.AsyncClient
SCENARIO = next(s for s in ALL_SCENARIOS if s.id == "TC-01")

LLAMA_PROPS = {
    "build_info": "b11418-9871df591",
    "default_generation_settings": {"n_ctx": 8192},
    "total_slots": 2,
}

# path -> body. A str is served as text, anything else as JSON; other paths 404.
Routes = dict[str, Any]
SERVERS: dict[str, Routes] = {
    "vllm": {
        "/metrics": "vllm:num_requests_running 0\n",
        "/version": {"version": "0.8.5"},
        "/v1/models": {"data": [{"id": "m", "root": "m", "max_model_len": 32768}]},
    },
    "llamacpp": {
        "/v1/models": {"data": [{"id": "m", "owned_by": "llamacpp"}]},
        "/props": LLAMA_PROPS,
    },
    # Halogen exports llama.cpp's metric namespace too; its own must win.
    "halogen": {
        "/metrics": "llamacpp:prompt_tokens_total 0\nhalogen:requests_total 0\n",
        "/v1/models": {"data": [{"id": "m"}]},
    },
}


def serve(
    monkeypatch: pytest.MonkeyPatch,
    routes: Routes | Callable[[httpx.Request], httpx.Response],
) -> list[httpx.Request]:
    """Answer every probe from *routes* and return the requests it saw."""
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if callable(routes):
            return routes(request)
        body = routes.get(request.url.path)
        if body is None:
            return httpx.Response(404)
        if isinstance(body, str):
            return httpx.Response(200, text=body)
        return httpx.Response(200, json=body)

    monkeypatch.setattr(
        metadata.httpx,
        "AsyncClient",
        lambda **kw: _REAL_CLIENT(transport=httpx.MockTransport(respond), **kw),
    )
    return requests


@pytest.fixture(autouse=True)
def stub_scenarios(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the model conversation; detection and metadata stay real."""
    summary = score_results([ScenarioResult("TC-01", ScenarioStatus.PASS, 2, "ok")], [SCENARIO])
    monkeypatch.setattr(service_module, "run_all_scenarios", AsyncMock(return_value=summary))
    monkeypatch.setattr(BenchmarkService, "_adapter_for", lambda *a, **k: object())


async def _run(**kwargs: Any) -> dict[str, Any]:
    options: dict[str, Any] = {
        "model": "m",
        "base_url": "http://test",
        "scenarios": [SCENARIO],
        "persist": False,
    }
    return await run_benchmark(**{**options, **kwargs})


@pytest.mark.parametrize(
    ("server", "engine"),
    [
        ("vllm", {"engine_name": "vLLM", "engine_version": "0.8.5", "max_model_len": 32768}),
        ("llamacpp", {"engine_name": "llama.cpp", "max_model_len": 8192, "slot_count": 2}),
        ("halogen", {"engine_name": "Halogen Flash"}),
    ],
)
async def test_default_backend_is_detected_and_recorded(monkeypatch, server, engine):
    serve(monkeypatch, SERVERS[server])

    result = await _run()

    assert result["config"]["backend"] == server
    assert result["metadata"]["backend"] == server
    assert result["metadata"]["server_model_id"] == "m"
    for key, value in engine.items():
        assert result["metadata"][key] == value
    assert result["metadata"]["scenario_selector"] == "TC-01"


async def test_explicit_backend_is_never_replaced(monkeypatch):
    serve(monkeypatch, SERVERS["vllm"])
    hint = AsyncMock(return_value=("vllm", "vLLM"))
    monkeypatch.setattr(metadata, "probe_backend_hint", hint)

    result = await _run(backend="sglang")

    hint.assert_not_called()
    assert result["config"]["backend"] == "sglang"
    assert result["metadata"]["backend"] == "sglang"
    # Engine facts are still read for an explicit label, as the CLI does.
    assert result["metadata"]["engine_name"] == "SGLang"
    assert result["metadata"]["server_model_id"] == "m"
    assert result["metadata"]["max_model_len"] == 32768


@pytest.mark.parametrize("backend", ["", "Unknown"])
async def test_unset_spellings_are_detected_and_unidentified_ones_record_unknown(
    monkeypatch, backend
):
    serve(monkeypatch, SERVERS["vllm"])
    assert (await _run(backend=backend))["metadata"]["backend"] == "vllm"

    serve(monkeypatch, {})
    assert (await _run(backend=backend))["metadata"]["backend"] == "unknown"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"wire_format": "bogus"},
        {"system_prompt": "   "},
    ],
)
async def test_invalid_arguments_fail_before_any_request(monkeypatch, kwargs):
    requests = serve(monkeypatch, SERVERS["vllm"])

    with pytest.raises(ValueError):
        await _run(**kwargs)

    assert requests == []


@pytest.mark.parametrize(
    ("base_url", "backend", "engine_name"),
    [
        ("https://generativelanguage.googleapis.com", "gemini", "Google Gemini API"),
        ("https://api.anthropic.com", "anthropic", "Anthropic Messages API"),
    ],
)
async def test_hosted_api_is_labelled_without_probing(monkeypatch, base_url, backend, engine_name):
    requests = serve(monkeypatch, SERVERS["vllm"])

    result = await _run(base_url=base_url)

    assert requests == []
    assert result["metadata"]["backend"] == backend
    assert result["metadata"]["engine_name"] == engine_name


async def test_explicit_gemini_format_on_a_custom_url_is_not_probed(monkeypatch):
    requests = serve(monkeypatch, SERVERS["vllm"])

    result = await _run(base_url="https://gemini-proxy.test", wire_format="gemini")

    assert requests == []
    assert result["metadata"]["backend"] == "gemini"


async def test_explicit_openai_format_on_a_google_url_is_probed(monkeypatch):
    # Pinning the OpenAI format makes the endpoint self-hosted as far as
    # labelling goes, so the server is asked who it is.
    requests = serve(monkeypatch, SERVERS["vllm"])

    result = await _run(base_url="https://generativelanguage.googleapis.com", wire_format="openai")

    assert any(r.url.path == "/metrics" for r in requests)
    assert result["metadata"]["backend"] == "vllm"


async def test_probe_engine_false_sends_no_detection_requests(monkeypatch):
    requests = serve(monkeypatch, SERVERS["vllm"])

    result = await _run(probe_engine=False, short=True, scenarios=None)

    assert requests == []
    assert result["metadata"]["backend"] == "unknown"
    assert "engine_name" not in result["metadata"]
    assert result["metadata"]["model"] == "m"
    assert result["metadata"]["scenario_selector"].startswith("short (")


async def test_detection_that_raises_leaves_the_run_unknown(monkeypatch):
    serve(monkeypatch, SERVERS["vllm"])
    monkeypatch.setattr(metadata, "probe_backend_hint", AsyncMock(side_effect=RuntimeError("x")))

    result = await _run()

    assert result["status"] == "completed"
    assert result["metadata"]["backend"] == "unknown"


async def test_unreachable_server_leaves_the_run_unknown(monkeypatch):
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    serve(monkeypatch, refuse)

    result = await _run()

    assert result["status"] == "completed"
    assert result["metadata"]["backend"] == "unknown"


async def test_context_failure_falls_back_to_legacy_metadata(monkeypatch):
    serve(monkeypatch, SERVERS["vllm"])
    monkeypatch.setattr(metadata, "collect_run_context", AsyncMock(side_effect=RuntimeError("x")))

    result = await _run()

    assert result["status"] == "completed"
    assert result["metadata"]["config"]["backend"] == "vllm"


def _cli_metadata(monkeypatch: pytest.MonkeyPatch, extra: list[str]) -> dict[str, Any]:
    """Run the CLI's own detection and context collection, then stop."""
    from tool_eval_bench.cli import dispatch

    captured: list[Any] = []

    def stop(target: Any) -> tuple[list[Any], bool]:
        captured.append(target)
        return [], True

    for name in ("TOOL_EVAL_BACKEND", "TOOL_EVAL_PROVIDER", "TOOL_EVAL_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(dispatch, "_load_dotenv", lambda: None)
    monkeypatch.setattr(dispatch, "_check_endpoint_ready", lambda *a, **k: None)
    monkeypatch.setattr(dispatch, "_run_throughput_mode", stop)
    argv = ["tool-eval-bench", "--model", "m", "--base-url", "http://test"]
    monkeypatch.setattr(sys, "argv", [*argv, "--scenarios", "TC-01", "--json", *extra])

    dispatch.main()

    assert captured[0].backend == captured[0].run_context.backend
    return dict(captured[0].run_context.to_dict())


NON_DEFAULT_CLI = [
    *("--no-think", "--seed", "7", "--temperature", "0.3", "--parallel", "2"),
    *("--max-turns", "5", "--timeout", "30", "--error-rate", "0.1"),
    *("--system-prompt", "  Be terse.  "),
]
NON_DEFAULT_API: dict[str, Any] = {
    "extra_params": {"chat_template_kwargs": {"enable_thinking": False}},
    "seed": 7,
    "temperature": 0.3,
    "concurrency": 2,
    "max_turns": 5,
    "timeout_seconds": 30.0,
    "error_rate": 0.1,
    "system_prompt": "  Be terse.  ",
}


@pytest.mark.parametrize(
    ("server", "cli_args", "api_kwargs"),
    [
        *((server, [], {}) for server in sorted(SERVERS)),
        ("llamacpp", NON_DEFAULT_CLI, NON_DEFAULT_API),
    ],
)
def test_cli_and_api_record_the_same_metadata(monkeypatch, capsys, server, cli_args, api_kwargs):
    serve(monkeypatch, SERVERS[server])

    cli = _cli_metadata(monkeypatch, cli_args)
    api = asyncio.run(_run(**api_kwargs))["metadata"]

    assert api == cli
    stderr_events = [json.loads(line) for line in capsys.readouterr().err.splitlines() if line]
    assert {"event": "backend_detected", "backend": server} in stderr_events
