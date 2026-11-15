"""A failed capability probe must not veto a working TC-45 request."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import open_repository
from tool_eval_bench import api
from tool_eval_bench.application import service as service_module
from tool_eval_bench.application.service import BenchmarkService
from tool_eval_bench.cli import dispatch
from tool_eval_bench.domain.adapters import BackendAdapter, ChatCompletionResult, ProviderToolCall
from tool_eval_bench.evals.scenarios import ALL_SCENARIOS


class ProbeFailureAdapter(BackendAdapter):
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    async def chat_completion(self, **kwargs: Any) -> ChatCompletionResult:
        self.requests.append(kwargs)
        tools = kwargs.get("tools") or []
        if tools and tools[0]["function"]["name"] == "probe_ping":
            return ChatCompletionResult(
                content="unfinished", finish_reason="length", completion_tokens=kwargs["max_tokens"]
            )
        if not tools:
            return ChatCompletionResult(content="server unavailable", transport_error_status=500)
        if any(message["role"] == "tool" for message in kwargs["messages"]):
            return ChatCompletionResult(content="56", finish_reason="stop")
        return ChatCompletionResult(
            content="",
            tool_calls=[
                ProviderToolCall("calc", "calculator", json.dumps({"expression": "7 * 8"}))
            ],
            finish_reason="tool_calls",
        )


async def _no_metadata(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return {}


@pytest.fixture
def adapter(monkeypatch: pytest.MonkeyPatch) -> ProbeFailureAdapter:
    adapter = ProbeFailureAdapter()
    monkeypatch.setattr(BenchmarkService, "_adapter_for", lambda *a, **kw: adapter)
    monkeypatch.setattr(service_module, "_collect_metadata_safe", _no_metadata)
    return adapter


@pytest.mark.parametrize(
    "mode_flags", [[], ["--no-live"], ["--json"]], ids=["live", "plain", "json"]
)
@pytest.mark.parametrize("disable_probe", [False, True])
def test_cli_runs_tc45_and_reports_actual_exclusions(
    mode_flags: list[str],
    disable_probe: bool,
    adapter: ProbeFailureAdapter,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for key in list(os.environ):
        if key.startswith("TOOL_EVAL_"):
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dispatch, "_load_dotenv", lambda: None)
    monkeypatch.setattr(dispatch, "_build_run_context", lambda *a, **kw: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tool-eval-bench",
            "--base-url",
            "http://localhost:9999",
            "--model",
            "probe-test",
            "--backend",
            "vllm",
            "--scenarios",
            "TC-45",
            "TC-64",
            "--parallel",
            "4",
            "--no-preflight",
            "--no-warmup",
            "--output-dir",
            str(tmp_path / "runs"),
            *(["--no-probe-engine"] if disable_probe else []),
            *mode_flags,
        ],
    )

    dispatch.main()

    output = capsys.readouterr().out
    (stored,) = open_repository(db_path=str(tmp_path / "data" / "benchmarks.sqlite")).list()
    scores = stored["scores"]
    assert scores["excluded_scenarios"] == ["TC-64"]
    assert scores["max_points"] == scores["total_points"] == 2
    results = {r["scenario_id"]: r for r in scores["scenario_results"]}
    assert results["TC-45"]["status"] == "pass"
    diagnostic = results["TC-45"]["diagnostics"]["tool_choice"]
    assert ("probe disabled" if disable_probe else "inconclusive") in diagnostic
    assert len(adapter.requests) == (3 if disable_probe else 5)
    if not disable_probe:
        assert [r["max_tokens"] for r in adapter.requests[:2]] == [256, 4096]
    assert any(p.suffix == ".md" for p in (tmp_path / "runs").rglob("*"))
    if "--json" in mode_flags:
        assert json.loads(output)["scores"]["excluded_scenarios"] == ["TC-64"]
    else:
        assert re.search(r"1 scenario\(s\) excluded.*?TC-64", output, re.DOTALL)


@pytest.mark.asyncio
@pytest.mark.parametrize("probe_engine", [False, True])
async def test_api_probe_setting_reaches_actual_execution(
    probe_engine: bool, adapter: ProbeFailureAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api, "build_run_context", _no_metadata)
    result = await api.run_benchmark(
        model="probe-test",
        backend="vllm",
        base_url="http://localhost:9999",
        scenarios=[s for s in ALL_SCENARIOS if s.id == "TC-45"],
        probe_engine=probe_engine,
        persist=False,
    )

    assert result["scores"]["max_points"] == result["scores"]["total_points"] == 2
    assert len(adapter.requests) == (4 if probe_engine else 2)
