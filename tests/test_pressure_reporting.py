"""Artifact-first finalization tests for context-pressure sweeps."""

from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from rich.console import Console

from tool_eval_bench.domain.models import RunContext
from tool_eval_bench.storage.reports import MarkdownReporter


def _scenario(scenario_id: str, raw_log: str, *, status: str = "pass") -> dict:
    return {
        "scenario_id": scenario_id,
        "status": status,
        "points": 2 if status == "pass" else 0,
        "summary": "trace-complete result",
        "expected_behavior": "call the expected tool",
        "tool_calls_made": ["get_weather(Berlin)"],
        "raw_log": raw_log,
    }


def test_pressure_sweep_report_contains_every_level_and_full_trace(tmp_path: Path) -> None:
    first_trace = "turn=1 assistant tool_call=get_weather\nturn=2 tool result=sunny"
    second_trace = "assistant included ``` in content\nfinal answer delivered"
    levels = [
        {
            "ratio": 0.5,
            "fill_tokens": 16000,
            "score_pct": 100.0,
            "scenario_results": [_scenario("TC-01", first_trace)],
        },
        {
            "ratio": 0.75,
            "fill_tokens": 24000,
            "score_pct": 0.0,
            "scenario_results": [_scenario("TC-01", second_trace, status="fail")],
        },
    ]

    path = MarkdownReporter(root=str(tmp_path)).write_pressure_sweep_report(
        run_id="pressure-run",
        model="Display Model",
        backend="vllm",
        display_url="http://redacted:8000",
        context_size=32768,
        level_results=levels,
        breaking_point=0.5,
        first_degradation=0.75,
    )

    assert path == next(tmp_path.rglob("pressure-run.md"))
    markdown = path.read_text(encoding="utf-8")
    assert "## Level 1 — 50%" in markdown
    assert "## Level 2 — 75%" in markdown
    assert markdown.count("### TC-01") == 2
    assert first_trace in markdown
    assert second_trace in markdown
    assert "get_weather(Berlin)" in markdown


def _engine_context() -> RunContext:
    return RunContext(
        tool_version="1.2.3",
        git_sha="abc1234",
        hostname="bench-host",
        platform_info="linux",
        python_version="3.13",
        model="served-model",
        backend="vllm",
        base_url="http://test/v1",
        temperature=0.7,
        engine_name="vLLM",
        engine_version="0.11.0",
        max_model_len=32768,
    )


def test_sweep_report_renders_the_probed_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tool_eval_bench.adapters import factory
    from tool_eval_bench.application import run_queries
    from tool_eval_bench.cli import pressure
    from tool_eval_bench.cli.pressure import run_pressure_sweep
    from tool_eval_bench.domain.scenarios import (
        Category,
        ModelScoreSummary,
        ScenarioDefinition,
        ScenarioResult,
        ScenarioStatus,
    )
    from tool_eval_bench.runner import orchestrator

    class _CalibrationFreeLoop:
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
    monkeypatch.setattr(orchestrator, "run_all_scenarios", AsyncMock(return_value=summary))
    monkeypatch.setattr(factory, "build_adapter", lambda *a, **k: AsyncMock())
    args = argparse.Namespace(
        context_pressure_sweep="0.5-1.0",
        sweep_steps=2,
        context_size=32768,
        seed=None,
        system_prompt=None,
        output_dir=str(tmp_path),
        timeout=60.0,
    )
    persisted: list[dict[str, Any]] = []
    monkeypatch.setattr(pressure, "resolve_scenarios", lambda _args: [scenario])
    monkeypatch.setattr(run_queries, "persist_run", persisted.append)

    with patch("asyncio.new_event_loop", return_value=_CalibrationFreeLoop()):
        run_pressure_sweep(
            Console(file=io.StringIO(), force_terminal=False, width=200),
            "served-model",
            "served-model",
            "vllm",
            "http://test/v1",
            None,
            args,
            run_context=_engine_context(),
        )

    markdown = Path(persisted[0]["report_path"]).read_text(encoding="utf-8")
    assert "- **tool-eval-bench**: `v1.2.3 abc1234`" in markdown
    assert "| Engine | vLLM 0.11.0 |" in markdown
    assert "| Max Model Length | 32,768 |" in markdown
    assert "| Host | `bench-host` |" in markdown
    # The sweep runs at temperature 0 whatever --temperature said, so the CLI
    # parameter table would misstate the run.
    assert "| Temperature |" not in markdown


def test_sweep_report_without_run_context_keeps_its_header(tmp_path: Path) -> None:
    path = MarkdownReporter(root=str(tmp_path)).write_pressure_sweep_report(
        run_id="pressure-run",
        model="Display Model",
        backend="vllm",
        display_url="http://redacted:8000",
        context_size=32768,
        level_results=[],
        breaking_point=None,
        first_degradation=None,
    )

    markdown = path.read_text(encoding="utf-8")
    assert "- **Context Window**: 32,768 tokens" in markdown
    assert "tool-eval-bench" not in markdown
    assert "## Inference Engine" not in markdown
    assert "## Environment" not in markdown
