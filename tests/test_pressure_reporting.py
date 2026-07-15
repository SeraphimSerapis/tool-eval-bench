"""Artifact-first finalization tests for context-pressure sweeps."""

from __future__ import annotations

from pathlib import Path

import pytest

from tool_eval_bench.cli.pressure import (
    _report_then_persist_pressure_sweep,
    _write_pressure_sweep_report,
)


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

    path = _write_pressure_sweep_report(
        run_id="pressure-run",
        model="Display Model",
        backend="vllm",
        display_url="http://redacted:8000",
        context_size=32768,
        level_results=levels,
        breaking_point=0.5,
        first_degradation=0.75,
        output_dir=str(tmp_path),
    )

    assert path == next(tmp_path.rglob("pressure-run.md"))
    markdown = path.read_text()
    assert "## Level 1 — 50%" in markdown
    assert "## Level 2 — 75%" in markdown
    assert markdown.count("### TC-01") == 2
    assert first_trace in markdown
    assert second_trace in markdown
    assert "get_weather(Berlin)" in markdown


def test_pressure_report_failure_prevents_completed_persistence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tool_eval_bench.cli import pressure

    persisted: list[dict] = []

    def fail_report(**_kwargs) -> Path:
        raise OSError("disk full")

    monkeypatch.setattr(pressure, "_write_pressure_sweep_report", fail_report)
    with pytest.raises(OSError, match="disk full"):
        _report_then_persist_pressure_sweep(
            report_kwargs={},
            run_data={"run_id": "pressure-run", "status": "completed"},
            persist_plugin_run=persisted.append,
        )

    assert persisted == []
