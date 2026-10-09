"""The context-pressure timeout must not depend on calibration noise."""

from __future__ import annotations

import sys
from typing import Any

import pytest

CONTEXT = "32768"


def _run(
    monkeypatch: pytest.MonkeyPatch,
    calibration_offset: int,
    *extra: str,
    stored: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run ``main`` at 50% pressure; calibration lands ``calibration_offset`` off target.

    Returns the keyword arguments the benchmark service received.
    """
    from tool_eval_bench.cli import dispatch, plugin_runners
    from tool_eval_bench.runner import context_pressure
    from tool_eval_bench.storage import db
    from tool_eval_bench.utils import metadata

    async def no_context(**kwargs: Any) -> None:
        return None

    async def calibrate(messages: Any, target: int, *args: Any, **kwargs: Any) -> Any:
        return messages, target + calibration_offset

    class Repo:
        def get(self, run_id: str) -> dict[str, Any] | None:
            return stored

        def get_checkpoints(self, run_id: str) -> list[dict[str, Any]]:
            return []

        def close(self) -> None:
            pass

    calls: list[dict[str, Any]] = []

    class Service:
        async def run_benchmark(self, **kwargs: Any) -> dict[str, Any]:
            calls.append(kwargs)
            return {"run_id": "r", "scores": {"scenario_results": []}}

    monkeypatch.setattr(dispatch, "_load_dotenv", lambda: None)
    monkeypatch.setattr(dispatch, "_preflight_model_check", lambda *a, **k: None)
    monkeypatch.setattr(metadata, "collect_run_context", no_context)
    monkeypatch.setattr(plugin_runners, "run_selected_plugins", lambda *a, **k: False)
    monkeypatch.setattr(context_pressure, "calibrate_pressure_messages", calibrate)
    monkeypatch.setattr(db, "RunRepository", Repo)
    monkeypatch.setattr(dispatch, "BenchmarkService", lambda **kwargs: Service())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tool-eval-bench",
            *extra,
            "--model",
            "m",
            "--backend",
            "vllm",
            "--base-url",
            "http://localhost:8000",
            "--scenarios",
            "TC-01",
            "--context-pressure",
            "0.5",
            "--context-size",
            CONTEXT,
            "--no-warmup",
            "--no-live",
        ],
    )
    dispatch.main()
    return calls[0]


def test_unseeded_runs_at_one_fill_target_get_one_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    first = _run(monkeypatch, 11)
    second = _run(monkeypatch, -7)

    assert (
        first["context_pressure_config"]["fill_tokens"]
        != second["context_pressure_config"]["fill_tokens"]
    )
    assert first["timeout_seconds"] > 60.0
    assert first["timeout_seconds"] == second["timeout_seconds"]


def test_resume_is_not_refused_over_calibration_noise(monkeypatch: pytest.MonkeyPatch) -> None:
    original = _run(monkeypatch, 11)
    stored = {
        "status": "interrupted",
        "config": {
            "model": "m",
            "backend": "vllm",
            "timeout_seconds": original["timeout_seconds"],
            "context_pressure": original["context_pressure_config"],
        },
        "scores": {},
    }

    resumed = _run(monkeypatch, -7, "resume", "r", stored=stored)

    assert resumed["resume_run_id"] == "r"
    assert resumed["timeout_seconds"] == original["timeout_seconds"]
