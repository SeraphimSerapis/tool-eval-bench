"""Resume refuses to merge results measured under different context pressure."""

from __future__ import annotations

import sys
from typing import Any

import pytest

from tool_eval_bench.cli.dispatch import _resume_config_mismatches
from tool_eval_bench.cli.legacy_parser import _make_parser
from tool_eval_bench.runner.context_pressure import compute_fill_budget

CONTEXT = 32_768
HALF = {
    "ratio": 0.5,
    "fill_tokens": 15_190,
    "fill_tokens_target": compute_fill_budget(CONTEXT, 0.5),
    "context_size": CONTEXT,
}


def _pressure(ratio: float, context_size: int = CONTEXT, fill_tokens: int = 15_190) -> dict:
    return {
        "ratio": ratio,
        "fill_tokens": fill_tokens,
        "fill_tokens_target": compute_fill_budget(context_size, ratio),
        "context_size": context_size,
    }


def _mismatches(previous: dict[str, Any], current: dict[str, Any] | None) -> list[str]:
    return _resume_config_mismatches(
        previous,
        model="m",
        backend="vllm",
        base_url="http://localhost:8000",
        scenarios=[],
        args=_make_parser().parse_args([]),
        extra_params=None,
        scenario_packs=None,
        context_pressure=current,
    )


@pytest.mark.parametrize(
    ("previous", "current", "expected"),
    [
        (HALF, _pressure(0.6), "context_pressure ratio (was 0.5, now 0.6)"),
        ({}, _pressure(0.5), "context_pressure (was off, now 0.5)"),
        (HALF, None, "context_pressure (was 0.5, now off)"),
        (
            HALF,
            _pressure(0.5, context_size=65_536),
            f"context_pressure fill (was {HALF['fill_tokens_target']:,} tokens, "
            f"now {compute_fill_budget(65_536, 0.5):,}; the context size changed)",
        ),
    ],
    ids=["ratio-differs", "added", "removed", "context-size-changes-fill"],
)
def test_resume_refuses_a_pressure_change(
    previous: dict[str, Any], current: dict[str, Any] | None, expected: str
) -> None:
    assert _mismatches({"context_pressure": previous} if previous else {}, current) == [expected]


def test_resume_accepts_the_same_pressure_after_a_server_restart() -> None:
    """Context drift that leaves the fill target alone, and calibration noise, are not changes."""
    drifted = CONTEXT - 64
    assert compute_fill_budget(drifted, 0.5) == HALF["fill_tokens_target"]

    restarted = _pressure(0.5, context_size=drifted, fill_tokens=15_201)

    assert _mismatches({"context_pressure": HALF}, restarted) == []


def test_resume_accepts_a_run_without_pressure() -> None:
    assert _mismatches({"model": "m"}, None) == []


def test_resume_compares_only_the_ratio_for_a_run_without_a_fill_target() -> None:
    """Runs from before tokenizer calibration recorded ratio, fill, and context only."""
    legacy = {"ratio": 0.5, "fill_tokens": 14_000, "context_size": 16_384}

    assert _mismatches({"context_pressure": legacy}, _pressure(0.5)) == []
    assert _mismatches({"context_pressure": legacy}, _pressure(0.7)) == [
        "context_pressure ratio (was 0.5, now 0.7)"
    ]


def _resume_interrupted_pressure_run(
    monkeypatch: pytest.MonkeyPatch, *pressure_flags: str
) -> list[dict[str, Any]]:
    """Run ``resume`` through ``main`` against a stored run made at 50% pressure."""
    from tool_eval_bench.cli import dispatch, plugin_runners
    from tool_eval_bench.runner import context_pressure
    from tool_eval_bench.storage import db
    from tool_eval_bench.utils import metadata

    async def no_context(**kwargs: Any) -> None:
        return None

    async def calibrate(messages: Any, target: int, *args: Any, **kwargs: Any) -> Any:
        return messages, target

    stored = {
        "status": "interrupted",
        "config": {
            "model": "m",
            "backend": "vllm",
            "timeout_seconds": 600.0,
            "context_pressure": HALF,
        },
        "scores": {},
    }

    class Repo:
        def get(self, run_id: str) -> dict[str, Any]:
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
            "resume",
            "r",
            "--model",
            "m",
            "--backend",
            "vllm",
            "--base-url",
            "http://localhost:8000",
            "--scenarios",
            "TC-01",
            "--timeout",
            "600",
            "--no-warmup",
            "--no-live",
            *pressure_flags,
        ],
    )
    dispatch.main()
    return calls


@pytest.mark.parametrize(
    ("flags", "named"),
    [
        (("--context-pressure", "0.6", "--context-size", str(CONTEXT)), "ratio (was 0.5, now 0.6)"),
        ((), "context_pressure (was 0.5, now off)"),
    ],
    ids=["ratio-differs", "removed"],
)
def test_cli_resume_refuses_a_pressure_change(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    flags: tuple[str, ...],
    named: str,
) -> None:
    with pytest.raises(SystemExit) as exited:
        _resume_interrupted_pressure_run(monkeypatch, *flags)

    assert exited.value.code == 1
    out = " ".join(capsys.readouterr().out.split())
    assert "configuration mismatch" in out
    assert named in out


def test_cli_resume_continues_under_the_same_pressure(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _resume_interrupted_pressure_run(
        monkeypatch, "--context-pressure", "0.5", "--context-size", str(CONTEXT)
    )

    assert calls[0]["resume_run_id"] == "r"
    assert calls[0]["context_pressure_config"]["ratio"] == 0.5
