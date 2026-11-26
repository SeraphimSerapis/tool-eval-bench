"""Deterministic tests for llama-benchy CLI progress tracking and invocation."""

from __future__ import annotations

import pytest
from rich.console import Console

from tool_eval_bench.cli.perf import (
    _BenchyProgressTracker,
    run_llama_benchy,
)
from tool_eval_bench.runner.throughput import ThroughputSample


def _sample(**overrides: object) -> ThroughputSample:
    values = {
        "pp_tokens": 100,
        "tg_tokens": 20,
        "depth": 1024,
        "concurrency": 2,
        "ttft_ms": 5.0,
        "total_ms": 100.0,
        "pp_tps": 1000.0,
        "tg_tps": 200.0,
        "requested_pp": 100,
        "requested_depth": 1024,
        "calibration_confidence": "heuristic",
    }
    values.update(overrides)
    return ThroughputSample(**values)


def test_llama_benchy_cli_success_and_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    import tool_eval_bench.runner.llama_benchy as benchy

    monkeypatch.setattr(benchy, "is_available", lambda: False)
    with pytest.raises(SystemExit):
        run_llama_benchy(
            Console(), "m", "d", "url", None, pp=[1], tg=[1], depths=[0], concurrency_levels=[1]
        )

    ok = _sample(calibration_confidence="llama-benchy")
    failed = _sample(error="failed")

    async def fake_run(*args: object, on_progress=None, **kwargs: object):
        if on_progress is not None:
            on_progress(
                {
                    "type": "request_start",
                    "request_id": 1,
                    "prompt_size": 100,
                    "response_size": 20,
                    "context_size": 0,
                    "concurrency": 1,
                    "run_index": 1,
                }
            )
            on_progress({"type": "request_end", "request_id": 1})
            on_progress({"type": "bench_complete"})
        return benchy.LlamaBenchyResult(version="1.2.3", latency_ms=2.0, samples=[ok, failed])

    monkeypatch.setattr(benchy, "is_available", lambda: True)
    monkeypatch.setattr(benchy, "run_llama_benchy", fake_run)
    console = Console(record=True, width=140)

    result = run_llama_benchy(
        console,
        "m",
        "Display",
        "url",
        None,
        pp=[100],
        tg=[20],
        depths=[0],
        concurrency_levels=[1],
        runs=1,
    )

    assert result == [ok, failed]
    output = console.export_text()
    assert "llama-benchy 1.2.3" in output
    assert "Incomplete" in output
    assert "\u2713 Complete" not in output
    assert "failed" in output


def test_benchy_progress_tracker_counts_measurement_runs_not_http_requests() -> None:
    """Concurrency>1 emits one request_end per HTTP request; progress is per run."""
    tracker = _BenchyProgressTracker()

    # One c4 measurement run → 4 request_end events, still one completed run.
    for rid in (10, 11, 12, 13):
        tracker.handle(
            {
                "type": "request_start",
                "request_id": rid,
                "prompt_size": 2048,
                "response_size": 128,
                "context_size": 0,
                "concurrency": 4,
                "run_index": 0,
            }
        )
        assert tracker.completed_runs == 0
        tracker.handle({"type": "request_end", "request_id": rid})

    assert tracker.completed_runs == 1
    assert "c4" in tracker.current_test

    # Second run at c2 → two more ends → completed_runs == 2 (not 6).
    for rid in (20, 21):
        tracker.handle(
            {
                "type": "request_start",
                "request_id": rid,
                "prompt_size": 2048,
                "response_size": 128,
                "context_size": 4096,
                "concurrency": 2,
                "run_index": 1,
            }
        )
        tracker.handle({"type": "request_end", "request_id": rid})

    assert tracker.completed_runs == 2

    # Default sweep shape: 3 depths × 3 conc × 3 runs, with conc in {1,2,4}.
    # Raw request_ends would be 63; measurement runs stay at 27.
    big = _BenchyProgressTracker()
    rid = 0
    for depth in (0, 4096, 8192):
        for conc in (1, 2, 4):
            for run_idx in (0, 1, 2):
                for _ in range(conc):
                    rid += 1
                    big.handle(
                        {
                            "type": "request_start",
                            "request_id": rid,
                            "prompt_size": 2048,
                            "response_size": 128,
                            "context_size": depth,
                            "concurrency": conc,
                            "run_index": run_idx,
                        }
                    )
                    big.handle({"type": "request_end", "request_id": rid})
    assert big.completed_runs == 27
    assert rid == 63


def test_benchy_progress_tracker_legacy_request_end_without_id() -> None:
    tracker = _BenchyProgressTracker()
    tracker.handle({"type": "request_end"})
    tracker.handle({"type": "request_end"})
    assert tracker.completed_runs == 2


@pytest.mark.usefixtures("no_dns")
def test_llama_benchy_cli_progress_never_exceeds_total_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI progress wiring must count measurement runs, not HTTP request_ends.

    A concurrency sweep emits far more request_end events than total_runs.
    Counting every end would report completed=4 after the first c4 run (and
    later climb past the task total). This drives the real Rich Progress path
    with a mocked llama-benchy event stream — no live server required.
    """
    import rich.progress as rich_progress

    import tool_eval_bench.runner.llama_benchy as benchy

    depths = [0, 4096]
    concurrency_levels = [1, 2, 4]
    runs = 2
    total_runs = len(depths) * len(concurrency_levels) * runs  # 12
    # Raw HTTP ends: 2 depths × 2 runs × (1+2+4) = 28
    expected_http_ends = len(depths) * runs * sum(concurrency_levels)

    completed_values: list[float] = []
    real_update = rich_progress.Progress.update

    def spy_update(self: object, task_id: object, *args: object, **kwargs: object) -> object:
        if "completed" in kwargs:
            completed_values.append(float(kwargs["completed"]))  # type: ignore[arg-type]
        return real_update(self, task_id, *args, **kwargs)

    monkeypatch.setattr(rich_progress.Progress, "update", spy_update)

    ok = _sample(calibration_confidence="llama-benchy", depth=0, concurrency=1)
    after_first_c4: list[float] = []
    http_ends_seen = {"n": 0}

    async def fake_run(*args: object, on_progress=None, **kwargs: object):
        assert on_progress is not None
        # First measurement run is c4 @ d0 — four HTTP ends must yield completed=1.
        for rid in (1, 2, 3, 4):
            on_progress(
                {
                    "type": "request_start",
                    "request_id": rid,
                    "prompt_size": 2048,
                    "response_size": 128,
                    "context_size": 0,
                    "concurrency": 4,
                    "run_index": 0,
                }
            )
            on_progress({"type": "request_end", "request_id": rid})
        after_first_c4.append(completed_values[-1])

        # Remaining sweep (skip the run we already emitted: d0/c4/run0).
        rid = 4
        for depth in depths:
            for conc in concurrency_levels:
                for run_idx in range(runs):
                    if depth == 0 and conc == 4 and run_idx == 0:
                        continue
                    for _ in range(conc):
                        rid += 1
                        on_progress(
                            {
                                "type": "request_start",
                                "request_id": rid,
                                "prompt_size": 2048,
                                "response_size": 128,
                                "context_size": depth,
                                "concurrency": conc,
                                "run_index": run_idx,
                            }
                        )
                        on_progress({"type": "request_end", "request_id": rid})
        on_progress({"type": "bench_complete", "status": "ok"})
        http_ends_seen["n"] = rid
        return benchy.LlamaBenchyResult(version="0.4.0", latency_ms=1.0, samples=[ok])

    monkeypatch.setattr(benchy, "is_available", lambda: True)
    monkeypatch.setattr(benchy, "run_llama_benchy", fake_run)

    console = Console(record=True, width=140)
    result = run_llama_benchy(
        console,
        "Laguna-S-2.1",
        "Laguna-S-2.1",
        "http://test/v1",
        None,
        pp=[2048],
        tg=[128],
        depths=depths,
        concurrency_levels=concurrency_levels,
        runs=runs,
        # An explicit tokenizer skips resolution, which reads the host's
        # HuggingFace cache and then probes the fake host's /props endpoint.
        tokenizer="fake/tokenizer",
    )

    assert result == [ok]
    assert http_ends_seen["n"] == expected_http_ends
    assert expected_http_ends > total_runs
    assert after_first_c4 == [1.0], (
        f"first c4 run must advance progress by 1, not by HTTP count; got {after_first_c4}"
    )
    assert completed_values, "expected progress.update(completed=...) calls"
    assert max(completed_values) <= total_runs
    assert completed_values[-1] == total_runs
