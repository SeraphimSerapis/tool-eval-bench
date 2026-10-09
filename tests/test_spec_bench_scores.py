"""Spec-bench stores its measurements, and the stored row reads back.

The goldens pin the exact row handed to ``persist_run``. These tests go one
step further, through real SQLite, to the queries ``history`` and ``export``
are built on.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from tests.test_mode_run_golden import _args, _context, _spec_samples, _target
from tool_eval_bench.runner.speculative import SpecDecodeSample


def _run_spec_bench(
    monkeypatch: pytest.MonkeyPatch, out: Path, samples: list[Any], *extra: str
) -> tuple[bool, int | None]:
    """Run spec-bench dispatch; return (stopped, exit code or None)."""
    from tool_eval_bench.cli.dispatch import _run_spec_bench_mode
    from tool_eval_bench.runner import speculative

    async def fake_run(*args: Any, on_sample: Callable[..., Any], **kwargs: Any) -> list[Any]:
        for index, sample in enumerate(samples):
            await on_sample(sample, index, len(samples))
        return samples

    monkeypatch.setattr(speculative, "run_spec_bench", fake_run)
    args = _args(out, "--spec-bench", "--depth", "0,4096", "--spec-prompts", "filler,code", *extra)
    try:
        return _run_spec_bench_mode(_target(args, _context())), None
    except SystemExit as exc:
        return True, int(exc.code or 0)


def test_stored_spec_bench_results_read_back_through_history_and_export(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from tool_eval_bench.application import run_queries
    from tool_eval_bench.cli.history import print_history
    from tool_eval_bench.cli.leaderboard import export_runs

    # RunRepository puts the database under ./data, so this isolates it.
    monkeypatch.chdir(tmp_path)
    ok = _spec_samples("response", 1)
    failed = SpecDecodeSample(
        depth=4096,
        prompt_type="structured",
        error="Server error '503' for url 'http://user:secret@gpu-box.internal:8000/v1'",
    )
    # A failed cell fails the run, as --perf-only does, once the row is stored.
    assert _run_spec_bench(monkeypatch, tmp_path / "runs", [ok[0], failed, ok[1]]) == (True, 1)

    [listed] = run_queries.recent_runs()
    stored = run_queries.get_run(listed["run_id"])
    assert stored is not None
    assert stored["status"] == "failed"
    scores = stored["scores"]
    assert scores == listed["scores"]
    assert scores["samples"] == 2
    assert scores["failed"] == 1
    assert scores["results"] == [sample.to_result() for sample in ok]
    assert scores["results"][0]["effective_tg_tps"] == ok[0].effective_tg_tps
    # Failed samples are a count only, so their error text never reaches the row.
    assert "secret" not in json.dumps(stored)
    assert "structured" not in json.dumps(scores)

    console = Console(record=True, width=200)
    print_history(console)
    assert listed["run_id"] in console.export_text()

    capsys.readouterr()
    # Export ranks tool-eval runs only; a spec-bench row is skipped, not an error.
    export_runs(Console(record=True), fmt="json")
    assert json.loads(capsys.readouterr().out) == []


def _failed(depth: int = 0) -> SpecDecodeSample:
    return SpecDecodeSample(depth=depth, prompt_type="filler", error="[server error 500] boom")


def test_every_cell_failing_is_stored_and_reported_as_a_failed_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Like --perf-only: store the row, say so in the report, exit 1."""
    from tool_eval_bench.application import run_queries

    monkeypatch.chdir(tmp_path)

    assert _run_spec_bench(monkeypatch, tmp_path / "runs", [_failed(0), _failed(4096)]) == (True, 1)

    [listed] = run_queries.recent_runs()
    assert listed["status"] == "failed"
    assert listed["scores"] == {"samples": 0, "failed": 2, "results": []}
    [report] = (tmp_path / "runs").rglob("*.md")
    text = report.read_text(encoding="utf-8")
    assert "2 of 2 cell(s) failed on every run" in text
    assert "No successful samples recorded." in text


def test_a_failed_cell_does_not_stop_a_combined_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With --perf still to run, spec-bench stores its failed row and carries on."""
    from tool_eval_bench.application import run_queries

    monkeypatch.chdir(tmp_path)

    assert _run_spec_bench(monkeypatch, tmp_path / "runs", [_failed()], "--perf") == (False, None)
    [listed] = run_queries.recent_runs()
    assert listed["status"] == "failed"


def test_a_method_alias_is_stored_under_its_canonical_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tool_eval_bench.application import run_queries

    monkeypatch.chdir(tmp_path)
    samples = _spec_samples("response", 1)

    _run_spec_bench(monkeypatch, tmp_path / "runs", samples, "--spec-method", "nextn")

    [listed] = run_queries.recent_runs()
    stored = run_queries.get_run(listed["run_id"])
    assert stored is not None
    assert stored["config"]["method"] == "mtp"


@pytest.mark.parametrize(
    ("hint", "canonical"),
    [
        ("draft", "draft_model"),
        ("standalone", "draft_model"),
        ("nextn", "mtp"),
        ("eagle3", "eagle3"),
        ("auto", None),
    ],
)
def test_canonical_spec_method_hint(hint: str, canonical: str | None) -> None:
    from tool_eval_bench.runner.spec_detection import canonical_spec_method_hint

    assert canonical_spec_method_hint(hint) == canonical


def test_detection_reports_the_canonical_method_for_an_alias() -> None:
    """Without /metrics, the hint is all there is; it must not be stored raw."""
    import asyncio

    from tool_eval_bench.runner.spec_detection import detect_spec_decoding

    class NoMetrics:
        async def metrics(self, metrics_url: str | None = None) -> Any:
            raise RuntimeError("no /metrics")

    info = asyncio.run(
        detect_spec_decoding(NoMetrics(), "http://x/v1", backend_hint="standalone")  # type: ignore[arg-type]
    )
    assert info.method == "draft_model"


def _fingerprint(**overrides: Any) -> str:
    from tool_eval_bench.cli.spec_bench import _spec_bench_config
    from tool_eval_bench.utils.fingerprint import with_config_fingerprint

    params: dict[str, Any] = {
        "spec_method": "auto",
        "runs": 3,
        "temperature": 0.0,
        "pp": 2048,
        "tg": 128,
        "depths": [0],
        "prompt_types": ["filler"],
        "baseline_tg_tps": None,
        "custom_prompts": None,
    }
    params.update(overrides)
    config = _spec_bench_config("m", "http://h:1/v1", **params)
    return str(with_config_fingerprint(config, {})["config_fingerprint"])


@pytest.mark.parametrize(
    "change",
    [
        {"pp": 512},
        {"tg": 1024},
        {"depths": [65536]},
        {"prompt_types": ["code"]},
        {"baseline_tg_tps": 30.0},
        {"prompt_types": ["mine"], "custom_prompts": {"mine": "Translate this"}},
    ],
)
def test_a_different_workload_is_a_different_cohort(change: dict[str, Any]) -> None:
    assert _fingerprint(**change) != _fingerprint()


def test_method_aliases_share_a_cohort() -> None:
    assert _fingerprint(spec_method="draft") == _fingerprint(spec_method="standalone")
    assert _fingerprint(spec_method="draft") == _fingerprint(spec_method="draft_model")


def test_custom_prompt_text_is_hashed_not_stored() -> None:
    from tool_eval_bench.cli.spec_bench import _spec_bench_config

    def config(text: str) -> dict[str, Any]:
        return _spec_bench_config(
            "m",
            "http://h:1/v1",
            spec_method="auto",
            runs=1,
            temperature=0.0,
            pp=2048,
            tg=128,
            depths=[0],
            prompt_types=["mine"],
            baseline_tg_tps=None,
            custom_prompts={"mine": text, "unused": "not selected"},
        )

    assert "Translate this" not in json.dumps(config("Translate this"))
    assert config("one")["custom_prompts_sha256"] != config("two")["custom_prompts_sha256"]


def test_to_result_keeps_derived_metrics_and_drops_per_step_arrays() -> None:
    sample = SpecDecodeSample(
        tg_tokens=100,
        ttft_ms=100.0,
        total_ms=1100.0,
        acceptance_rate=0.5,
        baseline_tg_tps=50.0,
        per_step_accepted=[2, 0],
        per_step_drafted=[2, 2],
        runs=2,
        acceptance_rate_range=(0.4, 0.6),
    )

    result = sample.to_result()

    assert result["effective_tg_tps"] == 99.0
    assert result["speedup_ratio"] == 1.98
    assert result["waste_ratio"] == 0.5
    assert result["per_position_acceptance"] == [0.5, 0.5]
    # A list, because the row is JSON; a tuple would not compare equal on read-back.
    assert result["acceptance_rate_range"] == [0.4, 0.6]
    assert "per_step_accepted" not in result
    assert "per_step_drafted" not in result
    assert "error" not in result


def test_to_result_leaves_unmeasured_metrics_empty() -> None:
    result = SpecDecodeSample(tg_tokens=10, total_ms=1000.0).to_result()

    assert result["effective_tg_tps"] == 10.0
    for key in (
        "acceptance_rate",
        "acceptance_rate_range",
        "goodput",
        "speedup_ratio",
        "draft_window",
        "draft_tps",
        "waste_ratio",
        "verify_steps_per_s",
        "per_position_acceptance",
    ):
        assert result[key] is None, key
