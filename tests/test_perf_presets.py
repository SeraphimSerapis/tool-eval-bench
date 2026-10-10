"""``--perf-preset``: grid resolution, flag precedence, the long preset's two passes.

llama-benchy is never run: dispatch's ``_run_llama_benchy`` is replaced by a
recorder that returns canned samples per pass.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from tests.conftest import open_repository
from tests.test_e2e import SmokeMockAdapter
from tool_eval_bench.application import run_queries
from tool_eval_bench.application import service as service_module
from tool_eval_bench.application.service import BenchmarkService
from tool_eval_bench.cli import dispatch
from tool_eval_bench.cli.legacy_parser import _make_parser
from tool_eval_bench.cli.perf import PerfPassFailed
from tool_eval_bench.domain.models import RunContext
from tool_eval_bench.runner import llama_benchy
from tool_eval_bench.runner.perf_presets import (
    CHAT_TEMPLATE_HEADROOM,
    PerfPass,
    depth0_speeds,
    estimate_pass_seconds,
    format_duration,
    plan_perf,
)
from tool_eval_bench.runner.throughput import ThroughputSample
from tool_eval_bench.utils.fingerprint import with_config_fingerprint

LONG_DEPTHS = (16384, 32768, 65536, 131072)

# -- plan_perf ----------------------------------------------------------------------


def test_short_preset_is_the_grid_perf_always_ran() -> None:
    plan = plan_perf("short")

    assert (plan.pp, plan.tg, plan.runs) == (2048, 128, 3)
    assert plan.passes == (PerfPass((0, 4096, 8192), (1, 2, 4)),)
    assert plan.depths == [0, 4096, 8192]
    assert plan.concurrency == [1, 2, 4]


def test_short_preset_ignores_the_context_window() -> None:
    assert plan_perf("short", context_window=4096).depths == [0, 4096, 8192]


def test_long_preset_splits_into_a_concurrency_pass_and_a_cached_depth_pass() -> None:
    plan = plan_perf("long")

    assert plan.passes == (
        PerfPass((0,), (1, 2, 4)),
        # Depth 0 at c1 is already in the first pass, so it is not repeated.
        PerfPass(LONG_DEPTHS, (1,), prefix_caching=True),
    )
    assert plan.depths == [0, *LONG_DEPTHS]
    assert plan.concurrency == [1, 2, 4]
    assert plan.dropped_depths == ()


@pytest.mark.parametrize("preset", ["short", "long"])
def test_explicit_flags_override_the_preset(preset: str) -> None:
    plan = plan_perf(preset, pp=512, tg=64, runs=5, depths=[0, 2048], concurrency=[1, 8])

    assert (plan.pp, plan.tg, plan.runs) == (512, 64, 5)
    assert plan.depths == [0, 2048]
    assert plan.concurrency == [1, 8]


def test_long_depth_pass_keeps_depth_zero_when_the_first_pass_skipped_c1() -> None:
    plan = plan_perf("long", concurrency=[2, 4])

    assert plan.passes[0] == PerfPass((0,), (2, 4))
    assert plan.passes[1].depths == (0, *LONG_DEPTHS)


def test_long_drops_depths_that_do_not_fit_the_window() -> None:
    plan = plan_perf("long", context_window=65536)

    assert plan.passes[1].depths == (16384, 32768)
    assert plan.dropped_depths == (65536, 131072)
    assert plan.context_window == 65536


@pytest.mark.parametrize(("slack", "kept"), [(0, True), (-1, False)])
def test_window_boundary_counts_pp_tg_and_template_headroom(slack: int, kept: bool) -> None:
    window = 32768 + 2048 + 128 + CHAT_TEMPLATE_HEADROOM + slack
    plan = plan_perf("long", depths=[32768], context_window=window)

    assert (32768 in plan.depths) is kept


def test_long_with_no_fitting_depth_is_one_plain_pass() -> None:
    plan = plan_perf("long", context_window=8192)

    assert plan.passes == (PerfPass((0,), (1, 2, 4)),)
    assert plan.dropped_depths == LONG_DEPTHS
    assert plan.request_fits is True


@pytest.mark.parametrize("concurrency", [None, [2, 4]])
def test_depth_zero_is_never_reported_dropped(concurrency: list[int] | None) -> None:
    # pp + tg alone overflows the window; the first pass still runs depth 0.
    plan = plan_perf("long", concurrency=concurrency, context_window=2000)

    assert plan.passes[0].depths == (0,)
    assert 0 not in plan.dropped_depths
    assert plan.dropped_depths == LONG_DEPTHS
    assert plan.request_fits is False


@pytest.mark.parametrize(
    ("window", "fits"), [(None, True), (2048 + 128 + 256, True), (2431, False)]
)
def test_request_fits_counts_pp_tg_and_headroom(window: int | None, fits: bool) -> None:
    assert plan_perf("short", context_window=window).request_fits is fits


def test_first_passes_keeps_only_what_ran() -> None:
    ran = plan_perf("long").first_passes(1)

    assert ran.passes == (PerfPass((0,), (1, 2, 4)),)
    assert (ran.depths, ran.concurrency) == ([0], [1, 2, 4])


def test_unknown_preset_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown perf preset"):
        plan_perf("medium")


# -- estimate -------------------------------------------------------------------------


def _sample(**fields: Any) -> ThroughputSample:
    values: dict[str, Any] = {
        "pp_tokens": 2048,
        "tg_tokens": 128,
        "depth": 0,
        "concurrency": 1,
        "ttft_ms": 500.0,
        "total_ms": 3000.0,
        "pp_tps": 4000.0,
        "tg_tps": 50.0,
        "requested_pp": 2048,
        "calibration_confidence": "llama-benchy",
    }
    values.update(fields)
    return ThroughputSample(**values)


def test_depth0_speeds_use_the_successful_single_stream_depth_zero_row() -> None:
    samples = [
        _sample(concurrency=2, pp_tps=9000.0),
        _sample(error="HTTP 500"),
        _sample(depth=4096, pp_tps=1.0),
        _sample(pp_tps=4000.0, tg_tps=50.0),
    ]

    assert depth0_speeds(samples) == (4000.0, 50.0)
    assert depth0_speeds([_sample(concurrency=2)]) is None


def test_estimate_counts_the_warm_up_run_and_the_cached_follow_up_decode() -> None:
    perf_pass = PerfPass((0, 8000), (1,), prefix_caching=True)
    seconds = estimate_pass_seconds(perf_pass, pp=2000, tg=100, runs=3, pp_tps=1000, tg_tps=50)

    # Depth 0 runs plain: 2 s prefill + 2 s decode. Depth 8000 prefills 8000
    # cold plus 2000 cached-prefix tokens (10 s) and decodes twice (4 s).
    assert seconds == pytest.approx((4 + 14) * 4)


def test_estimate_without_prefix_caching_decodes_once() -> None:
    perf_pass = PerfPass((8000,), (1,))
    seconds = estimate_pass_seconds(perf_pass, pp=2000, tg=100, runs=1, pp_tps=1000, tg_tps=50)

    assert seconds == pytest.approx(12 * 2)


@pytest.mark.parametrize(
    ("seconds", "text"), [(0.2, "1 s"), (45, "45 s"), (600, "10 min"), (7200, "2.0 h")]
)
def test_format_duration_is_coarse(seconds: float, text: str) -> None:
    assert format_duration(seconds) == text


# -- llama-benchy command -------------------------------------------------------------


def _command(**kwargs: Any) -> list[str]:
    return llama_benchy._build_command("http://host:8000", "m", **kwargs)


@pytest.fixture
def benchy_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llama_benchy, "_find_llama_benchy", lambda: "llama-benchy")


@pytest.mark.usefixtures("benchy_on_path")
def test_prefix_caching_adds_the_flag_and_keeps_llamacpp_caching_on() -> None:
    cmd = _command(enable_prefix_caching=True)

    assert cmd.count("--enable-prefix-caching") == 1
    assert "cache_prompt=true" in cmd
    assert "--enable-prefix-caching" not in _command()


@pytest.mark.usefixtures("benchy_on_path")
def test_prefix_caching_is_not_duplicated_from_benchy_args() -> None:
    cmd = _command(enable_prefix_caching=True, extra_args=["--enable-prefix-caching"])

    assert cmd.count("--enable-prefix-caching") == 1


# -- per-request decode rate ----------------------------------------------------------


def _entry(concurrency: int, tg_req: float) -> dict[str, Any]:
    return {
        "concurrency": concurrency,
        "context_size": 0,
        "prompt_size": 2048,
        "response_size": 128,
        "pp_throughput": {"mean": 4000.0},
        "tg_throughput": {"mean": 90.0},
        "pp_req_throughput": {"mean": 2000.0},
        "tg_req_throughput": {"mean": tg_req},
        "e2e_ttft": {"mean": 500.0},
    }


def test_concurrent_cell_stores_the_measured_per_request_decode_rate() -> None:
    sample = llama_benchy._parse_benchmark_entry(_entry(4, 25.0))

    assert sample.tg_tps == 90.0
    assert sample.tg_req_tps == 25.0
    assert sample.to_result()["tg_req_tps"] == 25.0


@pytest.mark.parametrize(("concurrency", "tg_req"), [(1, 45.0), (4, 0.0)])
def test_per_request_rate_is_left_out_when_it_adds_nothing(concurrency: int, tg_req: float) -> None:
    sample = llama_benchy._parse_benchmark_entry(_entry(concurrency, tg_req))

    assert sample.tg_req_tps is None
    assert "tg_req_tps" not in sample.to_result()


# -- cli.perf failure contract --------------------------------------------------------


@pytest.mark.parametrize(
    ("error", "interrupted", "text"),
    [
        (RuntimeError("HTTP 400"), False, "llama-benchy error"),
        (KeyboardInterrupt(), True, "Interrupted."),
        (OSError("killed"), False, "Error: killed"),
    ],
)
def test_cli_perf_raises_a_typed_failure_instead_of_exiting(
    monkeypatch: pytest.MonkeyPatch, error: BaseException, interrupted: bool, text: str
) -> None:
    from tool_eval_bench.cli import perf

    async def stopped(*_args: Any, **_kwargs: Any) -> None:
        raise error

    monkeypatch.setattr(llama_benchy, "is_available", lambda: True)
    monkeypatch.setattr(llama_benchy, "run_llama_benchy", stopped)
    monkeypatch.setattr(perf, "_resolve_benchy_tokenizer", lambda *_args: None)

    with pytest.raises(PerfPassFailed) as failed:
        perf.run_llama_benchy(
            Console(record=True),
            "m",
            "M",
            "url",
            None,
            pp=[1],
            tg=[1],
            depths=[0],
            concurrency_levels=[1],
        )

    assert failed.value.interrupted is interrupted
    assert text in failed.value.markup


# -- parser ---------------------------------------------------------------------------


def test_workload_flags_are_unset_until_the_preset_resolves_them() -> None:
    args = _make_parser().parse_args(["--perf"])

    assert args.perf_preset == "short"
    assert (args.pp, args.tg, args.depth, args.concurrency, args.benchy_runs) == (None,) * 5


def test_unknown_preset_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        _make_parser().parse_args(["--perf", "--perf-preset", "huge"])
    assert "invalid choice" in capsys.readouterr().err


# -- dispatch -------------------------------------------------------------------------


def _target(flags: list[str], tmp_path: Path, run_context: RunContext | None = None) -> Any:
    args = _make_parser().parse_args([*flags, "--output-dir", str(tmp_path)])
    return dispatch._Target(
        args=args,
        parser=argparse.ArgumentParser(),
        console=Console(record=True, width=200),
        model="m",
        display_name="Model",
        backend="vllm",
        base_url="http://bench.test/v1",
        display_url="http://bench.test/v1",
        api_key=None,
        wire_format="openai",
        extra_params={},
        run_context=run_context,
    )


class _Benchy:
    """Records each llama-benchy invocation and returns one sample per cell.

    ``raise_on_call`` makes that call (1-based) raise ``PerfPassFailed``, as
    ``cli.perf`` does when llama-benchy stops without results.
    """

    def __init__(
        self,
        *,
        fail_first_pass: bool = False,
        raise_on_call: int | None = None,
        interrupted: bool = False,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail_first_pass = fail_first_pass
        self.raise_on_call = raise_on_call
        self.interrupted = interrupted

    def __call__(self, *_args: Any, **kwargs: Any) -> list[ThroughputSample]:
        self.calls.append(kwargs)
        if len(self.calls) == self.raise_on_call:
            raise PerfPassFailed(
                "llama-benchy error: HTTP 400 from http://secret.test/v1",
                interrupted=self.interrupted,
            )
        error = "HTTP 503" if self.fail_first_pass and len(self.calls) == 1 else None
        return [
            _sample(depth=depth, concurrency=level, requested_depth=depth, error=error)
            for depth in kwargs["depths"]
            for level in kwargs["concurrency_levels"]
        ]


@pytest.fixture
def persisted(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    monkeypatch.setattr(run_queries, "persist_run", rows.append)
    return rows


def _context(max_model_len: int) -> RunContext:
    return RunContext(
        tool_version="0.0.0",
        git_sha=None,
        hostname="host",
        platform_info="Linux",
        python_version="3.13.0",
        model="m",
        backend="vllm",
        base_url="http://bench.test/v1",
        max_model_len=max_model_len,
    )


def test_long_runs_two_passes_and_stores_one_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, persisted: list[dict[str, Any]]
) -> None:
    benchy = _Benchy()
    monkeypatch.setattr(dispatch, "_run_llama_benchy", benchy)
    target = _target(["--perf-only", "--perf-preset", "long"], tmp_path, _context(40000))

    samples, finished = dispatch._run_throughput_mode(target)

    assert finished is True
    assert [(c["depths"], c["concurrency_levels"]) for c in benchy.calls] == [
        ([0], [1, 2, 4]),
        ([16384, 32768], [1]),
    ]
    assert [c["enable_prefix_caching"] for c in benchy.calls] == [False, True]
    assert [(s.depth, s.concurrency) for s in samples] == [
        (0, 1),
        (0, 2),
        (0, 4),
        (16384, 1),
        (32768, 1),
    ]
    (row,) = persisted
    config = row["config"]
    assert config["perf_preset"] == "long"
    assert config["depths"] == [0, 16384, 32768]
    assert config["passes"] == [
        {"depths": [0], "concurrency": [1, 2, 4], "prefix_caching": False},
        {"depths": [16384, 32768], "concurrency": [1], "prefix_caching": True},
    ]
    assert len(row["scores"]["results"]) == 5
    output = target.console.export_text()
    assert "Skipping depth 65,536, 131,072" in output
    assert "Depth pass estimate: at least" in output


def test_long_warns_when_the_window_is_unknown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    benchy = _Benchy()
    monkeypatch.setattr(dispatch, "_run_llama_benchy", benchy)
    target = _target(["--perf", "--perf-preset", "long"], tmp_path)

    dispatch._run_throughput_mode(target)

    assert benchy.calls[1]["depths"] == list(LONG_DEPTHS)
    assert "context window is unknown" in target.console.export_text()


def test_long_skips_the_depth_pass_when_the_first_pass_failed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    benchy = _Benchy(fail_first_pass=True)
    monkeypatch.setattr(dispatch, "_run_llama_benchy", benchy)
    target = _target(["--perf", "--perf-preset", "long"], tmp_path)

    samples, _ = dispatch._run_throughput_mode(target)

    assert len(benchy.calls) == 1
    assert all(sample.error for sample in samples)
    assert "Skipping the depth pass" in target.console.export_text()


def test_skipped_depth_pass_is_not_stored(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, persisted: list[dict[str, Any]]
) -> None:
    monkeypatch.setattr(dispatch, "_run_llama_benchy", _Benchy(fail_first_pass=True))
    target = _target(["--perf-only", "--perf-preset", "long"], tmp_path)

    with pytest.raises(SystemExit) as exited:
        dispatch._run_throughput_mode(target)

    assert exited.value.code == 1
    (row,) = persisted
    assert "passes" not in row["config"]
    assert (row["config"]["depths"], row["config"]["concurrency"]) == ([0], [1, 2, 4])


def test_failed_depth_pass_keeps_the_first_pass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, persisted: list[dict[str, Any]]
) -> None:
    monkeypatch.setattr(dispatch, "_run_llama_benchy", _Benchy(raise_on_call=2))
    target = _target(["--perf-only", "--perf-preset", "long", "--depth", "16384,32768"], tmp_path)

    with pytest.raises(SystemExit) as exited:
        dispatch._run_throughput_mode(target)

    assert exited.value.code == 1
    (row,) = persisted
    assert row["status"] == "failed"
    assert {key: row["scores"][key] for key in ("samples", "successful", "failed")} == {
        "samples": 5,
        "successful": 3,
        "failed": 2,
    }
    assert len(row["config"]["passes"]) == 2
    output = target.console.export_text()
    assert "Pass 2 stopped" in output
    assert "counting its 2 cell(s) as failed" in output


def test_failed_depth_pass_under_perf_hands_on_every_sample(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, persisted: list[dict[str, Any]]
) -> None:
    monkeypatch.setattr(dispatch, "_run_llama_benchy", _Benchy(raise_on_call=2))
    target = _target(["--perf", "--perf-preset", "long", "--depth", "16384"], tmp_path)

    samples, finished = dispatch._run_throughput_mode(target)

    assert finished is False
    assert persisted == []
    assert [(s.depth, s.concurrency, bool(s.error)) for s in samples] == [
        (0, 1, False),
        (0, 2, False),
        (0, 4, False),
        (16384, 1, True),
    ]
    # Fixed text: llama-benchy's own message can quote the server URL.
    assert "secret.test" not in (samples[-1].error or "")


def test_interrupted_depth_pass_saves_the_first_pass_then_exits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, persisted: list[dict[str, Any]]
) -> None:
    benchy = _Benchy(raise_on_call=2, interrupted=True)
    monkeypatch.setattr(dispatch, "_run_llama_benchy", benchy)
    # --perf would hand its samples to a scored run; Ctrl+C saves them instead.
    target = _target(["--perf", "--perf-preset", "long", "--depth", "16384"], tmp_path)

    with pytest.raises(SystemExit) as exited:
        dispatch._run_throughput_mode(target)

    assert exited.value.code == 1
    (row,) = persisted
    assert row["status"] == "failed"
    assert row["scores"]["successful"] == 3
    assert target.console.export_text().count("Interrupted.") == 1


def test_first_pass_failure_still_ends_the_cli_without_saving(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, persisted: list[dict[str, Any]]
) -> None:
    monkeypatch.setattr(dispatch, "_run_llama_benchy", _Benchy(raise_on_call=1))
    target = _target(["--perf-only", "--perf-preset", "long"], tmp_path)

    with pytest.raises(SystemExit) as exited:
        dispatch._run_throughput_mode(target)

    assert exited.value.code == 1
    assert persisted == []
    assert "HTTP 400" in target.console.export_text()


def test_window_too_small_for_one_request_is_flagged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(dispatch, "_run_llama_benchy", _Benchy())
    target = _target(["--perf", "--perf-preset", "long"], tmp_path, _context(2000))

    dispatch._run_throughput_mode(target)

    output = target.console.export_text()
    assert "does not fit the 2,000-token context window, so even depth-0" in output
    assert "Skipping depth 0" not in output


def test_short_run_passes_explicit_overrides_to_one_invocation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    benchy = _Benchy()
    monkeypatch.setattr(dispatch, "_run_llama_benchy", benchy)
    flags = [
        "--perf",
        "--pp",
        "512",
        "--depth",
        "0,1024",
        "--concurrency",
        "2",
        "--benchy-runs",
        "1",
    ]

    dispatch._run_throughput_mode(_target(flags, tmp_path))

    (call,) = benchy.calls
    assert (call["pp"], call["tg"], call["depths"], call["concurrency_levels"], call["runs"]) == (
        [512],
        [128],
        [0, 1024],
        [2],
        1,
    )
    assert call["enable_prefix_caching"] is False


def _stored_fingerprint(flags: list[str], tmp_path: Path) -> tuple[dict[str, Any], str]:
    config = dispatch._throughput_config(_target(flags, tmp_path))
    unhashed = {key: value for key, value in config.items() if key != "perf_preset"}
    return config, with_config_fingerprint(unhashed)["config_fingerprint"]


def test_short_preset_config_matches_the_explicit_pre_preset_flags(tmp_path: Path) -> None:
    default, default_fp = _stored_fingerprint(["--perf-only"], tmp_path)
    explicit, explicit_fp = _stored_fingerprint(
        [
            "--perf-only",
            "--pp",
            "2048",
            "--tg",
            "128",
            "--depth",
            "0,4096,8192",
            "--concurrency",
            "1,2,4",
            "--benchy-runs",
            "3",
        ],
        tmp_path,
    )

    assert default == explicit
    assert default_fp == explicit_fp
    assert "passes" not in default


def test_long_preset_lands_in_a_different_cohort(tmp_path: Path) -> None:
    _, short_fp = _stored_fingerprint(["--perf-only"], tmp_path)
    _, long_fp = _stored_fingerprint(["--perf-only", "--perf-preset", "long"], tmp_path)

    assert short_fp != long_fp


def test_spec_bench_keeps_the_short_workload_whatever_the_preset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}

    class _Stop(Exception):
        pass

    def fake_spec_bench(*_args: Any, **kwargs: Any) -> None:
        captured.update(kwargs)
        raise _Stop

    monkeypatch.setattr(dispatch, "_run_spec_bench", fake_spec_bench)
    with pytest.raises(_Stop):
        dispatch._run_spec_bench_mode(_target(["--spec-bench", "--perf-preset", "long"], tmp_path))

    assert (captured["pp"], captured["tg"], captured["depths"]) == (2048, 128, [0, 4096, 8192])


# -- scored run -----------------------------------------------------------------------


async def _no_metadata(*_args: Any) -> dict[str, Any]:
    return {}


@pytest.mark.parametrize(
    ("preset_flags", "stored"), [([], "short"), (["--perf-preset", "long"], "long")]
)
def test_scored_perf_run_records_the_preset(
    preset_flags: list[str],
    stored: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for key in list(os.environ):
        if key.startswith("TOOL_EVAL_"):
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dispatch, "_load_dotenv", lambda: None)
    monkeypatch.setattr(dispatch, "_run_llama_benchy", _Benchy())
    monkeypatch.setattr(dispatch, "_build_run_context", lambda *a, **k: None)
    monkeypatch.setattr(service_module, "_collect_metadata_safe", _no_metadata)
    monkeypatch.setattr(BenchmarkService, "_adapter_for", lambda *_a, **_k: SmokeMockAdapter())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tool-eval-bench",
            "--base-url",
            "http://localhost:9999",
            "--model",
            "perf-model",
            "--backend",
            "vllm",
            "--scenarios",
            "TC-01",
            "--perf",
            *preset_flags,
            "--no-preflight",
            "--no-warmup",
            "--no-probe-engine",
            "--no-live",
            "--output-dir",
            str(tmp_path / "runs"),
        ],
    )

    dispatch.main()

    capsys.readouterr()
    (row,) = open_repository(db_path=str(tmp_path / "data" / "benchmarks.sqlite")).list()
    assert row["config"]["perf_preset"] == stored
    assert row["scores"]["throughput"]["samples"] > 0


def test_service_stores_no_preset_without_samples() -> None:
    from tool_eval_bench.application.run_config import build_run_config
    from tool_eval_bench.cli.scored_run import ScoredRun

    args = _make_parser().parse_args(["--perf", "--perf-preset", "long"])
    request = ScoredRun.from_args(
        args,
        model="m",
        backend="vllm",
        base_url="http://bench.test/v1",
        api_key=None,
        wire_format="openai",
        scenarios=[],
        extra_params=None,
        scenario_packs=None,
    )
    assert request.service_kwargs()["perf_preset"] == "long"
    no_perf = _make_parser().parse_args(["--perf-preset", "long"])
    assert (
        ScoredRun.from_args(
            no_perf,
            model="m",
            backend="vllm",
            base_url="http://bench.test/v1",
            api_key=None,
            wire_format="openai",
            scenarios=[],
            extra_params=None,
            scenario_packs=None,
        ).perf_preset
        is None
    )
    # The builder writes whatever it is given; the service passes None without samples.
    assert "perf_preset" not in build_run_config(request.run_settings(), scenarios=[], metadata={})
