"""Golden artifacts for every non-scored run mode: the stored row and the report.

Each case drives a mode through its dispatch-level entry point with only the
network faked, then pins two things byte for byte: the run_data handed to
``run_queries.persist_run`` (config with its fingerprint, metadata, scores, key
order) and the Markdown report.  The random run ID and the wall-clock date are
the only values normalised.

The expected values live in ``tests/golden/mode_runs/<case>.json``.  To
regenerate after a deliberate, changelogged change, run::

    TOOL_EVAL_REGEN_GOLDENS=1 pytest tests/test_mode_run_golden.py

Without the variable a missing or different golden fails the test.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from rich.console import Console

from tool_eval_bench.domain.models import RunContext
from tool_eval_bench.domain.plugin import BenchmarkResult
from tool_eval_bench.domain.scenarios import ScenarioResult, ScenarioStatus

GOLDEN_DIR = Path(__file__).parent / "golden" / "mode_runs"
REGEN = os.environ.get("TOOL_EVAL_REGEN_GOLDENS") == "1"
BASE_URL = "http://user:secret@gpu-box.internal:8000/v1"
LABEL = "golden | label"


def _context() -> RunContext:
    return RunContext(
        tool_version="0.0.0+golden",
        git_sha="abc1234",
        hostname="golden-host",
        platform_info="Linux-golden",
        python_version="3.13.0",
        model="m",
        backend="vllm",
        base_url="http://***:8000/v1",
        temperature=0.0,
        max_turns=8,
        timeout_seconds=120.0,
        seed=7,
        scenario_selector="TC-01,TC-02",
        parallel=1,
        label=LABEL,
        server_model_id="org/model",
        server_model_root="/models/org/model",
        engine_name="vLLM",
        engine_version="0.11.0",
        max_model_len=32_768,
        quantization="fp8",
        gpu_count=2,
        spec_decoding="eagle3",
        slot_count=4,
    )


def _args(out: Path, *flags: str) -> argparse.Namespace:
    from tool_eval_bench.cli.legacy_parser import _make_parser

    return _make_parser().parse_args([*flags, "--output-dir", str(out)])


def _target(args: argparse.Namespace, run_context: RunContext | None) -> Any:
    from tool_eval_bench.cli.dispatch import _Target

    return _Target(
        args=args,
        parser=argparse.ArgumentParser(),
        console=Console(record=True, width=200),
        model="m",
        display_name="Golden Model",
        backend="vllm",
        base_url=BASE_URL,
        display_url="http://***:8000/v1",
        api_key=None,
        wire_format="openai",
        extra_params={},
        run_context=run_context,
    )


@pytest.fixture
def persisted(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    from tool_eval_bench.application import run_queries

    rows: list[dict[str, Any]] = []
    monkeypatch.setattr(run_queries, "persist_run", rows.append)
    return rows


def _normalise(row: dict[str, Any], out: Path) -> dict[str, Any]:
    run_id = row["run_id"]
    report_path = Path(row["report_path"])
    report = report_path.read_text(encoding="utf-8")
    report = report.replace(run_id, "<RUN_ID>")
    report = re.sub(r"^- \*\*Date\*\*: `[^`]+`$", "- **Date**: `<DATE>`", report, flags=re.M)
    relative = report_path.relative_to(out).as_posix().replace(run_id, "<RUN_ID>")
    relative = re.sub(r"^\d{4}/\d{2}/", "<YYYY>/<MM>/", relative)
    stored = json.loads(json.dumps(row).replace(run_id, "<RUN_ID>"))
    stored["report_path"] = relative
    return {"run_data": stored, "report": report}


def _check(name: str, actual: dict[str, Any]) -> None:
    path = GOLDEN_DIR / f"{name}.json"
    text = json.dumps(actual, indent=2, ensure_ascii=False) + "\n"
    if REGEN:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return
    assert path.exists(), f"missing golden {path}; regenerate with TOOL_EVAL_REGEN_GOLDENS=1"
    expected = json.loads(path.read_text(encoding="utf-8"))
    assert actual["run_data"] == expected["run_data"]
    # Key order is part of the stored JSON, so compare it as well as values.
    assert json.dumps(actual["run_data"]) == json.dumps(expected["run_data"])
    assert actual["report"] == expected["report"]


def _one_row(rows: list[dict[str, Any]], out: Path) -> dict[str, Any]:
    assert len(rows) == 1
    return _normalise(rows[0], out)


def _contexts() -> list[Any]:
    return [pytest.param(True, id="context"), pytest.param(False, id="no-context")]


# -- context-pressure sweep ---------------------------------------------------------


def _scenario_result(sid: str, status: ScenarioStatus) -> ScenarioResult:
    return ScenarioResult(
        scenario_id=sid,
        status=status,
        points={"pass": 2, "partial": 1, "fail": 0}[status.value],
        summary=f"{sid} {status.value}",
        raw_log=f"trace for {sid}",
        tool_calls_made=["get_weather(Berlin)"],
        expected_behavior="call get_weather",
        duration_seconds=1.5,
        turn_count=2,
    )


def _patch_sweep(monkeypatch: pytest.MonkeyPatch, *, fail_level: int | None = None) -> None:
    from tool_eval_bench.adapters import factory
    from tool_eval_bench.runner import context_pressure, orchestrator

    async def calibrate(messages: Any, target: int, *args: Any, **kwargs: Any) -> Any:
        return messages, target

    levels = iter(range(10))

    class Summary:
        def __init__(self, results: list[ScenarioResult]) -> None:
            self.scenario_results = results

    async def run_all(adapter: Any, **kwargs: Any) -> Summary:
        level = next(levels)
        if level == fail_level:
            raise RuntimeError("level exploded")
        second = ScenarioStatus.PASS if level == 0 else ScenarioStatus.PARTIAL
        return Summary(
            [
                _scenario_result("TC-01", ScenarioStatus.PASS),
                _scenario_result("TC-02", second),
            ]
        )

    monkeypatch.setattr(context_pressure, "calibrate_pressure_messages", calibrate)
    monkeypatch.setattr(orchestrator, "run_all_scenarios", run_all)
    monkeypatch.setattr(factory, "build_adapter", lambda *a, **k: object())


SWEEP_FLAGS = (
    "--context-pressure-sweep",
    "0.5-0.9",
    "--sweep-steps",
    "3",
    "--context-size",
    "32768",
    "--scenarios",
    "TC-01",
    "TC-02",
    "--seed",
    "7",
)


@pytest.mark.parametrize("with_context", _contexts())
@pytest.mark.parametrize(
    ("name", "flags", "fail_level"),
    [
        ("sweep_default", (), None),
        ("sweep_prompt_label", ("--system-prompt", "Be terse.", "--label", LABEL), None),
        ("sweep_level_error", (), 1),
    ],
)
def test_pressure_sweep(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    persisted: list[dict[str, Any]],
    name: str,
    flags: tuple[str, ...],
    fail_level: int | None,
    with_context: bool,
) -> None:
    from tool_eval_bench.cli.dispatch import _run_pressure_sweep_mode

    _patch_sweep(monkeypatch, fail_level=fail_level)
    args = _args(tmp_path, *SWEEP_FLAGS, *flags)

    assert _run_pressure_sweep_mode(_target(args, _context() if with_context else None))

    suffix = "" if with_context else "_no_context"
    _check(name + suffix, _one_row(persisted, tmp_path))


# -- spec-bench ---------------------------------------------------------------------


def _spec_samples(source: str, runs: int) -> list[Any]:
    from tool_eval_bench.runner.speculative import SpecDecodeSample

    def sample(prompt: str, depth: int, rate: float) -> SpecDecodeSample:
        return SpecDecodeSample(
            pp_tokens=512,
            tg_tokens=128,
            depth=depth,
            ttft_ms=120.0,
            total_ms=1500.0,
            pp_tps=4000.0,
            tg_tps=90.0,
            acceptance_rate=rate,
            acceptance_length=2.5,
            draft_tokens_delta=300,
            accepted_tokens_delta=int(300 * rate),
            num_drafts_delta=100,
            acceptance_source=source,
            num_spec_tokens=3,
            per_step_accepted=[90, 70, 50],
            per_step_drafted=[100, 100, 100],
            spec_method="eagle3",
            prompt_type=prompt,
            runs=runs,
        )

    return [sample("filler", 0, 0.8), sample("code", 4096, 0.6)]


@pytest.mark.parametrize("with_context", _contexts())
@pytest.mark.parametrize(
    ("name", "source", "runs", "flags"),
    [
        ("spec_response", "response", 1, ()),
        ("spec_prometheus_runs_label", "prometheus", 3, ("--spec-runs", "3", "--label", LABEL)),
    ],
)
def test_spec_bench(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    persisted: list[dict[str, Any]],
    name: str,
    source: str,
    runs: int,
    flags: tuple[str, ...],
    with_context: bool,
) -> None:
    from tool_eval_bench.cli.dispatch import _run_spec_bench_mode
    from tool_eval_bench.runner import speculative

    samples = _spec_samples(source, runs)

    async def fake_run(*args: Any, on_sample: Callable[..., Any], **kwargs: Any) -> list[Any]:
        for index, sample in enumerate(samples):
            await on_sample(sample, index, len(samples))
        return samples

    monkeypatch.setattr(speculative, "run_spec_bench", fake_run)
    args = _args(
        tmp_path, "--spec-bench", "--depth", "0,4096", "--spec-prompts", "filler,code", *flags
    )

    assert _run_spec_bench_mode(_target(args, _context() if with_context else None))

    suffix = "" if with_context else "_no_context"
    _check(name + suffix, _one_row(persisted, tmp_path))


# -- throughput (--perf-only) -------------------------------------------------------


def _throughput_samples(failed: bool) -> list[Any]:
    from tool_eval_bench.runner.throughput import ThroughputSample

    samples = [
        ThroughputSample(
            pp_tokens=512,
            tg_tokens=128,
            depth=0,
            concurrency=1,
            ttft_ms=100.0,
            total_ms=1400.0,
            pp_tps=5000.0,
            tg_tps=95.0,
            requested_pp=512,
            requested_depth=0,
        )
    ]
    if failed:
        samples.append(ThroughputSample(concurrency=4, error="HTTP 503", requested_pp=512))
    return samples


@pytest.mark.parametrize("with_context", _contexts())
@pytest.mark.parametrize("failed", [False, True], ids=["ok", "failed-cell"])
def test_throughput(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    persisted: list[dict[str, Any]],
    failed: bool,
    with_context: bool,
) -> None:
    from tool_eval_bench.cli import dispatch

    monkeypatch.setattr(dispatch, "_run_llama_benchy", lambda *a, **k: _throughput_samples(failed))
    args = _args(tmp_path, "--perf-only", "--label", LABEL)
    target = _target(args, _context() if with_context else None)

    if failed:
        with pytest.raises(SystemExit) as exited:
            dispatch._run_throughput_mode(target)
        assert exited.value.code == 1
    else:
        assert dispatch._run_throughput_mode(target)[1] is True

    name = "throughput_failed_cell" if failed else "throughput_ok"
    suffix = "" if with_context else "_no_context"
    _check(name + suffix, _one_row(persisted, tmp_path))


# -- dataset plugins ----------------------------------------------------------------


def _result(name: str, details: dict[str, Any], **extra: Any) -> BenchmarkResult:
    return BenchmarkResult(
        name,
        66.7,
        "66.7%",
        "★★★ Adequate",
        details=details,
        duration_seconds=12.0,
        total_tokens=900,
        **extra,
    )


def _patch_plugin(monkeypatch: pytest.MonkeyPatch, cls: type, result: BenchmarkResult) -> None:
    async def fake_run(self: Any, adapter: Any, **kwargs: Any) -> BenchmarkResult:
        return result

    monkeypatch.setattr(cls, "run", fake_run)


def _gsm8k(monkeypatch: pytest.MonkeyPatch) -> tuple[Callable[..., Any], tuple[str, ...]]:
    from tool_eval_bench.cli import plugin_runners
    from tool_eval_bench.plugins.gsm8k.plugin import GSM8KPlugin

    _patch_plugin(
        monkeypatch,
        GSM8KPlugin,
        _result(
            "gsm8k",
            {"total": 3, "correct": 2, "errors": 0, "answered": 3, "completion_rate": 100.0},
            item_results=[
                {"index": 0, "question": "q0", "correct": True, "extracted_answer": 1.0},
                {"index": 1, "question": "q1", "correct": True, "extracted_answer": 2.0},
                {
                    "index": 2,
                    "question": "q2",
                    "correct": False,
                    "extracted_answer": 9.0,
                    "ground_truth": 3.0,
                    "extraction_method": "hash",
                    "response": "work #### 9",
                },
            ],
        ),
    )
    return plugin_runners._run_gsm8k_benchmark, ("--gsm8k-only", "--gsm8k-limit", "3")


def _mmlu(monkeypatch: pytest.MonkeyPatch) -> tuple[Callable[..., Any], tuple[str, ...]]:
    from tool_eval_bench.cli import plugin_runners
    from tool_eval_bench.plugins.mmlu.plugin import MMLUPlugin

    _patch_plugin(
        monkeypatch,
        MMLUPlugin,
        _result(
            "mmlu",
            {
                "total": 3,
                "correct": 2,
                "errors": 0,
                "answered": 3,
                "completion_rate": 100.0,
                "categories": {"STEM": {"correct": 2, "total": 3, "accuracy": 66.7}},
            },
        ),
    )
    return plugin_runners._run_mmlu_benchmark, ("--mmlu-only", "--mmlu-limit", "3")


def _ifeval(monkeypatch: pytest.MonkeyPatch) -> tuple[Callable[..., Any], tuple[str, ...]]:
    from tool_eval_bench.cli import plugin_runners
    from tool_eval_bench.plugins.ifeval.plugin import IFEvalPlugin

    _patch_plugin(
        monkeypatch,
        IFEvalPlugin,
        _result(
            "ifeval",
            {
                "total": 3,
                "prompts_passed": 2,
                "errors": 0,
                "answered": 3,
                "completion_rate": 100.0,
                "prompt_accuracy": 66.7,
                "instructions_passed": 5,
                "instructions_total": 6,
                "instruction_accuracy": 83.3,
            },
        ),
    )
    return plugin_runners._run_ifeval_benchmark, ("--ifeval-only", "--ifeval-limit", "3")


def _needle(monkeypatch: pytest.MonkeyPatch) -> tuple[Callable[..., Any], tuple[str, ...]]:
    from tool_eval_bench.cli import plugin_runners
    from tool_eval_bench.plugins.needle.plugin import NeedlePlugin

    _patch_plugin(
        monkeypatch,
        NeedlePlugin,
        _result(
            "needle",
            {
                "total": 2,
                "retrieved": 1,
                "errors": 0,
                "answered": 2,
                "completion_rate": 100.0,
                "accuracy": 50.0,
                "context_size": 8192,
                "context_lengths": [1024, 5120],
                "depths": [0.0, 1.0],
                "effective_context": 1024,
            },
            item_results=[
                {"context_tokens": 1024, "depth_percent": 0.0, "found": True},
                {
                    "context_tokens": 5120,
                    "depth_percent": 1.0,
                    "found": False,
                    "cell_id": "5120@100%",
                    "expected": "MAGIC-42",
                    "model_response": "I could not find it | sorry",
                },
            ],
        ),
    )
    flags = ("--needle-only", "--context-size", "8192", "--needle-lengths", "2")
    return plugin_runners._run_needle_benchmark, (*flags, "--needle-depths", "2")


def _decision(monkeypatch: pytest.MonkeyPatch) -> tuple[Callable[..., Any], tuple[str, ...]]:
    from test_decision_typed import FakeTypedBackend

    import tool_eval_bench.adapters.factory as factory
    from tool_eval_bench.cli import plugin_runners
    from tool_eval_bench.plugins.decision import plugin as decision_plugin
    from tool_eval_bench.plugins.decision import typed_decisions

    cases = typed_decisions.load_cases()[:3]
    monkeypatch.setattr(typed_decisions, "load_cases", lambda *a, **k: cases)
    monkeypatch.setattr(factory, "build_decision_adapter", lambda **_kw: FakeTypedBackend(cases))
    # The duration is wall-clock; pin it so the report and row are reproducible.
    ticks = iter([100.0, 112.0])
    monkeypatch.setattr(decision_plugin, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    return plugin_runners._run_decision_benchmark, ("--decision-bench-only",)


PLUGINS = {
    "gsm8k": _gsm8k,
    "mmlu": _mmlu,
    "ifeval": _ifeval,
    "needle": _needle,
    "decision": _decision,
}


@pytest.fixture
def _offline_datasets(monkeypatch: pytest.MonkeyPatch) -> None:
    from tool_eval_bench.adapters import factory
    from tool_eval_bench.cli import plugin_runners

    monkeypatch.setattr(plugin_runners, "load_dataset_with_progress", lambda *a, **k: [object()])
    monkeypatch.setattr(factory, "build_adapter", lambda *a, **k: object())


@pytest.mark.usefixtures("_offline_datasets")
@pytest.mark.parametrize("with_context", _contexts())
@pytest.mark.parametrize("plugin", list(PLUGINS))
def test_plugin(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    persisted: list[dict[str, Any]],
    plugin: str,
    with_context: bool,
) -> None:
    runner, flags = PLUGINS[plugin](monkeypatch)
    args = _args(tmp_path, *flags, "--seed", "7", "--label", LABEL)

    runner(
        Console(record=True, width=200),
        "m",
        "Golden Model",
        BASE_URL,
        None,
        args,
        extra_params=None,
        output_dir=str(tmp_path),
        run_context=_context() if with_context else None,
    )

    suffix = "" if with_context else "_no_context"
    _check(f"plugin_{plugin}{suffix}", _one_row(persisted, tmp_path))


def test_every_golden_file_has_a_case() -> None:
    """A renamed case must not leave a stale golden behind."""
    expected = (
        {
            f"{base}{suffix}"
            for base in ("sweep_default", "sweep_prompt_label", "sweep_level_error")
            for suffix in ("", "_no_context")
        }
        | {
            f"{base}{suffix}"
            for base in ("spec_response", "spec_prometheus_runs_label")
            for suffix in ("", "_no_context")
        }
        | {
            f"{base}{suffix}"
            for base in ("throughput_ok", "throughput_failed_cell")
            for suffix in ("", "_no_context")
        }
        | {f"plugin_{name}{suffix}" for name in PLUGINS for suffix in ("", "_no_context")}
    )
    assert {path.stem for path in GOLDEN_DIR.glob("*.json")} == expected
