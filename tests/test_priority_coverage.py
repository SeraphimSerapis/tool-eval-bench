"""High-value deterministic coverage for reporting, performance CLI, and datasets."""

from __future__ import annotations

from pathlib import Path

import pytest
from rich.console import Console

from tool_eval_bench.cli.perf import run_llama_benchy, run_throughput
from tool_eval_bench.cli.spec_bench import run_spec_bench as run_spec_bench_cli
from tool_eval_bench.compare_reports import summary, tool_eval
from tool_eval_bench.runner.speculative import SpecDecodeSample
from tool_eval_bench.runner.throughput import ThroughputMatrixResult, ThroughputSample


def test_tool_eval_parser_and_helpers_cover_complete_report(tmp_path: Path) -> None:
    report = tmp_path / "run.md"
    report.write_text(
        """# Tool-Call Benchmark — org/model-q8.gguf
**Run ID**: `run-1`
**Date**: `2026-07-15T12:00:00Z`
**tool-eval-bench**: `2.1.0`
**Final Score**: **88**
**Total Points**: 88 / 100
**Rating**: Great
**Deployability**: **84**
**Quality**: 88 / 100
**Responsiveness**: 76 / 100
median turn: 1.5s
**Backend** | vllm
**Model (API)** | `org/model-q8.gguf`
**Model (Root)** | `org/root`
**Temperature** | 0
**Thinking** | off

## Category Scores

| Category | Earned | Max | Percent |
| --- | --- | --- | --- |
| Safety | 8 | 10 | 80% |

## Scenario Results

| ID | Title | Difficulty | Status | Points | Summary |
| --- | --- | --- | --- | --- | --- |
| TC-01 | Pass case | ★ | ✅ pass | 2/2 | fine |
| TC-02 | Partial case | ★★ | ⚠ partial | 1/2 | close |
| TC-03 | Fail case | ★★★ | ❌ fail | 0/2 | bad |

## Performance by Difficulty

| Tier | Scenarios | Passed | Rate |
| --- | --- | --- | --- |
| Easy | 2 | 1 | 50% |
| Hard | 1 | 0 | invalid |

> [!WARNING]
> **1 safety-critical failure**
> - **TC-03** (unsafe): leaked <secret>

End
""",
        encoding="utf-8",
    )

    parsed = tool_eval.parse_md(str(report))

    assert parsed["final_score"] == 88
    assert [s["status"] for s in parsed["scenarios"]] == ["pass", "partial", "fail"]
    assert parsed["difficulties"][1]["rate"] == 0
    assert parsed["safety_critical"] == [
        {"id": "TC-03", "type": "unsafe", "desc": "leaked <secret>"}
    ]
    assert tool_eval.short_label("model", "path/model-Q8_K_XL.gguf")[0] == "GGUF Q8"
    assert tool_eval.short_label("model", "path/model.gguf")[1] == "Q8_K_XL GGUF quantization"
    assert tool_eval.short_label("name", "api") == ("api", "api")
    assert tool_eval.dname({"model_api": "api", "model_name": "name"}) == "api"
    assert tool_eval.esc('<a x="1">&') == "&lt;a x=&quot;1&quot;&gt;&amp;"
    assert tool_eval.sign(1) == "+1"
    assert tool_eval.sign(-1) == "-1"
    assert tool_eval.pct_cls(2, 1).startswith("font-semibold")
    assert tool_eval.pct_cls(1, 2) == "text-rose-600"
    assert tool_eval.pct_cls(1, 1) == ""
    assert tool_eval.diff_display(2, 1) == ("+1", "diff-positive")
    assert tool_eval.diff_display(1, 2) == ("-1", "diff-negative")
    assert tool_eval.diff_display(1, 1)[0] == "—"
    assert tool_eval.turn_time_display(1.0, 2.0, "", "")[3] == "diff-positive"
    assert tool_eval.turn_time_display(None, None, "raw", "")[0:2] == ("raw", "—")


def test_summary_parser_and_helpers_cover_reliability_sections(tmp_path: Path) -> None:
    report = tmp_path / "summary.md"
    report.write_text(
        """# Cross-Trial Summary — deepseek-ai/deepseek-v4-flash-dspark
**Run ID**: `sum-1`
**Date**: `2026-07-15T12:00:00Z`
**tool-eval-bench**: `2.1.0`
**Trials**: 2
**Backend** | llamacpp
**Model (API)** | `model-nvfp4`
**Model (Root)** | `root`
**Temperature** | 0
**Thinking** | on
| **Final Score** | 80 | **85.5 ± 1.5** |
| **Total Points** | 80 | **90.0 ± 2.0** |
| **Rating** | Good | **Great** |
| **Safety Warnings** | 1 |
| **Pass@8** | 95.0% |
| **Pass^8** | 70.0% |
| **Reliability Gap** | 25.0pp |
| **95% CI** | [82.0, 89.0] |
| **Quality** | 86 / 100 |
| **Responsiveness** | 75 / 100 |
| **Deployability** | **82** |
| **Median Turn** | 1.2s |

## Category Variance

| Category | T1 | T2 | Variance |
| --- | --- | --- | --- |
| Safety | 80% | 100% | low |

## Per-Scenario Results

| ID | T1 | T2 | Pass@k | Pass^k |
| --- | --- | --- | --- | --- |
| TC-01 | ✅ | ⚠️ | ✓ | ✗ |
| TC-02 | ❌ | ❌ | ✗ | ✗ |

### ❌ Never Passes

| **TC-02** | connection error for url host |

### 🔀 Flaky

| **TC-03** | ✅ ❌ |

### ⚠️ Consistently Partial

| TC-04 | almost there |
""",
        encoding="utf-8",
    )

    parsed = summary.parse_summary(str(report))

    assert parsed["mean_score"] == 85.5
    assert parsed["rating"] == "**Great**"
    assert parsed["categories"] == [{"name": "Safety", "mean": 90, "variance": ""}]
    assert parsed["scenarios"][0]["passes"] == 1
    assert parsed["never_passes"][0]["id"] == "TC-02"
    assert parsed["flaky"][0]["id"] == "TC-03"
    assert parsed["consistent_partials"][0]["id"] == "TC-04"
    assert summary.short_label("nvidia fp4", "api")[0] == "NVFP4"
    assert summary.short_label("gguf", "api")[0] == "GGUF Q8"
    assert summary.short_label("deepseek-ai/model", "api")[0] == "model"
    assert summary.sign(0) == "+0"
    assert summary.pct_cls(2, 1).startswith("font-semibold")
    assert summary.pct_cls(1, 2) == "text-rose-600"
    assert summary.diff_display(1, 1)[0] == "—"
    assert summary.turn_time_display(2.0, 1.0, "", "")[3] == "diff-negative"
    assert summary._pct_or_dash(None) == "—"
    assert summary._pct_or_dash("2") == "2.0"
    assert summary._pp_or_dash(None) == "—"
    assert summary._pp_or_dash(2) == "2.0pp"
    assert summary._is_infrastructure_failure(parsed) is False
    parsed["mean_score"] = 0.0
    assert summary._is_infrastructure_failure(parsed) is True
    assert "1 scenarios" in summary._infrastructure_summary(parsed)
    assert summary._infrastructure_summary({"never_passes": []}).startswith("All scenarios")


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


def test_perf_cli_renders_success_error_and_spec_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    import tool_eval_bench.runner.throughput as throughput

    ok = _sample()
    failed = _sample(error="boom")

    async def fake_matrix(*args: object, on_sample=None, **kwargs: object):
        await on_sample(ok, 0, 2)
        await on_sample(failed, 1, 2)
        return ThroughputMatrixResult(
            samples=[ok, failed],
            spec_decoding_detected=True,
            spec_decoding_method="mtp",
        )

    monkeypatch.setattr(throughput, "run_throughput_matrix", fake_matrix)
    console = Console(record=True, width=140)

    result = run_throughput(
        console,
        "model",
        "Display",
        "http://test/v1",
        None,
        pp=100,
        tg=20,
        depths=[1024],
        concurrency_levels=[2],
    )

    output = console.export_text()
    assert result == [ok, failed]
    assert "Throughput Results" in output
    assert "boom" in output
    assert "Speculative decoding detected (mtp)" in output
    assert "heuristic" in output


def test_llama_benchy_cli_success_and_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    import tool_eval_bench.runner.llama_benchy as benchy

    monkeypatch.setattr(benchy, "is_available", lambda: False)
    with pytest.raises(SystemExit):
        run_llama_benchy(
            Console(), "m", "d", "url", None, pp=[1], tg=[1], depths=[0], concurrency_levels=[1]
        )

    ok = _sample(calibration_confidence="llama-benchy")
    failed = _sample(error="failed")

    async def fake_run(*args: object, on_output=None, **kwargs: object):
        for line in (
            "",
            "Running test: pp100",
            "Run 1/1",
            "Warming up",
            "Measuring latency",
            "Average latency 2ms",
        ):
            on_output(line)
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

    assert result == [ok]
    assert "llama-benchy 1.2.3" in console.export_text()


def test_spec_bench_cli_renders_metrics_and_persists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import tool_eval_bench.runner.speculative as speculative
    from tool_eval_bench.storage import reports

    rich_sample = SpecDecodeSample(
        pp_tokens=100,
        tg_tokens=20,
        depth=1024,
        ttft_ms=10,
        total_ms=110,
        tg_tps=180,
        acceptance_rate=0.7,
        acceptance_length=3.5,
        draft_tokens_delta=40,
        accepted_tokens_delta=28,
        num_drafts_delta=5,
        baseline_tg_tps=100,
        prompt_type="code",
    )
    low_sample = SpecDecodeSample(
        tg_tokens=10,
        total_ms=200,
        acceptance_rate=0.2,
        draft_tokens_delta=10,
        accepted_tokens_delta=2,
        num_drafts_delta=2,
        baseline_tg_tps=100,
        prompt_type="filler",
    )
    failed = SpecDecodeSample(error="failed", prompt_type="structured")

    async def fake_run(*args: object, on_sample=None, **kwargs: object):
        for idx, sample in enumerate((rich_sample, low_sample, failed)):
            await on_sample(sample, idx, 3)
        return [rich_sample, low_sample, failed]

    report_path = tmp_path / "spec.md"
    monkeypatch.setattr(speculative, "run_spec_bench", fake_run)
    monkeypatch.setattr(
        reports.MarkdownReporter,
        "write_spec_decode_report",
        lambda *args, **kwargs: report_path,
    )
    persisted: list[dict] = []
    console = Console(record=True, width=160)

    result = run_spec_bench_cli(
        console,
        "model",
        "Display",
        "http://test/v1",
        None,
        pp=100,
        tg=20,
        depths=[1024],
        baseline_tg_tps=100,
        output_dir=str(tmp_path),
        metadata_for_storage=lambda _: {"source": "test"},
        with_config_fingerprint=lambda config: {**config, "config_fingerprint": "fp"},
        persist_plugin_run=persisted.append,
    )

    output = console.export_text()
    assert result == [rich_sample, low_sample, failed]
    assert "Speculative Decoding Results" in output
    assert "Highest acceptance" in output
    assert "Consider reducing" in output
    assert "failed" in output
    assert persisted[0]["config"]["config_fingerprint"] == "fp"


def test_dataset_cache_roundtrips_and_download_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tool_eval_bench.plugins import hf_utils
    from tool_eval_bench.plugins.gsm8k import dataset as gsm
    from tool_eval_bench.plugins.ifeval import dataset as ifeval
    from tool_eval_bench.plugins.mmlu import dataset as mmlu

    gsm_path = tmp_path / "gsm.jsonl"
    gsm_items = [gsm.GSM8KItem(0, "q", "work #### 1,200", 1200.0)]
    gsm._save_to_cache(gsm_path, gsm_items)
    assert gsm._load_from_cache(gsm_path)[0].to_dict()["ground_truth"] == 1200.0
    assert (
        gsm._rows_to_items(
            [{"question": "q", "answer": "x #### 3"}, {"question": "bad", "answer": "none"}]
        )[0].ground_truth
        == 3
    )

    if_path = tmp_path / "if.jsonl"
    if_items = [ifeval.IFEvalItem(1, "p", ["id"], [{"x": 1}])]
    ifeval._save_to_cache(if_path, if_items)
    assert ifeval._load_from_cache(if_path)[0].to_dict()["key"] == 1
    assert (
        ifeval._rows_to_items(
            [{"key": "2", "prompt": "p2", "instruction_id_list": ["x"], "kwargs": [{}]}]
        )[0].key
        == 2
    )

    mm_path = tmp_path / "mm.jsonl"
    mm_items = [mmlu.MMLUItem(0, "q", "anatomy", ["a", "b", "c", "d"], 1)]
    mmlu._save_to_cache(mm_path, mm_items)
    loaded = mmlu._load_from_cache(mm_path)[0]
    assert loaded.answer_letter == "B"
    assert loaded.category == "STEM"
    assert loaded.to_dict()["answer_letter"] == "B"
    assert mmlu.MMLUItem(1, "q", "unknown", ["a", "b", "c", "d"], 0).category == "Other"

    rows = [
        {
            "question": "q",
            "subject": "anatomy",
            "choices": ["a", "b", "c", "d"],
            "answer": "2",
        }
    ]

    def library_rows(dataset: str, *args: object, **kwargs: object):
        if dataset == "openai/gsm8k":
            return [{"question": "q", "answer": "work #### 3"}]
        if dataset == "google/IFEval":
            return [{"key": "2", "prompt": "p", "instruction_id_list": ["x"], "kwargs": [{}]}]
        return rows

    monkeypatch.setattr(hf_utils, "load_via_datasets_lib", library_rows)
    assert mmlu._download_dataset()[0][0].answer == 2
    assert gsm._download_dataset()[0][0].ground_truth == 3
    assert ifeval._download_dataset()[0][0].key == 2

    monkeypatch.setattr(hf_utils, "load_via_datasets_lib", lambda *args, **kwargs: None)
    monkeypatch.setattr(hf_utils, "download_rows_paginated", lambda *args, **kwargs: rows)
    assert mmlu._download_dataset()[1] == "rest_api"


def test_dataset_load_uses_cache_or_download_and_cleans_partial(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tool_eval_bench.plugins.ifeval import dataset as ifeval
    from tool_eval_bench.plugins.mmlu import dataset as mmlu

    if_cache = tmp_path / "ifeval" / "prompts.jsonl"
    monkeypatch.setattr(ifeval, "_CACHE_DIR", if_cache.parent)
    monkeypatch.setattr(ifeval, "_CACHE_FILE", if_cache)
    monkeypatch.setattr(ifeval, "_find_cache_file", lambda: if_cache)
    item = ifeval.IFEvalItem(1, "p", [], [])
    monkeypatch.setattr(ifeval, "_download_dataset", lambda **kwargs: ([item], "test"))
    assert ifeval.load_dataset(force_download=True) == [item]
    assert ifeval.load_dataset() == [item]

    mm_cache = tmp_path / "mmlu" / "test.jsonl"
    monkeypatch.setattr(mmlu, "_CACHE_DIR", mm_cache.parent)
    monkeypatch.setattr(mmlu, "_find_cache_file", lambda split="test": mm_cache)
    mm_item = mmlu.MMLUItem(0, "q", "anatomy", ["a", "b", "c", "d"], 0)
    monkeypatch.setattr(mmlu, "_download_dataset", lambda **kwargs: ([mm_item], "test"))
    assert mmlu.load_dataset(force_download=True) == [mm_item]
    assert mmlu.load_dataset() == [mm_item]


def test_dispatch_legacy_parser_builds_every_argument_group() -> None:
    from tool_eval_bench.cli.dispatch import _make_parser

    parser = _make_parser()
    args = parser.parse_args(
        [
            "--model",
            "m",
            "--base-url",
            "http://test",
            "--categories",
            "A",
            "--perf",
            "--gsm8k",
            "--mmlu",
            "--ifeval",
            "--spec-bench",
            "--context-pressure",
            "0.5",
        ]
    )

    assert args.model == "m"
    assert args.categories == ["A"]
    assert args.perf and args.gsm8k and args.mmlu and args.ifeval and args.spec_bench


@pytest.mark.parametrize("json_mode", [False, True])
def test_dispatch_main_dry_run_without_server(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], json_mode: bool
) -> None:
    import sys

    from tool_eval_bench.cli import dispatch

    argv = ["tool-eval-bench", "--dry-run", "--short"]
    if json_mode:
        argv.append("--json")
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(dispatch, "_load_dotenv", lambda: None)

    with pytest.raises(SystemExit) as exc:
        dispatch.main()

    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert ("total_scenarios" if json_mode else "Dry run") in output


@pytest.mark.parametrize(
    ("argv", "target"),
    [
        (["--history"], "_print_history"),
        (["--leaderboard"], "_print_leaderboard"),
        (["--export", "json"], "_export_runs"),
        (["--compare", "a", "b"], "_compare_runs"),
    ],
)
def test_dispatch_main_routes_storage_commands(
    monkeypatch: pytest.MonkeyPatch, argv: list[str], target: str
) -> None:
    import sys

    from tool_eval_bench.cli import dispatch

    called: list[tuple] = []
    monkeypatch.setattr(sys, "argv", ["tool-eval-bench", *argv])
    monkeypatch.setattr(dispatch, "_load_dotenv", lambda: None)
    monkeypatch.setattr(dispatch, target, lambda *args, **kwargs: called.append((args, kwargs)))

    dispatch.main()

    assert called


@pytest.mark.asyncio
async def test_speculative_measurement_prometheus_and_llamacpp_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tool_eval_bench.runner import speculative

    counters = iter(
        [
            speculative.SpecDecodeCounters(accepted_tokens=10, draft_tokens=20, num_drafts=5),
            speculative.SpecDecodeCounters(accepted_tokens=24, draft_tokens=40, num_drafts=9),
        ]
    )

    async def scrape(*args: object, **kwargs: object):
        return next(counters)

    async def stream(*args: object, **kwargs: object):
        return _sample(draft_n=10, draft_n_accepted=6)

    monkeypatch.setattr(speculative, "scrape_spec_metrics", scrape)
    monkeypatch.setattr(speculative, "_stream_one", stream)
    sample = await speculative.measure_spec_single(
        MagicAsyncClient(),
        "url",
        "model",
        prompt_type="code",
        spec_info=speculative.SpecDecodeInfo(has_prometheus=True, method="mtp"),
    )
    assert sample.acceptance_rate == 0.7
    assert sample.acceptance_length == 3.5

    monkeypatch.setattr(speculative, "scrape_spec_metrics", lambda *args, **kwargs: None)
    fallback = await speculative.measure_spec_single(
        MagicAsyncClient(),
        "url",
        "model",
        prompt_type="structured",
        spec_info=speculative.SpecDecodeInfo(),
    )
    assert fallback.acceptance_rate == 0.6


class MagicAsyncClient:
    """Marker client for functions whose HTTP collaborators are monkeypatched."""


@pytest.mark.asyncio
async def test_speculative_full_sweep_invokes_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    from tool_eval_bench.runner import speculative

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args: object):
            return None

    monkeypatch.setattr(speculative.httpx, "AsyncClient", lambda **kwargs: Client())
    monkeypatch.setattr(
        speculative,
        "calibrate",
        async_return(speculative.TokenizerConfig()),
    )
    monkeypatch.setattr(
        speculative,
        "detect_spec_decoding",
        async_return(speculative.SpecDecodeInfo(has_per_request_timings=True)),
    )

    async def measure(*args: object, depth=0, prompt_type="filler", **kwargs: object):
        return SpecDecodeSample(depth=depth, prompt_type=prompt_type)

    seen: list[tuple[int, int]] = []

    async def callback(sample: SpecDecodeSample, idx: int, total: int):
        seen.append((idx, total))

    monkeypatch.setattr(speculative, "measure_spec_single", measure)
    samples = await speculative.run_spec_bench(
        "url", "model", depths=[0, 1024], prompt_types=["filler", "code"], on_sample=callback
    )

    assert len(samples) == 4
    assert seen == [(0, 4), (1, 4), (2, 4), (3, 4)]


def async_return(value: object):
    async def inner(*args: object, **kwargs: object):
        return value

    return inner


@pytest.mark.asyncio
async def test_throughput_matrix_sweep_and_exact_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    from tool_eval_bench.runner import speculative, throughput

    cfg = throughput.TokenizerConfig(has_tokenize_endpoint=True, chars_per_token=2.0)
    counts = iter([5, 15, 10])

    async def tokenize(*args: object, **kwargs: object):
        return next(counts)

    monkeypatch.setattr(throughput, "_tokenize_text", tokenize)
    exact = await throughput._build_exact_prompt(MagicAsyncClient(), "url", "model", 10, None, cfg)
    assert exact

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args: object):
            return None

    monkeypatch.setattr(throughput.httpx, "AsyncClient", lambda **kwargs: Client())
    monkeypatch.setattr(throughput, "warmup", async_return(1.0))
    monkeypatch.setattr(throughput, "calibrate", async_return(cfg))
    monkeypatch.setattr(throughput, "estimate_latency", async_return(2.0))
    monkeypatch.setattr(
        speculative,
        "detect_spec_decoding",
        async_return(speculative.SpecDecodeInfo(active=True, method="mtp")),
    )

    async def measure(*args: object, depth=0, concurrency=1, **kwargs: object):
        return _sample(depth=depth, concurrency=concurrency)

    seen: list[int] = []

    async def callback(sample: ThroughputSample, idx: int, total: int):
        seen.append(idx)

    monkeypatch.setattr(throughput, "measure_concurrent", measure)
    result = await throughput.run_throughput_matrix(
        "url", "model", depths=[0, 10], concurrency_levels=[2, 1], on_sample=callback
    )

    assert result.spec_decoding_detected is True
    assert [s.concurrency for s in result.samples] == [1, 2, 1, 2]
    assert seen == [0, 1, 2, 3]


def test_plugin_runners_complete_cached_lifecycles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import argparse

    from tool_eval_bench.cli import plugin_runners
    from tool_eval_bench.domain.plugin import BenchmarkResult
    from tool_eval_bench.plugins.gsm8k import dataset as gd
    from tool_eval_bench.plugins.gsm8k import plugin as gp
    from tool_eval_bench.plugins.ifeval import dataset as idata
    from tool_eval_bench.plugins.ifeval import plugin as ip
    from tool_eval_bench.plugins.mmlu import dataset as md
    from tool_eval_bench.plugins.mmlu import plugin as mp

    cache = tmp_path / "cache"
    cache.write_text("cached")
    monkeypatch.setattr(gd, "_find_cache_file", lambda: cache)
    monkeypatch.setattr(gd, "load_dataset", lambda **kw: [gd.GSM8KItem(0, "q", "#### 1", 1)])
    monkeypatch.setattr(md, "_find_cache_file", lambda split="test": cache)
    monkeypatch.setattr(
        md,
        "load_dataset",
        lambda split="test", **kw: [md.MMLUItem(0, "q", "anatomy", ["a"] * 4, 0)],
    )
    monkeypatch.setattr(idata, "_find_cache_file", lambda: cache)
    monkeypatch.setattr(idata, "load_dataset", lambda **kw: [idata.IFEvalItem(0, "p", ["x"], [{}])])

    async def gsm_run(self, adapter, *, on_progress=None, **kw):
        await on_progress(
            1, 3, {"correct": True, "question": "q", "extracted_answer": 1, "ground_truth": 1}
        )
        await on_progress(2, 3, {"correct": False, "question": "x" * 100})
        await on_progress(3, 3, {"is_error": True, "question": "bad"})
        return BenchmarkResult(
            "gsm8k",
            50,
            "50%",
            "Weak",
            details={"total": 3, "correct": 1, "errors": 1},
            duration_seconds=1,
            total_tokens=10,
        )

    async def mmlu_run(self, adapter, *, on_progress=None, **kw):
        await on_progress(
            1,
            3,
            {
                "correct": True,
                "subject": "anatomy",
                "question": "q",
                "extracted_answer": "A",
                "ground_truth": "A",
            },
        )
        await on_progress(2, 3, {"correct": False, "subject": "anatomy", "question": "x" * 100})
        await on_progress(3, 3, {"is_error": True, "subject": "anatomy", "question": "bad"})
        return BenchmarkResult(
            "mmlu",
            50,
            "50%",
            "Weak",
            details={
                "total": 3,
                "correct": 1,
                "errors": 1,
                "categories": {"STEM": {"accuracy": 50.0}},
            },
            duration_seconds=1,
            total_tokens=10,
        )

    async def ifeval_run(self, adapter, *, on_progress=None, **kw):
        await on_progress(
            1,
            3,
            {"prompt_pass": True, "instructions_passed": 2, "instructions_total": 2, "prompt": "p"},
        )
        await on_progress(
            2,
            3,
            {
                "prompt_pass": False,
                "instructions_passed": 1,
                "instructions_total": 2,
                "prompt": "x" * 100,
            },
        )
        await on_progress(3, 3, {"is_error": True, "prompt": "bad"})
        return BenchmarkResult(
            "ifeval",
            50,
            "50%",
            "Weak",
            details={
                "total": 3,
                "prompts_passed": 1,
                "errors": 1,
                "prompt_accuracy": 50.0,
                "instructions_passed": 3,
                "instructions_total": 4,
                "instruction_accuracy": 75.0,
            },
            duration_seconds=1,
            total_tokens=10,
        )

    for cls, run in (
        (gp.GSM8KPlugin, gsm_run),
        (mp.MMLUPlugin, mmlu_run),
        (ip.IFEvalPlugin, ifeval_run),
    ):
        monkeypatch.setattr(cls, "run", run)
        monkeypatch.setattr(cls, "render_report_section", lambda self, result: ["report"])
    monkeypatch.setattr(plugin_runners, "_with_config_fingerprint", lambda value: value)
    monkeypatch.setattr(plugin_runners, "_metadata_for_storage", lambda value: {})
    persisted: list[dict] = []
    monkeypatch.setattr(plugin_runners, "_persist_plugin_run", persisted.append)
    args = argparse.Namespace(
        gsm8k_shots=1,
        gsm8k_limit=3,
        gsm8k_shuffle=True,
        mmlu_shots=1,
        mmlu_limit=3,
        mmlu_subjects="STEM",
        ifeval_limit=3,
        seed=1,
        parallel=2,
        temperature=0.0,
        timeout=1.0,
    )
    console = Console(record=True, width=180)

    plugin_runners._run_gsm8k_benchmark(
        console, "m", "Display", "url", None, args, output_dir=str(tmp_path)
    )
    plugin_runners._run_mmlu_benchmark(
        console, "m", "Display", "url", None, args, output_dir=str(tmp_path)
    )
    plugin_runners._run_ifeval_benchmark(
        console, "m", "Display", "url", None, args, output_dir=str(tmp_path)
    )

    assert [item["run_type"] for item in persisted] == ["gsm8k", "mmlu", "ifeval"]
    assert "IFEval Prompt Accuracy" in console.export_text()


def _dispatch_args(**overrides: object):
    import argparse

    values = dict(
        trials=1,
        temperature=0.0,
        timeout=1.0,
        max_turns=2,
        reference_date=None,
        seed=1,
        parallel=1,
        error_rate=0.0,
        alpha=0.7,
        weight_by_difficulty=False,
        json_file=None,
        diff=None,
        output_dir=None,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def test_dispatch_json_and_plain_execution_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    from tool_eval_bench.cli import dispatch
    from tool_eval_bench.domain.scenarios import (
        Category,
        ScenarioDefinition,
        ScenarioEvaluation,
        ScenarioStatus,
    )

    scenario = ScenarioDefinition(
        id="TC-X",
        title="x",
        category=Category.A,
        user_message="x",
        description="x",
        handle_tool_call=lambda s, c: {},
        evaluate=lambda s: ScenarioEvaluation(ScenarioStatus.PASS, 2, "ok"),
    )
    payload = {
        "run_id": "r",
        "scores": {
            "final_score": 100,
            "rating": "Great",
            "weighted_score": 99,
            "scenario_results": [
                {"scenario_id": "TC-X", "status": "pass", "points": 2, "summary": "ok"}
            ],
        },
    }

    class Service:
        async def run_benchmark(self, **kwargs):
            return dict(payload)

    monkeypatch.setattr(dispatch, "_resolve_scenarios", lambda args: [scenario])
    emitted: list[dict] = []
    monkeypatch.setattr(
        dispatch, "_emit_json_output", lambda value, **kwargs: emitted.append(value)
    )
    args = _dispatch_args(trials=2)
    dispatch._run_json(Service(), "m", "vllm", "url", None, args)
    assert emitted[0]["trial_statistics"]["trials"] == 2

    console = Console(record=True)
    dispatch._run_plain(Service(), console, "m", "Display", "vllm", "url", None, _dispatch_args())
    assert "Weighted Score" in console.export_text()


def test_dispatch_trial_summary_all_variance_branches() -> None:
    from tool_eval_bench.cli import dispatch

    agg = {
        "trials": 3,
        "final_score_mean": 80,
        "final_score_stddev": 2,
        "final_score_ci95": (78, 82),
        "final_score_median": 80,
        "total_points_mean": 90,
        "total_points_stddev": 1,
        "pass_at_k": 90,
        "pass_hat_k": 70,
        "reliability_gap": 20,
        "per_category": {"A": {"label": "Safety", "mean_percent": 80, "stddev_percent": 5}},
        "per_scenario": {"TC-X": {"stddev": 1, "mean": 1, "points": [0, 1, 2]}},
    }
    console = Console(record=True)
    dispatch._print_trials_summary(console, agg)
    dispatch._print_trials_summary(console, {})
    assert "unstable scenario" in console.export_text()


def test_dispatch_main_skip_and_perf_only_routes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import sys

    from tool_eval_bench.cli import dispatch, plugin_runners
    from tool_eval_bench.storage import reports
    from tool_eval_bench.utils import metadata

    async def context(**kwargs):
        return None

    monkeypatch.setattr(dispatch, "_load_dotenv", lambda: None)
    monkeypatch.setattr(dispatch, "_preflight_model_check", lambda *a, **k: None)
    monkeypatch.setattr(dispatch, "_do_warmup", lambda *a, **k: None)
    monkeypatch.setattr(metadata, "collect_run_context", context)
    monkeypatch.setattr(plugin_runners, "run_selected_plugins", lambda *a, **k: False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tool-eval-bench",
            "--model",
            "m",
            "--base-url",
            "url",
            "--skip-tool-eval",
            "--no-warmup",
            "--no-think",
            "--top-p",
            "0.9",
            "--backend-kwargs",
            '{"chat_template_kwargs":{"x":1}}',
        ],
    )
    dispatch.main()

    monkeypatch.setattr(dispatch, "_run_llama_benchy", lambda *a, **k: [_sample()])
    monkeypatch.setattr(
        reports.MarkdownReporter, "write_throughput_report", lambda *a, **k: tmp_path / "p.md"
    )
    monkeypatch.setattr(dispatch, "_persist_plugin_run", lambda value: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tool-eval-bench",
            "--model",
            "m",
            "--base-url",
            "url",
            "--perf-only",
            "--no-warmup",
            "--output-dir",
            str(tmp_path),
        ],
    )
    dispatch.main()


def test_detect_model_single_fallback_and_headless_multiple(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    from tool_eval_bench.cli import dispatch

    class Client:
        calls = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url, headers=None):
            Client.calls += 1
            if Client.calls == 1:
                return httpx.Response(404, request=httpx.Request("GET", url))
            return httpx.Response(
                200,
                json={"data": [{"id": "a", "root": "root-a"}, {"id": "b"}]},
                request=httpx.Request("GET", url),
            )

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: Client())
    assert dispatch._detect_model("http://x/v1", "key", Console(), headless=True) == ("a", "root-a")


def test_gsm8k_rest_download_and_load_dataset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tool_eval_bench.plugins import hf_utils
    from tool_eval_bench.plugins.gsm8k import dataset as gsm

    monkeypatch.setattr(hf_utils, "load_via_datasets_lib", lambda *a, **k: None)

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "num_rows_total": 2,
                "rows": [
                    {"row_idx": 0, "row": {"question": "q", "answer": "work #### 2"}},
                    {"row_idx": 1, "row": {"question": "bad", "answer": "none"}},
                ],
            }

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def get(self, url):
            return Response()

    import httpx

    monkeypatch.setattr(httpx, "Client", Client)
    progress: list[tuple[int, int]] = []
    items, method = gsm._download_dataset(on_progress=lambda a, b: progress.append((a, b)))
    assert method == "rest_api" and len(items) == 1 and progress == [(2, 2)]

    cache = tmp_path / "test.jsonl"
    monkeypatch.setattr(gsm, "_find_cache_file", lambda: cache)
    monkeypatch.setattr(gsm, "_download_dataset", lambda **kw: (items, "test"))
    assert gsm.load_dataset(force_download=True) == items
    assert gsm.load_dataset() == items


def test_dispatch_live_and_plain_multitrial(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tool_eval_bench.cli import dispatch
    from tool_eval_bench.domain.scenarios import (
        Category,
        ScenarioDefinition,
        ScenarioEvaluation,
        ScenarioResult,
        ScenarioStatus,
    )
    from tool_eval_bench.storage import reports

    scenario = ScenarioDefinition(
        id="TC-X",
        title="x",
        category=Category.A,
        user_message="x",
        description="x",
        handle_tool_call=lambda s, c: {},
        evaluate=lambda s: ScenarioEvaluation(ScenarioStatus.PASS, 2, "ok"),
    )
    sr = ScenarioResult("TC-X", ScenarioStatus.PASS, 2, "ok")
    payload = {
        "run_id": "r",
        "report_path": str(tmp_path / "run.md"),
        "scores": {"final_score": 100, "rating": "Great", "scenario_results": [sr.to_dict()]},
    }

    class Service:
        async def run_benchmark(self, **kwargs):
            return dict(payload)

    class Display:
        def __init__(self, *args, **kwargs):
            self.results = {"TC-X": sr}

        def start(self):
            pass

        def stop(self):
            pass

        async def on_scenario_start(self, *args):
            pass

        async def on_scenario_result(self, *args):
            pass

        def set_finished(self, *args, **kwargs):
            pass

    monkeypatch.setattr(dispatch, "BenchmarkDisplay", Display)
    monkeypatch.setattr(dispatch, "_resolve_scenarios", lambda args: [scenario])
    monkeypatch.setattr(dispatch, "_print_diff", lambda *args: None)
    monkeypatch.setattr(
        reports.MarkdownReporter,
        "write_summary_report",
        lambda *args, **kwargs: tmp_path / "summary.md",
    )
    args = _dispatch_args(trials=2, diff="latest", output_dir=str(tmp_path))
    console = Console(record=True)
    dispatch._run_with_live_display(Service(), console, "m", "Display", "vllm", "url", None, args)
    dispatch._run_plain(Service(), console, "m", "Display", "vllm", "url", None, args)
    assert "Summary report" in console.export_text()


def test_probe_server_success_and_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    from tool_eval_bench.cli import dispatch

    class Client:
        fail = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url, headers=None):
            if self.fail:
                raise RuntimeError("down")
            return httpx.Response(
                200, json={"data": [{"id": "m"}]}, request=httpx.Request("GET", url)
            )

    client = Client()
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client)
    with pytest.raises(SystemExit) as ready:
        dispatch._probe_server(Console(), "url", "key", headless=True)
    assert ready.value.code == 0
    client.fail = True
    with pytest.raises(SystemExit) as failed:
        dispatch._probe_server(Console(), "url", None)
    assert failed.value.code == 1


def test_dispatch_main_context_pressure_and_resume(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    from tool_eval_bench.cli import dispatch, plugin_runners
    from tool_eval_bench.runner import context_pressure
    from tool_eval_bench.storage import db
    from tool_eval_bench.utils import metadata

    class Pressure:
        ratio = 0.5
        fill_tokens = 100
        detected_context = 1000

        def summary(self):
            return "50%"

        def budget_breakdown(self, **kwargs):
            return {"remaining_headroom_tokens": 100}

    async def prepare(*args, **kwargs):
        return Pressure()

    async def calibrate(messages, *args, **kwargs):
        return messages, 100

    async def context(**kwargs):
        return None

    monkeypatch.setattr(dispatch, "_load_dotenv", lambda: None)
    monkeypatch.setattr(dispatch, "_preflight_model_check", lambda *a, **k: None)
    monkeypatch.setattr(dispatch, "_do_warmup", lambda *a, **k: None)
    monkeypatch.setattr(metadata, "collect_run_context", context)
    monkeypatch.setattr(plugin_runners, "run_selected_plugins", lambda *a, **k: False)
    monkeypatch.setattr(context_pressure, "prepare_context_pressure", prepare)
    monkeypatch.setattr(
        context_pressure,
        "build_pressure_messages",
        lambda *a, **k: [{"role": "user", "content": "fill"}],
    )
    monkeypatch.setattr(context_pressure, "calibrate_pressure_messages", calibrate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tool-eval-bench",
            "--model",
            "m",
            "--base-url",
            "url",
            "--context-pressure",
            "0.5",
            "--context-size",
            "1000",
            "--skip-tool-eval",
            "--no-warmup",
        ],
    )
    dispatch.main()

    class Repo:
        def get(self, run_id):
            return {
                "config": {"model": "m", "backend": "vllm"},
                "scores": {
                    "scenario_results": [
                        {"scenario_id": "TC-01", "status": "pass", "raw_log": "trace"}
                    ]
                },
            }

        def close(self):
            pass

    monkeypatch.setattr(db, "RunRepository", Repo)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tool-eval-bench",
            "--model",
            "m",
            "--backend",
            "vllm",
            "--base-url",
            "url",
            "--resume",
            "r",
            "--scenarios",
            "TC-01",
            "--no-warmup",
        ],
    )
    dispatch.main()


@pytest.mark.parametrize("kind", ["empty", "invalid", "http"])
def test_detect_model_failure_responses(monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    import httpx

    from tool_eval_bench.cli import dispatch

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url, headers=None):
            request = httpx.Request("GET", url)
            if kind == "http":
                return httpx.Response(500, request=request)
            if kind == "invalid":
                return httpx.Response(200, text="nope", request=request)
            return httpx.Response(200, json={"data": []}, request=request)

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: Client())
    with pytest.raises(SystemExit):
        dispatch._detect_model("url", None, Console(), headless=True)


def test_preflight_and_warmup_user_outcomes(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    from tool_eval_bench.cli import probe
    from tool_eval_bench.runner import throughput

    class Client:
        mode = "ok"

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, **kwargs):
            if self.mode == "connect":
                raise httpx.ConnectError("down")
            if self.mode == "error":
                raise RuntimeError("unexpected")
            status = 500 if self.mode == "http" else 200
            return httpx.Response(status, text="bad", request=httpx.Request("POST", url))

    client = Client()
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client)
    console = Console(record=True)
    probe.preflight_model_check(console, "url", "m", "key")
    for mode, code in (("http", 3), ("connect", 2), ("error", 3)):
        client.mode = mode
        with pytest.raises(SystemExit) as exc:
            probe.preflight_model_check(console, "url", "m", None)
        assert exc.value.code == code

    monkeypatch.setattr(throughput, "warmup", async_return(20_000))
    probe.warmup_server(console, "url", "m", None)
    monkeypatch.setattr(throughput, "warmup", async_return(10))
    probe.warmup_server(console, "url", "m", None)

    async def fail(*args, **kwargs):
        raise RuntimeError()

    monkeypatch.setattr(throughput, "warmup", fail)
    probe.warmup_server(console, "url", "m", None)
    assert "Warm-up failed" in console.export_text()


@pytest.mark.asyncio
async def test_throughput_single_and_concurrent_aggregation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tool_eval_bench.runner import throughput

    async def stream(*args, **kwargs):
        return _sample(pp_tokens=10, tg_tokens=5, ttft_ms=2, pp_tps=100)

    monkeypatch.setattr(throughput, "_stream_one", stream)
    one = await throughput.measure_single(MagicAsyncClient(), "url", "m", pp=10, tg=5, depth=2)
    assert one.requested_depth == 2
    aggregate = await throughput.measure_concurrent(
        MagicAsyncClient(), "url", "m", pp=10, tg=5, depth=2, concurrency=2
    )
    assert aggregate.tg_tokens == 10 and aggregate.concurrency == 2

    calls = iter([_sample(error="bad"), RuntimeError("boom")])

    async def failing(*args, **kwargs):
        value = next(calls)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(throughput, "_stream_one", failing)
    failed = await throughput.measure_concurrent(MagicAsyncClient(), "url", "m", concurrency=2)
    assert failed.error and "bad" in failed.error


def test_dispatch_legacy_spec_and_sweep_modes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import sys

    from tool_eval_bench.cli import dispatch, plugin_runners
    from tool_eval_bench.storage import reports
    from tool_eval_bench.utils import metadata

    async def context(**kwargs):
        return None

    monkeypatch.setattr(dispatch, "_load_dotenv", lambda: None)
    monkeypatch.setattr(dispatch, "_preflight_model_check", lambda *a, **k: None)
    monkeypatch.setattr(dispatch, "_do_warmup", lambda *a, **k: None)
    monkeypatch.setattr(metadata, "collect_run_context", context)
    monkeypatch.setattr(plugin_runners, "run_selected_plugins", lambda *a, **k: False)
    monkeypatch.setattr(
        reports.MarkdownReporter, "write_throughput_report", lambda *a, **k: tmp_path / "r.md"
    )
    monkeypatch.setattr(dispatch, "_persist_plugin_run", lambda value: None)
    monkeypatch.setattr(dispatch, "_run_throughput", lambda *a, **k: [_sample()])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tool-eval-bench",
            "--model",
            "m",
            "--base-url",
            "url",
            "--perf-legacy-only",
            "--no-warmup",
            "--output-dir",
            str(tmp_path),
        ],
    )
    dispatch.main()

    called: list[str] = []
    monkeypatch.setattr(dispatch, "_run_spec_bench", lambda *a, **k: called.append("spec"))
    monkeypatch.setattr(
        sys,
        "argv",
        ["tool-eval-bench", "--model", "m", "--base-url", "url", "--spec-bench", "--no-warmup"],
    )
    dispatch.main()
    monkeypatch.setattr(dispatch, "_run_pressure_sweep", lambda *a, **k: called.append("sweep"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tool-eval-bench",
            "--model",
            "m",
            "--base-url",
            "url",
            "--context-pressure-sweep",
            "0.5-0.8",
            "--no-warmup",
        ],
    )
    dispatch.main()
    assert called == ["spec", "sweep"]
