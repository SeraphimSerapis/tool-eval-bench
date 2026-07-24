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

    async def fake_run(*args: object, on_progress=None, **kwargs: object):
        if on_progress is not None:
            on_progress(
                {
                    "type": "request_start",
                    "prompt_size": 100,
                    "response_size": 20,
                    "context_size": 0,
                    "concurrency": 1,
                    "run_index": 1,
                }
            )
            on_progress({"type": "request_end"})
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
