from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

from tool_eval_bench.cli.compare_report import (
    detect_report_kind,
    generate_compare_report,
    run_compare_report_command,
)
from tool_eval_bench.compare_reports.summary import generate_html as generate_summary_html
from tool_eval_bench.compare_reports.tool_eval import generate_html as generate_tool_eval_html


def _tool_eval_run(**overrides):
    data = {
        "model_name": "base",
        "model_api": "base",
        "date_short": "2026-07-04",
        "tool_eval_version": "test",
        "final_score": 80,
        "total_points_earned": 80,
        "total_points_max": 100,
        "total_points_str": "80 / 100",
        "rating": "Good",
        "deployability": 80,
        "quality": 80,
        "responsiveness": 90,
        "median_turn_time": "1.0",
        "backend": "test",
        "temperature": "0",
        "thinking": "off",
        "categories": [],
        "scenarios": [],
        "difficulties": [],
        "tier_order": ["Trivial", "Easy", "Moderate", "Hard", "Very Hard"],
        "safety_critical": [],
        "safety_critical_count": 0,
    }
    data.update(overrides)
    return data


def _summary_run(**overrides):
    data = {
        "model_name": "base",
        "model_api": "base",
        "date_short": "2026-07-04",
        "version": "test",
        "trials": 1,
        "mean_score": 80.0,
        "std_score": 1.0,
        "mean_points": 80.0,
        "rating": "Good",
        "quality": 80,
        "responsiveness": 90,
        "deployability": 80,
        "median_turn": "1.0",
        "safety_warnings": [],
        "pass_hat_k": "80",
        "pass_at_k": "80",
        "pass_k": 1,
        "reliability_gap": "0",
        "backend": "test",
        "temperature": "0",
        "thinking": "off",
        "categories": [],
        "scenarios": [],
        "never_passes": [],
        "flaky": [],
        "consistent_partials": [],
    }
    data.update(overrides)
    return data


def _html(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_detect_report_kind_from_markdown_heading(tmp_path: Path) -> None:
    summary = tmp_path / "summary.md"
    summary.write_text("# Cross-Trial Summary — Model\n\n", encoding="utf-8")
    single = tmp_path / "single.md"
    single.write_text("# Tool-Call Benchmark — Model\n\n", encoding="utf-8")

    assert detect_report_kind(str(summary)) == "summary"
    assert detect_report_kind(str(single)) == "tool-eval"


def test_generate_compare_report_rejects_mixed_report_types(tmp_path: Path) -> None:
    summary = tmp_path / "summary.md"
    summary.write_text("# Cross-Trial Summary — Model\n\n", encoding="utf-8")
    single = tmp_path / "single.md"
    single.write_text("# Tool-Call Benchmark — Model\n\n", encoding="utf-8")

    try:
        generate_compare_report(str(summary), str(single), str(tmp_path / "out.html"))
    except ValueError as exc:
        assert "Report types do not match" in str(exc)
    else:
        raise AssertionError("expected mixed report types to fail")


def test_generate_compare_report_routes_tool_eval_and_writes_html(tmp_path: Path) -> None:
    report_a = tmp_path / "a.md"
    report_b = tmp_path / "b.md"
    output = tmp_path / "comparison.html"
    report_a.write_text(
        "# Tool-Call Benchmark — Alpha\n\n- **Date**: `2026-07-15`\n- **Final Score**: **90**\n",
        encoding="utf-8",
    )
    report_b.write_text(
        "# Tool-Call Benchmark — Beta\n\n- **Date**: `2026-07-15`\n- **Final Score**: **70**\n",
        encoding="utf-8",
    )

    selected = generate_compare_report(str(report_a), str(report_b), str(output))

    assert selected == "tool-eval"
    assert "Winner:" in output.read_text(encoding="utf-8")
    assert "Alpha" in output.read_text(encoding="utf-8")


def test_generate_compare_report_validates_paths_and_explicit_kind(tmp_path: Path) -> None:
    report = tmp_path / "run.md"
    report.write_text("# Tool-Call Benchmark — Alpha\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="file not found"):
        generate_compare_report(str(report), str(tmp_path / "missing.md"), "out.html")
    with pytest.raises(ValueError, match="unknown report kind"):
        generate_compare_report(str(report), str(report), "out.html", kind="unsupported")


def test_compare_report_command_reports_success_and_user_errors(tmp_path: Path) -> None:
    report_a = tmp_path / "a.md"
    report_b = tmp_path / "b.md"
    output = tmp_path / "comparison.html"
    for path, model in ((report_a, "Alpha"), (report_b, "Beta")):
        path.write_text(
            f"# Tool-Call Benchmark — {model}\n\n"
            "- **Date**: `2026-07-15`\n- **Final Score**: **80**\n",
            encoding="utf-8",
        )
    console = Console(record=True, width=120)

    run_compare_report_command(
        SimpleNamespace(
            report_a=str(report_a),
            report_b=str(report_b),
            output=str(output),
            kind="auto",
        ),
        console,
    )

    assert output.is_file()
    assert "Generated tool-eval comparison report" in console.export_text()

    with pytest.raises(SystemExit) as exc_info:
        run_compare_report_command(
            SimpleNamespace(
                report_a=str(report_a),
                report_b=str(tmp_path / "missing.md"),
                output=str(output),
                kind="auto",
            ),
            console,
        )
    assert exc_info.value.code == 1
    assert "file not found" in console.export_text()


def test_tool_eval_median_turn_stays_with_model_columns(tmp_path: Path) -> None:
    out = tmp_path / "tool.html"
    winner = _tool_eval_run(
        model_name="winner",
        model_api="winner",
        final_score=90,
        total_points_earned=90,
        total_points_str="90 / 100",
        median_turn_time="5.0",
    )
    runner = _tool_eval_run(model_name="runner", model_api="runner", median_turn_time="1.0")

    generate_tool_eval_html(winner, runner, str(out))
    html = _html(out)

    row = html[html.index("Median Turn Time") : html.index("Safety Warnings")]
    assert ">5.0s<" in row
    assert ">1.0s<" in row
    assert 'class="diff-negative">+4.0s<' in row
    assert row.index(">5.0s<") < row.index(">1.0s<")


def test_tool_eval_html_explains_category_safety_and_failure_differences(tmp_path: Path) -> None:
    out = tmp_path / "rich-tool-comparison.html"
    winner = _tool_eval_run(
        model_name="winner",
        model_api="winner",
        final_score=94,
        total_points_earned=94,
        total_points_str="94 / 100",
        quality=93,
        deployability=92,
        median_turn_time="1.0",
        categories=[
            {"name": "Hard Mode", "percent": 90},
            {"name": "Tool Selection", "percent": 50},
        ],
        difficulties=[
            {"tier": "Hard", "rate": 90, "passed": "9", "scenarios": "10"},
            {"tier": "Easy", "rate": 50, "passed": "5", "scenarios": "10"},
        ],
        scenarios=[
            {"id": "TC-01", "status": "fail", "summary": "one failure"},
            {"id": "TC-02", "status": "partial", "summary": "one partial"},
        ],
    )
    runner = _tool_eval_run(
        model_name="runner",
        model_api="runner",
        final_score=70,
        total_points_earned=70,
        total_points_str="70 / 100",
        quality=70,
        deployability=68,
        median_turn_time="3.0",
        categories=[
            {"name": "Hard Mode", "percent": 40},
            {"name": "Tool Selection", "percent": 80},
        ],
        difficulties=[
            {"tier": "Hard", "rate": 40, "passed": "4", "scenarios": "10"},
            {"tier": "Easy", "rate": 80, "passed": "8", "scenarios": "10"},
        ],
        scenarios=[
            {"id": "TC-34", "status": "fail", "summary": "unsafe action"},
            {"id": "TC-35", "status": "fail", "summary": "ordinary failure"},
            {"id": "TC-36", "status": "partial", "summary": "partial result"},
        ],
        safety_critical=[{"id": "TC-34", "type": "unsafe", "desc": "unsafe action"}],
        safety_critical_count=1,
    )

    generate_tool_eval_html(winner, runner, str(out))
    html = _html(out)

    assert "Stronger hard-mode performance" in html
    assert "Faster" in html and "Safer" in html
    assert "Superior hard mode" in html
    assert "Lower tool selection" in html
    assert "TC-34" in html and "safety-critical" in html
    assert "faster inference (1.0s median vs 3.0s)" in html


def test_compare_reports_escape_short_model_labels(tmp_path: Path) -> None:
    out = tmp_path / "summary.html"
    payload = "<img src=x onerror=alert(1)>"
    winner = _summary_run(model_name=payload, model_api=payload, mean_score=90.0)
    runner = _summary_run(model_name="runner", model_api="runner", mean_score=80.0)

    generate_summary_html(winner, runner, str(out))
    html = _html(out)

    assert payload not in html
    assert "&lt;img src=x onerror=alert(1)&gt;" in html


def test_summary_median_turn_and_missing_gap_are_robust(tmp_path: Path) -> None:
    out = tmp_path / "summary.html"
    winner = _summary_run(
        model_name="winner",
        model_api="winner",
        mean_score=90.0,
        median_turn="5.0",
        reliability_gap="",
    )
    runner = _summary_run(
        model_name="runner",
        model_api="runner",
        mean_score=80.0,
        median_turn="1.0",
        reliability_gap="12.5",
    )

    generate_summary_html(winner, runner, str(out))
    html = _html(out)

    row = html[html.index("Median Turn Time") : html.index("Safety Warnings")]
    assert ">5.0s<" in row
    assert ">1.0s<" in row
    assert 'class="diff-negative">+4.0s<' in row
    assert "winner: \u2014" in html
    assert "runner: 12.5pp" in html


def test_parse_reports_and_render_with_held_out_tiers_infra_failure_and_escaping(
    tmp_path: Path,
) -> None:
    from tool_eval_bench.compare_reports.summary import parse_summary
    from tool_eval_bench.compare_reports.tool_eval import parse_md
    from tool_eval_bench.domain.scenarios import (
        Category as ScenarioCategory,
    )
    from tool_eval_bench.domain.scenarios import (
        CategoryScore,
        ModelScoreSummary,
        ScenarioReportMetadata,
        ScenarioResult,
        ScenarioStatus,
    )
    from tool_eval_bench.storage.reports import MarkdownReporter

    summary_a = ModelScoreSummary(
        total_points=2,
        max_points=2,
        final_score=100,
        rating="Excellent",
        category_scores=[
            CategoryScore(ScenarioCategory.A, "Tool Selection", 2, 2, 100.0, pass_count=1)
        ],
        scenario_results=[
            ScenarioResult(
                scenario_id="HOLD-01",
                status=ScenarioStatus.PASS,
                points=2,
                summary="secret",
                expected_behavior="exp",
                raw_log="trace",
            )
        ],
    )
    summary_b = ModelScoreSummary(
        total_points=0,
        max_points=2,
        final_score=0,
        rating="Poor",
        category_scores=[
            CategoryScore(ScenarioCategory.A, "Tool Selection", 0, 2, 0.0, fail_count=1)
        ],
        scenario_results=[
            ScenarioResult(
                scenario_id="HOLD-01",
                status=ScenarioStatus.FAIL,
                points=0,
                summary="secret",
                expected_behavior="exp",
                raw_log="trace",
            )
        ],
    )
    meta = {
        "HOLD-01": ScenarioReportMetadata(
            "Secret Title", ScenarioCategory.A, difficulty=4, held_out=True
        )
    }
    reporter = MarkdownReporter(root=str(tmp_path))
    pa = reporter.write_scenario_report("ra", "ModelA", summary_a, scenario_metadata=meta)
    pb = reporter.write_scenario_report("rb", "ModelB", summary_b, scenario_metadata=meta)

    da = parse_md(str(pa))
    db_ = parse_md(str(pb))
    assert [d["tier"] for d in da["difficulties"]] == ["Hard"]
    assert da["difficulties"][0]["rate"] == 100
    assert [s["id"] for s in da["scenarios"]] == ["HOLD-01"]
    assert da["scenarios"][0]["title"] == "_held out_"

    da["median_turn_time"] = "<script>alert(1)</script>"
    out_html = tmp_path / "cmp.html"
    generate_tool_eval_html(da, db_, str(out_html))
    html_text = _html(out_html)
    assert "<script>alert(1)</script>" not in html_text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html_text
    assert "Secret Title" not in html_text

    sa = reporter.write_summary_report(
        "sa",
        "ModelA",
        [summary_b],
        {
            "trials": 1,
            "final_score_mean": 0.0,
            "final_score_stddev": 0.0,
            "total_points_mean": 0.0,
            "total_points_stddev": 0.0,
            "per_scenario": {"HOLD-01": {"pass_at_k": False, "pass_hat_k": False}},
        },
        scenario_metadata=meta,
    )
    parsed_sum = parse_summary(str(sa))
    assert [s["id"] for s in parsed_sum["scenarios"]] == ["HOLD-01"]
    assert parsed_sum["never_passes"] == [{"id": "HOLD-01", "issue": "_held out_"}]

    # Non-numeric median_turn on one side must escape safely instead of crashing Faster badge check.
    bad_turn = dict(parsed_sum)
    bad_turn["median_turn"] = "<img src=x onerror=alert(1)>"
    good_turn = dict(parsed_sum)
    good_turn["median_turn"] = "1.5"
    good_turn["mean_score"] = 50.0
    good_turn["reliability_gap"] = "10.0"
    sum_html = tmp_path / "sum_cmp.html"
    generate_summary_html(good_turn, bad_turn, str(sum_html))
    sum_text = _html(sum_html)
    assert "<img src=x onerror=alert(1)>" not in sum_text
    assert "&lt;img src=x onerror=alert(1)&gt;" in sum_text


def test_parse_summary_reliability_sections_and_tool_eval_invalid_tier(tmp_path: Path) -> None:
    from tool_eval_bench.compare_reports.summary import parse_summary
    from tool_eval_bench.compare_reports.tool_eval import parse_md

    te_md = tmp_path / "run.md"
    te_md.write_text(
        """# Tool-Call Benchmark \u2014 path/model-Q8_K_XL.gguf
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
**Model (API)** | `path/model-Q8_K_XL.gguf`
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
    parsed_te = parse_md(str(te_md))
    assert parsed_te["final_score"] == 88
    assert [s["status"] for s in parsed_te["scenarios"]] == ["pass", "partial", "fail"]
    assert parsed_te["difficulties"][1]["rate"] == 0
    assert parsed_te["safety_critical"] == [
        {"id": "TC-03", "type": "unsafe", "desc": "leaked <secret>"}
    ]
    out_te = tmp_path / "te.html"
    generate_tool_eval_html(parsed_te, _tool_eval_run(final_score=70), str(out_te))
    assert "GGUF Q8" in _html(out_te)

    sum_md = tmp_path / "summary.md"
    sum_md.write_text(
        """# Cross-Trial Summary \u2014 deepseek-ai/deepseek-v4-flash-dspark
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
    parsed_sum = parse_summary(str(sum_md))
    assert parsed_sum["mean_score"] == 85.5
    assert parsed_sum["categories"] == [{"name": "Safety", "mean": 90, "variance": ""}]
    assert parsed_sum["scenarios"][0]["passes"] == 1
    assert parsed_sum["never_passes"][0]["id"] == "TC-02"
    assert parsed_sum["flaky"][0]["id"] == "TC-03"
    assert parsed_sum["consistent_partials"][0]["id"] == "TC-04"

    zero_infra = _summary_run(
        model_name="nvidia fp4",
        model_api="nvidia fp4",
        mean_score=0.0,
        never_passes=[{"id": "TC-02", "issue": "connection error for url host"}],
        pass_at_k="",
        pass_hat_k="",
        reliability_gap="",
    )
    out_sum = tmp_path / "sum.html"
    generate_summary_html(parsed_sum, zero_infra, str(out_sum))
    sum_html = _html(out_sum)
    assert "NVFP4" in sum_html
    assert "Invalid or infrastructure-failed run detected" in sum_html
    assert "connection error (1 scenarios)" in sum_html


def test_summary_all_zero_warning_and_both_infra_failed_runs(tmp_path: Path) -> None:
    from tool_eval_bench.compare_reports.summary import _infrastructure_summary

    assert (
        _infrastructure_summary(_summary_run(mean_score=0.0, never_passes=[]))
        == "All scenarios scored 0 — this run may be invalid."
    )

    infra_a = _summary_run(
        model_name="ModelA",
        model_api="ModelA",
        date_short="",
        mean_score=0.0,
        reliability_gap="not-a-number",
        never_passes=[{"id": "TC-01", "issue": "server error for url http://host"}],
    )
    infra_b = _summary_run(
        model_name="ModelB",
        model_api="ModelB",
        date_short="",
        mean_score=0.0,
        never_passes=[{"id": "TC-02", "issue": "timed out waiting for response"}],
    )
    out = tmp_path / "both_infra.html"
    generate_summary_html(infra_a, infra_b, str(out))
    html = _html(out)
    assert "Invalid or infrastructure-failed run detected" in html
    assert "ModelA" in html and "ModelB" in html
    assert "server error (1 scenarios)" in html
    assert "timed out waiting for response (1 scenarios)" in html


def test_summary_reliability_speed_safety_verdicts_and_strengths_weaknesses(tmp_path: Path) -> None:
    winner = _summary_run(
        model_name="FastSafe",
        model_api="FastSafe",
        mean_score=92.0,
        std_score=1.0,
        median_turn="1.2",
        safety_warnings=[0, 0],
        **{"pass_hat_k": "90.0", "pass_at_k": "96.0"},
        pass_k=2,
        reliability_gap="6.0",
        categories=[
            {"name": "Tool Selection", "mean": 100, "variance": "low"},
            {"name": "Planning", "mean": 60, "variance": "med"},
        ],
        never_passes=[],
        flaky=[],
        consistent_partials=[],
    )
    runner = _summary_run(
        model_name="SlowRisky",
        model_api="SlowRisky",
        mean_score=74.0,
        std_score=4.5,
        median_turn="3.5",
        safety_warnings=[0, 2],
        **{"pass_hat_k": "65.0", "pass_at_k": "85.0"},
        pass_k=2,
        reliability_gap="20.0",
        categories=[
            {"name": "Tool Selection", "mean": 70, "variance": "high"},
            {"name": "Planning", "mean": 85, "variance": "low"},
        ],
        never_passes=[{"id": "TC-34", "issue": "obeyed injection"}],
        flaky=[{"id": "TC-08"}],
        consistent_partials=[{"id": "TC-19", "issue": "missed secondary constraint"}],
    )
    out = tmp_path / "verdict_branches.html"
    generate_summary_html(winner, runner, str(out))
    html = _html(out)

    # Verdict banner badges
    assert "More reliable" in html
    assert "Faster" in html
    assert "Safer" in html
    # Runner-up consistent partials section
    assert "Consistent partials (1):" in html
    assert "TC-19" in html and "missed secondary constraint" in html
    # Winner strengths
    assert "Superior tool selection" in html
    assert "Higher reliability floor" in html and "90.0% vs 65.0%" in html
    assert "Faster inference" in html and "1.2s median vs 3.5s" in html
    assert "Stronger safety" in html
    # Winner weaknesses vs runner-up
    assert "Lower planning" in html and "60% vs 85%" in html
    assert "Has never-pass scenarios (1 vs 0)." in html
    # Safety conclusion
    assert "outmatched in safety and reliability testing" in html


def test_summary_cli_main_handles_usage_missing_files_and_valid_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from tool_eval_bench.compare_reports import summary as summary_mod

    # 1. Wrong arg count -> exits 1 with usage message.
    monkeypatch.setattr("sys.argv", ["compare_summary.py", "only_one.md"])
    with pytest.raises(SystemExit) as bad_argc:
        summary_mod.main()
    assert bad_argc.value.code == 1
    assert "Usage:" in capsys.readouterr().out

    # 2. Missing file -> exits 1 with file-not-found error.
    summary_a = tmp_path / "a.md"
    summary_b = tmp_path / "b.md"
    out_html = tmp_path / "cli_summary.html"
    summary_a.write_text(
        "# Cross-Trial Summary \u2014 Alpha\n\n"
        "- **Date**: `2026-07-15`\n"
        "| **Final Score** | 90 | **90.0 ± 1.0** |\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv", ["compare_summary.py", str(summary_a), str(summary_b), str(out_html)]
    )
    with pytest.raises(SystemExit) as missing_exc:
        summary_mod.main()
    assert missing_exc.value.code == 1
    assert "Error: file not found" in capsys.readouterr().out

    # 3. Valid inputs -> parses both summaries and writes HTML output.
    summary_b.write_text(
        "# Cross-Trial Summary \u2014 Beta\n\n"
        "- **Date**: `2026-07-15`\n"
        "| **Final Score** | 75 | **75.0 ± 2.0** |\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv", ["compare_summary.py", str(summary_a), str(summary_b), str(out_html)]
    )
    summary_mod.main()
    out_text = capsys.readouterr().out
    assert "Summary A: Alpha" in out_text
    assert "Summary B: Beta" in out_text
    assert out_html.is_file()
    assert "Alpha" in _html(out_html) and "Beta" in _html(out_html)
