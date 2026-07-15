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
        "pass_8": "80",
        "pass_at_8": "80",
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
