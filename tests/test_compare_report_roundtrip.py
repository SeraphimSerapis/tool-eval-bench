"""Compare-report parsers read what the current report writers emit.

The parsers are regex scrapers of the Markdown artifacts, so the only real
guard against drift is to feed them real writer output.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest

from tool_eval_bench.compare_reports import summary as cmp_summary
from tool_eval_bench.compare_reports import tool_eval as cmp_tool_eval
from tool_eval_bench.compare_reports._common import config_note, deployability_label
from tool_eval_bench.domain.models import RunContext
from tool_eval_bench.domain.scenarios import ScenarioResult, ScenarioStatus
from tool_eval_bench.evals.scenarios import SCENARIOS
from tool_eval_bench.runner.orchestrator import score_results
from tool_eval_bench.storage.reports import MarkdownReporter

SCENARIO = SCENARIOS[0]


def _ctx(model: str, temperature: float = 0.0, thinking: bool = False) -> RunContext:
    return RunContext(
        tool_version="2.9.0",
        git_sha=None,
        hostname="h",
        platform_info="p",
        python_version="3.13",
        model=model,
        backend="vllm",
        base_url="http://***:8000",
        temperature=temperature,
        thinking_enabled=thinking,
    )


def _result(status: ScenarioStatus, safety_violation: str | None = None) -> ScenarioResult:
    result = ScenarioResult(
        scenario_id=SCENARIO.id,
        status=status,
        points={ScenarioStatus.PASS: 2, ScenarioStatus.PARTIAL: 1}.get(status, 0),
        summary="ok" if status == ScenarioStatus.PASS else "bad <thing> | here",
    )
    result.safety_violation = safety_violation
    return result


def _scenario_report(tmp_path: Path, run_id: str, ctx: RunContext, result: ScenarioResult) -> str:
    path = MarkdownReporter(root=str(tmp_path)).write_scenario_report(
        run_id, ctx.model, score_results([result], [SCENARIO]), run_context=ctx
    )
    return str(path)


def test_tool_eval_parser_reads_the_run_context_the_writer_emits(tmp_path: Path) -> None:
    path = _scenario_report(
        tmp_path, "run-a", _ctx("org/model-a", 0.7, thinking=True), _result(ScenarioStatus.PASS)
    )
    parsed = cmp_tool_eval.parse_md(path)
    assert parsed["model_api"] == "org/model-a"
    assert parsed["backend"] == "vllm"
    assert parsed["temperature"] == "0.7"
    assert parsed["thinking"] == "enabled"


def test_tool_eval_parser_names_safety_critical_scenarios(tmp_path: Path) -> None:
    path = _scenario_report(
        tmp_path,
        "run-a",
        _ctx("a"),
        _result(ScenarioStatus.FAIL, safety_violation="deleted <prod> data"),
    )
    parsed = cmp_tool_eval.parse_md(path)
    assert parsed["safety_critical_count"] == 1
    assert parsed["safety_critical"] == [
        {"id": SCENARIO.id, "type": SCENARIO.title, "desc": "deleted <prod> data"}
    ]
    # Cells come back unescaped, so the HTML layer escapes them exactly once.
    (row,) = parsed["scenarios"]
    assert row["summary"] == "bad <thing> | here"


def test_safety_title_containing_parentheses_is_not_split(tmp_path: Path) -> None:
    report = tmp_path / "r.md"
    report.write_text(
        "# Tool-Call Benchmark — m\n\n"
        "> [!WARNING]\n"
        "> **1 safety-critical failure(s) detected:**\n"
        "> - TC-07 (Foo (bar): baz): sent money\n\n"
        "## Category Scores\n",
        encoding="utf-8",
    )
    (item,) = cmp_tool_eval.parse_md(str(report))["safety_critical"]
    assert item == {"id": "TC-07", "type": "Foo (bar): baz", "desc": "sent money"}


def test_tool_eval_html_names_differing_settings_and_does_not_crown_a_tie(
    tmp_path: Path,
) -> None:
    ok = _result(ScenarioStatus.PASS)
    a = cmp_tool_eval.parse_md(_scenario_report(tmp_path, "run-a", _ctx("a", 0.0), ok))
    b = cmp_tool_eval.parse_md(_scenario_report(tmp_path, "run-b", _ctx("b", 1.0), ok))
    out = tmp_path / "cmp.html"
    cmp_tool_eval.generate_html(a, b, str(out))
    html = out.read_text(encoding="utf-8")
    assert "same backend" not in html
    assert "temperature 0.0 (a) vs 1.0 (b)" in html
    assert "clear winner" not in html
    assert "tie at 100 / 100" in html
    assert "Tie:" in html
    assert "<span>a vs. b: Strengths &amp; Weaknesses</span>" in html
    assert ">Even</span>" in html  # safety row
    _assert_cards_not_crowned(html)


def test_tool_eval_html_keeps_the_winner_when_scores_differ(tmp_path: Path) -> None:
    a = cmp_tool_eval.parse_md(
        _scenario_report(tmp_path, "run-a", _ctx("a"), _result(ScenarioStatus.PASS))
    )
    b = cmp_tool_eval.parse_md(
        _scenario_report(tmp_path, "run-b", _ctx("b"), _result(ScenarioStatus.FAIL))
    )
    out = tmp_path / "cmp.html"
    cmp_tool_eval.generate_html(a, b, str(out))
    html = out.read_text(encoding="utf-8")
    assert "clear winner" in html
    assert ">WINNER</span>" in html and ">RUNNER-UP</div>" in html
    assert 'class="winner-badge"' in html
    assert "(Winner)</th>" in html
    assert ">TIED</div>" not in html
    assert "Winner vs. Runner-up" in html
    # b has the extra failure, so it is not listed among a's weaknesses.
    assert "More outright failures" not in html
    assert "Both runs used the same backend vllm, temperature 0.0, thinking disabled." in html


def test_pipe_in_model_id_survives_the_round_trip(tmp_path: Path) -> None:
    path = _scenario_report(tmp_path, "run-a", _ctx("a/model|x"), _result(ScenarioStatus.PASS))
    assert "| Model (API) | `a/model\\|x` |" in Path(path).read_text(encoding="utf-8")
    assert cmp_tool_eval.parse_md(path)["model_api"] == "a/model|x"
    summary = _summary_report(tmp_path / "s", 2, "a/model|x")
    assert cmp_summary.parse_summary(summary)["model_api"] == "a/model|x"


def test_config_note_claims_nothing_when_no_settings_were_recorded() -> None:
    blank = {"model_api": "a", "model_name": "a"}
    assert "Neither report" in config_note(blank, {**blank, "model_api": "b"})


@pytest.mark.parametrize(
    ("a", "b", "label"),
    [
        ("0.7", "0.7", "Deployability (\u03b1=0.7)"),
        ("0.5", "0.7", "Deployability (\u03b1=0.5 vs 0.7)"),
        ("", "0.7", "Deployability"),
    ],
)
def test_deployability_label_uses_the_reported_alpha(a: str, b: str, label: str) -> None:
    assert deployability_label({"alpha": a}, {"alpha": b}) == label


def _summary_report(tmp_path: Path, trials: int, model: str = "M") -> str:
    # Trial 1 is clean; every later trial has a safety warning.
    first = dataclasses.replace(
        score_results([_result(ScenarioStatus.PASS)], [SCENARIO]),
        responsiveness=64,
        deployability=89,
        alpha=0.7,
        median_turn_ms=2300.0,
    )
    summaries = [first] + [
        score_results([_result(ScenarioStatus.FAIL, "sent money")], [SCENARIO])
        for _ in range(trials - 1)
    ]
    agg = {
        "trials": trials,
        "final_score_mean": 33.3,
        "final_score_stddev": 47.1,
        "total_points_mean": 0.7,
        "total_points_stddev": 0.9,
        "pass_at_k": 100.0,
        "pass_hat_k": 0.0,
        "reliability_gap": 100.0,
        "final_score_ci95": (0.0, 100.0),
        "per_scenario": {},
        "per_category": {},
    }
    path = MarkdownReporter(root=str(tmp_path)).write_summary_report(
        "run-" + re.sub(r"\W", "_", model), model, summaries, agg, run_context=_ctx(model)
    )
    return str(path)


@pytest.mark.parametrize("trials", [2, 3, 8])
def test_summary_parser_reads_pass_at_k_for_any_trial_count(tmp_path: Path, trials: int) -> None:
    parsed = cmp_summary.parse_summary(_summary_report(tmp_path, trials))
    assert parsed["pass_k"] == trials
    assert parsed["pass_at_k"] == "100.0"
    assert parsed["pass_hat_k"] == "0.0"
    # Every trial's count, not only trial 1's; the aggregate dash is skipped.
    assert parsed["safety_warnings"] == [0] + [1] * (trials - 1)
    assert parsed["temperature"] == "0.0"


def test_summary_parser_reads_the_unbolded_deployability_rows(tmp_path: Path) -> None:
    parsed = cmp_summary.parse_summary(_summary_report(tmp_path, 2))
    assert parsed["quality"] == 100
    assert parsed["responsiveness"] == 64
    assert parsed["deployability"] == 89
    assert parsed["alpha"] == "0.7"
    assert parsed["median_turn"] == "2.3"


def test_summary_html_labels_reliability_with_the_trial_count(tmp_path: Path) -> None:
    a = cmp_summary.parse_summary(_summary_report(tmp_path / "a", 3, "a"))
    b = cmp_summary.parse_summary(_summary_report(tmp_path / "b", 3, "b"))
    out = tmp_path / "cmp.html"
    cmp_summary.generate_html(a, b, str(out))
    html = out.read_text(encoding="utf-8")
    assert "Pass^3" in html
    assert "Pass@3" in html
    assert "\u2078" not in html and "\u2088" not in html
    assert "clear winner" not in html  # identical runs tie
    _assert_cards_not_crowned(html)


def _assert_cards_not_crowned(html: str) -> None:
    assert "Winner" not in re.sub(r'class="[^"]*"', "", html)
    assert html.count(">TIED</div>") == 2
    assert ">WINNER<" not in html and ">RUNNER-UP<" not in html
    assert 'class="winner-badge"' not in html
    assert "(Winner)</th>" not in html
