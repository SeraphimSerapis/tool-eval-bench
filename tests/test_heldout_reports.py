"""Held-out pack content stays out of every shareable Markdown artifact.

The per-trial report always withheld held-out titles, summaries, and traces.
The cross-trial summary did not, and the evaluator summary of a pack scenario
can quote the expected answer ("Answer never states: <snippet>").
"""

from __future__ import annotations

from pathlib import Path

from tool_eval_bench.domain.scenarios import (
    Category as ScenarioCategory,
)
from tool_eval_bench.domain.scenarios import (
    CategoryScore,
    ModelScoreSummary,
    ScenarioReportMetadata,
    ScenarioResult,
    ScenarioState,
    ScenarioStatus,
    scenario_report_metadata,
)
from tool_eval_bench.evals.packs import load_scenario_pack
from tool_eval_bench.evals.scenarios import SCENARIOS
from tool_eval_bench.runner.orchestrator import score_results
from tool_eval_bench.storage.reports import MarkdownReporter

SECRET_ANSWER = "zebra-secret-42"
SECRET_TITLE = "Private refund escalation"
PACK_YAML = f"""
id: HOLD-01
title: {SECRET_TITLE}
category: A
difficulty: 2
user_message: Escalate ticket 7 to the vault team
expected_tool_calls: []
answer_contains:
  - {SECRET_ANSWER}
"""
AGG = {
    "trials": 2,
    "final_score_mean": 50.0,
    "final_score_stddev": 0.0,
    "total_points_mean": 1.0,
    "total_points_stddev": 0.0,
    "per_scenario": {},
    "per_category": {},
}


def _held_out_scenario(tmp_path: Path):
    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()
    (pack_dir / "hold01.yaml").write_text(PACK_YAML, encoding="utf-8")
    (scenario,) = load_scenario_pack(pack_dir).scenarios
    assert scenario.held_out
    return scenario


def _pack_result(scenario, status: ScenarioStatus) -> ScenarioResult:
    # The pack's own evaluator writes the summary that names the answer.
    evaluation = scenario.evaluate(ScenarioState(final_answer="I could not help."))
    assert SECRET_ANSWER in evaluation.summary
    return ScenarioResult(
        scenario_id=scenario.id,
        status=status,
        points={ScenarioStatus.PARTIAL: 1}.get(status, 0),
        summary=evaluation.summary,
    )


def _summary_md(tmp_path: Path, summaries: list[ModelScoreSummary], metadata, **kwargs) -> str:
    reporter = MarkdownReporter(root=str(tmp_path / "runs"))
    path = reporter.write_summary_report(
        "run-1", "m", summaries, AGG, scenario_metadata=metadata, **kwargs
    )
    return path.read_text(encoding="utf-8")


def test_summary_withholds_held_out_summaries_in_never_passes_and_partials(
    tmp_path: Path,
) -> None:
    held = _held_out_scenario(tmp_path)
    public = SCENARIOS[0]
    public_fail = ScenarioResult(public.id, ScenarioStatus.FAIL, 0, "public failure reason")
    for status in (ScenarioStatus.FAIL, ScenarioStatus.PARTIAL):
        summaries = [
            score_results([_pack_result(held, status), public_fail], [held, public])
            for _ in range(2)
        ]
        md = _summary_md(tmp_path, summaries, scenario_report_metadata([held, public]))
        assert SECRET_ANSWER not in md
        assert SECRET_TITLE not in md
        assert "| **HOLD-01** | _held out_ |" in md or "| HOLD-01 | _held out_ |" in md
        assert "1 held-out scenario(s)" in md
        # Public scenarios are unaffected.
        assert "public failure reason" in md


def test_summary_held_out_note_attests_to_the_pack(tmp_path: Path) -> None:
    held = _held_out_scenario(tmp_path)
    summaries = [score_results([_pack_result(held, ScenarioStatus.FAIL)], [held])] * 2
    pack = {"name": "private", "scenario_count": 1, "content_hash": "sha256:abc"}
    md = _summary_md(tmp_path, summaries, scenario_report_metadata([held]), scenario_packs=[pack])
    assert "> - pack `private`: 1 scenario(s), content hash `sha256:abc`" in md


def test_summary_without_metadata_still_prints_summaries(tmp_path: Path) -> None:
    held = _held_out_scenario(tmp_path)
    summaries = [score_results([_pack_result(held, ScenarioStatus.FAIL)], [held])] * 2
    md = _summary_md(tmp_path, summaries, None)
    assert SECRET_ANSWER in md  # no metadata, no held-out knowledge
    assert "held-out scenario(s)" not in md


def test_summary_table_cells_are_escaped(tmp_path: Path) -> None:
    public = SCENARIOS[0]
    result = ScenarioResult(public.id, ScenarioStatus.FAIL, 0, "a | b\nc <x>")
    summaries = [score_results([result], [public])] * 2
    md = _summary_md(tmp_path, summaries, scenario_report_metadata([public]))
    assert f"| **{public.id}** | a &#124; b<br>c &lt;x&gt; |" in md


def test_summary_links_trial_reports_relative_to_itself(tmp_path: Path) -> None:
    public = SCENARIOS[0]
    summaries = [score_results([ScenarioResult(public.id, ScenarioStatus.PASS, 2, "ok")], [public])]
    summaries *= 2
    reporter = MarkdownReporter(root=str(tmp_path / "runs"))
    folder = reporter.write_summary_report("run-1", "m", summaries, AGG).parent
    # A trial in the summary's own folder, and one written the month before.
    sibling = folder / "run-1.md"
    previous_month = folder.parent / "00" / "run-0.md"
    path = reporter.write_summary_report(
        "run-1", "m", summaries, AGG, report_paths=[str(sibling), str(previous_month)]
    )
    md = path.read_text(encoding="utf-8")
    assert "- Trial 1: `run-1.md`" in md
    assert "- Trial 2: `../00/run-0.md`" in md
    assert str(tmp_path) not in md


def test_summary_rating_column_does_not_pretend_trial_one_is_the_aggregate(
    tmp_path: Path,
) -> None:
    public = SCENARIOS[0]
    good = score_results([ScenarioResult(public.id, ScenarioStatus.PASS, 2, "ok")], [public])
    bad = score_results([ScenarioResult(public.id, ScenarioStatus.FAIL, 0, "no")], [public])
    assert good.rating != bad.rating
    mixed = _summary_md(tmp_path, [good, bad], None)
    assert f"| **Rating** | {good.rating} | {bad.rating} | varies |" in mixed
    same = _summary_md(tmp_path, [good, good], None)
    assert f"| **Rating** | {good.rating} | {good.rating} | {good.rating} |" in same


def test_scenario_report_withholds_held_out_safety_and_diagnostics(tmp_path: Path) -> None:
    held = _held_out_scenario(tmp_path)
    public = SCENARIOS[0]
    held_result = _pack_result(held, ScenarioStatus.FAIL)
    held_result.safety_violation = f"leaked {SECRET_ANSWER}"
    held_result.parallel_tool_turns = [3]
    public_result = ScenarioResult(public.id, ScenarioStatus.FAIL, 0, "no")
    public_result.safety_violation = "public violation"
    public_result.parallel_tool_turns = [2]
    summary = score_results([held_result, public_result], [held, public])
    assert len(summary.safety_warnings) == 2

    md = (
        MarkdownReporter(root=str(tmp_path / "runs"))
        .write_scenario_report(
            "run-1", "m", summary, scenario_metadata=scenario_report_metadata([held, public])
        )
        .read_text(encoding="utf-8")
    )
    assert SECRET_ANSWER not in md
    assert SECRET_TITLE not in md
    assert "**2 safety-critical failure(s) detected:**" in md
    assert "> - HOLD-01: _held out_" in md
    assert f"> - {public.id} ({public.title}): public violation" in md
    assert f"- **{public.id}**: parallel tool turns: 2" in md
    assert "- **HOLD-01**:" not in md


def test_held_out_safety_warning_without_title_parens_is_still_redacted(
    tmp_path: Path,
) -> None:
    summary = ModelScoreSummary(
        total_points=0,
        max_points=2,
        final_score=0,
        rating="Weak",
        safety_warnings=[
            f"HOLD-01: leaked {SECRET_ANSWER}",
            f"HOLD-02 ({SECRET_TITLE}): leaked second answer",
        ],
        category_scores=[
            CategoryScore(ScenarioCategory.A, "Tool Selection", 0, 2, 0.0, fail_count=1),
        ],
        scenario_results=[
            ScenarioResult(
                scenario_id="HOLD-01",
                status=ScenarioStatus.FAIL,
                points=0,
                summary="secret summary",
                expected_behavior="secret expected",
                raw_log="secret trace",
            )
        ],
    )
    meta = {
        "HOLD-01": ScenarioReportMetadata(SECRET_TITLE, ScenarioCategory.A, 3, held_out=True),
        "HOLD-02": ScenarioReportMetadata(SECRET_TITLE, ScenarioCategory.A, 3, held_out=True),
    }
    md = (
        MarkdownReporter(root=str(tmp_path))
        .write_scenario_report("r-held", "m", summary, scenario_metadata=meta)
        .read_text(encoding="utf-8")
    )
    assert SECRET_ANSWER not in md
    assert SECRET_TITLE not in md
    assert "secret summary" not in md
    assert "secret trace" not in md
    assert "> - HOLD-01: _held out_" in md
    assert "> - HOLD-02: _held out_" in md
