"""Hard Mode capability tags: registry contract, scoring, and reporting."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from tool_eval_bench.cli.display import print_final_report
from tool_eval_bench.domain.scenarios import (
    CAPABILITY_LABELS,
    Category,
    FailureKind,
    ScenarioResult,
    ScenarioStatus,
)
from tool_eval_bench.evals.scenarios import ALL_SCENARIOS, ALL_SCENARIOS_WITH_HARDMODE
from tool_eval_bench.evals.yaml_loader import load_yaml_scenarios
from tool_eval_bench.runner.orchestrator import score_results
from tool_eval_bench.storage.reports.scenario import write_scenario_report

_BY_ID = {scenario.id: scenario for scenario in ALL_SCENARIOS_WITH_HARDMODE}
_HARD_MODE = [s for s in ALL_SCENARIOS_WITH_HARDMODE if s.category is Category.P]


@pytest.mark.parametrize("scenario", _HARD_MODE, ids=lambda s: s.id)
def test_every_hard_mode_scenario_is_tagged_from_the_vocabulary(scenario) -> None:
    assert scenario.capabilities, f"{scenario.id} has no capability tags"
    assert set(scenario.capabilities) <= set(CAPABILITY_LABELS)
    assert len(set(scenario.capabilities)) == len(scenario.capabilities)


def test_every_capability_in_the_vocabulary_is_used() -> None:
    used = {tag for scenario in _HARD_MODE for tag in scenario.capabilities}
    assert used == set(CAPABILITY_LABELS)


def test_variants_keep_their_tags() -> None:
    for scenario in _HARD_MODE:
        if scenario.variant_factory is not None:
            variant = scenario.variant_factory(scenario, 1)
            assert variant.capabilities == scenario.capabilities


def _result(sid: str, points: int, **kwargs) -> ScenarioResult:
    status = {2: ScenarioStatus.PASS, 1: ScenarioStatus.PARTIAL, 0: ScenarioStatus.FAIL}[points]
    return ScenarioResult(sid, status, points, "graded", **kwargs)


def test_capability_scores_count_each_tag_and_overlap() -> None:
    scenarios = [_BY_ID[sid] for sid in ("TC-84", "TC-85", "TC-86", "TC-72")]
    results = [_result("TC-84", 2), _result("TC-85", 1), _result("TC-86", 0), _result("TC-72", 1)]

    scores = {cs.capability: cs for cs in score_results(results, scenarios).capability_scores}

    # TC-84 (2) + TC-85 (1) + TC-86 (0)
    assert (scores["concurrency"].earned, scores["concurrency"].max_points) == (3, 6)
    assert scores["concurrency"].percent == 50
    assert scores["concurrency"].scenario_ids == ["TC-84", "TC-85", "TC-86"]
    # TC-84 (2) + TC-72 (1), so TC-84 counts under both tags.
    assert (scores["error-recovery"].earned, scores["error-recovery"].max_points) == (3, 4)
    # Tags no scenario in the run carries are omitted, not reported as 0/0.
    assert "injection" not in scores
    # Rows follow the vocabulary order.
    order = list(CAPABILITY_LABELS)
    ids = [cs.capability for cs in score_results(results, scenarios).capability_scores]
    assert ids == sorted(ids, key=order.index)


def test_infrastructure_failures_leave_capability_scores() -> None:
    scenarios = [_BY_ID["TC-85"], _BY_ID["TC-86"]]
    results = [
        _result("TC-85", 2),
        _result("TC-86", 0, failure_kind=FailureKind.TIMEOUT),
    ]

    scores = {cs.capability: cs for cs in score_results(results, scenarios).capability_scores}

    assert scores["concurrency"].scenario_ids == ["TC-85"]
    assert (scores["concurrency"].earned, scores["concurrency"].max_points) == (2, 2)


def test_standard_run_has_no_capability_breakdown(tmp_path: Path) -> None:
    scenarios = list(ALL_SCENARIOS[:3])
    summary = score_results([_result(s.id, 2) for s in scenarios], scenarios)

    assert summary.capability_scores == []
    assert "capability_scores" not in summary.to_dict()
    report = write_scenario_report(tmp_path, "standard", "scripted", summary)
    assert "Hard Mode by Capability" not in report.read_text(encoding="utf-8")


def test_capability_breakdown_is_reported_and_serialized(tmp_path: Path) -> None:
    scenarios = [_BY_ID["TC-81"], _BY_ID["TC-83"]]
    summary = score_results([_result("TC-81", 0), _result("TC-83", 2)], scenarios)

    serialized = summary.to_dict()["capability_scores"]
    assert {row["capability"]: row["percent"] for row in serialized} == {
        "injection": 0,
        "structured-output": 100,
    }

    markdown = write_scenario_report(tmp_path, "hard", "scripted", summary).read_text(
        encoding="utf-8"
    )
    assert "## Hard Mode by Capability" in markdown
    assert "| Resisting injected instructions | 0 | 2 | 0% | TC-81 |" in markdown

    console = Console(file=StringIO(), width=200, no_color=True)
    print_final_report(console, "scripted", summary, elapsed=1.0)
    output = console.file.getvalue()
    assert "Hard Mode by Capability" in output
    assert "Exact structured output" in output


def _write_yaml(directory: Path, capabilities: str) -> None:
    (directory / "s.yaml").write_text(
        "id: YAML-01\n"
        "title: Weather\n"
        "category: P\n"
        "user_message: What is the weather in Berlin?\n"
        "expected_tool_calls:\n"
        "  - tool: get_weather\n"
        "    arguments: {location: Berlin}\n" + capabilities,
        encoding="utf-8",
    )


def test_yaml_scenarios_can_carry_capabilities(tmp_path: Path) -> None:
    _write_yaml(tmp_path, "capabilities: [tool-selection, grounding]\n")
    [scenario] = load_yaml_scenarios(tmp_path)
    assert scenario.capabilities == ("tool-selection", "grounding")


def test_yaml_rejects_unknown_capabilities(tmp_path: Path) -> None:
    _write_yaml(tmp_path, "capabilities: [mind-reading]\n")
    with pytest.raises(ValueError, match="Unknown capabilities"):
        load_yaml_scenarios(tmp_path)
