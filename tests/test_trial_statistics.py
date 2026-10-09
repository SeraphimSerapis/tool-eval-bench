"""Trial statistics score each trial the way the service scored the row it stored.

The live, ``--no-live`` and ``--json`` runners must agree: an infrastructure
failure stays out of the score, and a resumed trial is scored against the
whole original protocol, not the rerun subset.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from tool_eval_bench.cli import dispatch
from tool_eval_bench.cli.legacy_parser import make_parser
from tool_eval_bench.domain.scenarios import ModelScoreSummary, ScenarioResult, ScenarioStatus
from tool_eval_bench.evals.scenarios import SCENARIOS
from tool_eval_bench.runner.orchestrator import score_results

TC01 = next(s for s in SCENARIOS if s.id == "TC-01")  # category A
TC04 = next(s for s in SCENARIOS if s.id == "TC-04")  # category B
PASS_01 = ScenarioResult("TC-01", ScenarioStatus.PASS, 2, "ok")
PASS_04 = ScenarioResult("TC-04", ScenarioStatus.PASS, 2, "ok")
TIMEOUT_04 = ScenarioResult("TC-04", ScenarioStatus.FAIL, 0, "timed out", failure_kind="timeout")


class _Service:
    """Reports ``executed`` through the callbacks and returns ``stored`` as the row."""

    def __init__(self, executed: list[ScenarioResult], stored: list[ScenarioResult]) -> None:
        self.executed = {result.scenario_id: result for result in executed}
        self.stored = stored

    async def run_benchmark(self, **kwargs: Any) -> dict[str, Any]:
        scenarios = kwargs["scenarios"]
        for idx, scenario in enumerate(scenarios):
            if on_result := kwargs.get("on_scenario_result"):
                await on_result(scenario, self.executed[scenario.id], idx, len(scenarios))
        return {
            "run_id": "run-1",
            "scores": {
                "final_score": 0,
                "scenario_results": [copy.deepcopy(r.to_dict()) for r in self.stored],
            },
        }


class _Display:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.results: dict[str, ScenarioResult] = {}

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    async def on_scenario_start(self, *args: Any) -> None:
        pass

    async def on_scenario_result(
        self, scenario: Any, result: ScenarioResult, idx: int, total: int
    ) -> None:
        self.results[scenario.id] = result

    def on_rate_limit(self, status: Any) -> None:
        pass

    def set_finished(self, *args: Any, **kwargs: Any) -> None:
        pass


def _run(
    mode: str, args: argparse.Namespace, service: _Service, monkeypatch: pytest.MonkeyPatch
) -> list[ModelScoreSummary]:
    """Run ``mode`` and return the per-trial summaries it aggregated."""
    summaries: list[ModelScoreSummary] = []

    def aggregate(trials: list[ModelScoreSummary]) -> dict[str, Any]:
        summaries.extend(trials)
        return {}

    monkeypatch.setattr(dispatch, "_aggregate_trials", aggregate)
    monkeypatch.setattr(dispatch, "BenchmarkDisplay", _Display)
    monkeypatch.setattr(dispatch, "_emit_json_output", lambda *a, **k: None)
    console = Console(record=True)
    if mode == "live":
        dispatch._run_with_live_display(service, console, "m", "M", "vllm", "url", None, args)  # type: ignore[arg-type]
    elif mode == "plain":
        dispatch._run_plain(service, console, "m", "M", "vllm", "url", None, args)  # type: ignore[arg-type]
    else:
        dispatch._run_json(service, "m", "vllm", "url", None, args)  # type: ignore[arg-type]
    return summaries


def _args(tmp_path: Path, *flags: str) -> argparse.Namespace:
    return make_parser().parse_args(
        ["--scenarios", "TC-01", "TC-04", "--trials", "2", "--output-dir", str(tmp_path), *flags]
    )


def _as_service_scores(results: list[ScenarioResult], weighted: bool) -> ModelScoreSummary:
    stored = [ScenarioResult.from_dict(r.to_dict()) for r in results]
    return score_results(stored, [TC01, TC04], alpha=0.7, weight_by_difficulty=weighted)


def _key(summary: ModelScoreSummary) -> tuple[Any, ...]:
    categories = [(c.category, c.earned, c.max_points) for c in summary.category_scores]
    return (
        summary.final_score,
        summary.total_points,
        summary.max_points,
        summary.weighted_score,
        summary.excluded_scenarios,
        categories,
    )


MODES = ["live", "plain", "json"]


@pytest.mark.parametrize("mode", MODES)
def test_a_timeout_stays_out_of_every_trial_score(
    mode: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = [PASS_01, TIMEOUT_04]

    summaries = _run(mode, _args(tmp_path), _Service(results, results), monkeypatch)

    expected = _as_service_scores(results, weighted=False)
    assert expected.final_score == 100 and expected.excluded_scenarios == ["TC-04"]
    assert [_key(s) for s in summaries] == [_key(expected)] * 2


@pytest.mark.parametrize("mode", MODES)
def test_a_resumed_trial_is_scored_against_the_whole_protocol(
    mode: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The state _plan_resume leaves: TC-01 preserved, only TC-04 rerun.
    args = _args(tmp_path, "--weight-by-difficulty")
    args.scenarios = ["TC-04"]
    args._resume_remaining_scenarios = [TC04]
    args._resume_scenarios = [TC01, TC04]
    args._resume_prior_results = [PASS_01.to_dict()]
    args._resume_run_id = "run-1"
    merged = [PASS_01, PASS_04]

    summaries = _run(mode, args, _Service([PASS_04], merged), monkeypatch)

    expected = _as_service_scores(merged, weighted=True)
    assert expected.max_points == 4 and expected.weighted_score is not None
    assert [_key(s) for s in summaries] == [_key(expected)] * 2
