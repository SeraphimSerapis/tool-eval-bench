"""Tests for the scenario evaluation logic.

These test the evaluators directly without needing a live model,
by constructing ScenarioState objects manually.
"""


from tool_eval_bench.domain.scenarios import (
    Category,
    ScenarioState,
    ScenarioStatus,
    ToolCallRecord,
)
from tool_eval_bench.evals.scenarios import SCENARIOS
from tool_eval_bench.runner.orchestrator import score_results


def _make_state(
    tool_calls: list[dict] | None = None,
    final_answer: str = "",
    assistant_messages: list[str] | None = None,
) -> ScenarioState:
    state = ScenarioState()
    state.final_answer = final_answer
    state.assistant_messages = assistant_messages or ([final_answer] if final_answer else [])
    if tool_calls:
        for tc in tool_calls:
            state.tool_calls.append(
                ToolCallRecord(
                    id=tc.get("id", f"call_{len(state.tool_calls)}"),
                    name=tc["name"],
                    raw_arguments="{}",
                    arguments=tc.get("arguments", {}),
                    turn=tc.get("turn", 1),
                )
            )
    return state


class TestScenarioRegistry:
    def test_has_15_scenarios(self) -> None:
        assert len(SCENARIOS) == 15

    def test_all_categories_covered(self) -> None:
        cats = {s.category for s in SCENARIOS}
        assert cats == {Category.A, Category.B, Category.C, Category.D, Category.E}

    def test_3_scenarios_per_base_category(self) -> None:
        base_cats = {Category.A, Category.B, Category.C, Category.D, Category.E}
        for cat in base_cats:
            count = sum(1 for s in SCENARIOS if s.category == cat)
            assert count == 3, f"Category {cat.value} has {count} scenarios, expected 3"

    def test_ids_are_unique(self) -> None:
        ids = [s.id for s in SCENARIOS]
        assert len(ids) == len(set(ids))

    def test_all_scenarios_total(self) -> None:
        from tool_eval_bench.evals.scenarios import ALL_SCENARIOS
        assert len(ALL_SCENARIOS) == 69  # 15 base + 6 extended + 25 agentic + 4 large-toolset + 9 planning + 4 adversarial + 6 structured

    def test_all_ids_unique(self) -> None:
        from tool_eval_bench.evals.scenarios import ALL_SCENARIOS
        ids = [s.id for s in ALL_SCENARIOS]
        assert len(ids) == len(set(ids))

    def test_all_categories_present(self) -> None:
        from tool_eval_bench.evals.scenarios import ALL_SCENARIOS
        cats = {s.category for s in ALL_SCENARIOS}
        expected = {Category.A, Category.B, Category.C, Category.D, Category.E,
                    Category.F, Category.G, Category.H, Category.I, Category.J, Category.K,
                    Category.L, Category.M, Category.N, Category.O}
        assert cats == expected


class TestTC01DirectSpecialistMatch:
    scenario = next(s for s in SCENARIOS if s.id == "TC-01")

    def test_pass_get_weather(self) -> None:
        state = _make_state(
            tool_calls=[{"name": "get_weather", "arguments": {"location": "Berlin"}}],
            final_answer="Berlin is 8C and overcast.",
        )
        result = self.scenario.evaluate(state)
        assert result.status == ScenarioStatus.PASS
        assert result.points == 2

    def test_partial_web_search(self) -> None:
        state = _make_state(
            tool_calls=[{"name": "web_search", "arguments": {"query": "Berlin weather"}}],
            final_answer="Berlin is 8C and overcast.",
        )
        result = self.scenario.evaluate(state)
        assert result.status == ScenarioStatus.PARTIAL

    def test_fail_no_tools(self) -> None:
        state = _make_state(final_answer="I think it's cold in Berlin.")
        result = self.scenario.evaluate(state)
        assert result.status == ScenarioStatus.FAIL


class TestTC10TrivialKnowledge:
    scenario = next(s for s in SCENARIOS if s.id == "TC-10")

    def test_pass_direct_answer(self) -> None:
        state = _make_state(final_answer="World War II ended in 1945.")
        result = self.scenario.evaluate(state)
        assert result.status == ScenarioStatus.PASS

    def test_fail_used_tool(self) -> None:
        state = _make_state(
            tool_calls=[{"name": "web_search", "arguments": {"query": "WWII end"}}],
            final_answer="1945",
        )
        result = self.scenario.evaluate(state)
        assert result.status == ScenarioStatus.FAIL


class TestTC12ImpossibleRequest:
    scenario = next(s for s in SCENARIOS if s.id == "TC-12")

    def test_pass_refusal(self) -> None:
        state = _make_state(
            final_answer="I cannot delete your emails. The available tools don't include email deletion."
        )
        result = self.scenario.evaluate(state)
        assert result.status == ScenarioStatus.PASS

    def test_fail_tool_used(self) -> None:
        state = _make_state(
            tool_calls=[{"name": "send_email", "arguments": {"to": "x", "subject": "y", "body": "z"}}],
            final_answer="Done!",
        )
        result = self.scenario.evaluate(state)
        assert result.status == ScenarioStatus.FAIL


class TestTC35ContradictoryParams:
    """SCEN-04: Model should recognize a trivial/nonsensical conversion."""

    @staticmethod
    def _get_scenario():
        from tool_eval_bench.evals.scenarios import ALL_SCENARIOS
        return next(s for s in ALL_SCENARIOS if s.id == "TC-35")

    def test_pass_no_calc_with_explanation(self) -> None:
        s = self._get_scenario()
        state = _make_state(final_answer="500 Kelvin is already in Kelvin — no conversion needed.")
        result = s.evaluate(state)
        assert result.status == ScenarioStatus.PASS

    def test_fail_blind_calculation(self) -> None:
        s = self._get_scenario()
        state = _make_state(
            tool_calls=[{"name": "calculator", "arguments": {"expression": "500"}}],
            final_answer="I have computed the conversion for you.",
        )
        result = s.evaluate(state)
        assert result.status == ScenarioStatus.FAIL

    def test_partial_calc_with_insight(self) -> None:
        s = self._get_scenario()
        state = _make_state(
            tool_calls=[{"name": "calculator", "arguments": {"expression": "500"}}],
            final_answer="The answer is 500. Since it's the same unit, no conversion is needed.",
        )
        result = s.evaluate(state)
        assert result.status == ScenarioStatus.PARTIAL


class TestTC36MissingInfo:
    """SCEN-04: Model must ask for missing details, not guess."""

    @staticmethod
    def _get_scenario():
        from tool_eval_bench.evals.scenarios import ALL_SCENARIOS
        return next(s for s in ALL_SCENARIOS if s.id == "TC-36")

    def test_pass_asks_for_details(self) -> None:
        s = self._get_scenario()
        state = _make_state(
            final_answer="I'd be happy to send an email. Could you please provide the recipient and subject?"
        )
        result = s.evaluate(state)
        assert result.status == ScenarioStatus.PASS

    def test_fail_sends_without_confirmation(self) -> None:
        s = self._get_scenario()
        state = _make_state(
            tool_calls=[{"name": "send_email", "arguments": {"to": "boss@co.com", "subject": "Hello", "body": "..."}}],
            final_answer="Email sent!",
        )
        result = s.evaluate(state)
        assert result.status == ScenarioStatus.FAIL


class TestScoring:
    def test_perfect_score(self) -> None:
        from tool_eval_bench.domain.scenarios import ScenarioResult

        results = [
            ScenarioResult(
                scenario_id=s.id,
                status=ScenarioStatus.PASS,
                points=2,
                summary="Perfect",
            )
            for s in SCENARIOS
        ]
        summary = score_results(results, SCENARIOS)
        assert summary.final_score == 100
        assert summary.total_points == 30
        assert summary.rating == "★★★★★ Excellent"

    def test_zero_score(self) -> None:
        from tool_eval_bench.domain.scenarios import ScenarioResult

        results = [
            ScenarioResult(
                scenario_id=s.id,
                status=ScenarioStatus.FAIL,
                points=0,
                summary="Failed",
            )
            for s in SCENARIOS
        ]
        summary = score_results(results, SCENARIOS)
        assert summary.final_score == 0
        assert summary.rating == "★ Poor"

    def test_all_scenarios_score(self) -> None:
        """Test scoring with all 69 scenarios including 13-scenario Category K."""
        from tool_eval_bench.domain.scenarios import ScenarioResult
        from tool_eval_bench.evals.scenarios import ALL_SCENARIOS

        results = [
            ScenarioResult(
                scenario_id=s.id,
                status=ScenarioStatus.PASS,
                points=2,
                summary="Perfect",
            )
            for s in ALL_SCENARIOS
        ]
        summary = score_results(results, ALL_SCENARIOS)
        assert summary.final_score == 100
        assert summary.total_points == 138  # 69 × 2
        assert summary.max_points == 138
        assert len(summary.category_scores) == 15


class TestSafetyWarnings:
    """SCORE-03: Safety-critical failures in Category K generate explicit warnings."""

    def test_category_k_fail_generates_warning(self) -> None:
        from tool_eval_bench.domain.scenarios import ScenarioResult
        from tool_eval_bench.evals.scenarios import ALL_SCENARIOS

        results = [
            ScenarioResult(
                scenario_id=s.id,
                status=ScenarioStatus.PASS if s.category != Category.K else ScenarioStatus.FAIL,
                points=2 if s.category != Category.K else 0,
                summary="pass" if s.category != Category.K else "obeyed injected instructions",
            )
            for s in ALL_SCENARIOS
        ]
        summary = score_results(results, ALL_SCENARIOS)
        assert len(summary.safety_warnings) == 13  # all 13 Cat K scenarios
        assert any("TC-34" in w for w in summary.safety_warnings)
        # Safety gate: K at 0% → rating should be capped
        assert "safety-capped" in summary.rating

    def test_no_warnings_when_safety_passes(self) -> None:
        from tool_eval_bench.domain.scenarios import ScenarioResult
        from tool_eval_bench.evals.scenarios import ALL_SCENARIOS

        results = [
            ScenarioResult(
                scenario_id=s.id,
                status=ScenarioStatus.PASS,
                points=2,
                summary="pass",
            )
            for s in ALL_SCENARIOS
        ]
        summary = score_results(results, ALL_SCENARIOS)
        assert summary.safety_warnings == []
        # No safety cap when everything passes
        assert "safety-capped" not in summary.rating

    def test_non_safety_fail_no_warning(self) -> None:
        """Category A failures should NOT generate safety warnings."""
        from tool_eval_bench.domain.scenarios import ScenarioResult

        results = [
            ScenarioResult(
                scenario_id=s.id,
                status=ScenarioStatus.FAIL,
                points=0,
                summary="failed",
            )
            for s in SCENARIOS  # base 15, categories A-E only
        ]
        summary = score_results(results, SCENARIOS)
        assert summary.safety_warnings == []  # no Category K in base scenarios


class TestSafetyGating:
    """METH-01: Safety score gating prevents misleading ratings."""

    def test_high_score_capped_when_safety_fails(self) -> None:
        """A model scoring 90+ overall but failing all safety scenarios
        should NOT receive ★★★★★ Excellent."""
        from tool_eval_bench.domain.scenarios import ScenarioResult
        from tool_eval_bench.evals.scenarios import ALL_SCENARIOS

        results = [
            ScenarioResult(
                scenario_id=s.id,
                status=ScenarioStatus.PASS if s.category != Category.K else ScenarioStatus.FAIL,
                points=2 if s.category != Category.K else 0,
                summary="pass" if s.category != Category.K else "failed",
            )
            for s in ALL_SCENARIOS
        ]
        summary = score_results(results, ALL_SCENARIOS)
        # Overall score is still high (14/15 categories at 100%)
        # 100 earned / 126 max = ~79%
        assert summary.final_score > 75
        # But rating is capped
        assert "safety-capped" in summary.rating
        assert "Excellent" not in summary.rating

    def test_no_cap_when_safety_above_threshold(self) -> None:
        """If Category K scores ≥50%, no cap is applied."""
        from tool_eval_bench.domain.scenarios import ScenarioResult
        from tool_eval_bench.evals.scenarios import ALL_SCENARIOS

        # Pass 7/13 K scenarios → ~54% (above 50% threshold)
        k_scenarios = [s for s in ALL_SCENARIOS if s.category == Category.K]
        results = []
        for s in ALL_SCENARIOS:
            if s.category == Category.K:
                # Pass first 7, fail the rest
                idx = k_scenarios.index(s)
                if idx < 7:
                    results.append(ScenarioResult(scenario_id=s.id, status=ScenarioStatus.PASS, points=2, summary="ok"))
                else:
                    results.append(ScenarioResult(scenario_id=s.id, status=ScenarioStatus.FAIL, points=0, summary="fail"))
            else:
                results.append(ScenarioResult(scenario_id=s.id, status=ScenarioStatus.PASS, points=2, summary="ok"))

        summary = score_results(results, ALL_SCENARIOS)
        assert "safety-capped" not in summary.rating

    def test_rating_for_score_direct(self) -> None:
        """Test rating_for_score with and without safety_capped flag."""
        from tool_eval_bench.domain.scenarios import rating_for_score

        assert rating_for_score(95) == "★★★★★ Excellent"
        assert rating_for_score(95, safety_capped=True) == "★★★ Adequate (safety-capped)"
        assert rating_for_score(80, safety_capped=True) == "★★★ Adequate (safety-capped)"
        assert rating_for_score(60, safety_capped=True) == "★★★ Adequate (safety-capped)"
        # Below 60, rating is already ≤ Adequate, so cap doesn't change it
        assert rating_for_score(50, safety_capped=True) == "★★ Weak"
        assert rating_for_score(30, safety_capped=True) == "★ Poor"

    def test_safety_warnings_in_to_dict(self) -> None:
        from tool_eval_bench.domain.scenarios import ScenarioResult
        from tool_eval_bench.evals.scenarios import ALL_SCENARIOS

        results = [
            ScenarioResult(
                scenario_id=s.id,
                status=ScenarioStatus.FAIL if s.id == "TC-34" else ScenarioStatus.PASS,
                points=0 if s.id == "TC-34" else 2,
                summary="obeyed injection" if s.id == "TC-34" else "pass",
            )
            for s in ALL_SCENARIOS
        ]
        summary = score_results(results, ALL_SCENARIOS)
        d = summary.to_dict()
        assert "safety_warnings" in d
        assert len(d["safety_warnings"]) == 1


class TestTrialAggregation:
    """METH-02: Multi-trial statistical aggregation."""

    def test_single_trial_returns_empty(self) -> None:
        from tool_eval_bench.cli.bench import _aggregate_trials
        from tool_eval_bench.domain.scenarios import ModelScoreSummary

        summary = ModelScoreSummary(
            scenario_results=[], category_scores=[],
            final_score=80, total_points=40, max_points=50, rating="Good",
        )
        assert _aggregate_trials([summary]) == {}

    def test_identical_trials_zero_stddev(self) -> None:
        from tool_eval_bench.cli.bench import _aggregate_trials
        from tool_eval_bench.domain.scenarios import CategoryScore, ModelScoreSummary, ScenarioResult

        sr = ScenarioResult(scenario_id="TC-01", status=ScenarioStatus.PASS, points=2, summary="ok")
        cs = CategoryScore(category=Category.A, label="Tool Selection", earned=6, max_points=6, percent=100.0)
        s1 = ModelScoreSummary(
            scenario_results=[sr], category_scores=[cs],
            final_score=100, total_points=6, max_points=6, rating="Excellent",
        )
        s2 = ModelScoreSummary(
            scenario_results=[sr], category_scores=[cs],
            final_score=100, total_points=6, max_points=6, rating="Excellent",
        )
        agg = _aggregate_trials([s1, s2])
        assert agg["trials"] == 2
        assert agg["final_score_mean"] == 100.0
        assert agg["final_score_stddev"] == 0.0
        assert agg["per_scenario"]["TC-01"]["stddev"] == 0.0

    def test_different_trials_nonzero_stddev(self) -> None:
        from tool_eval_bench.cli.bench import _aggregate_trials
        from tool_eval_bench.domain.scenarios import CategoryScore, ModelScoreSummary, ScenarioResult

        sr1 = ScenarioResult(scenario_id="TC-01", status=ScenarioStatus.PASS, points=2, summary="ok")
        sr2 = ScenarioResult(scenario_id="TC-01", status=ScenarioStatus.FAIL, points=0, summary="fail")
        cs1 = CategoryScore(category=Category.A, label="Tool Selection", earned=6, max_points=6, percent=100.0)
        cs2 = CategoryScore(category=Category.A, label="Tool Selection", earned=4, max_points=6, percent=67.0)

        s1 = ModelScoreSummary(
            scenario_results=[sr1], category_scores=[cs1],
            final_score=100, total_points=6, max_points=6, rating="Excellent",
        )
        s2 = ModelScoreSummary(
            scenario_results=[sr2], category_scores=[cs2],
            final_score=67, total_points=4, max_points=6, rating="Adequate",
        )
        agg = _aggregate_trials([s1, s2])
        assert agg["trials"] == 2
        assert agg["final_score_stddev"] > 0
        assert agg["per_scenario"]["TC-01"]["stddev"] > 0
        assert agg["per_scenario"]["TC-01"]["points"] == [2, 0]
        assert agg["per_category"]["A"]["stddev_percent"] > 0


class TestWorstCategory:
    """SCORE-02: Worst category floor metric."""

    def test_perfect_score_worst_is_100(self) -> None:
        from tool_eval_bench.domain.scenarios import ScenarioResult

        results = [
            ScenarioResult(scenario_id=s.id, status=ScenarioStatus.PASS, points=2, summary="ok")
            for s in SCENARIOS
        ]
        summary = score_results(results, SCENARIOS)
        assert summary.worst_category_percent == 100
        assert summary.worst_category is not None

    def test_mixed_scores_identifies_weakest(self) -> None:
        from tool_eval_bench.domain.scenarios import ScenarioResult

        results = []
        for s in SCENARIOS:
            # Fail all category A scenarios, pass everything else
            if s.category == Category.A:
                results.append(ScenarioResult(scenario_id=s.id, status=ScenarioStatus.FAIL, points=0, summary="fail"))
            else:
                results.append(ScenarioResult(scenario_id=s.id, status=ScenarioStatus.PASS, points=2, summary="pass"))

        summary = score_results(results, SCENARIOS)
        assert summary.worst_category_percent == 0
        assert "A " in summary.worst_category  # Category A label
        assert "(0%)" in summary.worst_category

    def test_worst_in_to_dict(self) -> None:
        from tool_eval_bench.domain.scenarios import ScenarioResult

        results = [
            ScenarioResult(scenario_id=s.id, status=ScenarioStatus.PASS, points=2, summary="ok")
            for s in SCENARIOS
        ]
        summary = score_results(results, SCENARIOS)
        d = summary.to_dict()
        assert "worst_category" in d
        assert d["worst_category_percent"] == 100


class TestSQLitePath:
    """ARCH-05: SQLite path is resolved relative to project root."""

    def test_default_path_is_absolute(self) -> None:
        from tool_eval_bench.storage.db import _default_db_path

        path = _default_db_path()
        assert path.startswith("/")
        assert "data/benchmarks.sqlite" in path

    def test_custom_path_accepted(self) -> None:
        import tempfile
        from tool_eval_bench.storage.db import RunRepository

        with tempfile.TemporaryDirectory() as tmp:
            custom = f"{tmp}/test.db"
            repo = RunRepository(db_path=custom)
            assert str(repo.db_path) == custom
            repo.close()


class TestReferenceDate:
    """METH-03: Configurable reference date for date-relative scenarios."""

    def test_default_reference_date_in_messages(self) -> None:
        from tool_eval_bench.runner.orchestrator import _initial_messages

        msgs = _initial_messages("test")
        system = msgs[0]["content"]
        assert "2026-03-20" in system
        assert "Friday" in system

    def test_custom_reference_date(self) -> None:
        from tool_eval_bench.runner.orchestrator import _initial_messages

        msgs = _initial_messages("test", reference_date="2025-12-25", reference_day="Thursday")
        system = msgs[0]["content"]
        assert "2025-12-25" in system
        assert "Thursday" in system
        assert "2026-03-20" not in system

    def test_service_computes_day_name(self) -> None:
        from datetime import datetime

        # Verify the day computation logic
        day = datetime.strptime("2025-12-25", "%Y-%m-%d").strftime("%A")
        assert day == "Thursday"


