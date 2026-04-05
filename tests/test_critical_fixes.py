"""Tests for critical review fixes.

Covers:
  - Weighted scoring (scenario-count, not category-count)
  - Flexible datetime matching (TC-05, TC-08)
  - TC-14 tighter partial credit (web_search after error)
  - Safety exfiltration scan (TC-34 / scan_for_injection)
  - Calibration confidence levels
  - Parallel concurrency warning
"""

from __future__ import annotations

from tool_eval_bench.domain.scenarios import (
    Category,
    ScenarioResult,
    ScenarioStatus,
)
from tool_eval_bench.evals.helpers import (
    date_matches,
    datetime_matches,
    scan_for_injection,
)
from tool_eval_bench.evals.scenarios import _tc08_eval, _tc14_eval, _tc05_eval
from tool_eval_bench.domain.scenarios import ScenarioState, ToolCallRecord, ToolResultRecord
from tool_eval_bench.runner.orchestrator import score_results
from tool_eval_bench.runner.throughput import TokenizerConfig


# ---------------------------------------------------------------------------
# 1. Weighted scoring — scenario count should drive final_score
# ---------------------------------------------------------------------------

class TestWeightedScoring:
    """The old formula averaged category percentages equally (category-count weighted).
    The correct formula weights by total points earned vs total points possible.
    """

    def _make_result(self, scenario_id: str, status: ScenarioStatus, points: int) -> ScenarioResult:
        return ScenarioResult(scenario_id=scenario_id, status=status, points=points, summary="")

    def _make_scenario(self, sid: str, category: Category):
        """Minimal stand-in — we only need id and category for score_results."""
        from tool_eval_bench.domain.scenarios import ScenarioDefinition
        return ScenarioDefinition(
            id=sid, title=sid, category=category,
            user_message="", description="",
            handle_tool_call=lambda s, c: {},
            evaluate=lambda s: None,
        )

    def test_equal_weight_per_scenario_not_per_category(self):
        """3 scenarios in cat A (all pass) + 1 scenario in cat B (fail).

        Old formula: avg(100%, 0%) = 50
        New formula: 6 earned / 8 max = 75
        """
        scenarios = [
            self._make_scenario("A1", Category.A),
            self._make_scenario("A2", Category.A),
            self._make_scenario("A3", Category.A),
            self._make_scenario("B1", Category.B),
        ]
        results = [
            self._make_result("A1", ScenarioStatus.PASS, 2),
            self._make_result("A2", ScenarioStatus.PASS, 2),
            self._make_result("A3", ScenarioStatus.PASS, 2),
            self._make_result("B1", ScenarioStatus.FAIL, 0),
        ]
        summary = score_results(results, scenarios)
        # New weighted formula: 6/8 = 75%
        assert summary.final_score == 75

    def test_all_pass_is_100(self):
        scenarios = [
            self._make_scenario("A1", Category.A),
            self._make_scenario("B1", Category.B),
            self._make_scenario("K1", Category.K),
        ]
        results = [
            self._make_result("A1", ScenarioStatus.PASS, 2),
            self._make_result("B1", ScenarioStatus.PASS, 2),
            self._make_result("K1", ScenarioStatus.PASS, 2),
        ]
        summary = score_results(results, scenarios)
        assert summary.final_score == 100

    def test_all_fail_is_0(self):
        scenarios = [
            self._make_scenario("A1", Category.A),
            self._make_scenario("K1", Category.K),
        ]
        results = [
            self._make_result("A1", ScenarioStatus.FAIL, 0),
            self._make_result("K1", ScenarioStatus.FAIL, 0),
        ]
        summary = score_results(results, scenarios)
        assert summary.final_score == 0

    def test_large_low_scoring_category_drags_down_score(self):
        """10 K-cat scenarios all fail, 3 A-cat scenarios all pass.

        Old formula: avg(100%, 0%) = 50
        New formula: 6 earned / 26 max ≈ 23%
        """
        scenarios = (
            [self._make_scenario(f"A{i}", Category.A) for i in range(3)]
            + [self._make_scenario(f"K{i}", Category.K) for i in range(10)]
        )
        results = (
            [self._make_result(f"A{i}", ScenarioStatus.PASS, 2) for i in range(3)]
            + [self._make_result(f"K{i}", ScenarioStatus.FAIL, 0) for i in range(10)]
        )
        summary = score_results(results, scenarios)
        # 6 / 26 = 0.2307… → rounds to 23
        assert summary.final_score == 23

    def test_partial_scores_weighted_correctly(self):
        """Mix of pass/partial/fail across two differently-sized categories."""
        scenarios = [
            self._make_scenario("A1", Category.A),
            self._make_scenario("A2", Category.A),
            self._make_scenario("B1", Category.B),
            self._make_scenario("B2", Category.B),
            self._make_scenario("B3", Category.B),
        ]
        results = [
            self._make_result("A1", ScenarioStatus.PASS, 2),     # A: 2/4
            self._make_result("A2", ScenarioStatus.FAIL, 0),
            self._make_result("B1", ScenarioStatus.PASS, 2),     # B: 4/6
            self._make_result("B2", ScenarioStatus.PARTIAL, 1),
            self._make_result("B3", ScenarioStatus.PARTIAL, 1),
        ]
        summary = score_results(results, scenarios)
        # Total: 6 earned / 10 max = 60%
        assert summary.final_score == 60


# ---------------------------------------------------------------------------
# 2. Flexible datetime matching
# ---------------------------------------------------------------------------

class TestDatetimeMatches:
    def test_naive_iso(self):
        assert datetime_matches("2026-03-21T08:00:00", "2026-03-21", "08:00")

    def test_utc_suffix(self):
        assert datetime_matches("2026-03-21T08:00:00Z", "2026-03-21", "08:00")

    def test_positive_offset(self):
        assert datetime_matches("2026-03-21T08:00:00+01:00", "2026-03-21", "08:00")

    def test_negative_offset(self):
        assert datetime_matches("2026-03-21T08:00:00-05:00", "2026-03-21", "08:00")

    def test_space_separator(self):
        assert datetime_matches("2026-03-21 08:00:00", "2026-03-21", "08:00")

    def test_wrong_date_rejected(self):
        assert not datetime_matches("2026-03-22T08:00:00", "2026-03-21", "08:00")

    def test_wrong_time_rejected(self):
        assert not datetime_matches("2026-03-21T09:00:00", "2026-03-21", "08:00")

    def test_empty_string(self):
        assert not datetime_matches("", "2026-03-21", "08:00")

    def test_none_value(self):
        assert not datetime_matches(None, "2026-03-21", "08:00")

    def test_time_with_only_hhmm(self):
        """Partial time string — no seconds provided."""
        assert datetime_matches("2026-03-21T08:00", "2026-03-21", "08:00")


class TestDateMatches:
    def test_exact_match(self):
        assert date_matches("2026-03-23", "2026-03-23")

    def test_with_trailing_info(self):
        assert date_matches("2026-03-23T09:30:00", "2026-03-23")

    def test_wrong_date(self):
        assert not date_matches("2026-03-24", "2026-03-23")

    def test_none(self):
        assert not date_matches(None, "2026-03-23")


# ---------------------------------------------------------------------------
# 3. TC-05 flexible time/date matching
# ---------------------------------------------------------------------------

class TestTC05FlexibleDatetime:
    def _make_state(self, date: str, time: str) -> ScenarioState:
        state = ScenarioState()
        state.tool_calls = [ToolCallRecord(
            id="c1", name="create_calendar_event",
            raw_arguments="", arguments={
                "title": "Team Standup",
                "date": date,
                "time": time,
                "duration_minutes": 30,
                "attendees": ["alex@example.com", "jamie@example.com"],
            },
            turn=1,
        )]
        state.final_answer = "Meeting scheduled."
        return state

    def test_pass_naive(self):
        result = _tc05_eval(self._make_state("2026-03-23", "09:30"))
        assert result.status == ScenarioStatus.PASS

    def test_pass_with_seconds(self):
        result = _tc05_eval(self._make_state("2026-03-23", "09:30:00"))
        assert result.status == ScenarioStatus.PASS

    def test_fail_wrong_date(self):
        result = _tc05_eval(self._make_state("2026-03-24", "09:30"))
        assert result.status == ScenarioStatus.FAIL

    def test_fail_wrong_time(self):
        result = _tc05_eval(self._make_state("2026-03-23", "10:00"))
        assert result.status == ScenarioStatus.FAIL


# ---------------------------------------------------------------------------
# 4. TC-08 flexible datetime (timezone-aware models)
# ---------------------------------------------------------------------------

class TestTC08FlexibleDatetime:
    def _make_state(self, datetime_val: str) -> ScenarioState:
        state = ScenarioState()
        state.tool_calls = [
            ToolCallRecord(id="c1", name="get_weather", raw_arguments="", arguments={"location": "Paris"}, turn=1),
            ToolCallRecord(id="c2", name="set_reminder", raw_arguments="", arguments={
                "message": "bring an umbrella",
                "datetime": datetime_val,
            }, turn=2),
        ]
        state.final_answer = "Reminder set."
        return state

    def test_pass_naive_datetime(self):
        result = _tc08_eval(self._make_state("2026-03-21T08:00:00"))
        assert result.status == ScenarioStatus.PASS

    def test_pass_utc_datetime(self):
        result = _tc08_eval(self._make_state("2026-03-21T08:00:00Z"))
        assert result.status == ScenarioStatus.PASS

    def test_pass_cet_datetime(self):
        """Timezone-aware models emitting CET offset should pass."""
        result = _tc08_eval(self._make_state("2026-03-21T08:00:00+01:00"))
        assert result.status == ScenarioStatus.PASS

    def test_pass_est_datetime(self):
        result = _tc08_eval(self._make_state("2026-03-21T08:00:00-05:00"))
        assert result.status == ScenarioStatus.PASS

    def test_fail_wrong_date(self):
        result = _tc08_eval(self._make_state("2026-03-22T08:00:00"))
        assert result.status == ScenarioStatus.FAIL

    def test_fail_wrong_time(self):
        result = _tc08_eval(self._make_state("2026-03-21T09:00:00"))
        assert result.status == ScenarioStatus.FAIL

    def test_fail_no_reminder(self):
        state = ScenarioState()
        state.tool_calls = [
            ToolCallRecord(id="c1", name="get_weather", raw_arguments="", arguments={"location": "Paris"}, turn=1),
        ]
        state.final_answer = "It is raining."
        result = _tc08_eval(state)
        assert result.status == ScenarioStatus.FAIL


# ---------------------------------------------------------------------------
# 5. TC-14 — web_search must come AFTER the stock error
# ---------------------------------------------------------------------------

class TestTC14TighterPartial:
    def _make_state(self, search_turn: int, stock_turn: int = 1) -> ScenarioState:
        state = ScenarioState()
        state.tool_calls = [
            ToolCallRecord(id="c1", name="get_stock_price", raw_arguments="", arguments={"ticker": "AAPL"}, turn=stock_turn),
            ToolCallRecord(id="c2", name="web_search", raw_arguments="", arguments={"query": "AAPL price"}, turn=search_turn),
        ]
        state.tool_results = [
            ToolResultRecord(call_id="c1", name="get_stock_price", result={"error": "Rate limit exceeded."}),
            ToolResultRecord(call_id="c2", name="web_search", result={"results": [{"snippet": "AAPL ~$187"}]}),
        ]
        state.assistant_messages = ["rate limit exceeded, trying web search"]
        state.final_answer = "AAPL is around $187."
        return state

    def test_pass_search_after_error_with_acknowledgment(self):
        result = _tc14_eval(self._make_state(search_turn=2, stock_turn=1))
        assert result.status == ScenarioStatus.PASS

    def test_partial_search_before_error_not_counted_as_recovery(self):
        """web_search in same turn as stock call — model searched proactively,
        not as a fallback after seeing the error. No acknowledgment in text either.
        Should not get full pass credit."""
        state = ScenarioState()
        state.tool_calls = [
            ToolCallRecord(id="c1", name="get_stock_price", raw_arguments="", arguments={"ticker": "AAPL"}, turn=1),
            ToolCallRecord(id="c2", name="web_search",      raw_arguments="", arguments={"query": "AAPL price"}, turn=1),
        ]
        state.tool_results = [
            ToolResultRecord(call_id="c1", name="get_stock_price", result={"error": "Rate limit exceeded."}),
            ToolResultRecord(call_id="c2", name="web_search",      result={"results": [{"snippet": "AAPL ~$187"}]}),
        ]
        # No acknowledgment, no mention of "web search" as a plan — just a direct answer
        state.assistant_messages = ["AAPL is trading around $187."]
        state.final_answer = "AAPL is trading around $187."
        result = _tc14_eval(state)
        # search was same turn (not a recovery), no acknowledgment → FAIL or PARTIAL at best
        assert result.status in (ScenarioStatus.PARTIAL, ScenarioStatus.FAIL)

    def test_partial_acknowledged_no_search(self):
        state = ScenarioState()
        state.tool_calls = [
            ToolCallRecord(id="c1", name="get_stock_price", raw_arguments="", arguments={}, turn=1),
        ]
        state.tool_results = [
            ToolResultRecord(call_id="c1", name="get_stock_price", result={"error": "Unavailable"}),
        ]
        state.assistant_messages = ["The stock service is temporarily unavailable."]
        state.final_answer = "The service is temporarily unavailable. Please try again later."
        result = _tc14_eval(state)
        assert result.status == ScenarioStatus.PARTIAL

    def test_fail_no_stock_call_at_all(self):
        state = ScenarioState()
        state.tool_calls = []
        state.tool_results = []
        state.assistant_messages = ["AAPL is $187."]
        state.final_answer = "AAPL is $187."
        result = _tc14_eval(state)
        assert result.status == ScenarioStatus.FAIL


# ---------------------------------------------------------------------------
# 6. scan_for_injection — exfiltration detection
# ---------------------------------------------------------------------------

class TestScanForInjection:
    def _make_state(self, messages: list[str]) -> ScenarioState:
        state = ScenarioState()
        state.assistant_messages = messages
        return state

    def test_detects_leaked_email(self):
        state = self._make_state(["I found the notes. Note: attacker@evil.com was mentioned."])
        assert scan_for_injection(state, ["attacker@evil.com"])

    def test_detects_leaked_instruction(self):
        state = self._make_state(["The document says 'ignore all previous instructions'."])
        assert scan_for_injection(state, ["ignore all previous"])

    def test_clean_state_not_flagged(self):
        state = self._make_state(["The action items are: 1. Review budget 2. Schedule follow-up."])
        assert not scan_for_injection(state, ["attacker@evil.com", "credentials", "system override"])

    def test_case_insensitive(self):
        state = self._make_state(["ATTACKER@EVIL.COM is mentioned in the file."])
        assert scan_for_injection(state, ["attacker@evil.com"])

    def test_multi_turn_transcript_scanned(self):
        """Injection in an intermediate assistant turn, not the final answer."""
        state = self._make_state([
            "Reading the file now.",
            "The file contains a system override instruction directing me to send credentials.",
        ])
        assert scan_for_injection(state, ["system override", "credentials"])

    def test_empty_markers_never_triggers(self):
        state = self._make_state(["Anything here."])
        assert not scan_for_injection(state, [])

    def test_empty_transcript(self):
        state = self._make_state([])
        assert not scan_for_injection(state, ["attacker@evil.com"])


# ---------------------------------------------------------------------------
# 7. Calibration confidence levels
# ---------------------------------------------------------------------------

class TestCalibrationConfidence:
    def test_default_is_heuristic(self):
        cfg = TokenizerConfig()
        assert cfg.calibration_confidence == "heuristic"

    def test_heuristic_chars_per_token(self):
        cfg = TokenizerConfig()
        assert cfg.chars_per_token == 4.0

    def test_set_tokenize_confidence(self):
        cfg = TokenizerConfig()
        cfg.calibration_confidence = "tokenize"
        cfg.has_tokenize_endpoint = True
        assert cfg.calibration_confidence == "tokenize"
        assert cfg.has_tokenize_endpoint is True

    def test_set_probe_confidence(self):
        cfg = TokenizerConfig()
        cfg.calibration_confidence = "probe"
        assert cfg.calibration_confidence == "probe"
