"""Tests for withheld scores on 0-graded runs, degenerate loop detection, and answer-in-thinking diagnostics."""

from __future__ import annotations

import json
import math
from io import StringIO
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from tool_eval_bench.api import format_result
from tool_eval_bench.application.card_data import build_card_data
from tool_eval_bench.application.service import BenchmarkService
from tool_eval_bench.cli.dispatch import _print_trials_summary
from tool_eval_bench.cli.display import _print_final_panel
from tool_eval_bench.cli.history import compare_runs, print_history
from tool_eval_bench.cli.leaderboard import export_runs, print_leaderboard
from tool_eval_bench.cli.run_io import aggregate_trials
from tool_eval_bench.domain.adapters import (
    BackendAdapter,
    ChatCompletionResult,
    ProviderToolCall,
)
from tool_eval_bench.domain.loop_detection import (
    LoopDetectionResult,
    detect_degenerate_loop,
)
from tool_eval_bench.domain.scenarios import (
    INFRASTRUCTURE_FAILURE_KINDS,
    Category,
    FailureKind,
    ModelScoreSummary,
    ScenarioDefinition,
    ScenarioEvaluation,
    ScenarioResult,
    ScenarioStatus,
    compute_deployability,
    rating_for_score,
)
from tool_eval_bench.runner.orchestrator import run_scenario, score_results
from tool_eval_bench.storage.reports import MarkdownReporter
from tool_eval_bench.storage.reports.card import render_card_svg
from tool_eval_bench.web.app import _build_comparison_payload, _format_export_payload

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ScriptedAdapter(BackendAdapter):
    def __init__(self, turns: list[ChatCompletionResult]) -> None:
        self._turns = list(turns)
        self._idx = 0

    async def chat_completion(self, **kwargs: Any) -> ChatCompletionResult:
        if self._idx >= len(self._turns):
            return ChatCompletionResult(content="fallback", finish_reason="stop")
        turn = self._turns[self._idx]
        self._idx += 1
        return turn

    async def health_check(self, base_url: str, api_key: str | None = None) -> bool:
        return True

    async def list_models(self, base_url: str, api_key: str | None = None) -> list[str]:
        return ["test-model"]


def _failing_scenario(
    scenario_id: str = "LOOP-01",
    *,
    category: Category = Category.A,
    difficulty: int = 3,
    follow_ups: list[str] | None = None,
) -> ScenarioDefinition:
    return ScenarioDefinition(
        id=scenario_id,
        title=f"Scenario {scenario_id}",
        category=category,
        user_message="Check the status and report back.",
        description="Expected tool call and answer.",
        difficulty=difficulty,
        follow_up_messages=follow_ups or [],
        handle_tool_call=lambda state, call: {"status": "ok"},
        evaluate=lambda state: ScenarioEvaluation(
            ScenarioStatus.FAIL, 0, "Did not complete the task."
        ),
    )


def _withheld_summary(
    scenario_ids: list[str] | None = None,
    *,
    weight_by_difficulty: bool = False,
) -> ModelScoreSummary:
    ids = scenario_ids or ["S1", "S2"]
    scenarios = [_failing_scenario(sid) for sid in ids]
    results = [
        ScenarioResult(
            scenario_id=sid,
            status=ScenarioStatus.FAIL,
            points=0,
            summary="Request timed out",
            failure_kind=FailureKind.TIMEOUT,
            turn_latencies_ms=[120_000.0],
        )
        for sid in ids
    ]
    return score_results(results, scenarios, weight_by_difficulty=weight_by_difficulty)


# ---------------------------------------------------------------------------
# 1. Loop detection unit tests
# ---------------------------------------------------------------------------


class TestDegenerateLoopDetection:
    def test_empty_and_none_inputs_do_not_trigger(self) -> None:
        assert detect_degenerate_loop(None) == LoopDetectionResult(False)
        assert detect_degenerate_loop("") == LoopDetectionResult(False)
        assert detect_degenerate_loop("Short normal reply.") == LoopDetectionResult(False)

    def test_detects_repeated_20_gram_word_cycle(self) -> None:
        sentence = (
            "I need to double check the user constraints before calling the weather "
            "service so that every required argument matches cleanly."
        )
        assert len(sentence.split()) == 20
        # Twice is not enough; three times triggers.
        assert detect_degenerate_loop(f"{sentence} {sentence}").detected is False
        res = detect_degenerate_loop(f"{sentence} {sentence} {sentence}")
        assert res.detected is True
        assert res.kind == "ngram_repetition"
        assert "20-word" in res.detail

    def test_detects_shorter_phrase_loop_forming_20_gram_repeats(self) -> None:
        phrase = "Wait let me verify the tool parameters before proceeding now."
        text = " ".join([phrase] * 4)
        res = detect_degenerate_loop(text)
        assert res.detected is True
        assert res.kind == "ngram_repetition"

    def test_detects_single_word_repetition_loop(self) -> None:
        res = detect_degenerate_loop(" ".join(["wait"] * 60))
        assert res.detected is True
        assert res.kind == "ngram_repetition"

    @pytest.mark.parametrize(
        "payload",
        [
            "a" * 120,
            " " * 120,
            "\n" * 120,
            "." * 120,
            "-" * 200,
            "=" * 200,
            "ab" * 40,
            "\n " * 40,
            "ok! " * 40,
        ],
    )
    def test_detects_degenerate_char_and_whitespace_runs(self, payload: str) -> None:
        res = detect_degenerate_loop(f"prefix {payload} suffix")
        assert res.detected is True
        assert res.kind == "char_run"

    def test_does_not_false_positive_on_short_single_char_or_rule_runs(self) -> None:
        assert detect_degenerate_loop("x" * 100).detected is False
        assert detect_degenerate_loop("-" * 150).detected is False
        assert detect_degenerate_loop("=" * 150).detected is False
        assert detect_degenerate_loop("_" * 150).detected is False

    def test_zero_false_positives_on_structured_json(self) -> None:
        records = [
            {
                "id": f"rec-{i:03d}",
                "customer": f"Customer {i}",
                "status": "active" if i % 2 == 0 else "pending",
                "amount": round(19.95 + i * 1.37, 2),
                "tags": [f"region-{i % 7}", f"tier-{i % 4}"],
                "metadata": {"source": "import", "sequence": i, "verified": True},
            }
            for i in range(40)
        ]
        pretty_json = json.dumps({"version": 1, "records": records}, indent=2)
        assert detect_degenerate_loop(pretty_json).detected is False

    def test_zero_false_positives_on_markdown_tables_and_rules(self) -> None:
        cols = 15
        header = "| " + " | ".join(f"Col{i}" for i in range(cols)) + " |"
        sep = "| " + " | ".join("---" for _ in range(cols)) + " |"
        row_na = "| " + " | ".join("N/A" for _ in range(cols)) + " |"
        rows = ["| " + " | ".join(f"v{r}_{c}" for c in range(cols)) + " |" for r in range(15)]
        md = "\n".join(
            [
                "-" * 120,
                header,
                sep,
                row_na,
                *rows,
                "=" * 120,
            ]
        )
        assert detect_degenerate_loop(md).detected is False

    def test_zero_false_positives_on_code_blocks_and_matrices(self) -> None:
        banner = "# " + "=" * 100
        code = "\n".join(
            [
                "```python",
                banner,
                "def compute_matrix(scale: int) -> list[list[int]]:",
                "    # " + "-" * 90,
                *[f"    row_{i} = [{i}, {i + 1}, {i + 2}, {i + 3}, {i + 4}]" for i in range(25)],
                "    return [row_0]",
                "```",
                "Matrix baseline: " + " ".join(["0"] * 25),
            ]
        )
        assert detect_degenerate_loop(code).detected is False

    def test_zero_false_positives_on_long_valid_reasoning(self) -> None:
        reasoning = " ".join(
            f"Step {i}: comparing invoice {1000 + i} against ledger entry {2000 + i} "
            f"for department D{i % 13} yields delta {(i * 17) % 97}."
            for i in range(300)
        )
        assert len(reasoning) > 20_000
        assert detect_degenerate_loop(reasoning).detected is False


# ---------------------------------------------------------------------------
# 2. Domain serialization & withheld score helpers
# ---------------------------------------------------------------------------


class TestDomainSerializationAndHelpers:
    def test_degenerate_loop_failure_kind_is_not_infrastructure(self) -> None:
        assert FailureKind.DEGENERATE_LOOP == "degenerate_loop"
        assert FailureKind.DEGENERATE_LOOP not in INFRASTRUCTURE_FAILURE_KINDS

    def test_rating_and_deployability_for_withheld_score(self) -> None:
        assert rating_for_score(None) == "Withheld"
        assert rating_for_score(None, safety_capped=True) == "Withheld"
        assert compute_deployability(None, 500.0) == (None, None, None)

    def test_scenario_result_answer_in_thinking_round_trip(self) -> None:
        clean = ScenarioResult("S1", ScenarioStatus.PASS, 2, "ok")
        assert "answer_in_thinking_turns" not in clean.to_dict()
        assert ScenarioResult.from_dict(clean.to_dict()).answer_in_thinking_turns == 0

        flagged = ScenarioResult(
            "S2",
            ScenarioStatus.FAIL,
            0,
            "empty",
            failure_kind=FailureKind.EMPTY_TURN,
            answer_in_thinking_turns=2,
        )
        d = flagged.to_dict()
        assert d["answer_in_thinking_turns"] == 2
        assert ScenarioResult.from_dict(d).answer_in_thinking_turns == 2

    def test_summary_to_dict_omits_zero_diagnostic_counts_and_includes_nonzero(self) -> None:
        clean = score_results(
            [ScenarioResult("S1", ScenarioStatus.PASS, 2, "ok")],
            [_failing_scenario("S1")],
        )
        d_clean = clean.to_dict()
        assert "answer_in_thinking_turns" not in d_clean
        assert "answer_in_thinking_scenarios" not in d_clean
        assert "degenerate_loop_scenarios" not in d_clean

        diag = score_results(
            [
                ScenarioResult(
                    "S1",
                    ScenarioStatus.FAIL,
                    0,
                    "loop",
                    failure_kind=FailureKind.DEGENERATE_LOOP,
                ),
                ScenarioResult(
                    "S2",
                    ScenarioStatus.FAIL,
                    0,
                    "empty",
                    failure_kind=FailureKind.EMPTY_TURN,
                    answer_in_thinking_turns=3,
                ),
            ],
            [_failing_scenario("S1"), _failing_scenario("S2")],
        )
        d_diag = diag.to_dict()
        assert d_diag["degenerate_loop_scenarios"] == 1
        assert d_diag["answer_in_thinking_turns"] == 3
        assert d_diag["answer_in_thinking_scenarios"] == 1


# ---------------------------------------------------------------------------
# 3. Orchestrator: degenerate loops, answer_in_thinking, and withheld scores
# ---------------------------------------------------------------------------


class TestOrchestratorLoopAndThinkingDiagnostics:
    @pytest.mark.asyncio
    async def test_length_truncated_reasoning_loop_classified_as_degenerate_loop(self) -> None:
        repeated_sentence = (
            "Let me reconsider whether the user asked for the weather in Berlin or "
            "the forecast in Munich before picking a tool now."
        )
        looping_reasoning = " ".join([repeated_sentence] * 4)
        adapter = _ScriptedAdapter(
            [ChatCompletionResult(content="", reasoning=looping_reasoning, finish_reason="length")]
        )
        result = await run_scenario(
            adapter,
            model="test-model",
            base_url="http://localhost:8000",
            api_key=None,
            scenario=_failing_scenario("LOOP-REASONING"),
            max_turns=4,
        )
        assert result.failure_kind == FailureKind.DEGENERATE_LOOP
        assert result.failure_kind != FailureKind.REASONING_TRUNCATED
        assert result.answer_in_thinking_turns == 0
        assert "degenerate_loop=" in result.raw_log
        assert result.note is not None and "degenerate repetition loop in reasoning" in result.note

    @pytest.mark.asyncio
    async def test_length_truncated_visible_content_loop_classified_as_degenerate_loop(
        self,
    ) -> None:
        adapter = _ScriptedAdapter(
            [ChatCompletionResult(content="ab" * 60, reasoning=None, finish_reason="length")]
        )
        result = await run_scenario(
            adapter,
            model="test-model",
            base_url="http://localhost:8000",
            api_key=None,
            scenario=_failing_scenario("LOOP-CONTENT"),
            max_turns=4,
        )
        assert result.failure_kind == FailureKind.DEGENERATE_LOOP
        assert "degenerate_loop=" in result.raw_log
        assert result.note is not None and "in content" in result.note

    @pytest.mark.asyncio
    async def test_cut_off_tool_args_with_char_run_classified_as_degenerate_loop(self) -> None:
        adapter = _ScriptedAdapter(
            [
                ChatCompletionResult(
                    content="",
                    tool_calls=[
                        ProviderToolCall(
                            id="c1",
                            name="get_weather",
                            arguments_str='{"city": "' + ("a" * 150),
                        )
                    ],
                    finish_reason="length",
                )
            ]
        )
        result = await run_scenario(
            adapter,
            model="test-model",
            base_url="http://localhost:8000",
            api_key=None,
            scenario=_failing_scenario("LOOP-ARGS"),
            max_turns=4,
        )
        assert result.failure_kind == FailureKind.DEGENERATE_LOOP
        assert "degenerate_loop=" in result.raw_log

    @pytest.mark.asyncio
    async def test_clean_stop_with_degenerate_loop_in_reasoning_is_not_answer_in_thinking(
        self,
    ) -> None:
        adapter = _ScriptedAdapter(
            [ChatCompletionResult(content="", reasoning="z" * 200, finish_reason="stop")]
        )
        result = await run_scenario(
            adapter,
            model="test-model",
            base_url="http://localhost:8000",
            api_key=None,
            scenario=_failing_scenario("LOOP-STOP"),
            max_turns=4,
        )
        assert result.failure_kind == FailureKind.DEGENERATE_LOOP
        assert result.answer_in_thinking_turns == 0

    @pytest.mark.asyncio
    async def test_answer_in_thinking_tracks_clean_reasoning_only_turns(self) -> None:
        adapter = _ScriptedAdapter(
            [
                ChatCompletionResult(
                    content="",
                    reasoning="The weather in Berlin is 18C and clear; I will report that.",
                    finish_reason="stop",
                ),
                ChatCompletionResult(
                    content="",
                    reasoning="Confirming my previous summary for the follow-up turn.",
                    finish_reason="stop",
                ),
            ]
        )
        result = await run_scenario(
            adapter,
            model="test-model",
            base_url="http://localhost:8000",
            api_key=None,
            scenario=_failing_scenario("AIT-01", follow_ups=["Anything else?"]),
            max_turns=4,
        )
        assert result.failure_kind == FailureKind.EMPTY_TURN
        assert result.answer_in_thinking_turns == 2
        assert result.note is not None
        assert "Answer emitted in reasoning block with empty visible content" in result.note
        assert "answer_in_thinking" in result.diagnostics
        assert "2 turns" in result.diagnostics["answer_in_thinking"]
        assert "answer_in_thinking=Turn 1" in result.raw_log
        assert "answer_in_thinking=Turn 2" in result.raw_log

    def test_score_results_withholds_score_when_zero_gradable_scenarios(self) -> None:
        summary = _withheld_summary(["S1", "S2"], weight_by_difficulty=True)
        assert summary.max_points == 0
        assert summary.total_points == 0
        assert summary.final_score is None
        assert summary.rating == "Withheld"
        assert summary.weighted_score is None
        assert summary.deployability is None
        assert summary.responsiveness is None
        assert summary.excluded_scenarios == ["S1", "S2"]
        assert summary.completion_rate == 0.0
        d = summary.to_dict()
        assert d["final_score"] is None
        assert d["rating"] == "Withheld"
        assert "weighted_score" not in d
        assert "deployability" not in d


# ---------------------------------------------------------------------------
# 4. Consumer null-safety & diagnostic surfacing
# ---------------------------------------------------------------------------


class TestConsumersWithWithheldScoresAndDiagnostics:
    def test_cli_display_renders_withheld_and_diagnostics(self) -> None:
        withheld = _withheld_summary(["TC-01"])
        buf = StringIO()
        console = Console(file=buf, width=200, no_color=True)
        _print_final_panel(console, "test-model", withheld, elapsed=12.3)
        out = buf.getvalue()
        assert "Withheld" in out
        assert "None / 100" not in out

        diag = score_results(
            [
                ScenarioResult(
                    "TC-01",
                    ScenarioStatus.FAIL,
                    0,
                    "looped",
                    failure_kind=FailureKind.DEGENERATE_LOOP,
                ),
                ScenarioResult(
                    "TC-02",
                    ScenarioStatus.FAIL,
                    0,
                    "empty",
                    failure_kind=FailureKind.EMPTY_TURN,
                    answer_in_thinking_turns=2,
                ),
            ],
            [_failing_scenario("TC-01"), _failing_scenario("TC-02")],
        )
        buf2 = StringIO()
        console2 = Console(file=buf2, width=200, no_color=True)
        _print_final_panel(console2, "test-model", diag, elapsed=5.0)
        out2 = buf2.getvalue()
        assert "degenerate_loop" in out2
        assert "raising max_tokens will not help" in out2
        assert "Answer in thinking: 2 turns across 1 scenario" in out2

    def test_aggregate_trials_handles_all_withheld_and_mixed_trials(self) -> None:
        w1 = _withheld_summary(["S1"])
        w2 = _withheld_summary(["S1"])
        agg_all = aggregate_trials([w1, w2])
        assert agg_all["trials"] == 2
        assert agg_all["final_score_mean"] is None
        assert agg_all["final_score_stddev"] is None
        assert agg_all["final_score_median"] is None
        assert agg_all["final_score_ci95"] is None
        assert agg_all["withheld_trials"] == 2

        scored = score_results(
            [ScenarioResult("S1", ScenarioStatus.PASS, 2, "ok")],
            [_failing_scenario("S1")],
        )
        agg_one = aggregate_trials([scored, w1])
        assert agg_one["final_score_mean"] == 100.0
        assert agg_one["final_score_stddev"] == 0.0
        assert agg_one["final_score_median"] == 100.0
        assert agg_one["final_score_ci95"] == (100.0, 100.0)
        assert agg_one["withheld_trials"] == 1

        scored_zero = score_results(
            [ScenarioResult("S1", ScenarioStatus.FAIL, 0, "fail")],
            [_failing_scenario("S1")],
        )
        agg_two = aggregate_trials([scored, w1, scored_zero])
        assert agg_two["final_score_mean"] == 50.0
        assert agg_two["final_score_stddev"] > 0
        assert agg_two["withheld_trials"] == 1

        # _print_trials_summary renders cleanly for all-withheld trials.
        buf = StringIO()
        console = Console(file=buf, width=200, no_color=True)
        _print_trials_summary(console, agg_all)
        rendered = buf.getvalue()
        assert "Withheld" in rendered
        assert "None" not in rendered

    def test_markdown_scenario_and_summary_reports_handle_withheld_and_diagnostics(
        self, tmp_path: Path
    ) -> None:
        reporter = MarkdownReporter(root=str(tmp_path))

        w1 = _withheld_summary(["TC-01"])
        w2 = _withheld_summary(["TC-01"])
        report_path = reporter.write_scenario_report("run-withheld", "test-model", w1)
        md = report_path.read_text(encoding="utf-8")
        assert "**Final Score**: **Withheld**" in md
        assert "**None**" not in md
        assert "**Rating**: Withheld" in md

        agg = aggregate_trials([w1, w2])
        summary_path = reporter.write_summary_report("run-withheld", "test-model", [w1, w2], agg)
        smd = summary_path.read_text(encoding="utf-8")
        assert "| **Final Score** | Withheld | Withheld | **Withheld** |" in smd
        assert "| **95% CI** | — |" in smd
        assert "None" not in smd

        diag = score_results(
            [
                ScenarioResult(
                    "TC-01",
                    ScenarioStatus.FAIL,
                    0,
                    "looped",
                    failure_kind=FailureKind.DEGENERATE_LOOP,
                ),
                ScenarioResult(
                    "TC-02",
                    ScenarioStatus.FAIL,
                    0,
                    "empty",
                    failure_kind=FailureKind.EMPTY_TURN,
                    answer_in_thinking_turns=1,
                ),
            ],
            [_failing_scenario("TC-01"), _failing_scenario("TC-02")],
        )
        diag_path = reporter.write_scenario_report("run-diag", "test-model", diag)
        dmd = diag_path.read_text(encoding="utf-8")
        assert "**Degenerate Loops**: 1 scenario" in dmd
        assert "**Answer in Thinking**: 1 turn across 1 scenario" in dmd
        assert "| degenerate_loop |" in dmd

    def test_history_and_compare_runs_render_withheld_cleanly(self) -> None:
        withheld_run = {
            "run_id": "20261125T100000Z_w1",
            "status": "completed",
            "model": "test-model",
            "created_at": "2026-11-25T10:00:00Z",
            "config": {"model": "test-model", "backend": "vllm", "scenario_count": 1},
            "metadata": {},
            "scores": {
                "final_score": None,
                "rating": "Withheld",
                "max_points": 0,
                "total_points": 0,
                "scenario_results": [
                    {
                        "scenario_id": "TC-01",
                        "status": "fail",
                        "points": 0,
                        "failure_kind": "timeout",
                    }
                ],
            },
        }
        scored_run = {
            "run_id": "20261125T110000Z_s1",
            "status": "completed",
            "model": "test-model",
            "created_at": "2026-11-25T11:00:00Z",
            "config": {"model": "test-model", "backend": "vllm", "scenario_count": 1},
            "metadata": {},
            "scores": {
                "final_score": 100,
                "rating": "★★★★★ Excellent",
                "max_points": 2,
                "total_points": 2,
                "scenario_results": [
                    {"scenario_id": "TC-01", "status": "pass", "points": 2},
                ],
            },
        }

        repo_mock = MagicMock()
        repo_mock.list.return_value = [withheld_run, scored_run]
        repo_mock.get.side_effect = lambda rid, **_: (
            withheld_run if rid == withheld_run["run_id"] else scored_run
        )

        with patch("tool_eval_bench.storage.db.RunRepository", return_value=repo_mock):
            buf = StringIO()
            print_history(Console(file=buf, width=200, no_color=True))
            hist_out = buf.getvalue()
            assert "Withheld" in hist_out
            assert "—" in hist_out
            assert "None" not in hist_out

            buf2 = StringIO()
            compare_runs(
                Console(file=buf2, width=200, no_color=True),
                withheld_run["run_id"],
                scored_run["run_id"],
            )
            cmp_out = buf2.getvalue()
            assert "Withheld → 100" in cmp_out
            assert "None" not in cmp_out

    def test_cli_leaderboard_and_export_handle_none_score_defensively(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        row = {
            "cohort_rank": 1,
            "model": "test-model",
            "run_id": "r1",
            "date": "2026-11-25",
            "final_score": None,
            "rating": "Withheld",
            "total_points": 0,
            "max_points": 0,
            "passes": 0,
            "partials": 0,
            "fails": 1,
            "completion_count": 1,
            "cat_scores": {},
            "scenario_count": 1,
            "backend": "vllm",
            "total_tokens": 0,
            "num_runs": 1,
            "cohort_label": "vllm/1",
            "cohort_fingerprint": "fp1",
        }
        with (
            patch("tool_eval_bench.application.run_queries.recent_runs", return_value=[{"id": 1}]),
            patch("tool_eval_bench.cli.leaderboard.extract_leaderboard_rows", return_value=[row]),
        ):
            buf = StringIO()
            print_leaderboard(Console(file=buf, width=200, no_color=True))
            lb_out = buf.getvalue()
            assert "None" not in lb_out
            assert "—" in lb_out

            export_runs(Console(file=StringIO(), no_color=True), fmt="csv")
            csv_out = capsys.readouterr().out
            assert "None" not in csv_out

            export_runs(Console(file=StringIO(), no_color=True), fmt="json")
            json_out = json.loads(capsys.readouterr().out)
            assert json_out[0]["final_score"] is None

    def test_card_data_and_svg_render_withheld_runs(self) -> None:
        def _row(final_score: int | None, minute: int, *, trials: int = 1) -> dict[str, Any]:
            return {
                "run_id": f"r-{minute}",
                "status": "completed",
                "run_type": "tool_eval",
                "model": "Qwen/Qwen3-8B",
                "session_id": "sess-12345678",
                "created_at": f"2026-11-25T12:{minute:02d}:00+00:00",
                "config": {
                    "model": "Qwen/Qwen3-8B",
                    "config_fingerprint": "fp1234567890",
                    "scenario_ids": ["TC-01"],
                },
                "metadata": {
                    "tool_version": "1.0.0",
                    "session_id": "sess-12345678",
                    "trials": trials,
                },
                "scores": {
                    "final_score": final_score,
                    "rating": "Withheld" if final_score is None else "★★★★★ Excellent",
                    "total_points": 0 if final_score is None else 2,
                    "max_points": 0 if final_score is None else 2,
                    "excluded_scenarios": ["TC-01"] if final_score is None else [],
                    "category_scores": []
                    if final_score is None
                    else [
                        {
                            "category": "A",
                            "label": "Tool Selection",
                            "percent": 100.0,
                            "pass_count": 1,
                        }
                    ],
                    "scenario_results": [
                        {
                            "scenario_id": "TC-01",
                            "status": "fail" if final_score is None else "pass",
                            "points": 0 if final_score is None else 2,
                        }
                    ],
                },
            }

        # Single withheld run
        card_single = build_card_data([_row(None, 1, trials=1)])
        assert card_single.tool_eval is not None
        assert math.isnan(card_single.tool_eval.score)
        assert card_single.tool_eval.rating == "Withheld"
        for layout in ("portrait", "landscape"):
            svg = render_card_svg(card_single, layout=layout)  # type: ignore[arg-type]
            assert "Withheld" in svg
            assert "nan" not in svg.lower()

        # All-withheld multi-trial
        card_all_w = build_card_data([_row(None, 1, trials=2), _row(None, 2, trials=2)])
        assert card_all_w.tool_eval is not None
        assert math.isnan(card_all_w.tool_eval.score)
        assert card_all_w.tool_eval.rating == "Withheld"

        # Mixed withheld + scored multi-trial averages only the scored trial
        card_mixed = build_card_data([_row(100, 1, trials=2), _row(None, 2, trials=2)])
        assert card_mixed.tool_eval is not None
        assert card_mixed.tool_eval.score == pytest.approx(100.0)
        assert card_mixed.tool_eval.rating == "★★★★★ Excellent"

    def test_api_and_web_helpers_preserve_none_cleanly(self) -> None:
        summary = _withheld_summary(["TC-01"])
        env = format_result({"run_id": "r1", "scores": summary.to_dict()})
        assert env["final_score"] is None
        assert env["rating"] == "Withheld"
        assert env["deployability"] is None
        assert env["weighted_score"] is None

        row = {
            "cohort_rank": 1,
            "model": "test-model",
            "run_id": "r1",
            "date": "2026-11-25",
            "final_score": None,
            "rating": "Withheld",
            "total_points": 0,
            "max_points": 0,
            "passes": 0,
            "partials": 0,
            "fails": 1,
            "scenario_count": 1,
            "backend": "vllm",
            "total_tokens": 0,
            "num_runs": 1,
            "cohort_label": "vllm/1",
            "cohort_fingerprint": "fp1",
            "cat_scores": {},
        }
        csv_body, _, _ = _format_export_payload([row], "csv")
        assert "None" not in csv_body
        json_body, _, _ = _format_export_payload([row], "json")
        assert json.loads(json_body)[0]["final_score"] is None

        run_a = {
            "run_id": "ra",
            "model": "m",
            "status": "completed",
            "config": {"model": "m", "backend": "vllm", "scenario_count": 1},
            "metadata": {},
            "scores": summary.to_dict(),
        }
        run_b = {
            "run_id": "rb",
            "model": "m",
            "status": "completed",
            "config": {"model": "m", "backend": "vllm", "scenario_count": 1},
            "metadata": {},
            "scores": {
                "final_score": 100,
                "rating": "★★★★★ Excellent",
                "total_points": 2,
                "max_points": 2,
                "scenario_results": [{"scenario_id": "TC-01", "status": "pass", "points": 2}],
            },
        }
        cmp_payload = _build_comparison_payload(run_a, run_b)
        assert cmp_payload["summary"]["score_a"] is None
        assert cmp_payload["summary"]["score_b"] == 100
        assert cmp_payload["summary"]["score_delta"] is None

    @pytest.mark.asyncio
    async def test_service_resume_merge_handles_withheld_final_score(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Resuming a run where every scenario timed out must not crash logger formatting."""
        sc1 = _failing_scenario("TC-01")
        sc2 = _failing_scenario("TC-02")
        prior_timeout = ScenarioResult(
            scenario_id="TC-01",
            status=ScenarioStatus.FAIL,
            points=0,
            summary="timed out",
            failure_kind=FailureKind.TIMEOUT,
        ).to_dict()

        async def _fake_run_all(*args: Any, **kwargs: Any) -> ModelScoreSummary:
            return score_results(
                [
                    ScenarioResult(
                        scenario_id="TC-02",
                        status=ScenarioStatus.FAIL,
                        points=0,
                        summary="timed out",
                        failure_kind=FailureKind.TIMEOUT,
                    )
                ],
                [sc2],
            )

        monkeypatch.setattr("tool_eval_bench.application.service.run_all_scenarios", _fake_run_all)
        service = BenchmarkService(repo=None, reporter=None)
        run_data = await service.run_benchmark(
            model="test-model",
            backend="vllm",
            base_url="http://localhost:8000",
            scenarios=[sc2],
            resume_run_id="resumed-withheld-run",
            resume_prior_results=[prior_timeout],
            resume_scenarios=[sc1, sc2],
            probe_engine=False,
        )
        assert run_data["scores"]["final_score"] is None
        assert run_data["scores"]["rating"] == "Withheld"
