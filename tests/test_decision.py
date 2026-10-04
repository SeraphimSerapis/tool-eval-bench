"""Decision models: wire format, scoring, calibration metrics, plugin run, CLI wiring."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from typing import Any

import httpx
import pytest

from tool_eval_bench.adapters.factory import build_adapter, build_decision_adapter
from tool_eval_bench.adapters.openai_compat import OpenAICompatibleAdapter
from tool_eval_bench.adapters.systemone import SystemOneAdapter
from tool_eval_bench.domain.decision import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionBackend,
    DecisionQuestion,
    DecisionRequestError,
    DecisionResult,
    DecisionUnsupportedError,
    ScoreAnswer,
    ScoreQuestion,
    YesNoAnswer,
    YesNoQuestion,
    parse_answers,
)
from tool_eval_bench.plugins.decision.dataset import (
    CATEGORY_ROUTING,
    QUESTION_NAME,
    VARIANT_BASE,
    VARIANT_OPAQUE,
    VARIANT_SHUFFLED,
    DecisionItem,
    build_items,
)
from tool_eval_bench.plugins.decision.evaluator import score_item
from tool_eval_bench.plugins.decision.metrics import (
    confusion_matrix,
    expected_calibration_error,
    percentile,
    reliability_bins,
    robustness,
)
from tool_eval_bench.plugins.decision.plugin import DecisionPlugin
from tool_eval_bench.plugins.decision.render import bar
from tool_eval_bench.plugins.registry import available_plugins, get_plugin
from tool_eval_bench.utils.urls import systemone_url

_CHOICE = ChoiceQuestion("Which team?", {"billing": "payments", "shipping": "delivery"})
_SCORE = ScoreQuestion("How urgent?", ("can wait", "this week", "today", "now"))
_YES_NO = YesNoQuestion("Is the customer angry?")


def _item(question: DecisionQuestion, gold: Any, **kw: Any) -> DecisionItem:
    return DecisionItem(id="x", category="c", state="s", question=question, gold=gold, **kw)


# ---------------------------------------------------------------------------
# Wire format
# ---------------------------------------------------------------------------


class TestWireFormat:
    def test_questions_serialise_to_the_documented_shape(self) -> None:
        assert _CHOICE.to_wire() == {
            "type": "choice",
            "instructions": "Which team?",
            "criteria": {"billing": "payments", "shipping": "delivery"},
        }
        assert _SCORE.to_wire()["criteria"] == ["can wait", "this week", "today", "now"]
        # The wire type for a yes/no question is "noul".
        assert _YES_NO.to_wire() == {
            "type": "noul",
            "instructions": "Is the customer angry?",
        }

    def test_parses_each_answer_type(self) -> None:
        payload = {
            "answers": {
                "a": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.9}},
                "b": {"score": 2.04, "probabilities": {"0": 0.1, "2": 0.9}, "confidence": 0.5},
                "c": {"noul": 0.7},
            }
        }
        answers = parse_answers(payload, {"a": _CHOICE, "b": _SCORE, "c": _YES_NO})
        assert answers["a"] == ChoiceAnswer("billing", {"billing": 0.9}, None)
        assert answers["b"] == ScoreAnswer(2.04, {"0": 0.1, "2": 0.9}, 0.5)
        assert answers["c"] == YesNoAnswer(0.7)

    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"answers": []},
            {"answers": {}},
            {"answers": {"a": {"choice": "billing"}}},
            {"answers": {"a": {"choice": "billing", "probabilities": None}}},
        ],
    )
    def test_a_missing_or_malformed_answer_is_an_error_not_a_zero(
        self, payload: dict[str, Any]
    ) -> None:
        with pytest.raises(ValueError):
            parse_answers(payload, {"a": _CHOICE})

    def test_systemone_url_accepts_both_base_forms(self) -> None:
        assert systemone_url("http://h:8080") == "http://h:8080/v1/systemone"
        assert systemone_url("http://h:8080/v1/") == "http://h:8080/v1/systemone"


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


def _adapter(handler: Any) -> SystemOneAdapter:
    adapter = SystemOneAdapter()
    adapter._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return adapter


def _ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "laya",
            "answers": {
                "q": {"type": "choice", "choice": "billing", "probabilities": {"billing": 1.0}}
            },
            "usage": {"input_tokens": 12, "output_tokens": 0},
        },
    )


class TestSystemOneAdapter:
    @pytest.mark.asyncio
    async def test_posts_state_and_questions_and_parses_the_answer(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["body"] = json.loads(request.content)
            seen["auth"] = request.headers.get("authorization")
            return _ok(request)

        result = await _adapter(handler).decide(
            model="laya",
            state="charged twice",
            questions={"q": _CHOICE},
            api_key="secret",
            base_url="http://host:8084",
        )

        assert seen["url"] == "http://host:8084/v1/systemone"
        assert seen["body"] == {
            "model": "laya",
            "state": "charged twice",
            "questions": {"q": _CHOICE.to_wire()},
        }
        assert seen["auth"] == "Bearer secret"
        assert result.input_tokens == 12
        assert result.elapsed_ms > 0
        assert result.answers["q"] == ChoiceAnswer("billing", {"billing": 1.0}, None)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [404, 405, 501])
    async def test_a_server_without_the_endpoint_is_unsupported(self, status: int) -> None:
        adapter = _adapter(lambda r: httpx.Response(status, text="nope"))
        with pytest.raises(DecisionUnsupportedError, match="does not serve decision models"):
            await adapter.decide(
                model="m", state="s", questions={"q": _CHOICE}, base_url="http://h"
            )

    @pytest.mark.asyncio
    async def test_a_rejected_request_names_the_server_message(self) -> None:
        adapter = _adapter(lambda r: httpx.Response(400, text='{"error":"instructions"}'))
        with pytest.raises(DecisionRequestError, match="instructions"):
            await adapter.decide(
                model="m", state="s", questions={"q": _CHOICE}, base_url="http://h"
            )

    @pytest.mark.asyncio
    async def test_a_non_object_body_is_rejected(self) -> None:
        adapter = _adapter(lambda r: httpx.Response(200, json=[1, 2]))
        with pytest.raises(ValueError, match="not a JSON object"):
            await adapter.decide(
                model="m", state="s", questions={"q": _CHOICE}, base_url="http://h"
            )

    def test_the_decision_adapter_is_also_a_chat_adapter(self) -> None:
        adapter = build_decision_adapter(wire_format="gemini")
        assert isinstance(adapter, DecisionBackend)
        assert isinstance(adapter, OpenAICompatibleAdapter)

    def test_the_chat_factory_does_not_return_a_decision_backend(self) -> None:
        assert not isinstance(build_adapter("http://h:1"), DecisionBackend)


# ---------------------------------------------------------------------------
# Scoring one answer
# ---------------------------------------------------------------------------


class TestScoreItem:
    def test_correct_choice_uses_the_probability_of_the_predicted_option(self) -> None:
        score = score_item(
            _item(_CHOICE, "billing"), ChoiceAnswer("billing", {"billing": 0.8, "shipping": 0.2})
        )
        assert score.correct
        assert score.confidence == 0.8 and score.p_gold == 0.8
        # (0.8-1)^2 + (0.2-0)^2
        assert score.brier == pytest.approx(0.08)
        assert score.log_loss == pytest.approx(-math.log(0.8))

    def test_confident_wrong_choice_is_penalised_hard(self) -> None:
        score = score_item(
            _item(_CHOICE, "billing"), ChoiceAnswer("shipping", {"billing": 0.1, "shipping": 0.9})
        )
        assert not score.correct
        assert score.confidence == 0.9 and score.p_gold == 0.1
        assert score.brier == pytest.approx(1.62)

    def test_zero_probability_for_gold_keeps_log_loss_finite(self) -> None:
        score = score_item(
            _item(_CHOICE, "billing"), ChoiceAnswer("shipping", {"billing": 0.0, "shipping": 1.0})
        )
        assert math.isfinite(score.log_loss)

    def test_an_option_outside_the_list_is_wrong_not_a_crash(self) -> None:
        score = score_item(_item(_CHOICE, "billing"), ChoiceAnswer("refunds", {"billing": 1.0}))
        assert not score.correct and score.confidence == 0.0

    def test_score_question_predicts_the_most_likely_level_and_reports_distance(self) -> None:
        answer = ScoreAnswer(2.04, {"0": 0.05, "1": 0.08, "2": 0.64, "3": 0.23})
        right = score_item(_item(_SCORE, 2), answer)
        wrong = score_item(_item(_SCORE, 3), answer)
        assert right.correct and right.predicted == "2"
        assert right.level_error == pytest.approx(0.04)
        assert not wrong.correct and wrong.level_error == pytest.approx(0.96)

    @pytest.mark.parametrize(
        ("p_yes", "gold", "correct", "confidence"),
        [(0.95, True, True, 0.95), (0.05, False, True, 0.95), (0.3, True, False, 0.7)],
    )
    def test_yes_no_is_a_two_way_distribution(
        self, p_yes: float, gold: bool, correct: bool, confidence: float
    ) -> None:
        score = score_item(_item(_YES_NO, gold), YesNoAnswer(p_yes))
        assert score.correct is correct
        assert score.confidence == pytest.approx(confidence)

    def test_exactly_one_half_counts_as_yes(self) -> None:
        assert score_item(_item(_YES_NO, True), YesNoAnswer(0.5)).correct

    def test_an_answer_of_the_wrong_type_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="does not answer"):
            score_item(_item(_CHOICE, "billing"), YesNoAnswer(0.9))


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_a_confidence_of_one_lands_in_the_last_bin(self) -> None:
        bins = reliability_bins([1.0, 0.0], [True, False], n_bins=10)
        assert [(b.low, b.count) for b in bins] == [(0.0, 1), (0.9, 1)]

    def test_empty_bins_are_omitted(self) -> None:
        assert len(reliability_bins([0.95, 0.96], [True, True])) == 1

    def test_length_mismatch_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            reliability_bins([0.5], [True, False])

    def test_perfectly_calibrated_bins_have_zero_ece(self) -> None:
        bins = reliability_bins([0.5, 0.5], [True, False])
        assert expected_calibration_error(bins) == pytest.approx(0.0)

    def test_overconfidence_shows_up_as_ece(self) -> None:
        # Always 95% sure, right half the time.
        bins = reliability_bins([0.95] * 4, [True, True, False, False])
        assert expected_calibration_error(bins) == pytest.approx(0.45)

    def test_ece_of_nothing_is_zero(self) -> None:
        assert expected_calibration_error([]) == 0.0

    def test_percentile_interpolates(self) -> None:
        assert percentile([10, 20, 30, 40], 50) == pytest.approx(25.0)
        assert percentile([7], 95) == 7
        assert percentile([], 50) == 0.0

    def test_confusion_keeps_a_hallucinated_label_visible(self) -> None:
        matrix = confusion_matrix([("a", "a"), ("a", "b"), ("b", "zzz")], labels=["a", "b"])
        assert matrix == {"a": {"a": 1, "b": 1}, "b": {"zzz": 1}}

    def test_invariance_is_measured_against_the_base_answer_not_the_gold(self) -> None:
        # Wrong the same way twice: invariant, but not accurate.
        base = {"r1": ("billing", False), "r2": ("shipping", True)}
        result = robustness(
            base,
            {
                "shuffled": [("r1", "billing", False), ("r2", "billing", False)],
                "opaque": [],
            },
        )
        assert result.shuffled_accuracy == 0.0
        assert result.shuffled_invariance == 50.0
        assert result.opaque_accuracy is None and result.opaque_invariance is None

    def test_variants_of_unanswered_base_items_are_ignored(self) -> None:
        result = robustness({}, {"shuffled": [("r1", "a", True)]})
        assert result.shuffled_accuracy is None


# ---------------------------------------------------------------------------
# Built-in items
# ---------------------------------------------------------------------------


class TestDataset:
    def test_ids_are_unique(self) -> None:
        ids = [i.id for i in build_items()]
        assert len(ids) == len(set(ids))

    def test_every_gold_is_a_valid_answer_for_its_question(self) -> None:
        for item in build_items():
            if isinstance(item.question, ChoiceQuestion):
                assert item.gold in item.question.options, item.id
            elif isinstance(item.question, ScoreQuestion):
                assert isinstance(item.gold, int) and not isinstance(item.gold, bool)
                assert 0 <= item.gold < len(item.question.levels), item.id
            else:
                assert isinstance(item.gold, bool), item.id

    def test_variants_are_optional_and_cover_routing_only(self) -> None:
        base = build_items(variants=False)
        everything = build_items()
        assert all(i.variant == VARIANT_BASE for i in base)
        extra = everything[len(base) :]
        assert {i.category for i in extra} == {CATEGORY_ROUTING}
        routing = sum(1 for i in base if i.category == CATEGORY_ROUTING)
        assert len(extra) == 2 * routing

    def test_the_shuffled_variant_reverses_the_options_and_keeps_the_gold(self) -> None:
        items = {i.id: i for i in build_items()}
        base, shuffled = items["route-01"], items["route-01~shuffled"]
        assert isinstance(base.question, ChoiceQuestion)
        assert isinstance(shuffled.question, ChoiceQuestion)
        assert list(shuffled.question.options) == list(reversed(list(base.question.options)))
        assert shuffled.gold == base.gold
        assert shuffled.variant == VARIANT_SHUFFLED and shuffled.base_id == "route-01"

    def test_the_opaque_variant_keeps_descriptions_and_maps_names_back(self) -> None:
        items = {i.id: i for i in build_items()}
        base, opaque = items["route-05"], items["route-05~opaque"]
        assert isinstance(base.question, ChoiceQuestion)
        assert isinstance(opaque.question, ChoiceQuestion)
        assert opaque.variant == VARIANT_OPAQUE
        assert set(opaque.question.options).isdisjoint(base.question.options)
        assert sorted(opaque.question.options.values()) == sorted(base.question.options.values())
        assert opaque.to_base[str(opaque.gold)] == base.gold
        # Labels are not alphabetical in presentation order, so neither "first
        # option" nor "alphabetical" is a shortcut to the answer.
        assert list(opaque.question.options) != sorted(opaque.question.options)


# ---------------------------------------------------------------------------
# Plugin run
# ---------------------------------------------------------------------------


class FakeDecisionBackend(DecisionBackend):
    """Answers every item from its gold label, or by a configurable defect."""

    def __init__(
        self,
        items: list[DecisionItem],
        *,
        confidence: float = 0.9,
        always_first: bool = False,
        fail_states: frozenset[str] = frozenset(),
        unsupported: bool = False,
    ) -> None:
        self._by_key = {(i.state, id(i.question)): i for i in items}
        self.confidence = confidence
        self.always_first = always_first
        self.fail_states = fail_states
        self.unsupported = unsupported
        self.calls = 0

    async def decide(
        self,
        *,
        model: str,
        state: str,
        questions: Mapping[str, DecisionQuestion],
        timeout_seconds: float = 60.0,
        api_key: str | None = None,
        base_url: str = "",
    ) -> DecisionResult:
        self.calls += 1
        if self.unsupported:
            raise DecisionUnsupportedError("no endpoint")
        if state in self.fail_states:
            raise TimeoutError("synthetic")
        question = questions[QUESTION_NAME]
        item = self._by_key[(state, id(question))]
        c = self.confidence
        answer: ChoiceAnswer | ScoreAnswer | YesNoAnswer
        if isinstance(question, ChoiceQuestion):
            names = list(question.options)
            pick = names[0] if self.always_first else str(item.gold)
            rest = (1 - c) / max(len(names) - 1, 1)
            answer = ChoiceAnswer(pick, {n: c if n == pick else rest for n in names})
        elif isinstance(question, ScoreQuestion):
            level = int(item.gold)
            rest = (1 - c) / (len(question.levels) - 1)
            answer = ScoreAnswer(
                float(level),
                {str(i): c if i == level else rest for i in range(len(question.levels))},
            )
        else:
            answer = YesNoAnswer(c if item.gold else 1 - c)
        return DecisionResult({QUESTION_NAME: answer}, input_tokens=10, elapsed_ms=5.0)


class NotADecisionAdapter:
    pass


class TestPluginRun:
    @pytest.mark.asyncio
    async def test_a_perfect_model_scores_everything(self) -> None:
        items = build_items()
        result = await DecisionPlugin().run(
            FakeDecisionBackend(items),  # type: ignore[arg-type]
            model="m",
            base_url="u",
            items=items,
            concurrency=3,
        )
        d = result.details
        base_total = len(build_items(variants=False))
        assert result.score == 100.0
        assert (d["correct"], d["total"], d["requests"]) == (base_total, base_total, len(items))
        assert d["errors"] == 0 and d["status"] == "completed"
        assert result.total_tokens == 10 * len(items)
        # 90% confident and always right: under-confident by exactly 0.1.
        assert d["calibration"]["ece"] == pytest.approx(0.1, abs=0.02)
        assert d["calibration"]["high_confidence_errors"] == 0
        assert d["ordinal"]["mae"] == 0.0 and d["ordinal"]["within_one"] == 100.0
        assert d["robustness"]["shuffled_invariance"] == 100.0
        assert d["robustness"]["opaque_accuracy"] == 100.0
        assert d["latency_ms"]["p50"] == 5.0

    @pytest.mark.asyncio
    async def test_headline_accuracy_ignores_variants(self) -> None:
        items = build_items()
        backend = FakeDecisionBackend(items)
        # Break only the variants: wrong answers there must not move the headline.
        variants = {i.state for i in items if i.variant != VARIANT_BASE}
        assert variants  # sanity
        result = await DecisionPlugin().run(
            backend,  # type: ignore[arg-type]
            model="m",
            base_url="u",
            items=[i for i in items if i.variant == VARIANT_BASE],
        )
        assert result.score == 100.0
        assert result.details["robustness"]["shuffled_accuracy"] is None

    @pytest.mark.asyncio
    async def test_position_bias_is_visible_as_low_invariance(self) -> None:
        items = build_items()
        result = await DecisionPlugin().run(
            FakeDecisionBackend(items, always_first=True),  # type: ignore[arg-type]
            model="m",
            base_url="u",
            items=items,
        )
        rob = result.details["robustness"]
        # The first option of the reversed list is never the base's first option.
        assert rob["shuffled_invariance"] == 0.0
        assert result.details["categories"]["routing"]["accuracy"] < 50

    @pytest.mark.asyncio
    async def test_confident_mistakes_are_counted(self) -> None:
        items = build_items(variants=False)
        result = await DecisionPlugin().run(
            FakeDecisionBackend(items, confidence=0.95, always_first=True),  # type: ignore[arg-type]
            model="m",
            base_url="u",
            items=items,
        )
        assert result.details["calibration"]["high_confidence_errors"] > 0

    @pytest.mark.asyncio
    async def test_a_failed_request_is_an_error_and_the_run_is_incomplete(self) -> None:
        items = build_items(variants=False)
        target = items[0]
        result = await DecisionPlugin().run(
            FakeDecisionBackend(items, fail_states=frozenset({target.state})),  # type: ignore[arg-type]
            model="m",
            base_url="u",
            items=items,
        )
        d = result.details
        assert d["errors"] == 1 and d["incomplete"] and d["status"] == "incomplete"
        assert d["correct"] == len(items) - 1
        assert d["error_kinds"] == {"TimeoutError": 1}
        failed = next(r for r in result.item_results if r["id"] == target.id)
        assert failed["is_error"] and not failed["correct"]

    @pytest.mark.asyncio
    async def test_an_unsupported_endpoint_aborts_instead_of_scoring_zero(self) -> None:
        items = build_items(variants=False)
        with pytest.raises(DecisionUnsupportedError):
            await DecisionPlugin().run(
                FakeDecisionBackend(items, unsupported=True),  # type: ignore[arg-type]
                model="m",
                base_url="u",
                items=items,
            )

    @pytest.mark.asyncio
    async def test_a_chat_only_adapter_is_refused_with_a_pointer(self) -> None:
        with pytest.raises(ValueError, match="build_decision_adapter"):
            await DecisionPlugin().run(NotADecisionAdapter(), model="m", base_url="u")  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_concurrency_must_be_positive(self) -> None:
        items = build_items(variants=False)
        with pytest.raises(ValueError, match="concurrency"):
            await DecisionPlugin().run(
                FakeDecisionBackend(items),  # type: ignore[arg-type]
                model="m",
                base_url="u",
                items=items,
                concurrency=0,
            )

    @pytest.mark.asyncio
    async def test_progress_reports_every_item(self) -> None:
        items = build_items(variants=False)
        seen: list[tuple[int, int, str]] = []

        async def on_progress(current: int, total: int, info: dict[str, Any]) -> None:
            seen.append((current, total, info["id"]))

        await DecisionPlugin().run(
            FakeDecisionBackend(items),  # type: ignore[arg-type]
            model="m",
            base_url="u",
            items=items,
            on_progress=on_progress,
        )
        assert [c for c, _, _ in seen] == list(range(1, len(items) + 1))
        assert {t for _, t, _ in seen} == {len(items)}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


class TestReport:
    def test_bar_is_fixed_width_and_clamped(self) -> None:
        assert bar(0.0) == "░" * 10
        assert bar(1.0) == "█" * 10
        assert bar(0.5) == "█" * 5 + "░" * 5
        assert bar(7.0) == "█" * 10 and bar(-1.0) == "░" * 10

    @pytest.mark.asyncio
    async def test_report_has_every_section_and_lists_confident_mistakes_first(self) -> None:
        items = build_items()
        plugin = DecisionPlugin()
        result = await plugin.run(
            FakeDecisionBackend(items, always_first=True),  # type: ignore[arg-type]
            model="m",
            base_url="u",
            items=items,
        )
        report = "\n".join(plugin.render_report_section(result))
        for heading in (
            "### Accuracy by Category",
            "### Calibration",
            "### Routing Confusion Matrix",
            "### Urgency Scale",
            "### Robustness",
            "### Latency",
            "### Mistakes",
            "### Full Trace",
        ):
            assert heading in report, heading
        assert "◀ gold" in report and "◀ pred" in report

    @pytest.mark.asyncio
    async def test_score_levels_are_traced_in_scale_order(self) -> None:
        items = build_items(variants=False)
        plugin = DecisionPlugin()
        result = await plugin.run(
            FakeDecisionBackend(items),  # type: ignore[arg-type]
            model="m",
            base_url="u",
            items=items,
        )
        trace = "\n".join(plugin.render_report_section(result))
        block = trace.split("urg-01")[1].split("urg-02")[0]
        levels = [line.split()[0] for line in block.splitlines() if line.startswith("    ")]
        assert levels == ["0", "1", "2", "3"]

    @pytest.mark.asyncio
    async def test_an_error_row_renders_without_a_distribution(self) -> None:
        items = build_items(variants=False)
        plugin = DecisionPlugin()
        result = await plugin.run(
            FakeDecisionBackend(items, fail_states=frozenset({items[0].state})),  # type: ignore[arg-type]
            model="m",
            base_url="u",
            items=items,
        )
        report = "\n".join(plugin.render_report_section(result))
        assert f"{items[0].id}  ERROR" in report

    @pytest.mark.asyncio
    async def test_message_text_cannot_break_the_mistakes_table(self) -> None:
        # always_first answers "billing", so a gold of "shipping" forces a mistake row.
        item = DecisionItem(
            id="pipe",
            category="c",
            state="a | b\nc",
            question=_CHOICE,
            gold="shipping",
        )
        plugin = DecisionPlugin()
        result = await plugin.run(
            FakeDecisionBackend([item], always_first=True),  # type: ignore[arg-type]
            model="m",
            base_url="u",
            items=[item],
        )
        report = "\n".join(plugin.render_report_section(result))
        assert "| pipe | a \\| b c |" in report


# ---------------------------------------------------------------------------
# Registration and CLI
# ---------------------------------------------------------------------------


def _parse(argv: list[str]) -> Any:
    from tool_eval_bench.cli.legacy_parser import make_parser
    from tool_eval_bench.cli.parser import parse_cli_args

    _, args = parse_cli_args(make_parser, argv)
    return args


class TestRegistration:
    def test_is_a_registered_plugin(self) -> None:
        assert "decision" in available_plugins()
        assert get_plugin("decision").name == "decision"

    def test_flat_flags_default_off(self) -> None:
        args = _parse(["--short"])
        assert args.decision is False and args.decision_only is False

    def test_plugin_subcommand_selects_decision_only(self) -> None:
        args = _parse(["plugin", "decision"])
        assert args.decision_only is True

    @pytest.mark.parametrize("flag", ["--shots", "--limit"])
    def test_flags_the_plugin_does_not_have_are_rejected(self, flag: str) -> None:
        with pytest.raises(SystemExit):
            _parse(["plugin", "decision", flag, "3"])

    def test_decision_only_stops_before_tool_scenarios_and_runs_after_needle(self) -> None:
        from rich.console import Console

        from tool_eval_bench.cli.plugin_runners import run_selected_plugins

        calls: list[str] = []

        def runner(name: str) -> Any:
            return lambda *_a, **_k: calls.append(name)

        stopped = run_selected_plugins(
            Console(),
            "m",
            "d",
            "u",
            None,
            _parse(["--needle", "--decision-only"]),
            runners={n: runner(n) for n in ("gsm8k", "mmlu", "ifeval", "needle", "decision")},
            extra_params=None,
            output_dir=None,
            run_context=None,
        )
        assert calls == ["needle", "decision"] and stopped is True


class TestDecisionRunner:
    def test_run_prints_the_tables_persists_and_reports(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        from rich.console import Console

        import tool_eval_bench.adapters.factory as factory
        from tool_eval_bench.cli import plugin_runners

        items = build_items()
        monkeypatch.setattr(
            factory,
            "build_decision_adapter",
            lambda **_kw: FakeDecisionBackend(items, confidence=0.8),
        )
        monkeypatch.setattr(plugin_runners, "_with_config_fingerprint", lambda value: value)
        monkeypatch.setattr(plugin_runners, "_metadata_for_storage", lambda value: {})
        persisted: list[dict[str, Any]] = []
        monkeypatch.setattr(plugin_runners, "_persist_plugin_run", persisted.append)

        args = argparse.Namespace(parallel=2, timeout=5.0, format=None)
        console = Console(record=True, width=160)
        plugin_runners._run_decision_benchmark(
            console, "m", "Display", "http://h", None, args, output_dir=str(tmp_path)
        )

        output = console.export_text()
        assert "Decision Accuracy" in output
        assert "Accuracy by category" in output
        assert "Reliability" in output
        assert "Routing confusion" in output
        assert [p["run_type"] for p in persisted] == ["decision"]
        assert list(tmp_path.rglob("*.md"))

    def test_an_unsupported_endpoint_exits_with_a_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from rich.console import Console

        import tool_eval_bench.adapters.factory as factory
        from tool_eval_bench.cli import plugin_runners

        monkeypatch.setattr(
            factory,
            "build_decision_adapter",
            lambda **_kw: FakeDecisionBackend(build_items(), unsupported=True),
        )
        console = Console(record=True, width=160)
        with pytest.raises(SystemExit):
            plugin_runners._run_decision_benchmark(
                console,
                "m",
                "Display",
                "http://h",
                None,
                argparse.Namespace(parallel=1, timeout=5.0, format=None),
            )
        assert "no endpoint" in console.export_text()
