"""Decision models: wire format, adapter, calibration metrics, registration."""

from __future__ import annotations

import json
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
    DecisionRequestError,
    DecisionUnsupportedError,
    ScoreAnswer,
    ScoreQuestion,
    YesNoAnswer,
    YesNoQuestion,
    parse_answers,
)
from tool_eval_bench.plugins.decision.metrics import (
    expected_calibration_error,
    percentile,
    reliability_bins,
)
from tool_eval_bench.plugins.decision.render import bar
from tool_eval_bench.plugins.registry import available_plugins, get_plugin
from tool_eval_bench.utils.urls import systemone_url

_CHOICE = ChoiceQuestion("Which team?", {"billing": "payments", "shipping": "delivery"})
_SCORE = ScoreQuestion("How urgent?", ("can wait", "this week", "today", "now"))
_YES_NO = YesNoQuestion("Is the customer angry?")


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


class TestBar:
    def test_bar_is_fixed_width_and_clamped(self) -> None:
        assert bar(0.0) == "░" * 10
        assert bar(1.0) == "█" * 10
        assert bar(0.5) == "█" * 5 + "░" * 5
        assert bar(7.0) == "█" * 10 and bar(-1.0) == "░" * 10


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
        assert args.decision_bench is False and args.decision_bench_only is False

    def test_plugin_subcommand_selects_decision_bench_only(self) -> None:
        args = _parse(["plugin", "decision"])
        assert args.decision_bench_only is True

    @pytest.mark.parametrize("flag", ["--shots", "--limit"])
    def test_flags_the_plugin_does_not_have_are_rejected(self, flag: str) -> None:
        with pytest.raises(SystemExit):
            _parse(["plugin", "decision", flag, "3"])

    def test_decision_bench_only_stops_before_tool_scenarios_and_runs_after_needle(self) -> None:
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
            _parse(["--needle", "--decision-bench-only"]),
            runners={n: runner(n) for n in ("gsm8k", "mmlu", "ifeval", "needle", "decision")},
            extra_params=None,
            output_dir=None,
            run_context=None,
        )
        assert calls == ["needle", "decision"] and stopped is True
