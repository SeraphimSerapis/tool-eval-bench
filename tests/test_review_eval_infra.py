"""Regression tests from the evaluation infrastructure review."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tool_eval_bench.domain.adapters import BackendAdapter, ChatCompletionResult, ProviderToolCall
from tool_eval_bench.domain.scenarios import (
    Category,
    FailureKind,
    ScenarioDefinition,
    ScenarioEvaluation,
    ScenarioState,
    ScenarioStatus,
    ToolCallRecord,
    ToolResultRecord,
)
from tool_eval_bench.evals.helpers import (
    answer_affirms_number,
    fail_eval,
    forbid_unrequested_side_effects,
    parse_math_expression,
    pass_eval,
    positive_argument_contains,
)
from tool_eval_bench.evals.locales import normalize_answer
from tool_eval_bench.evals.noise import (
    enrich_contacts,
    enrich_file_search,
    enrich_search,
    enrich_stock,
    enrich_weather,
)
from tool_eval_bench.evals.packs import load_scenario_pack, pack_content_hash
from tool_eval_bench.evals.variants import apply_variants
from tool_eval_bench.evals.yaml_loader import _load_yaml_file, load_yaml_scenarios_with_bytes
from tool_eval_bench.runner import orchestrator
from tool_eval_bench.runner.orchestrator import _INJECTED_ERRORS, run_scenario, score_results


class _ScriptedAdapter(BackendAdapter):
    def __init__(self, responses: list[ChatCompletionResult]) -> None:
        self._responses = list(responses)
        self._index = 0
        self.captured: list[dict[str, Any]] = []

    async def chat_completion(self, **kwargs: Any) -> ChatCompletionResult:
        self.captured.append(dict(kwargs))
        if self._index >= len(self._responses):
            return ChatCompletionResult(content="done", finish_reason="stop")
        resp = self._responses[self._index]
        self._index += 1
        return resp


def _tool_call(name: str, args: dict[str, Any], call_id: str = "c1") -> ProviderToolCall:
    return ProviderToolCall(id=call_id, name=name, arguments_str=json.dumps(args))


def _base_scenario(**overrides: Any) -> ScenarioDefinition:
    defaults: dict[str, Any] = {
        "id": "INFRA-01",
        "title": "Infra test scenario",
        "category": Category.A,
        "user_message": "Run a lookup.",
        "description": "Test scenario",
        "handle_tool_call": lambda state, call: {"ok": True},
        "evaluate": lambda state: pass_eval("passed"),
    }
    defaults.update(overrides)
    return ScenarioDefinition(**defaults)


# ---------------------------------------------------------------------------
# 1. Orchestrator fixes
# ---------------------------------------------------------------------------


class TestOrchestratorTransportClassification:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
    async def test_post_tool_server_status_is_infrastructure_failure(self, status: int) -> None:
        adapter = _ScriptedAdapter(
            [
                ChatCompletionResult(content="", tool_calls=[_tool_call("get_weather", {})]),
                ChatCompletionResult(
                    content=f"[server error {status}] backend overloaded",
                    transport_error_status=status,
                ),
            ]
        )
        scenario = _base_scenario()

        result = await run_scenario(
            adapter,
            model="m",
            base_url="http://localhost:8000",
            api_key=None,
            scenario=scenario,
        )

        assert result.status is ScenarioStatus.FAIL
        assert result.failure_kind == FailureKind.SERVER_ERROR
        assert result.is_infrastructure_failure is True
        summary = score_results([result], [scenario])
        assert summary.excluded_scenarios == [scenario.id]

    @pytest.mark.asyncio
    async def test_post_tool_400_without_infra_flag_stays_model_crash(self) -> None:
        adapter = _ScriptedAdapter(
            [
                ChatCompletionResult(content="", tool_calls=[_tool_call("get_weather", {})]),
                ChatCompletionResult(
                    content="[server error 400] invalid arguments",
                    transport_error_status=400,
                ),
            ]
        )
        scenario = _base_scenario()

        result = await run_scenario(
            adapter,
            model="m",
            base_url="http://localhost:8000",
            api_key=None,
            scenario=scenario,
        )

        assert result.status is ScenarioStatus.FAIL
        assert result.failure_kind == FailureKind.MODEL_CRASH
        assert result.is_infrastructure_failure is False


class TestOrchestratorErrorInjectionInteractions:
    @pytest.mark.asyncio
    async def test_tool_choice_after_first_call_waits_for_non_injected_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        draws = iter([dict(_INJECTED_ERRORS[0]), None])
        monkeypatch.setattr(
            orchestrator, "_draw_injected_error", lambda rate, rng=None: next(draws)
        )

        scenario = _base_scenario(
            tool_choice_override="required",
            tool_choice_after_first_call="none",
        )
        adapter = _ScriptedAdapter(
            [
                ChatCompletionResult(content="", tool_calls=[_tool_call("calculator", {}, "c1")]),
                ChatCompletionResult(content="", tool_calls=[_tool_call("calculator", {}, "c2")]),
                ChatCompletionResult(content="42", finish_reason="stop"),
            ]
        )

        result = await run_scenario(
            adapter,
            model="m",
            base_url="http://localhost:8000",
            api_key=None,
            scenario=scenario,
            error_rate=0.5,
        )

        assert result.status is ScenarioStatus.PASS
        assert [p["tool_choice"] for p in adapter.captured] == ["required", "required", "none"]

    @pytest.mark.asyncio
    async def test_injected_retries_do_not_trigger_loop_guard(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        draws = iter(
            [
                dict(_INJECTED_ERRORS[0]),
                dict(_INJECTED_ERRORS[0]),
                dict(_INJECTED_ERRORS[0]),
                None,
            ]
        )
        monkeypatch.setattr(
            orchestrator, "_draw_injected_error", lambda rate, rng=None: next(draws)
        )

        scenario = _base_scenario()
        call = _tool_call("get_weather", {"location": "Berlin"})
        adapter = _ScriptedAdapter(
            [
                ChatCompletionResult(content="", tool_calls=[call]),
                ChatCompletionResult(content="", tool_calls=[call]),
                ChatCompletionResult(content="", tool_calls=[call]),
                ChatCompletionResult(content="", tool_calls=[call]),
                ChatCompletionResult(content="Berlin is sunny.", finish_reason="stop"),
            ]
        )

        result = await run_scenario(
            adapter,
            model="m",
            base_url="http://localhost:8000",
            api_key=None,
            scenario=scenario,
            error_rate=0.5,
            max_turns=8,
        )

        assert result.status is ScenarioStatus.PASS
        assert result.failure_kind is None
        assert "loop_guard=" not in result.raw_log

    @pytest.mark.asyncio
    async def test_real_repeated_errors_still_trigger_loop_guard(self) -> None:
        scenario = _base_scenario(
            handle_tool_call=lambda state, call: {"error": "permanent failure"},
            evaluate=lambda state: fail_eval("never recovered"),
        )
        call = _tool_call("get_weather", {"location": "Berlin"})
        adapter = _ScriptedAdapter(
            [ChatCompletionResult(content="", tool_calls=[call]) for _ in range(5)]
        )

        result = await run_scenario(
            adapter,
            model="m",
            base_url="http://localhost:8000",
            api_key=None,
            scenario=scenario,
            max_turns=8,
        )

        assert result.failure_kind == FailureKind.REPEATED_CALL_LOOP
        assert result.turn_count == 3


class TestOrchestratorDependencyNotes:
    @pytest.mark.asyncio
    async def test_dependency_violation_preserves_evaluator_and_harness_notes(self) -> None:
        scenario = _base_scenario(
            dependencies=(("get_contacts", "send_email"),),
            evaluate=lambda state: pass_eval("looks fine", note="evaluator-note"),
        )
        adapter = _ScriptedAdapter(
            [
                ChatCompletionResult(
                    content="",
                    tool_calls=[
                        _tool_call("get_contacts", {"query": "Alex"}, "c1"),
                        _tool_call("send_email", {"to": "alex@company.com"}, "c2"),
                    ],
                ),
                ChatCompletionResult(content="", finish_reason="stop"),
            ]
        )

        result = await run_scenario(
            adapter,
            model="m",
            base_url="http://localhost:8000",
            api_key=None,
            scenario=scenario,
        )

        assert result.status is ScenarioStatus.FAIL
        assert result.safety_violation is not None
        assert result.note is not None
        assert "evaluator-note" in result.note
        assert "ended with no visible answer" in result.note

    @pytest.mark.asyncio
    async def test_ordered_dependency_calls_still_pass(self) -> None:
        scenario = _base_scenario(
            dependencies=(("get_contacts", "send_email"),),
            evaluate=lambda state: pass_eval("ordered", note="ok-note"),
        )
        adapter = _ScriptedAdapter(
            [
                ChatCompletionResult(
                    content="", tool_calls=[_tool_call("get_contacts", {"query": "Alex"}, "c1")]
                ),
                ChatCompletionResult(
                    content="",
                    tool_calls=[_tool_call("send_email", {"to": "alex@company.com"}, "c2")],
                ),
                ChatCompletionResult(content="Sent.", finish_reason="stop"),
            ]
        )

        result = await run_scenario(
            adapter,
            model="m",
            base_url="http://localhost:8000",
            api_key=None,
            scenario=scenario,
        )

        assert result.status is ScenarioStatus.PASS
        assert result.note == "ok-note"


# ---------------------------------------------------------------------------
# 2. Eval helpers fixes
# ---------------------------------------------------------------------------


class TestEvalHelperFixes:
    @pytest.mark.parametrize(
        "answer",
        [
            "No value was 30.",
            "None of them were 30.",
            "Nothing was 30.",
            "Nothing was 20 or 30.",
        ],
    )
    def test_answer_affirms_number_rejects_negative_subjects(self, answer: str) -> None:
        assert not answer_affirms_number(answer, "30")

    @pytest.mark.parametrize(
        ("answer", "value"),
        [
            ("With no discounts on the invoice, the total is $4,400,000.", "4400000"),
            ("No conflicts were found; the total is 30.", "30"),
            ("Nothing else changed: the final price is 187.42.", "187.42"),
        ],
    )
    def test_answer_affirms_number_accepts_positive_claims_after_unrelated_negation(
        self, answer: str, value: str
    ) -> None:
        assert answer_affirms_number(answer, value)

    def test_positive_argument_contains_handles_nfd_and_cjk_negation(self) -> None:
        assert positive_argument_contains("Mu\u0308nchen, DE", "München")
        assert positive_argument_contains("Berlin会议", "berlin")
        assert not positive_argument_contains("not Berlin会议", "berlin")
        assert not positive_argument_contains("Berlin", "")
        assert not positive_argument_contains("Berlin", "   ")

    def test_parse_math_expression_rejects_non_finite_overflow(self) -> None:
        huge = "1" + "0" * 200
        assert parse_math_expression(f"{huge} * {huge}") is None
        assert parse_math_expression("15 * 200 / 100") == 30.0

    def test_forbid_unrequested_side_effects_tags_forbidden_action(self) -> None:
        @forbid_unrequested_side_effects({"send_email": 1})
        def evaluator(state: ScenarioState) -> ScenarioEvaluation:
            return pass_eval("sent")

        bad_state = ScenarioState(
            tool_calls=[
                ToolCallRecord("c1", "send_email", "{}", {}, 1),
                ToolCallRecord("c2", "set_reminder", "{}", {}, 2),
            ],
            tool_results=[
                ToolResultRecord("c1", "send_email", {"status": "sent"}),
                ToolResultRecord("c2", "set_reminder", {"status": "set"}),
            ],
        )
        demoted = evaluator(bad_state)
        assert demoted.status is ScenarioStatus.FAIL
        assert demoted.failure_kind == FailureKind.FORBIDDEN_ACTION

        clean_state = ScenarioState(
            tool_calls=[ToolCallRecord("c1", "send_email", "{}", {}, 1)],
            tool_results=[ToolResultRecord("c1", "send_email", {"status": "sent"})],
        )
        assert evaluator(clean_state).status is ScenarioStatus.PASS


# ---------------------------------------------------------------------------
# 3. Noise enrichment fixes
# ---------------------------------------------------------------------------


class TestNoiseEnrichmentFixes:
    def test_enrich_contacts_keeps_payload_key_order_first(self) -> None:
        row = {"name": "Alice", "email": "alice@company.com"}
        enriched = enrich_contacts({"results": [row]})
        first = enriched["results"][0]
        assert list(first)[:2] == ["name", "email"]
        assert first["title"] == "Team Member"

        with_role = {"name": "Bob", "role": "manager"}
        enriched_role = enrich_contacts({"results": [with_role]})
        assert list(enriched_role["results"][0])[:2] == ["name", "role"]
        assert "title" not in enriched_role["results"][0]

    def test_enrichers_do_not_crash_on_non_numeric_or_non_dict_payload_values(self) -> None:
        wx = enrich_weather({"temperature": "22C", "condition": "sunny"})
        assert wx["temperature"] == "22C"
        assert wx["feels_like"] == -2
        assert enrich_weather({"temperature": None})["dew_point"] == -5

        st = enrich_stock({"price": None, "change": "+2.5%"})
        assert st["price"] is None
        assert st["previous_close"] == -1.23
        assert enrich_stock({"price": True, "change": "nan"})["previous_close"] == -1.23
        assert enrich_stock({"price": 100.0, "change": 2.5})["previous_close"] == 97.5

        assert enrich_search({"results": ["plain-item", {"snippet": "ok"}]})["results"][0] == (
            "plain-item"
        )
        assert enrich_file_search({"results": None})["results"] == []
        assert enrich_contacts({"results": ["raw-entry"]})["results"] == ["raw-entry"]


# ---------------------------------------------------------------------------
# 4. Variant wrapper fixes
# ---------------------------------------------------------------------------


class TestVariantWrapperFixes:
    def test_identity_variant_inverts_checkpoint_call_and_substitutes_diagnostic(self) -> None:
        observed_calls: list[dict[str, Any]] = []

        def checkpoint(state: ScenarioState, call: ToolCallRecord) -> str | None:
            observed_calls.append(dict(call.arguments))
            state.meta["checked_target"] = call.arguments.get("target")
            if call.arguments.get("target") == "AlphaProject":
                return "Observed mutation on AlphaProject"
            return None

        base = _base_scenario(
            id="VAR-CHK",
            user_message="Update AlphaProject.",
            variant_literals=("AlphaProject",),
            checkpoint=checkpoint,
        )
        (variant,) = apply_variants([base], seed=3)
        assert "AlphaProject" not in variant.user_message
        fixture_token = variant.user_message.removeprefix("Update ").removesuffix(".")

        state = ScenarioState()
        call = ToolCallRecord(
            id="c1",
            name="update_item",
            raw_arguments=json.dumps({"target": fixture_token}),
            arguments={"target": fixture_token},
            turn=1,
        )
        state.tool_calls.append(call)

        assert variant.checkpoint is not None
        diagnostic = variant.checkpoint(state, call)

        assert observed_calls == [{"target": "AlphaProject"}]
        assert state.meta["checked_target"] == fixture_token
        assert diagnostic == f"Observed mutation on {fixture_token}"

        other_call = ToolCallRecord(
            id="c2",
            name="update_item",
            raw_arguments='{"target": "Other"}',
            arguments={"target": "Other"},
            turn=2,
        )
        state.tool_calls.append(other_call)
        assert variant.checkpoint(state, other_call) is None

    def test_identity_variant_handle_distinguishes_calls_with_empty_ids(self) -> None:
        seen_targets: list[str] = []

        def handler(state: ScenarioState, call: ToolCallRecord) -> dict[str, Any]:
            seen_targets.append(str(call.arguments.get("target")))
            return {"target": call.arguments.get("target")}

        base = _base_scenario(
            id="VAR-ID",
            user_message="Inspect AlphaProject and BetaProject.",
            variant_literals=("AlphaProject", "BetaProject"),
            handle_tool_call=handler,
        )
        (variant,) = apply_variants([base], seed=1)
        tokens = variant.user_message.removeprefix("Inspect ").removesuffix(".").split(" and ")
        alpha_fixture, beta_fixture = tokens

        state = ScenarioState()
        first = ToolCallRecord("", "lookup", "{}", {"target": alpha_fixture}, 1)
        second = ToolCallRecord("", "lookup", "{}", {"target": beta_fixture}, 2)
        state.tool_calls.extend([first, second])

        res1 = variant.handle_tool_call(state, first)
        res2 = variant.handle_tool_call(state, second)

        assert seen_targets == ["AlphaProject", "BetaProject"]
        assert res1 == {"target": alpha_fixture}
        assert res2 == {"target": beta_fixture}


# ---------------------------------------------------------------------------
# 5. Locale answer normalization fixes
# ---------------------------------------------------------------------------


class TestLocaleNormalizationFixes:
    @pytest.mark.parametrize(
        ("locale", "answer", "expected_value"),
        [
            ("zh", "结果是－３０。", "-30"),
            ("es", "La temperatura es −3,5 °C.", "-3.5"),
            ("ja", "合計は１，０２０ドルです。", "1020"),
            ("zh", "总计４，２５０，０００美元。", "4250000"),
            ("de", "Der Kurs ist 187,42 USD.", "187.42"),
        ],
    )
    def test_fullwidth_minus_and_group_commas_affirm_expected_numbers(
        self, locale: str, answer: str, expected_value: str
    ) -> None:
        assert answer_affirms_number(normalize_answer(answer, locale), expected_value)

    def test_fullwidth_clause_comma_still_separates_negation(self) -> None:
        normalized = normalize_answer("答案不是30，而是25。", "zh")
        assert not answer_affirms_number(normalized, "30")
        assert answer_affirms_number(normalized, "25")

    def test_trailing_dollar_does_not_swallow_sentence_period(self) -> None:
        normalized = normalize_answer("Der Kurs ist 187.42. USD ist stark.", "de")
        assert normalized == "Der Kurs ist 187.42. USD ist stark."


# ---------------------------------------------------------------------------
# 6. YAML loader and pack fixes
# ---------------------------------------------------------------------------


class TestYamlLoaderAndPackFixes:
    def _write_weather_yaml(self, path: Path, *, with_answer_contains: bool = False) -> Path:
        body = (
            "id: HOLD-901\n"
            "title: Weather check\n"
            "category: A\n"
            "difficulty: 1\n"
            "user_message: What is the weather in Berlin?\n"
            "expected_tool_calls:\n"
            "  - tool: get_weather\n"
            "    arguments:\n"
            "      location: Berlin\n"
            "tool_responses:\n"
            "  get_weather:\n"
            "    - match:\n"
            "        location: Berlin\n"
            "      response:\n"
            "        nested:\n"
            "          temp: 18\n"
        )
        if with_answer_contains:
            body += 'answer_contains:\n  - "18"\n'
        path.write_text(body, encoding="utf-8")
        return path

    @pytest.mark.parametrize("with_answer_contains", [False, True])
    def test_unretried_injected_error_fails_yaml_scenario(
        self, tmp_path: Path, with_answer_contains: bool
    ) -> None:
        sc = _load_yaml_file(
            self._write_weather_yaml(
                tmp_path / "weather.yaml", with_answer_contains=with_answer_contains
            )
        )
        state = ScenarioState(
            tool_calls=[
                ToolCallRecord(
                    id="c1",
                    name="get_weather",
                    raw_arguments='{"location": "Berlin"}',
                    arguments={"location": "Berlin"},
                    turn=1,
                    injected=True,
                )
            ],
            tool_results=[
                ToolResultRecord("c1", "get_weather", dict(_INJECTED_ERRORS[0]), injected=True)
            ],
            final_answer="Berlin is 18 degrees.",
        )

        evaluation = sc.evaluate(state)
        assert evaluation.status is ScenarioStatus.FAIL
        assert "injected error" in evaluation.summary

    def test_retried_injected_error_still_passes_yaml_scenario(self, tmp_path: Path) -> None:
        sc = _load_yaml_file(
            self._write_weather_yaml(tmp_path / "weather.yaml", with_answer_contains=True)
        )
        state = ScenarioState(
            tool_calls=[
                ToolCallRecord("c1", "get_weather", "{}", {"location": "Berlin"}, 1, injected=True),
                ToolCallRecord(
                    "c2", "get_weather", "{}", {"location": "Berlin"}, 2, injected=False
                ),
            ],
            tool_results=[
                ToolResultRecord("c1", "get_weather", dict(_INJECTED_ERRORS[0]), injected=True),
                ToolResultRecord("c2", "get_weather", {"nested": {"temp": 18}}, injected=False),
            ],
            final_answer="Berlin is 18 degrees.",
        )

        assert sc.evaluate(state).status is ScenarioStatus.PASS

    def test_yaml_handler_deep_copies_response_payloads(self, tmp_path: Path) -> None:
        sc = _load_yaml_file(self._write_weather_yaml(tmp_path / "weather.yaml"))
        call = ToolCallRecord("c1", "get_weather", "{}", {"location": "Berlin"}, 1)

        first = sc.handle_tool_call(ScenarioState(), call)
        first["nested"]["temp"] = 999
        second = sc.handle_tool_call(ScenarioState(), call)

        assert second["nested"]["temp"] == 18

    def test_pack_loader_and_hash_ignore_yaml_subdirectories(self, tmp_path: Path) -> None:
        self._write_weather_yaml(tmp_path / "a.yaml")
        (tmp_path / "ignored.yaml").mkdir()

        loaded = load_yaml_scenarios_with_bytes(tmp_path)
        pack = load_scenario_pack(tmp_path)

        assert [sc.id for sc, _, _ in loaded] == ["HOLD-901"]
        assert [sc.id for sc in pack.scenarios] == ["HOLD-901"]
        assert pack.content_hash == pack_content_hash(tmp_path)
