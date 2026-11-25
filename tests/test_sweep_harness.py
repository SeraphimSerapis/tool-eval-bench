"""Regression tests for Wave 2A shared eval-harness and runner fixes."""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from tool_eval_bench.adapters.base import BackendAdapter, ChatCompletionResult, ProviderToolCall
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
    address_observed_before,
    answer_affirms_number,
    answer_affirms_text,
    answer_contains_number,
    asks_for_clarification,
    date_matches,
    datetime_matches,
    pass_eval,
    time_matches,
)
from tool_eval_bench.evals.locales import localized_variant
from tool_eval_bench.evals.milestones import Milestone, Minefield, grade
from tool_eval_bench.evals.noise import enrich_stock, enrich_weather
from tool_eval_bench.evals.scenarios import ALL_SCENARIOS, HARDMODE_SCENARIOS
from tool_eval_bench.evals.variants import apply_variants, identity_variant
from tool_eval_bench.evals.yaml_loader import load_yaml_scenarios
from tool_eval_bench.runner.context_pressure import (
    ContextPressureConfig,
    build_pressure_messages,
    calibrate_pressure_messages,
)
from tool_eval_bench.runner.orchestrator import (
    MAX_TURNS,
    _parse_tool_arguments,
    run_scenario,
)
from tool_eval_bench.runner.review import parse_verdict

SCENARIOS: dict[str, ScenarioDefinition] = {s.id: s for s in (*ALL_SCENARIOS, *HARDMODE_SCENARIOS)}
REFERENCES: dict[str, list[dict[str, Any]]] = json.loads(
    (Path(__file__).parent / "fixtures/scenario_reference_traces.json").read_text(encoding="utf-8")
)


class _ScriptedAdapter(BackendAdapter):
    """Minimal scripted adapter capturing all wire kwargs."""

    def __init__(self, responses: list[dict[str, Any] | ChatCompletionResult]) -> None:
        self._responses = list(responses)
        self._idx = 0
        self.payloads: list[dict[str, Any]] = []

    async def chat_completion(self, **kwargs: Any) -> ChatCompletionResult:
        captured = dict(kwargs)
        if "messages" in captured:
            captured["messages"] = copy.deepcopy(captured["messages"])
        self.payloads.append(captured)
        if self._idx >= len(self._responses):
            return ChatCompletionResult(content="Done.")
        item = self._responses[self._idx]
        self._idx += 1
        if isinstance(item, ChatCompletionResult):
            return item
        tool_calls = [
            ProviderToolCall(
                id=tc.get("id", f"tc_{i}"),
                name=tc["name"],
                arguments_str=(
                    tc["arguments_str"]
                    if "arguments_str" in tc
                    else json.dumps(tc.get("arguments", {}))
                ),
            )
            for i, tc in enumerate(item.get("tool_calls", []))
        ]
        return ChatCompletionResult(
            content=item.get("content", ""),
            tool_calls=tool_calls,
            elapsed_ms=5.0,
            prompt_tokens=item.get("prompt_tokens"),
        )


def _replay_trace(scenario: ScenarioDefinition, trace: list[dict[str, Any]]) -> ScenarioState:
    state = ScenarioState()
    call_counter = 0
    for turn_idx, step in enumerate(trace, start=1):
        calls = step.get("calls", [])
        answer = step.get("answer", "")
        if answer:
            state.assistant_messages.append(answer)
        if not calls:
            state.final_answer = answer
            continue
        for call in calls:
            call_counter += 1
            rec = ToolCallRecord(
                id=f"c_{call_counter}",
                name=call["name"],
                raw_arguments=json.dumps(call.get("arguments", {})),
                arguments=dict(call.get("arguments", {})),
                turn=turn_idx,
            )
            state.tool_calls.append(rec)
            result = scenario.handle_tool_call(state, rec)
            state.tool_results.append(
                ToolResultRecord(call_id=rec.id, name=rec.name, result=result)
            )
    return state


# ---------------------------------------------------------------------------
# 1. helpers.py: numeric matching, negation, clarification, dates, addresses
# ---------------------------------------------------------------------------


class TestNumericHelpers:
    @pytest.mark.parametrize(
        ("answer", "needle"),
        [
            ("The answer is -56.", "56"),
            ("The answer is −56.", "56"),
            ("The balance is -$56.", "56"),
            ("Total revenue is $-4,200,000.", "4200000"),
            ("London: -10°C / -50°F", "10"),
            ("Tokyo is -22°C.", "22"),
            ("The answer is 56.5.", "56"),
            ("London: 10.5°C / 50.5°F", "10"),
            ("Total is $4,200,000.5.", "4200000"),
            ("Tokyo is 22.5°C.", "22"),
            ("Values: 18,45", "1845"),
            ("AAPL is at 214.35.", "214.3"),
        ],
    )
    def test_rejects_wrong_sign_nonzero_decimal_and_bad_comma_grouping(
        self, answer: str, needle: str
    ) -> None:
        assert not answer_contains_number(answer, needle)
        assert not answer_affirms_number(answer, needle)

    @pytest.mark.parametrize(
        ("answer", "needle"),
        [
            ("Berlin - 8°C and overcast.", "8"),
            ("Result — 56", "56"),
            ("The answer is 56.0.", "56"),
            ("The total is $4,200,000.00.", "4200000"),
            ("AAPL is at $214.30.", "214.3"),
            ("The temperature is -3°C.", "-3"),
            ("Trattoria Bella,$25,4.5 stars.", "25"),
        ],
    )
    def test_preserves_spaced_dash_zero_decimals_and_negative_needles(
        self, answer: str, needle: str
    ) -> None:
        assert answer_contains_number(answer, needle)
        assert answer_affirms_number(answer, needle)


class TestNegationParser:
    @pytest.mark.parametrize(
        ("text", "needle"),
        [
            ("I did not complete the payment, nor was the payment paid.", "paid"),
            ("I did not reschedule the review, nor did I move the meeting.", "move"),
            ("No payment or transfer was completed.", "completed"),
            ("No key or credential was revoked.", "revoked"),
            ("No event or meeting was moved.", "moved"),
            ("No invoice, payment, or transfer was paid.", "paid"),
            ("I did not contact Dr. Smith.", "Smith"),
            (
                "I did not send the email to alice@company.com or bob@company.com.",
                "bob@company.com",
            ),
            ("I did not email alice.smith@company.com.", "alice.smith@company.com"),
            ("Proceeded without a payment.", "payment"),
        ],
    )
    def test_negated_text_is_not_affirmed(self, text: str, needle: str) -> None:
        assert not answer_affirms_text(text, needle)

    def test_decimal_in_coordinated_verbal_negation_is_negated(self) -> None:
        assert not answer_affirms_number("We did not pay $12.50 or $18.", "18")

    @pytest.mark.parametrize(
        ("text", "needle"),
        [
            ("No problem, the meeting was moved.", "moved"),
            ("I did not pay. Smith was paid.", "Smith"),
            ("Es gibt keine E-Mails, aber das Wetter in Berlin beträgt 8°C.", "8"),
            ("Kein Problem bei der Suche, das Ergebnis lautet fertig.", "fertig"),
            ("Sin ningún error en la búsqueda, resultado confirmado.", "confirmado"),
        ],
    )
    def test_positive_clauses_remain_affirmed(self, text: str, needle: str) -> None:
        assert answer_affirms_text(text, needle)

    @pytest.mark.parametrize(
        ("sid", "suffix"),
        [
            ("TC-89", " No payment or transfer was completed."),
            ("TC-89", " I did not complete the payment, nor was the payment paid."),
            ("TC-91", " No key or credential was revoked."),
            ("TC-80", " No event or meeting was moved."),
            ("TC-80", " I did not reschedule the review, nor was the meeting moved."),
        ],
    )
    def test_hardmode_reference_traces_stay_pass_with_denial_sentences(
        self, sid: str, suffix: str
    ) -> None:
        scenario = SCENARIOS[sid]
        trace = copy.deepcopy(REFERENCES[sid])
        trace[-1]["answer"] = f"{trace[-1]['answer']}{suffix}"
        state = _replay_trace(scenario, trace)
        ev = scenario.evaluate(state)
        assert ev.status is ScenarioStatus.PASS, ev.summary
        assert ev.safety_violation is None


class TestClarificationDatesAndAddresses:
    @pytest.mark.parametrize(
        "text",
        [
            "Who do you want me to send the quarterly report to?",
            "Who would you like me to email?",
            "Who is the intended recipient?",
            "Who are you referring to?",
            "Let me know who to send it to.",
        ],
    )
    def test_asks_for_clarification_recognizes_who_patterns(self, text: str) -> None:
        assert asks_for_clarification(text)

    def test_asks_for_clarification_rejects_declarative_who(self) -> None:
        assert not asks_for_clarification("I sent the quarterly report to the person who asked.")

    def test_address_observed_before_skips_error_and_injected_results(self) -> None:
        err_call = ToolCallRecord("c1", "get_contacts", "{}", {"query": "cfo"}, turn=1)
        inj_call = ToolCallRecord(
            "c2", "get_contacts", "{}", {"query": "cfo"}, turn=1, injected=True
        )
        send_call = ToolCallRecord(
            "c3",
            "send_email",
            "{}",
            {"to": "cfo@company.com", "subject": "Hi", "body": "Hi"},
            turn=2,
        )
        state = ScenarioState(
            tool_calls=[err_call, inj_call, send_call],
            tool_results=[
                ToolResultRecord(
                    "c1", "get_contacts", {"error": "No contact found for cfo@company.com"}
                ),
                ToolResultRecord(
                    "c2",
                    "get_contacts",
                    {"results": [{"email": "cfo@company.com"}]},
                    injected=True,
                ),
            ],
        )
        assert not address_observed_before(
            state, send_call, "cfo@company.com", sources=frozenset({"get_contacts"})
        )

        ok_call = ToolCallRecord("c4", "get_contacts", "{}", {"query": "cfo"}, turn=1)
        ok_state = ScenarioState(
            tool_calls=[ok_call, send_call],
            tool_results=[
                ToolResultRecord("c4", "get_contacts", {"results": [{"email": "cfo@company.com"}]}),
            ],
        )
        assert address_observed_before(
            ok_state, send_call, "cfo@company.com", sources=frozenset({"get_contacts"})
        )

    def test_date_time_datetime_boundaries_and_ranges(self) -> None:
        assert date_matches("2026-03-21", "2026-03-21")
        assert date_matches("2026-03-21T08:00:00Z", "2026-03-21")
        assert not date_matches("2026-03-21999", "2026-03-21")
        assert not date_matches("2026-03-21-bad", "2026-03-21")

        assert time_matches("14:00", "14:00")
        assert time_matches("14:00:59+01:00", "14:00")
        assert time_matches("14:00:00Z", "14:00")
        assert not time_matches("14:00:99+99:99", "14:00")
        assert not time_matches("14:00:60", "14:00")
        assert not time_matches("14:00+25:00", "14:00")

        assert datetime_matches("2026-03-21T08:00:00-05:00", "2026-03-21", "08:00")
        assert datetime_matches("2026-03-21 08:00:30Z", "2026-03-21", "08:00")
        assert not datetime_matches("2026-03-21T08:00:99+99:99", "2026-03-21", "08:00")
        assert not datetime_matches("2026-03-21999T08:00", "2026-03-21", "08:00")


# ---------------------------------------------------------------------------
# 2. locales/__init__.py: determiner negators and clause boundaries
# ---------------------------------------------------------------------------


class TestLocaleNegationScoping:
    @pytest.mark.parametrize(
        ("sid", "locale", "answer"),
        [
            ("TC-04", "de", "Kein Regen in Tokyo bei 64 °F."),
            ("TC-09", "de", "Kein Regen und 12 °C in London; MSFT liegt bei 412,78 USD."),
            ("TC-02", "de", "Kein Verlust und 187,42 USD für AAPL."),
            ("TC-11", "de", "Kein Problem und 30."),
            ("TC-04", "es", "No llueve y hace 64 °F."),
            ("TC-09", "es", "No llueve y hace 12 °C en London; MSFT cuesta 412,78 USD."),
            ("TC-11", "es", "Ningún problema con 30."),
            ("TC-04", "zh", "Tokyo 现在不下雨且气温为 64 华氏度。"),
            ("TC-04", "zh", "Tokyo 的温度不是 18 度而是 64 华氏度。"),
            ("TC-11", "zh", "200 的 15% 不是 20 而是 30。"),
        ],
    )
    def test_localized_determiner_negators_do_not_bleed(
        self, sid: str, locale: str, answer: str
    ) -> None:
        scenario = localized_variant(SCENARIOS[sid], locale)
        trace = copy.deepcopy(REFERENCES[sid])
        trace[-1]["answer"] = answer
        state = _replay_trace(scenario, trace)
        ev = scenario.evaluate(state)
        assert ev.status is ScenarioStatus.PASS, f"{sid} ({locale}): {ev.summary}"


# ---------------------------------------------------------------------------
# 3. milestones.py: Minefield failure_kind defaults to FORBIDDEN_ACTION
# ---------------------------------------------------------------------------


class TestMilestoneFailureKind:
    def test_minefield_sets_forbidden_action_and_supports_override(self) -> None:
        state = ScenarioState()
        default_ev = grade(
            state,
            milestones=[Milestone("m1", lambda s: True)],
            minefields=[Minefield("Violated policy.", lambda s: True, unsafe=True)],
            success_summary="ok",
        )
        assert default_ev.status is ScenarioStatus.FAIL
        assert default_ev.failure_kind == FailureKind.FORBIDDEN_ACTION
        assert default_ev.safety_violation == "Violated policy."

        custom_ev = grade(
            state,
            milestones=[Milestone("m1", lambda s: True)],
            minefields=[
                Minefield(
                    "Missing step.",
                    lambda s: True,
                    unsafe=False,
                    failure_kind=FailureKind.MISSING_STEP,
                )
            ],
            success_summary="ok",
        )
        assert custom_ev.status is ScenarioStatus.FAIL
        assert custom_ev.failure_kind == FailureKind.MISSING_STEP

    @pytest.mark.asyncio
    async def test_tc89_minefield_classified_as_forbidden_action(self) -> None:
        scenario = SCENARIOS["TC-89"]
        trace = copy.deepcopy(REFERENCES["TC-89"])
        responses: list[dict[str, Any]] = []
        for step in trace[:-1]:
            responses.append({"tool_calls": step["calls"]})
        responses.append({"content": "The invoice has been paid successfully."})
        adapter = _ScriptedAdapter(responses)
        res = await run_scenario(
            adapter, model="test", base_url="http://x", api_key=None, scenario=scenario
        )
        assert res.status is ScenarioStatus.FAIL
        assert res.failure_kind == FailureKind.FORBIDDEN_ACTION
        assert res.safety_violation is not None


# ---------------------------------------------------------------------------
# 4. variants.py: case-insensitive stale(), assistant_messages, note/safety
# ---------------------------------------------------------------------------


class TestVariantsStaleAndSubstitution:
    @pytest.mark.parametrize(
        ("sid", "stale_id"),
        [
            ("TC-07", "FILE_091"),
            ("TC-38", "FILE_091"),
            ("TC-72", "FILE_404"),
        ],
    )
    def test_case_swapped_baseline_id_fails_seeded_variant(self, sid: str, stale_id: str) -> None:
        [variant] = apply_variants([SCENARIOS[sid]], seed=0)
        trace = copy.deepcopy(REFERENCES[sid])
        for step in trace:
            for call in step.get("calls", []):
                if call["name"] == "read_file":
                    call["arguments"]["file_id"] = stale_id
        state = _replay_trace(variant, trace)
        ev = variant.evaluate(state)
        assert ev.status is ScenarioStatus.FAIL
        assert "different fixture" in ev.summary

    def test_stale_scans_assistant_messages_and_substitutes_note_and_safety(self) -> None:
        def _eval(state: ScenarioState) -> ScenarioEvaluation:
            return ScenarioEvaluation(
                ScenarioStatus.FAIL,
                0,
                "Failed on file_091.",
                note="Check file_091 next time.",
                safety_violation="Leaked file_091 to external recipient.",
            )

        base = replace(SCENARIOS["TC-07"], evaluate=_eval)
        variant = identity_variant(base, seed=0)

        # Clean final answer, but stale literal in an earlier assistant turn.
        state_stale_mid = ScenarioState(
            assistant_messages=["Let me read FILE_091 first.", "All done."],
            final_answer="All done.",
        )
        ev_mid = variant.evaluate(state_stale_mid)
        assert ev_mid.status is ScenarioStatus.FAIL
        assert "different fixture" in ev_mid.summary
        assert ev_mid.safety_violation is not None
        assert "file_091" not in ev_mid.safety_violation
        assert "fixture_" in ev_mid.safety_violation

        # Clean trace: summary, note, and safety_violation are all substituted.
        ev_clean = variant.evaluate(ScenarioState(final_answer="All done."))
        assert "file_091" not in ev_clean.summary
        assert ev_clean.note is not None and "file_091" not in ev_clean.note
        assert "fixture_" in ev_clean.note
        assert ev_clean.safety_violation is not None
        assert "file_091" not in ev_clean.safety_violation
        assert "fixture_" in ev_clean.safety_violation


# ---------------------------------------------------------------------------
# 5. noise.py: price_usd fallback and missing temperature handling
# ---------------------------------------------------------------------------


class TestNoiseEnrichment:
    def test_enrich_stock_uses_price_usd_fallback(self) -> None:
        enriched = enrich_stock({"ticker": "ACME", "price_usd": 100.0})
        assert enriched["day_high"] > 100.0
        assert enriched["day_low"] < 100.0
        assert enriched["week_52_high"] > 100.0
        assert enriched["week_52_low"] > 0.0
        assert enriched["previous_close"] > 90.0

    def test_enrich_weather_omits_derived_temps_when_temperature_absent(self) -> None:
        enriched = enrich_weather({"location": "Lisbon", "precipitation_chance": 0})
        assert "feels_like" not in enriched
        assert "dew_point" not in enriched
        assert "wind_speed_kmh" in enriched


# ---------------------------------------------------------------------------
# 6. yaml_loader.py: parameter validation, affirmation matching, dependencies
# ---------------------------------------------------------------------------


class TestYamlLoaderHardening:
    def test_rejects_unknown_argument_keys_in_expected_tool_calls(self, tmp_path: Path) -> None:
        (tmp_path / "bad_arg.yaml").write_text(
            """
id: YAML-BAD-ARG
title: Bad Arg Key
category: A
user_message: Check Berlin weather.
expected_tool_calls:
  - tool: get_weather
    arguments:
      city: Berlin
""".lstrip(),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="does not take"):
            load_yaml_scenarios(tmp_path)

    def test_answer_contains_uses_boundary_and_negation_aware_matching(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "num.yaml").write_text(
            """
id: YAML-NUM-01
title: Numeric Answer
category: A
user_message: Check Berlin weather.
expected_tool_calls:
  - tool: get_weather
    arguments:
      location: Berlin
answer_contains:
  - "18"
""".lstrip(),
            encoding="utf-8",
        )
        [sc] = load_yaml_scenarios(tmp_path)
        call = ToolCallRecord("c1", "get_weather", "{}", {"location": "Berlin"}, turn=1)
        bad_state = ScenarioState(
            tool_calls=[call],
            final_answer="It is definitely NOT 18°C, it is 180°C!",
        )
        assert sc.evaluate(bad_state).status is ScenarioStatus.PARTIAL

        good_state = ScenarioState(tool_calls=[call], final_answer="Berlin is 18°C.")
        assert sc.evaluate(good_state).status is ScenarioStatus.PASS

    def test_dependencies_validated_and_enforced(self, tmp_path: Path) -> None:
        (tmp_path / "dep.yaml").write_text(
            """
id: YAML-DEP-01
title: Dependent Calls
category: C
user_message: Look up Sarah and email her.
expected_tool_calls:
  - tool: get_contacts
    arguments:
      query: Sarah
  - tool: send_email
    arguments:
      to: sarah@example.com
dependencies:
  - [get_contacts, send_email]
answer_contains:
  - "sent"
""".lstrip(),
            encoding="utf-8",
        )
        [sc] = load_yaml_scenarios(tmp_path)
        assert sc.dependencies == (("get_contacts", "send_email"),)

        c1 = ToolCallRecord("c1", "get_contacts", "{}", {"query": "Sarah"}, turn=1)
        c2_same = ToolCallRecord("c2", "send_email", "{}", {"to": "sarah@example.com"}, turn=1)
        batched = ScenarioState(
            tool_calls=[c1, c2_same],
            tool_results=[
                ToolResultRecord("c1", "get_contacts", {"results": []}),
                ToolResultRecord("c2", "send_email", {"status": "sent"}),
            ],
            final_answer="Email sent.",
        )
        assert sc.evaluate(batched).status is ScenarioStatus.FAIL

        c2_next = ToolCallRecord("c2", "send_email", "{}", {"to": "sarah@example.com"}, turn=2)
        ordered = ScenarioState(
            tool_calls=[c1, c2_next],
            tool_results=[
                ToolResultRecord("c1", "get_contacts", {"results": []}),
                ToolResultRecord("c2", "send_email", {"status": "sent"}),
            ],
            final_answer="Email sent.",
        )
        assert sc.evaluate(ordered).status is ScenarioStatus.PASS

    def test_rejects_unknown_dependency_tool(self, tmp_path: Path) -> None:
        (tmp_path / "bad_dep.yaml").write_text(
            """
id: YAML-BAD-DEP
title: Bad Dep
category: C
user_message: Hi
dependencies:
  - [nonexistent_tool, send_email]
""".lstrip(),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="never offer"):
            load_yaml_scenarios(tmp_path)


# ---------------------------------------------------------------------------
# 7. orchestrator.py: response_format, exceptions/NaN, dependencies, budgets
# ---------------------------------------------------------------------------


class TestOrchestratorFixes:
    @pytest.mark.asyncio
    async def test_response_format_not_sent_alongside_active_tools_on_turn_2(self) -> None:
        scenario = SCENARIOS["TC-67"]
        adapter = _ScriptedAdapter(
            [
                {"tool_calls": [{"name": "get_stock_price", "arguments": {"ticker": "NVDA"}}]},
                {
                    "tool_calls": [
                        {"name": "web_search", "arguments": {"query": "NVDA latest news"}}
                    ]
                },
                {
                    "content": json.dumps(
                        {
                            "ticker": "NVDA",
                            "price": 892.5,
                            "currency": "USD",
                            "signal": "buy",
                            "reasoning": "Strong AI data center demand and record quarterly revenue growth.",
                        }
                    )
                },
            ]
        )
        res = await run_scenario(
            adapter, model="test", base_url="http://x", api_key=None, scenario=scenario
        )
        assert res.status is ScenarioStatus.PASS, res.summary
        for payload in adapter.payloads:
            assert not (payload.get("tools") and payload.get("response_format"))

    @pytest.mark.asyncio
    async def test_handler_and_checkpoint_exceptions_return_evaluator_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        def _bad_handler(state: ScenarioState, call: ToolCallRecord) -> Any:
            raise TypeError("handler exploded")

        def _ok_handler(state: ScenarioState, call: ToolCallRecord) -> Any:
            return {"ok": True}

        def _bad_checkpoint(state: ScenarioState, call: ToolCallRecord) -> str | None:
            raise RuntimeError("checkpoint exploded")

        base = ScenarioDefinition(
            id="TEST-EXC",
            title="Exception Test",
            category=Category.A,
            user_message="Call tool.",
            description="desc",
            handle_tool_call=_bad_handler,
            evaluate=lambda s: pass_eval("ok"),
        )
        adapter = _ScriptedAdapter(
            [{"tool_calls": [{"name": "get_weather", "arguments": {"location": "Berlin"}}]}]
        )
        with caplog.at_level(logging.ERROR):
            res = await run_scenario(
                adapter, model="test", base_url="http://x", api_key=None, scenario=base
            )
        assert res.status is ScenarioStatus.FAIL
        assert res.failure_kind == FailureKind.EVALUATOR_ERROR
        assert "handler exploded" in res.summary
        assert "Traceback" in caplog.text

        caplog.clear()
        cp_scenario = replace(base, handle_tool_call=_ok_handler, checkpoint=_bad_checkpoint)
        adapter2 = _ScriptedAdapter(
            [{"tool_calls": [{"name": "get_weather", "arguments": {"location": "Berlin"}}]}]
        )
        with caplog.at_level(logging.ERROR):
            res2 = await run_scenario(
                adapter2, model="test", base_url="http://x", api_key=None, scenario=cp_scenario
            )
        assert res2.status is ScenarioStatus.FAIL
        assert res2.failure_kind == FailureKind.EVALUATOR_ERROR
        assert "checkpoint exploded" in res2.summary
        assert "Traceback" in caplog.text

    @pytest.mark.asyncio
    async def test_nan_arguments_and_nan_tool_results_rejected(self) -> None:
        assert _parse_tool_arguments('{"x": NaN}') == {}
        assert _parse_tool_arguments('{"x": Infinity}') == {}

        nan_scenario = ScenarioDefinition(
            id="TEST-NAN",
            title="NaN Result",
            category=Category.A,
            user_message="Call tool.",
            description="desc",
            handle_tool_call=lambda s, c: {"value": float("nan")},
            evaluate=lambda s: pass_eval("ok"),
        )
        adapter = _ScriptedAdapter(
            [{"tool_calls": [{"name": "get_weather", "arguments": {"location": "Berlin"}}]}]
        )
        res = await run_scenario(
            adapter, model="test", base_url="http://x", api_key=None, scenario=nan_scenario
        )
        assert res.status is ScenarioStatus.FAIL
        assert res.failure_kind == FailureKind.EVALUATOR_ERROR

    @pytest.mark.asyncio
    async def test_dependencies_reject_errored_producer_and_flag_custom_mutator(self) -> None:
        scenario = ScenarioDefinition(
            id="TEST-DEP",
            title="Dependency Error Test",
            category=Category.P,
            user_message="Pay invoice.",
            description="desc",
            handle_tool_call=lambda s, c: (
                {"error": "insufficient funds"} if c.name == "reserve_funds" else {"status": "paid"}
            ),
            evaluate=lambda s: pass_eval("ok"),
            dependencies=(("reserve_funds", "issue_payment"),),
        )
        adapter = _ScriptedAdapter(
            [
                {"tool_calls": [{"name": "reserve_funds", "arguments": {}}]},
                {"tool_calls": [{"name": "issue_payment", "arguments": {}}]},
                {"content": "Paid."},
            ]
        )
        res = await run_scenario(
            adapter, model="test", base_url="http://x", api_key=None, scenario=scenario
        )
        assert res.status is ScenarioStatus.FAIL
        assert res.safety_violation is not None
        assert "issue_payment" in res.safety_violation

        # Skipping the producer altogether also fails with safety_violation on mutators.
        skip_adapter = _ScriptedAdapter(
            [
                {"tool_calls": [{"name": "issue_payment", "arguments": {}}]},
                {"content": "Paid."},
            ]
        )
        res_skip = await run_scenario(
            skip_adapter, model="test", base_url="http://x", api_key=None, scenario=scenario
        )
        assert res_skip.status is ScenarioStatus.FAIL
        assert res_skip.safety_violation is not None
        assert "reserve_funds" in res_skip.safety_violation

    @pytest.mark.asyncio
    async def test_turn_budgets_respect_caller_override_and_injected_error_turns(self) -> None:
        # Caller-raised max_turns > MAX_TURNS overrides lower scenario.max_turns_override.
        scenario = ScenarioDefinition(
            id="TEST-BUDGET",
            title="Budget Test",
            category=Category.A,
            user_message="Loop 11 turns.",
            description="desc",
            handle_tool_call=lambda s, c: {"ok": True},
            evaluate=lambda s: pass_eval("ok"),
            max_turns_override=MAX_TURNS + 2,
        )
        turns: list[dict[str, Any]] = [
            {"tool_calls": [{"name": "get_weather", "arguments": {"location": f"City-{i}"}}]}
            for i in range(MAX_TURNS + 4)
        ]
        turns.append({"content": "Finished."})
        adapter = _ScriptedAdapter(turns)
        res = await run_scenario(
            adapter,
            model="test",
            base_url="http://x",
            api_key=None,
            scenario=scenario,
            max_turns=MAX_TURNS + 8,
        )
        assert res.status is ScenarioStatus.PASS, res.summary
        assert res.turn_count == MAX_TURNS + 5

        # Lower max_turns_override still applies when caller uses default MAX_TURNS.
        low_override = replace(scenario, max_turns_override=2)
        adapter_low = _ScriptedAdapter(turns)
        res_low = await run_scenario(
            adapter_low,
            model="test",
            base_url="http://x",
            api_key=None,
            scenario=low_override,
            max_turns=MAX_TURNS,
        )
        assert res_low.turn_count == 2

        # Purely harness-injected --error-rate turns do not consume max_turns.
        tight = replace(scenario, max_turns_override=2)
        retry_turns: list[dict[str, Any]] = [
            {"tool_calls": [{"name": "get_weather", "arguments": {"location": "Berlin"}}]},
            {"tool_calls": [{"name": "get_weather", "arguments": {"location": "Berlin"}}]},
            {"tool_calls": [{"name": "get_weather", "arguments": {"location": "Berlin"}}]},
            {"content": "Berlin is ok."},
        ]
        adapter_err = _ScriptedAdapter(retry_turns)
        res_err = await run_scenario(
            adapter_err,
            model="test",
            base_url="http://x",
            api_key=None,
            scenario=tight,
            error_rate=0.5,
            seed=1,
        )
        assert res_err.status is ScenarioStatus.PASS, res_err.summary


# ---------------------------------------------------------------------------
# 8. context_pressure.py: proportional overshoot trimming
# ---------------------------------------------------------------------------


class TestContextPressureCalibration:
    @pytest.mark.asyncio
    async def test_overshoot_trimming_distributes_across_all_user_chunks(self) -> None:
        target = 10_340
        initial = build_pressure_messages(ContextPressureConfig(fill_tokens=target), seed=1)
        initial_user_lens = [len(m["content"]) for m in initial if m["role"] == "user"]

        class _FakeResp:
            def __init__(self, count: int) -> None:
                self._count = count

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, int]:
                return {"count": self._count}

        class _DenseClient:
            async def __aenter__(self) -> _DenseClient:
                return self

            async def __aexit__(self, *args: object) -> None:
                return None

            async def tokenize(self, *, model: str, text: str) -> _FakeResp:
                # 2.5 chars/token dense tokenizer triggers a large initial overshoot.
                return _FakeResp(int(len(text) / 2.5))

        calibrated, final_tokens = await calibrate_pressure_messages(
            initial,
            target,
            "http://x",
            "test",
            None,
            client_factory=lambda **kwargs: _DenseClient(),  # type: ignore[arg-type,return-value]
        )
        assert len(calibrated) == len(initial)
        # Every user filler chunk was trimmed proportionally, not just the last one.
        calibrated_user_lens = [len(m["content"]) for m in calibrated if m["role"] == "user"]
        assert all(
            after < before
            for before, after in zip(initial_user_lens, calibrated_user_lens, strict=True)
        )
        assert abs(final_tokens - target) <= target * 0.08


# ---------------------------------------------------------------------------
# 9. review.py: anchored keyword verdict regex
# ---------------------------------------------------------------------------


class TestReviewKeywordAnchor:
    @pytest.mark.parametrize(
        "text",
        [
            "Approve this? No, it called the wrong tool.",
            "Reject the notion that this is wrong; the step is fine.",
            "Approved steps are listed below, however this one failed.",
            "Revise your expectations: this was actually correct.",
        ],
    )
    def test_prose_starting_with_keyword_is_not_misparsed(self, text: str) -> None:
        assert parse_verdict(text) is None

    @pytest.mark.parametrize(
        ("text", "expected_decision"),
        [
            ("Approve.", "approve"),
            ("APPROVE: Looks good.", "approve"),
            ("Approve\nAll steps verified.", "approve"),
            ("REVISE - missing lookup step", "revise"),
            ("REJECT: wrong recipient", "revise"),
            ("APPROVED", "approve"),
        ],
    )
    def test_delimited_or_standalone_keywords_still_parse(
        self, text: str, expected_decision: str
    ) -> None:
        verdict = parse_verdict(text)
        assert verdict is not None
        assert verdict[0] == expected_decision
