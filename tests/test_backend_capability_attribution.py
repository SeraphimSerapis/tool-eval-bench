"""A stack's limits must not be recorded as a model's mistakes.

Three places where the benchmark graded the serving environment and reported
the result as model quality:

- a 4xx that rejected the request before the model produced anything, which the
  adapter degrades to a soft result whose ``content`` was then scored as an
  answer;
- ``tool_choice="required"``, which an endpoint may accept and silently drop,
  leaving a scenario unable to tell an ignored instruction from one that was
  never delivered;
- TC-88's PASS tier, which needs a reasoning channel most OpenAI-compatible
  endpoints do not expose.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import httpx
import pytest

from tool_eval_bench.adapters.base import BackendAdapter, ChatCompletionResult, ProviderToolCall
from tool_eval_bench.adapters.gemini import GeminiAdapter
from tool_eval_bench.adapters.openai_compat import OpenAICompatibleAdapter
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
from tool_eval_bench.evals.scenarios import ALL_SCENARIOS_WITH_HARDMODE
from tool_eval_bench.runner.orchestrator import (
    probe_tool_choice_required,
    run_all_scenarios,
    run_scenario,
    supports_tool_choice_required,
)

BY_ID = {s.id: s for s in ALL_SCENARIOS_WITH_HARDMODE}


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class ScriptedAdapter(BackendAdapter):
    """Returns queued results or raises, recording each request's tool_choice."""

    def __init__(self, results: list[ChatCompletionResult | Exception]) -> None:
        self._results = list(results)
        self.tool_choices: list[Any] = []
        self.calls = 0

    async def chat_completion(self, **kwargs: Any) -> ChatCompletionResult:
        self.tool_choices.append(kwargs.get("tool_choice"))
        self.calls += 1
        if not self._results:
            return ChatCompletionResult(content="[exhausted]")
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class RaisingAdapter(BackendAdapter):
    """Every request explodes, standing in for an endpoint that refuses one."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def chat_completion(self, **kwargs: Any) -> ChatCompletionResult:
        raise self._exc


def _tool_call(name: str = "probe_ping") -> ProviderToolCall:
    return ProviderToolCall(id="tc_1", name=name, arguments_str=json.dumps({"value": "ok"}))


def _trivial_scenario(
    scenario_id: str = "TC-TEST",
    *,
    tool_choice_override: str | None = None,
) -> ScenarioDefinition:
    return ScenarioDefinition(
        id=scenario_id,
        title="Test",
        category=Category.A,
        user_message="hello",
        description="test scenario",
        handle_tool_call=lambda state, call: {"ok": True},
        evaluate=lambda state: ScenarioEvaluation(ScenarioStatus.PASS, 2, "fine"),
        difficulty=1,
        tool_choice_override=tool_choice_override,
    )


async def _run(adapter: BackendAdapter, scenario: ScenarioDefinition) -> Any:
    return await run_scenario(
        adapter,
        model="m",
        base_url="http://x",
        api_key=None,
        scenario=scenario,
        max_turns=4,
    )


# ---------------------------------------------------------------------------
# 1. A rejected request is the stack's failure, not the model's
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rejected_first_request_is_infrastructure_not_model_failure() -> None:
    """A 400 before the model authored anything cannot be the model's fault."""
    adapter = ScriptedAdapter(
        [
            ChatCompletionResult(
                content="[server error 400] tool_choice 'required' is not supported",
                transport_error_status=400,
            )
        ]
    )
    result = await _run(adapter, _trivial_scenario())

    assert result.failure_kind == FailureKind.SERVER_ERROR
    assert result.is_infrastructure_failure
    assert "400" in result.summary


@pytest.mark.asyncio
async def test_rejected_request_never_reaches_the_evaluator() -> None:
    """The error string must not be graded as though the model wrote it."""
    seen: list[str] = []

    def _evaluate(state: ScenarioState) -> ScenarioEvaluation:
        seen.append(state.final_answer)
        return ScenarioEvaluation(ScenarioStatus.PASS, 2, "graded")

    scenario = _trivial_scenario()
    scenario.evaluate = _evaluate
    adapter = ScriptedAdapter(
        [ChatCompletionResult(content="[server error 422] bad", transport_error_status=422)]
    )

    result = await _run(adapter, scenario)

    assert seen == [], "the evaluator ran on a turn that produced no model output"
    assert result.status == ScenarioStatus.FAIL


@pytest.mark.asyncio
async def test_rejection_after_the_model_authored_a_tool_call_stays_a_model_failure() -> None:
    """A strict server refusing model-authored arguments is still the model.

    This is the case the adapter's soft result was built for, so it has to keep
    working: the 4xx is caused by what the model put in the history.
    """
    adapter = ScriptedAdapter(
        [
            ChatCompletionResult(content="", tool_calls=[_tool_call("anything")]),
            ChatCompletionResult(content="[server error 400] bad args", transport_error_status=400),
        ]
    )
    scenario = _trivial_scenario()
    scenario.evaluate = lambda state: ScenarioEvaluation(ScenarioStatus.FAIL, 0, "model failed")

    result = await _run(adapter, scenario)

    assert result.failure_kind != FailureKind.SERVER_ERROR
    assert not result.is_infrastructure_failure


@pytest.mark.asyncio
async def test_sampler_initialization_rejection_after_tool_call_is_infrastructure() -> None:
    """A backend sampler failure is not caused by valid tool-call history."""
    adapter = ScriptedAdapter(
        [
            ChatCompletionResult(content="", tool_calls=[_tool_call("anything")]),
            ChatCompletionResult(
                content=(
                    "[server error 400] Failed to initialize samplers: failed to parse grammar"
                ),
                transport_error_status=400,
                transport_error_is_infrastructure=True,
            ),
        ]
    )
    seen: list[str] = []
    scenario = _trivial_scenario()

    def evaluate(state: ScenarioState) -> ScenarioEvaluation:
        seen.append(state.final_answer)
        return ScenarioEvaluation(ScenarioStatus.PARTIAL, 1, "graded server output")

    scenario.evaluate = evaluate

    result = await _run(adapter, scenario)

    assert seen == []
    assert result.failure_kind == FailureKind.SERVER_ERROR
    assert result.is_infrastructure_failure
    assert "serving stack failed before inference" in result.summary


@pytest.mark.asyncio
async def test_infrastructure_rejection_drops_out_of_the_score() -> None:
    """An excluded scenario leaves the denominator, it does not score zero."""
    good = _trivial_scenario("TC-GOOD")
    bad = _trivial_scenario("TC-BAD")
    adapter = ScriptedAdapter(
        [
            ChatCompletionResult(content="done"),
            ChatCompletionResult(content="[server error 400] no", transport_error_status=400),
        ]
    )

    summary = await run_all_scenarios(
        adapter, model="m", base_url="http://x", scenarios=[good, bad]
    )

    assert summary.excluded_scenarios == ["TC-BAD"]
    assert summary.max_points == 2, "the excluded scenario must leave the denominator"
    assert summary.final_score == 100
    assert summary.completion_rate == 50.0


# ---------------------------------------------------------------------------
# 2. tool_choice="required" is probed, not assumed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_probe_reports_support_when_a_tool_call_comes_back() -> None:
    adapter = ScriptedAdapter([ChatCompletionResult(content="", tool_calls=[_tool_call()])])

    assert await supports_tool_choice_required(
        adapter, model="m", base_url="http://x", api_key=None, timeout_seconds=5
    )
    assert adapter.tool_choices == ["required"]


@pytest.mark.asyncio
async def test_probe_reports_no_support_when_the_parameter_is_silently_dropped() -> None:
    """The failure mode nothing else can detect: accepted, then ignored."""
    adapter = ScriptedAdapter(
        [ChatCompletionResult(content="56", finish_reason="stop", completion_tokens=1)]
    )

    probe = await probe_tool_choice_required(
        adapter, model="m", base_url="http://x", api_key=None, timeout_seconds=5
    )

    assert not probe.enforced
    assert adapter.calls == 1
    assert "not enforced" in probe.detail
    assert "finish_reason=stop" in probe.detail
    assert "completion_tokens=1" in probe.detail


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "token_overrides",
    [
        {},
        {"max_tokens": 1},
        {"max_completion_tokens": 1},
        {"max_tokens": 1, "max_completion_tokens": 1},
    ],
)
async def test_truncated_probe_retries_with_larger_wire_budget(token_overrides: dict) -> None:
    payloads: list[dict] = []
    extra = {**token_overrides, "top_p": 0.9}
    original = extra.copy()

    def respond(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        if len(payloads) == 1:
            message = {"content": "", "reasoning_content": "Still thinking about the constraint"}
            finish, tokens = "length", 256
        else:
            message = {
                "content": "",
                "tool_calls": [
                    {
                        "id": "probe",
                        "type": "function",
                        "function": {"name": "probe_ping", "arguments": '{"value":"ok"}'},
                    }
                ],
            }
            finish, tokens = "tool_calls", 42
        return httpx.Response(
            200,
            json={
                "choices": [{"message": message, "finish_reason": finish}],
                "usage": {"completion_tokens": tokens},
            },
        )

    adapter = OpenAICompatibleAdapter()
    adapter._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    try:
        probe = await probe_tool_choice_required(
            adapter,
            model="m",
            base_url="http://x",
            api_key=None,
            timeout_seconds=5,
            extra_params=extra,
        )
    finally:
        await adapter.aclose()

    assert probe.enforced and probe.arguments_intact
    assert [p["max_tokens"] for p in payloads] == [256, 4096]
    assert all("max_completion_tokens" not in p for p in payloads)
    assert all(p["top_p"] == 0.9 and p["tool_choice"] == "required" for p in payloads)
    assert extra == original
    assert "finish_reason=length" in probe.detail
    assert "completion_tokens=256" in probe.detail
    assert "retry" in probe.detail
    assert "finish_reason=tool_calls" in probe.detail
    assert "completion_tokens=42" in probe.detail


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "token_overrides",
    [
        {},
        {"maxOutputTokens": 1},
        {"generationConfig": {"maxOutputTokens": 1, "topP": 0.9}},
        {"max_tokens": 1, "generationConfig": {"maxOutputTokens": 1}},
    ],
)
async def test_gemini_truncated_probe_retries_with_larger_wire_budget(
    token_overrides: dict,
) -> None:
    configs: list[dict] = []
    original = copy.deepcopy(token_overrides)

    def respond(request: httpx.Request) -> httpx.Response:
        configs.append(json.loads(request.content)["generationConfig"])
        if len(configs) == 1:
            candidate = {"content": {"parts": []}, "finishReason": "MAX_TOKENS"}
        else:
            call = {"functionCall": {"name": "probe_ping", "args": {"value": "ok"}}}
            candidate = {"content": {"parts": [call]}, "finishReason": "STOP"}
        return httpx.Response(200, json={"candidates": [candidate]})

    adapter = GeminiAdapter()
    adapter._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    try:
        probe = await probe_tool_choice_required(
            adapter,
            model="gemini-test",
            base_url="http://x",
            api_key=None,
            timeout_seconds=5,
            extra_params=token_overrides,
        )
    finally:
        await adapter.aclose()

    assert probe.enforced and probe.arguments_intact
    assert [c["maxOutputTokens"] for c in configs] == [256, 4096]
    assert token_overrides == original
    if "topP" in original.get("generationConfig", {}):
        assert all(c["topP"] == 0.9 for c in configs)


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrency", [1, 2])
async def test_truncated_probe_recovery_keeps_forced_scenario_in_score(concurrency: int) -> None:
    adapter = ScriptedAdapter(
        [
            ChatCompletionResult(content="", finish_reason="length", completion_tokens=256),
            ChatCompletionResult(content="", tool_calls=[_tool_call()], finish_reason="tool_calls"),
            ChatCompletionResult(content="56"),
        ]
    )
    summary = await run_all_scenarios(
        adapter,
        model="m",
        base_url="http://x",
        scenarios=[_trivial_scenario("TC-45", tool_choice_override="required")],
        concurrency=concurrency,
    )

    assert adapter.calls == 3
    assert summary.excluded_scenarios == []
    assert summary.max_points == summary.total_points == 2
    assert summary.completion_rate == 100.0
    assert "retry" in summary.scenario_results[0].diagnostics["tool_choice"]


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrency", [1, 2])
async def test_repeated_probe_truncation_is_inconclusive(concurrency: int, caplog) -> None:
    adapter = ScriptedAdapter(
        [
            ChatCompletionResult(content="", finish_reason="length", completion_tokens=256),
            ChatCompletionResult(
                content="unfinished", finish_reason="length", completion_tokens=4096
            ),
        ]
    )
    summary = await run_all_scenarios(
        adapter,
        model="m",
        base_url="http://x",
        scenarios=[_trivial_scenario("TC-45", tool_choice_override="required")],
        concurrency=concurrency,
    )

    assert adapter.calls == 2
    assert summary.excluded_scenarios == ["TC-45"]
    assert summary.max_points == 0
    result = summary.scenario_results[0]
    assert "inconclusive" in result.diagnostics["tool_choice"]
    assert "completion_tokens=4096" in result.diagnostics["tool_choice"]
    assert "does not enforce" not in result.summary
    assert "does not enforce" not in caplog.text
    assert "inconclusive" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        (ChatCompletionResult(content="OK", finish_reason="stop"), "not enforced"),
        (ChatCompletionResult(content="", finish_reason="stop"), "inconclusive"),
        (
            ChatCompletionResult(content=" \n\t", finish_reason="stop"),
            "inconclusive; probe returned no tool call or answer",
        ),
        (
            ChatCompletionResult(content="", finish_reason="length"),
            "exhausted both token budgets without a tool call;",
        ),
        (
            ChatCompletionResult(content="Well, the word is", finish_reason="length"),
            "exhausted both token budgets without a tool call, leaving partial text;",
        ),
        (ChatCompletionResult(content="", transport_error_status=400), "HTTP 400"),
        (ChatCompletionResult(content="", malformed=True), "not valid JSON"),
        (RuntimeError("connection reset"), "inconclusive; probe request failed"),
    ],
)
async def test_probe_retry_classifies_final_reply(
    reply: ChatCompletionResult | Exception, expected: str
) -> None:
    adapter = ScriptedAdapter([ChatCompletionResult(content="", finish_reason="length"), reply])

    probe = await probe_tool_choice_required(
        adapter, model="m", base_url="http://x", api_key=None, timeout_seconds=5
    )

    assert not probe.enforced
    assert adapter.calls == 2
    assert expected in probe.detail
    assert "retry" in probe.detail


@pytest.mark.asyncio
async def test_probe_does_not_retry_after_a_first_attempt_exception() -> None:
    adapter = ScriptedAdapter([RuntimeError("connection reset")])

    probe = await probe_tool_choice_required(
        adapter, model="m", base_url="http://x", api_key=None, timeout_seconds=5
    )

    assert not probe.enforced
    assert adapter.calls == 1
    assert probe.detail.startswith("inconclusive; probe request failed (connection reset)")
    assert "probe(max_tokens=256, request failed)" in probe.detail
    assert "retry" not in probe.detail


@pytest.mark.asyncio
async def test_probe_does_not_retry_when_truncated_reply_contains_a_call() -> None:
    adapter = ScriptedAdapter(
        [ChatCompletionResult(content="", tool_calls=[_tool_call()], finish_reason="length")]
    )

    probe = await probe_tool_choice_required(
        adapter, model="m", base_url="http://x", api_key=None, timeout_seconds=5
    )

    assert probe.enforced
    assert adapter.calls == 1


@pytest.mark.asyncio
async def test_probe_reports_no_support_when_the_endpoint_rejects_the_parameter() -> None:
    adapter = ScriptedAdapter(
        [ChatCompletionResult(content="[server error 400] no", transport_error_status=400)]
    )

    assert not await supports_tool_choice_required(
        adapter, model="m", base_url="http://x", api_key=None, timeout_seconds=5
    )


@pytest.mark.asyncio
async def test_probe_treats_an_exception_as_unsupported() -> None:
    adapter = RaisingAdapter(RuntimeError("connection reset"))

    assert not await supports_tool_choice_required(
        adapter, model="m", base_url="http://x", api_key=None, timeout_seconds=5
    )


@pytest.mark.asyncio
async def test_forced_tool_choice_scenario_is_excluded_on_an_endpoint_that_ignores_it() -> None:
    """TC-45 scored zero for answering 7x8 correctly. Now it is excluded."""
    forced = _trivial_scenario("TC-45", tool_choice_override="required")
    adapter = ScriptedAdapter(
        [
            ChatCompletionResult(content="56"),  # the probe, no tool call
        ]
    )

    summary = await run_all_scenarios(adapter, model="m", base_url="http://x", scenarios=[forced])

    assert summary.excluded_scenarios == ["TC-45"]
    assert summary.max_points == 0
    assert summary.completion_rate == 0.0
    assert adapter.calls == 1, "the excluded scenario must not be run"


@pytest.mark.asyncio
async def test_forced_tool_choice_scenario_runs_when_the_endpoint_enforces_it() -> None:
    forced = _trivial_scenario("TC-45", tool_choice_override="required")
    adapter = ScriptedAdapter(
        [
            ChatCompletionResult(content="", tool_calls=[_tool_call()]),  # probe
            ChatCompletionResult(content="56"),  # the scenario itself
        ]
    )

    summary = await run_all_scenarios(adapter, model="m", base_url="http://x", scenarios=[forced])

    assert summary.excluded_scenarios == []
    assert summary.max_points == 2
    assert adapter.tool_choices[0] == "required"


@pytest.mark.asyncio
async def test_probe_tells_the_model_not_to_call() -> None:
    """A model that would call anyway proves nothing about the endpoint."""
    adapter = ScriptedAdapter([ChatCompletionResult(content="", tool_calls=[_tool_call()])])
    messages: list[Any] = []

    async def _capture(**kwargs: Any) -> ChatCompletionResult:
        messages.append(kwargs["messages"])
        return await ScriptedAdapter.chat_completion(adapter, **kwargs)

    adapter.chat_completion = _capture  # type: ignore[method-assign]
    probe = await probe_tool_choice_required(
        adapter, model="m", base_url="http://x", api_key=None, timeout_seconds=5
    )

    assert probe.enforced and probe.arguments_intact
    assert "Do not call any tools" in messages[0][0]["content"]


@pytest.mark.asyncio
async def test_probe_flags_a_forced_call_that_lost_its_arguments() -> None:
    """DeepSeek V4.1 Flash on SGLang: the forced call arrived as calculator {}."""
    empty = ProviderToolCall(id="tc_1", name="probe_ping", arguments_str="{}")
    adapter = ScriptedAdapter([ChatCompletionResult(content="", tool_calls=[empty])])

    probe = await probe_tool_choice_required(
        adapter, model="m", base_url="http://x", api_key=None, timeout_seconds=5
    )

    assert probe.enforced
    assert not probe.arguments_intact
    assert "empty arguments" in probe.detail


@pytest.mark.asyncio
async def test_probe_verdict_is_attached_to_forced_scenarios_only() -> None:
    forced = _trivial_scenario("TC-45", tool_choice_override="required")
    other = _trivial_scenario("TC-OTHER")
    adapter = ScriptedAdapter(
        [
            ChatCompletionResult(content="", tool_calls=[_tool_call()]),  # probe
            ChatCompletionResult(content="56"),
            ChatCompletionResult(content="done"),
        ]
    )

    summary = await run_all_scenarios(
        adapter, model="m", base_url="http://x", scenarios=[forced, other]
    )

    by_id = {r.scenario_id: r for r in summary.scenario_results}
    assert by_id["TC-45"].diagnostics["tool_choice"].startswith("enforced by the endpoint")
    assert "tool_choice" not in by_id["TC-OTHER"].diagnostics


@pytest.mark.asyncio
async def test_probe_is_skipped_when_no_scenario_forces_a_tool_call() -> None:
    """An ordinary run must not pay for a capability it never uses."""
    adapter = ScriptedAdapter([ChatCompletionResult(content="done")])

    await run_all_scenarios(
        adapter, model="m", base_url="http://x", scenarios=[_trivial_scenario()]
    )

    assert adapter.calls == 1
    assert "required" not in adapter.tool_choices


@pytest.mark.asyncio
async def test_exclusion_also_applies_on_the_parallel_path() -> None:
    forced = _trivial_scenario("TC-45", tool_choice_override="required")
    other = _trivial_scenario("TC-OTHER")
    adapter = ScriptedAdapter(
        [
            ChatCompletionResult(content="56"),  # probe: unsupported
            ChatCompletionResult(content="done"),
        ]
    )

    summary = await run_all_scenarios(
        adapter, model="m", base_url="http://x", scenarios=[forced, other], concurrency=2
    )

    assert summary.excluded_scenarios == ["TC-45"]
    assert summary.max_points == 2


def test_only_tc45_forces_a_tool_call() -> None:
    """If another scenario adopts ``required`` it inherits the probe by design."""
    forced = [s.id for s in ALL_SCENARIOS_WITH_HARDMODE if s.tool_choice_override == "required"]
    assert forced == ["TC-45"]


# ---------------------------------------------------------------------------
# 3. TC-88's PASS tier depends on a channel the stack may not have
# ---------------------------------------------------------------------------

# Three distinct 20-digit numbers with digit sums 73, 91 and 109, where each
# value's last six digits reverse the previous value's first six.
_TC88_VALID = (
    "11111199999994000000",
    "99999999940000111111",
    "99999910000000999999",
)


def _tc88_state(reasoning: list[str]) -> ScenarioState:
    state = ScenarioState()
    state.assistant_messages = list(_TC88_VALID)
    state.final_answer = _TC88_VALID[-1]
    state.assistant_reasoning = reasoning
    return state


def test_tc88_fixture_actually_satisfies_the_scenario_constraints() -> None:
    for number, expected in zip(_TC88_VALID, (73, 91, 109), strict=True):
        assert len(number) == 20 and number[0] != "0"
        assert sum(int(digit) for digit in number) == expected
    assert _TC88_VALID[1][-6:] == _TC88_VALID[0][:6][::-1]
    assert _TC88_VALID[2][-6:] == _TC88_VALID[1][:6][::-1]


def test_tc88_passes_when_the_provider_exposes_the_plan() -> None:
    plan = f"Plan: {' '.join(_TC88_VALID)}. Sums and reversals verified."
    result = BY_ID["TC-88"].evaluate(_tc88_state([plan, "", ""]))

    assert result.status == ScenarioStatus.PASS


def test_tc88_names_the_missing_channel_instead_of_blaming_the_model() -> None:
    result = BY_ID["TC-88"].evaluate(_tc88_state(["", "", ""]))
    assert result.status == ScenarioStatus.PASS
    assert result.diagnostics["reasoning_transport"] == "unavailable"


def test_tc88_still_reports_a_missing_plan_when_the_channel_exists() -> None:
    result = BY_ID["TC-88"].evaluate(_tc88_state(["thinking about it", "", ""]))
    assert result.status == ScenarioStatus.PASS
    assert result.diagnostics["reasoning_transport"] == "unconfirmed"


def test_tc88_constraint_violations_outrank_the_channel_message() -> None:
    """A wrong answer fails on its own terms, whatever the provider exposed."""
    state = _tc88_state(["", "", ""])
    state.assistant_messages = ["1" * 20, "2" * 20, "3" * 20]
    state.final_answer = "3" * 20
    result = BY_ID["TC-88"].evaluate(state)

    assert result.status == ScenarioStatus.FAIL


def test_tc88_tool_use_still_fails_regardless_of_reasoning() -> None:
    state = _tc88_state(["", "", ""])
    call = ToolCallRecord(id="c", name="web_search", raw_arguments="{}", arguments={}, turn=1)
    state.tool_calls.append(call)
    state.tool_results.append(
        ToolResultRecord("c", "web_search", BY_ID["TC-88"].handle_tool_call(state, call))
    )
    result = BY_ID["TC-88"].evaluate(state)

    assert result.status == ScenarioStatus.FAIL
