"""The reviewer: verdicts, the review loop, inheritance, credentials, startup, and wiring.

Sections follow the feature, not the history of its review: parsing and
rendering, the loop inside ``run_scenario``, text the harness stores, settings
and inheritance, credentials, the startup check, the service, and the
surfaces that label a paired run.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import httpx
import pytest

from tool_eval_bench.adapters.base import BackendAdapter, ChatCompletionResult, ProviderToolCall
from tool_eval_bench.application import service as service_module
from tool_eval_bench.application.reviewer import (
    benchmark_credentials,
    check_reviewer,
    mask_credentials_in_adapter_logs,
    reject_held_out,
    resolve_reviewer,
    reviewer_config,
    validate_reviewer,
)
from tool_eval_bench.application.service import BenchmarkService
from tool_eval_bench.domain.models import ChatMessage
from tool_eval_bench.domain.review import (
    ReviewerSettings,
    inheritable_extra_params,
    review_applies,
)
from tool_eval_bench.domain.scenarios import (
    Category,
    ScenarioDefinition,
    ScenarioEvaluation,
    ScenarioState,
    ScenarioStatus,
    ToolCallRecord,
)
from tool_eval_bench.runner.orchestrator import _wire_tool_choice, run_all_scenarios, run_scenario
from tool_eval_bench.runner.review import (
    TurnReviewer,
    credential_values,
    parse_verdict,
    render_review_request,
    revision_request,
    scrub,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _call(name: str, arguments: dict[str, Any], call_id: str = "c1") -> ProviderToolCall:
    return ProviderToolCall(id=call_id, name=name, arguments_str=json.dumps(arguments))


def _result(
    content: str = "",
    calls: list[ProviderToolCall] | None = None,
    *,
    finish_reason: str | None = None,
    **extra: Any,
) -> ChatCompletionResult:
    return ChatCompletionResult(
        content=content,
        tool_calls=list(calls or []),
        finish_reason=finish_reason or ("tool_calls" if calls else "stop"),
        **extra,
    )


class ScriptedAdapter(BackendAdapter):
    """Returns scripted results in order and records every request."""

    def __init__(self, steps: list[ChatCompletionResult | BaseException]) -> None:
        self.steps = list(steps)
        self.requests: list[dict[str, Any]] = []
        self.closed = False

    async def chat_completion(self, **kwargs: Any) -> ChatCompletionResult:
        self.requests.append({**kwargs, "messages": copy.deepcopy(kwargs["messages"])})
        if not self.steps:
            return _result("[exhausted]")
        step = self.steps.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step

    async def aclose(self) -> None:
        self.closed = True


def _verdict(verdict: str, feedback: str = "") -> ChatCompletionResult:
    return _result(
        json.dumps({"verdict": verdict, "feedback": feedback}),
        prompt_tokens=100,
        completion_tokens=10,
    )


def _resolved(**overrides: Any) -> ReviewerSettings:
    settings = {
        "model": "critic",
        "base_url": "http://reviewer.test",
        "temperature": 0.0,
        "timeout_seconds": 30.0,
        "wire_format": "openai",
        **overrides,
    }
    return ReviewerSettings(**settings)


def _handler(state: ScenarioState, call: ToolCallRecord) -> Any:
    return {"city": call.arguments.get("location"), "temperature": "22C"}


def _berlin_evaluator(state: ScenarioState) -> ScenarioEvaluation:
    locations = [call.arguments.get("location") for call in state.tool_calls]
    if locations == ["Berlin"]:
        return ScenarioEvaluation(ScenarioStatus.PASS, 2, "Asked for Berlin only.")
    return ScenarioEvaluation(ScenarioStatus.FAIL, 0, f"Called with {locations}.")


def _scenario(scenario_id: str = "REV-01", **overrides: Any) -> ScenarioDefinition:
    fields: dict[str, Any] = {
        "id": scenario_id,
        "title": "Weather in Berlin",
        "category": Category.A,
        "user_message": "What's the weather in Berlin?",
        "description": "Call get_weather for Berlin.",
        "handle_tool_call": _handler,
        "evaluate": _berlin_evaluator,
        **overrides,
    }
    return ScenarioDefinition(**fields)


SCENARIO = _scenario()
#: Passes whatever the model does; for tests about what the harness stores.
ANY_ANSWER = _scenario(
    "REV-02", evaluate=lambda state: ScenarioEvaluation(ScenarioStatus.PASS, 2, "ok")
)
HELD_OUT = _scenario("PACK-01", user_message="secret", held_out=True)
BERLIN = _call("get_weather", {"location": "Berlin"})
PARIS = _call("get_weather", {"location": "Paris"})


async def _run(main: ScriptedAdapter, reviewer: ScriptedAdapter | None, **kwargs: Any) -> Any:
    settings = kwargs.pop("settings", _resolved())
    return await run_scenario(
        main,
        model="m",
        base_url="http://bench.test",
        api_key=None,
        scenario=kwargs.pop("scenario", SCENARIO),
        reviewer=TurnReviewer(reviewer, settings) if reviewer is not None else None,
        **kwargs,
    )


RUN = {
    "base_url": "http://bench.test:8000",
    "api_key": "main-key",
    "wire_format": "anthropic",
    "extra_headers": {"X-Team": "bench"},
    "session_header": "X-Session",
    "temperature": 0.6,
    "timeout_seconds": 90.0,
    "seed": 3,
    "extra_params": {"top_p": 0.9},
}


def _capturing_adapter(adapter_cls: Any, sent: list[dict[str, Any]], status: int = 500) -> Any:
    """A real adapter whose requests land in ``sent`` and fail with ``status``."""

    def respond(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(status, json={"error": {"message": "stop here"}})

    adapter = adapter_cls()
    adapter._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    return adapter


# ---------------------------------------------------------------------------
# Verdict parsing, scope, and rendering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"verdict": "approve", "feedback": ""}', ("approve", "")),
        ('{"verdict": "REVISE", "feedback": " Use Celsius. "}', ("revise", "Use Celsius.")),
        ('```json\n{"verdict": "revise", "feedback": "x"}\n```', ("revise", "x")),
        ('Here is my verdict: {"verdict": "approved"} done.', ("approve", "")),
        ('{"verdict": "reject", "feedback": "wrong city"}', ("revise", "wrong city")),
        ("REVISE: ask for the city first", ("revise", "ask for the city first")),
        ("Approve.", ("approve", "")),
    ],
)
def test_parse_verdict_reads_json_and_keywords(text: str, expected: tuple[str, str]) -> None:
    assert parse_verdict(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "Looks fine to me",
        '{"feedback": "no verdict"}',
        '{"verdict": "maybe"}',
        "[" * 100_000 + "]" * 100_000,  # exhausts the JSON decoder's stack
    ],
    ids=["empty", "blank", "prose", "no-verdict-key", "unknown-verdict", "deeply-nested"],
)
def test_parse_verdict_returns_none_without_a_verdict(text: str) -> None:
    assert parse_verdict(text) is None


@pytest.mark.parametrize(
    ("scope", "tools", "answer", "expected"),
    [
        ("turns", True, False, True),
        ("turns", False, True, True),
        ("turns", False, False, False),
        ("tools", True, True, True),
        ("tools", False, True, False),
        ("final", False, True, True),
        ("final", True, True, False),
        ("final", False, False, False),
    ],
)
def test_review_scope(scope: str, tools: bool, answer: bool, expected: bool) -> None:
    assert review_applies(scope, has_tool_calls=tools, has_answer=answer) is expected


def test_review_request_omits_pressure_filler_and_shows_tools_and_draft() -> None:
    messages: list[ChatMessage] = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "FILLER-QUESTION"},
        {"role": "assistant", "content": "FILLER-ANSWER"},
        {"role": "user", "content": "Weather in Berlin?"},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Current weather",
                "parameters": {"type": "object"},
            },
        }
    ]

    text = render_review_request(messages, tools, _result(calls=[PARIS]), omitted_after_system=2)

    assert "FILLER" not in text
    assert "2 earlier messages of unrelated conversation are omitted" in text
    assert "You are helpful." in text
    assert "Weather in Berlin?" in text
    assert "- get_weather: Current weather" in text
    assert text.endswith('-> tool call: get_weather({"location": "Paris"})')


@pytest.mark.parametrize(
    ("tool_choice", "note"),
    [
        ("none", "Tool calls were disabled for this step"),
        ("required", "A tool call was required for this step."),
        (
            {"type": "function", "function": {"name": "get_weather"}},
            "required a call to get_weather",
        ),
        ("auto", None),
        ({"function": "lookup"}, None),  # malformed: no note, and no crash
    ],
    ids=["none", "required", "forced", "auto", "malformed"],
)
def test_the_reviewer_is_told_how_the_step_could_use_tools(
    tool_choice: Any, note: str | None
) -> None:
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
    messages: list[ChatMessage] = [{"role": "user", "content": "Weather?"}]

    text = render_review_request(messages, tools, _result("It is sunny."), tool_choice=tool_choice)

    if note is None:
        assert "for this step" not in text
    else:
        assert note in text


def test_revision_request_is_a_user_message_with_draft_and_feedback() -> None:
    draft = _result("Sending now.", [_call("send_email", {"to": "a@b.c"})])

    message = revision_request(draft, "Ask before sending.")

    assert message["role"] == "user"
    assert "NOT carried out" in message["content"]
    assert 'send_email({"to": "a@b.c"})' in message["content"]
    assert "Ask before sending." in message["content"]
    assert "do not mention it" in message["content"]


def test_turn_reviewer_requires_resolved_settings() -> None:
    with pytest.raises(ValueError, match="resolved"):
        TurnReviewer(ScriptedAdapter([]), ReviewerSettings(model="critic"))


# ---------------------------------------------------------------------------
# The review loop inside run_scenario
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_approved_turn_runs_as_proposed() -> None:
    main = ScriptedAdapter([_result(calls=[BERLIN]), _result("22C in Berlin.")])
    reviewer = ScriptedAdapter([_verdict("approve"), _verdict("approve")])

    result = await _run(main, reviewer)

    assert result.status == ScenarioStatus.PASS
    assert len(main.requests) == 2
    assert [r["tools"] for r in reviewer.requests] == [None, None]
    assert result.review["reviews"] == 2
    assert result.review["approved"] == 2
    assert result.review["prompt_tokens"] == 200
    assert "review_1.1=approve" in result.raw_log
    assert "rejected_draft" not in result.raw_log


@pytest.mark.asyncio
async def test_a_rejected_call_never_runs_and_the_revision_does() -> None:
    main = ScriptedAdapter(
        [
            _result(calls=[PARIS], prompt_tokens=5),
            _result(calls=[BERLIN], prompt_tokens=7),
            _result("22C in Berlin."),
        ]
    )
    reviewer = ScriptedAdapter(
        [_verdict("revise", "The user asked about Berlin."), _verdict("approve")]
    )

    result = await _run(main, reviewer)

    assert result.status == ScenarioStatus.PASS, result.summary
    assert result.tool_calls_made == ["get_weather(location=Berlin)"]
    # The revision request carries the draft and the feedback after the history.
    revision_messages = main.requests[1]["messages"]
    assert revision_messages[-1]["role"] == "user"
    assert "The user asked about Berlin." in revision_messages[-1]["content"]
    # The next turn's history holds the accepted call only, never the review.
    final_history = main.requests[2]["messages"]
    assert not any("independent reviewer" in str(m.get("content")) for m in final_history)
    assert [
        tc["function"]["arguments"] for m in final_history for tc in m.get("tool_calls", [])
    ] == ['{"location": "Berlin"}']
    assert result.prompt_tokens == 12
    assert result.review == {
        "reviews": 2,
        "approved": 1,
        "revise_requested": 1,
        "revised": 1,
        "reviewer_errors": 0,
        "revision_errors": 0,
        "prompt_tokens": 200,
        "completion_tokens": 20,
        "latency_ms": result.review["latency_ms"],
    }
    assert 'rejected_tool_call_1.1=get_weather {"location": "Paris"}' in result.raw_log
    assert "review_1.1=revise: The user asked about Berlin." in result.raw_log


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("max_rounds", "drafts", "verdicts", "reviews", "revised", "trace"),
    [
        # The revision after the last allowed review runs without another one.
        (1, [PARIS, BERLIN], ["Wrong city.", None], 1, 1, "rejected_tool_call_1.1="),
        (
            2,
            [PARIS, _call("get_weather", {"location": "Rome"}), BERLIN],
            ["Wrong.", "Still."],
            2,
            2,
            "rejected_tool_call_1.2=get_weather",
        ),
    ],
    ids=["one-round", "two-rounds"],
)
async def test_rounds_bound_the_reviews_per_turn(
    max_rounds: int,
    drafts: list[ProviderToolCall],
    verdicts: list[str | None],
    reviews: int,
    revised: int,
    trace: str,
) -> None:
    main = ScriptedAdapter([*(_result(calls=[call]) for call in drafts), _result("22C.")])
    reviewer = ScriptedAdapter(
        [_verdict("revise", text) if text else _verdict("approve") for text in verdicts]
        + [_verdict("approve")]
    )

    result = await _run(main, reviewer, settings=_resolved(max_rounds=max_rounds, scope="tools"))

    assert result.status == ScenarioStatus.PASS
    assert len(reviewer.requests) == reviews
    assert (result.review["reviews"], result.review["revised"]) == (reviews, revised)
    assert trace in result.raw_log


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "detail"),
    [
        (ConnectionError("token sk-live-456 rejected"), "ConnectionError"),
        (
            _result("Looks fine, Authorization: Bearer sk-live-123"),
            "no verdict in a 45-character reply",
        ),
        (_result("Authorization: Bearer sk-live-123", transport_error_status=401), "HTTP 401"),
        (_result("", transport_error_status=503), "HTTP 503"),
        (_result("[malformed response]", malformed=True), "malformed response"),
        (_result("{" * 100_000), "no verdict in a 100000-character reply"),
        (_result('[no content: {"verdict": "approve"}]'), "no reply"),
    ],
    ids=[
        "exception",
        "no-verdict",
        "http-401-echo",
        "http-503",
        "malformed",
        "parser-breaking",
        "placeholder-quoting-a-verdict",
    ],
)
async def test_a_reviewer_failure_lets_the_turn_run_and_stores_no_reply_text(
    failure: Any, detail: str
) -> None:
    main = ScriptedAdapter([_result(calls=[BERLIN]), _result("22C in Berlin.")])
    reviewer = ScriptedAdapter([failure])

    result = await _run(main, reviewer, settings=_resolved(scope="tools"))

    assert result.status == ScenarioStatus.PASS
    assert result.review["reviewer_errors"] == 1
    assert result.review["error_details"] == [detail]
    assert "review_1.1=error:" in result.raw_log
    assert "sk-live" not in json.dumps(result.to_dict())


def _cut_off(arguments: str) -> ChatCompletionResult:
    call = ProviderToolCall(id="c2", name="get_weather", arguments_str=arguments)
    return ChatCompletionResult(
        content="", tool_calls=[call], finish_reason="length", completion_tokens=900
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("revision", "reason"),
    [
        (_result("", transport_error_status=500), "HTTP 500"),
        (_result("[malformed]", malformed=True), "malformed response"),
        (RuntimeError("upstream echoed sk-live-777"), "RuntimeError"),
        (
            _result("", finish_reason="length", completion_tokens=900),
            "revision had no answer and no tool call (finish_reason=length)",
        ),
        (
            _result("", finish_reason="sk-live-finish-secret"),
            "revision had no answer and no tool call (finish_reason=other)",
        ),
        (_cut_off('{"loc'), "revision cut off its tool-call arguments at the token ceiling"),
        (_cut_off("[" * 100_000), "revision cut off its tool-call arguments at the token ceiling"),
        (
            _result(
                '<tool_call>{"name": "get_weather", "arguments": {"location": "Berlin"}}</tool_call>'
            ),
            "revision emitted tool-call markup the server did not parse",
        ),
    ],
    ids=[
        "http-500",
        "malformed",
        "exception",
        "empty-length-stop",
        "unknown-finish-reason",
        "cut-off-arguments",
        "deeply-nested-arguments",
        "unparsed-markup",
    ],
)
async def test_an_unusable_revision_keeps_the_working_draft(revision: Any, reason: str) -> None:
    main = ScriptedAdapter(
        [_result(calls=[BERLIN], completion_tokens=10), revision, _result("22C in Berlin.")]
    )
    reviewer = ScriptedAdapter([_verdict("revise", "Add units.")])

    result = await _run(main, reviewer, settings=_resolved(scope="tools"))

    assert result.status == ScenarioStatus.PASS, result.summary
    assert result.tool_calls_made == ["get_weather(location=Berlin)"]
    assert (result.review["revision_errors"], result.review["revised"]) == (1, 0)
    assert f"revision_error_1.1={reason}; kept the draft" in result.raw_log
    # The draft ran, so nothing is recorded as rejected, and no server text is kept.
    assert "rejected_" not in result.raw_log
    assert "sk-live" not in result.raw_log
    if isinstance(revision, ChatCompletionResult) and revision.completion_tokens:
        # A discarded revision still cost the benchmarked model its tokens.
        assert result.completion_tokens == 910


@pytest.mark.asyncio
async def test_a_revision_that_cannot_be_inspected_keeps_the_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tool_eval_bench.runner import orchestrator

    def explode(result: Any, tools: Any) -> str | None:
        raise RecursionError("deep")

    monkeypatch.setattr(orchestrator, "_unusable_revision", explode)
    main = ScriptedAdapter([_result(calls=[BERLIN]), _result(calls=[PARIS]), _result("22C.")])

    result = await _run(
        main, ScriptedAdapter([_verdict("revise", "x")]), settings=_resolved(scope="tools")
    )

    assert result.status == ScenarioStatus.PASS, result.summary
    assert "revision could not be inspected (RecursionError)" in result.raw_log


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scenario_choice", "extra_params", "used"),
    [
        # The wire says "none", the scenario "auto": the turn path would tag
        # the markup as unparsed, so the revision is not used.
        (None, {"tool_choice": "none"}, False),
        # The scenario itself switches tools off: the markup is an answer.
        ("none", None, True),
    ],
    ids=["wire-override", "scenario-none"],
)
async def test_revision_markup_is_judged_like_the_turn_path(
    scenario_choice: str | None, extra_params: dict[str, Any] | None, used: bool
) -> None:
    markup = '{"name": "get_weather", "arguments": {"location": "Berlin"}}'
    scenario = _scenario(
        "REV-05",
        evaluate=lambda state: ScenarioEvaluation(ScenarioStatus.PASS, 2, "ok"),
        tool_choice_override=scenario_choice,
    )
    main = ScriptedAdapter([_result("Which city do you mean?"), _result(markup)])
    reviewer = ScriptedAdapter([_verdict("revise", "Just look it up.")])

    result = await _run(
        main,
        reviewer,
        scenario=scenario,
        settings=_resolved(scope="final"),
        extra_params=extra_params,
    )

    if used:
        assert result.review["revised"] == 1
        assert f"final_answer={markup}" in result.raw_log
    else:
        assert "revision_error_1.1=revision emitted tool-call markup" in result.raw_log
        assert "final_answer=Which city do you mean?" in result.raw_log


@pytest.mark.asyncio
async def test_final_scope_reviews_only_replies_without_tool_calls() -> None:
    main = ScriptedAdapter(
        [_result(calls=[BERLIN]), _result("It is 22C."), _result("It is 22C in Berlin right now.")]
    )
    reviewer = ScriptedAdapter([_verdict("revise", "Name the city.")])

    result = await _run(main, reviewer, settings=_resolved(scope="final"))

    assert len(reviewer.requests) == 1
    assert (
        "Proposed next step (not carried out yet)\n\nIt is 22C."
        in reviewer.requests[0]["messages"][1]["content"]
    )
    assert result.review["revised"] == 1
    assert "final_answer=It is 22C in Berlin right now." in result.raw_log


@pytest.mark.asyncio
async def test_the_reviewer_never_sees_context_pressure_filler() -> None:
    filler: list[ChatMessage] = [
        {"role": "user", "content": "FILLER-1"},
        {"role": "assistant", "content": "FILLER-2"},
    ]
    main = ScriptedAdapter([_result("22C.")])
    reviewer = ScriptedAdapter([_verdict("approve")])

    await _run(main, reviewer, scenario=ANY_ANSWER, context_pressure_messages=filler)

    assert any("FILLER-1" in str(m["content"]) for m in main.requests[0]["messages"])
    assert "FILLER" not in reviewer.requests[0]["messages"][1]["content"]


@pytest.mark.asyncio
async def test_reviewer_requests_use_its_own_settings() -> None:
    main = ScriptedAdapter([_result("22C.")])
    reviewer = ScriptedAdapter([_verdict("approve")])
    settings = _resolved(
        api_key="rk",
        temperature=0.4,
        timeout_seconds=12.0,
        seed=9,
        extra_params={"top_p": 0.5},
        system_prompt="Custom reviewer.",
    )

    await _run(main, reviewer, settings=settings, seed=1, extra_params={"top_p": 0.9})

    request = reviewer.requests[0]
    assert request["model"] == "critic"
    assert request["base_url"] == "http://reviewer.test"
    assert request["api_key"] == "rk"
    assert request["temperature"] == 0.4
    assert request["timeout_seconds"] == 12.0
    assert request["extra_params"] == {"seed": 9, "top_p": 0.5}
    assert request["messages"][0] == {"role": "system", "content": "Custom reviewer."}
    assert request["conversation_id"] != main.requests[0]["conversation_id"]


@pytest.mark.asyncio
async def test_the_reviewer_sees_a_tool_choice_overridden_by_backend_kwargs() -> None:
    reviewer = ScriptedAdapter([_verdict("approve")])

    await _run(
        ScriptedAdapter([_result("It is sunny.")]), reviewer, extra_params={"tool_choice": "none"}
    )

    assert (
        "Tool calls were disabled for this step" in reviewer.requests[0]["messages"][1]["content"]
    )


def test_the_wire_tool_choice_follows_each_adapters_override() -> None:
    gemini_none = {"toolConfig": {"functionCallingConfig": {"mode": "NONE"}}}
    gemini_any = {"toolConfig": {"functionCallingConfig": {"mode": "ANY"}}}
    assert _wire_tool_choice({}, "auto") == "auto"
    assert _wire_tool_choice({"tool_choice": "none"}, "auto", "openai") == "none"
    assert _wire_tool_choice(gemini_none, "auto", "gemini") == "none"
    assert _wire_tool_choice(gemini_any, "auto", "gemini") == "required"
    # Each adapter honours only its own override; the Messages API drops both.
    assert _wire_tool_choice(gemini_none, "auto", "openai") == "auto"
    assert _wire_tool_choice({"tool_choice": "none"}, "auto", "gemini") == "auto"
    assert _wire_tool_choice({**gemini_none, "tool_choice": "none"}, "auto", "anthropic") == "auto"
    # Malformed overrides leave the scenario's choice, never crash.
    assert (
        _wire_tool_choice({"toolConfig": {"functionCallingConfig": "NONE"}}, "auto", "gemini")
        == "auto"
    )
    assert _wire_tool_choice({"toolConfig": "NONE"}, "auto", "gemini") == "auto"
    assert _wire_tool_choice({"tool_choice": 3}, "auto", "openai") == "auto"


@pytest.mark.asyncio
async def test_a_solo_run_records_no_review() -> None:
    result = await _run(ScriptedAdapter([_result(calls=[BERLIN]), _result("ok")]), None)

    assert result.review is None
    assert "review" not in result.to_dict()


@pytest.mark.asyncio
async def test_held_out_scenarios_never_reach_a_reviewer() -> None:
    main, reviewer = ScriptedAdapter([]), ScriptedAdapter([])

    with pytest.raises(ValueError, match="held-out"):
        reject_held_out([SCENARIO, HELD_OUT])
    reject_held_out([SCENARIO])
    with pytest.raises(ValueError, match="held-out"):
        await _run(main, reviewer, scenario=HELD_OUT)
    with pytest.raises(ValueError, match="held-out"):
        await run_all_scenarios(
            main,
            model="m",
            base_url="http://bench.test",
            scenarios=[SCENARIO, HELD_OUT],
            reviewer=TurnReviewer(reviewer, _resolved()),
        )
    assert main.requests == [] and reviewer.requests == []


# ---------------------------------------------------------------------------
# Text the harness stores and replays
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_malformed_usage_counts_never_fail_a_scenario() -> None:
    main = ScriptedAdapter(
        [
            _result(calls=[PARIS]),
            ChatCompletionResult(
                content="",
                tool_calls=[BERLIN],
                finish_reason="tool_calls",
                prompt_tokens="7",  # type: ignore[arg-type]
                completion_tokens=-3,
            ),
            _result("22C."),
        ]
    )
    reviewer = ScriptedAdapter(
        [
            ChatCompletionResult(
                content='{"verdict": "revise", "feedback": "Berlin."}',
                prompt_tokens="12",  # type: ignore[arg-type]
                completion_tokens=True,  # type: ignore[arg-type]
                finish_reason="stop",
            ),
            _verdict("approve"),
        ]
    )

    result = await _run(main, reviewer)

    assert result.status == ScenarioStatus.PASS, result.summary
    assert result.review["prompt_tokens"] == 100  # the approve verdict only


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("revision", "stored"),
    [
        (_result("Fixed \ud800 answer."), "final_answer=Fixed ? answer."),
        (
            ChatCompletionResult(
                content="",
                reasoning="thinking \ud800 hard",
                tool_calls=[
                    ProviderToolCall(
                        id="c2", name="get_weather", arguments_str='{"location": "Berlin\ud800"}'
                    )
                ],
                finish_reason="tool_calls",
            ),
            'get_weather {"location": "Berlin?"}',
        ),
        (
            ChatCompletionResult(
                content="",
                tool_calls=[
                    ProviderToolCall(
                        id="call\ud800",
                        name="get_weather",
                        arguments_str='{"location": "Berlin"}',
                        extra_content={"sig": "x\ud800"},
                    )
                ],
                message_extra_content={"meta": ["y\ud800"]},
                finish_reason="tool_calls",
            ),
            'get_weather {"location": "Berlin"}',
        ),
    ],
    ids=["content", "reasoning-and-arguments", "tool-id-and-metadata"],
)
async def test_an_unencodable_revision_is_stored_and_replayed_safely(
    revision: ChatCompletionResult, stored: str
) -> None:
    draft = _result(calls=[BERLIN]) if revision.tool_calls else _result("A draft.")
    main = ScriptedAdapter([draft, revision, _result("22C.")])
    # A revised call is graded like any call: "Berlin?" is not Berlin.
    scenario = SCENARIO if revision.tool_calls else ANY_ANSWER

    result = await _run(main, ScriptedAdapter([_verdict("revise", "x")]), scenario=scenario)

    assert result.review["revised"] == 1
    assert stored in result.raw_log
    result.raw_log.encode("utf-8")
    # Any request after the revision replays it; it must encode.
    json.dumps(main.requests[-1]["messages"], ensure_ascii=False).encode("utf-8")


# ---------------------------------------------------------------------------
# Settings: resolution, validation, inheritance, and the stored block
# ---------------------------------------------------------------------------


def test_a_reviewer_on_the_benchmarked_server_inherits_everything() -> None:
    resolved = resolve_reviewer(ReviewerSettings(model="critic"), **RUN)

    assert resolved.base_url == "http://bench.test:8000"
    assert resolved.api_key == "main-key"
    assert resolved.wire_format == "anthropic"
    assert resolved.extra_headers == {"X-Team": "bench"}
    assert resolved.session_header == "X-Session"
    assert (resolved.temperature, resolved.timeout_seconds, resolved.seed) == (0.6, 90.0, 3)
    assert resolved.extra_params == {"top_p": 0.9}


def test_a_reviewer_elsewhere_inherits_sampling_but_never_credentials() -> None:
    resolved = resolve_reviewer(
        ReviewerSettings(model="critic", base_url="http://other.test:8001", temperature=0.0),
        **RUN,
    )

    assert (resolved.api_key, resolved.extra_headers, resolved.session_header) == (None, None, None)
    assert resolved.wire_format == "openai"
    assert resolved.temperature == 0.0
    assert (resolved.timeout_seconds, resolved.seed) == (90.0, 3)
    assert resolved.extra_params == {"top_p": 0.9}


def test_a_reviewer_key_of_its_own_drops_the_inherited_authorization() -> None:
    resolved = resolve_reviewer(
        ReviewerSettings(model="critic", api_key="rk"),
        **{**RUN, "extra_headers": {"Authorization": "Bearer main-key", "X-Team": "bench"}},
    )

    assert (resolved.api_key, resolved.extra_headers) == ("rk", {"X-Team": "bench"})


def test_the_stored_block_holds_the_origin_and_format_but_no_secret() -> None:
    settings = ReviewerSettings(
        model="critic",
        base_url="https://user:pw@critic.test:8443/t/SECRET?k=1",
        api_key="secret-key",
        wire_format="anthropic",
        extra_headers={"Authorization": "Bearer t"},
    )
    resolved = resolve_reviewer(settings, **RUN)

    config = reviewer_config(resolved)

    assert config["base_url"] == "https://***:8443"
    assert (config["model"], config["scope"], config["wire_format"]) == (
        "critic",
        "turns",
        "anthropic",
    )
    for secret in ("secret-key", "pw", "SECRET", "Bearer"):
        assert secret not in json.dumps(config)
    assert all(secret not in repr(settings) for secret in ("secret-key", "pw@", "Bearer"))
    assert "critic" in repr(settings)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"model": " "}, "model"),
        ({"scope": "everything"}, "scope"),
        ({"max_rounds": 0}, "max_rounds"),
        ({"max_rounds": True}, "max_rounds"),
        ({"temperature": -1.0}, "temperature"),
        ({"timeout_seconds": 0}, "timeout_seconds"),
        ({"seed": 1.5}, "seed"),
        ({"base_url": "ftp://x"}, "base_url"),
        ({"extra_params": ["top_p"]}, "extra_params"),
        *(
            ({"extra_params": {key: "x"}}, key)
            for key in ("model", "messages", "system", "contents", "systemInstruction")
        ),
        *(
            ({"extra_params": value}, "must not carry credentials")
            for value in (
                {"api_key": "sk-x"},
                {"metadata": {"Authorization": "Bearer y"}},
                {"a": [{"token": 1}]},
                {"metadata": ({"api_key": "s"},)},
                *(
                    {"metadata": {key: "s"}}
                    for key in (
                        "client_secret",
                        "hf_token",
                        "X-Api-Key",
                        "db_password",
                        "X-Goog-Api-Key",
                        "accessToken",
                        "apiKey",
                        "clientSecret",
                        "refreshToken",
                        "Cookie",
                        "Set-Cookie",
                    )
                ),
            )
        ),
    ],
)
def test_invalid_reviewer_settings_are_rejected(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_reviewer(ReviewerSettings(**{"model": "critic", **overrides}))


@pytest.mark.parametrize(
    ("extra", "inherited"),
    [
        # Request structure, native tool settings, caches, and output schemas.
        ({"top_p": 0.9, "model": "other", "tools": []}, {"top_p": 0.9}),
        (
            {
                "toolConfig": {"functionCallingConfig": {"mode": "ANY"}},
                "output_config": {"format": {"type": "json_schema"}, "effort": "low"},
                "generationConfig": {"responseSchema": {}, "responseMimeType": "x", "topK": 4},
                "thinkingConfig": {"thinkingBudget": 128},
            },
            {
                "output_config": {"effort": "low"},
                "generationConfig": {"topK": 4},
                "thinkingConfig": {"thinkingBudget": 128},
            },
        ),
        (
            {"cachedContent": "c/1", "functions": [], "function_call": "auto", "top_p": 0.5},
            {"top_p": 0.5},
        ),
        # Credential-like names, at any depth and in any spelling.
        ({"api_key": "s", "Authorization": "t", "token": "u", "max_tokens": 5}, {"max_tokens": 5}),
        (
            {"metadata": {"api_key": "sk-x", "user": "u", "inner": [{"token": "t", "a": 1}]}},
            {"metadata": {"user": "u", "inner": [{"a": 1}]}},
        ),
        ({"metadata": ({"api_key": "s", "a": 1},)}, {"metadata": [{"a": 1}]}),
        *(
            ({"metadata": {key: "s", "userId": "u"}}, {"metadata": {"userId": "u"}})
            for key in (
                "client_secret",
                "hf_token",
                "X-Api-Key",
                "db_password",
                "X-Goog-Api-Key",
                "accessToken",
                "apiKey",
                "clientSecret",
                "refreshToken",
                "Cookie",
                "Set-Cookie",
                "cookie",
            )
        ),
        # Sampling keys that merely mention tokens stay.
        (
            {"max_tokens": 5, "max_completion_tokens": 6},
            {"max_tokens": 5, "max_completion_tokens": 6},
        ),
    ],
)
def test_inheritance_drops_request_structure_and_credentials(
    extra: dict[str, Any], inherited: dict[str, Any]
) -> None:
    assert inheritable_extra_params(extra) == inherited


def test_an_inherited_extra_params_filter_reaches_resolution() -> None:
    resolved = resolve_reviewer(
        ReviewerSettings(model="critic"),
        **{**RUN, "extra_params": {"top_p": 0.9, "model": "other", "tools": []}},
    )

    assert resolved.extra_params == {"top_p": 0.9}


@pytest.mark.parametrize("key", ["messages", "contents"])
def test_a_conversation_override_cannot_be_combined_with_a_reviewer(key: str) -> None:
    with pytest.raises(ValueError, match="replaces the conversation"):
        resolve_reviewer(ReviewerSettings(model="critic"), **{**RUN, "extra_params": {key: []}})


@pytest.mark.asyncio
async def test_the_api_refuses_a_conversation_override_before_any_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tool_eval_bench import api

    async def no_requests(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a request was sent")

    monkeypatch.setattr(api, "detect_backend", no_requests)
    monkeypatch.setattr(api, "build_run_context", no_requests)

    with pytest.raises(ValueError, match="replaces the conversation"):
        await api.run_benchmark(
            model="m",
            base_url="http://bench.test",
            scenarios=[SCENARIO],
            extra_params={"messages": []},
            reviewer=ReviewerSettings(model="critic"),
            persist=False,
        )


def test_an_explicit_reviewer_temperature_and_seed_beat_inherited_kwargs() -> None:
    resolved = resolve_reviewer(
        ReviewerSettings(model="critic", temperature=0.0, seed=7),
        **{**RUN, "extra_params": {"temperature": 0.9, "seed": 13, "top_p": 0.5}},
    )

    assert resolved.extra_params == {"top_p": 0.5}
    assert (resolved.temperature, resolved.seed) == (0.0, 7)


@pytest.mark.asyncio
async def test_an_explicit_reviewer_temperature_and_seed_reach_the_gemini_wire() -> None:
    from tool_eval_bench.adapters.gemini import GeminiAdapter

    resolved = resolve_reviewer(
        ReviewerSettings(
            model="gemini-critic",
            base_url="https://generativelanguage.googleapis.com",
            temperature=0.0,
            seed=7,
        ),
        **{
            **RUN,
            "extra_params": {"generationConfig": {"temperature": 0.9, "seed": 13, "topK": 4}},
        },
    )
    sent: list[dict[str, Any]] = []
    adapter = _capturing_adapter(GeminiAdapter, sent)
    try:
        await TurnReviewer(adapter, resolved).review(messages=[], tools=None, draft=_result("x"))
    finally:
        await adapter.aclose()

    config = sent[0]["generationConfig"]
    assert (config["temperature"], config["seed"], config["topK"]) == (0.0, 7, 4)


@pytest.mark.asyncio
async def test_an_explicit_reviewer_top_p_and_top_k_reach_the_gemini_wire() -> None:
    from tool_eval_bench.adapters.gemini import GeminiAdapter
    from tool_eval_bench.cli.legacy_parser import make_parser
    from tool_eval_bench.cli.reviewer import build_reviewer_settings, prepare_reviewer_args

    parser = make_parser()
    args = parser.parse_args(
        [
            "--reviewer-base-url",
            "https://generativelanguage.googleapis.com",
            "--reviewer-model",
            "gemini-critic",
            "--reviewer-top-p",
            "0.3",
            "--reviewer-top-k",
            "5",
        ]
    )
    prepare_reviewer_args(args, parser)
    settings = build_reviewer_settings(
        args,
        model="gemini-critic",
        main_base_url="http://bench.test:8000",
        main_api_key=None,
        main_headers=None,
        main_session_header=None,
        main_extra_params={"top_p": 0.8, "generationConfig": {"topP": 0.9, "topK": 40}},
    )
    resolved = resolve_reviewer(settings, **{**RUN, "extra_params": None})
    sent: list[dict[str, Any]] = []
    adapter = _capturing_adapter(GeminiAdapter, sent)
    try:
        await TurnReviewer(adapter, resolved).review(messages=[], tools=None, draft=_result("x"))
    finally:
        await adapter.aclose()

    config = sent[0]["generationConfig"]
    assert (config["topP"], config["topK"]) == (0.3, 5)


# ---------------------------------------------------------------------------
# Credentials: feedback, outbound transcripts, and adapter logs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("settings", "feedback", "secret", "masked"),
    [
        (
            {
                "api_key": "sk-critic-key-999",
                "extra_headers": {"Authorization": "Bearer sk-critic-key-999"},
            },
            "Gateway saw Bearer sk-critic-key-999; use Berlin.",
            "sk-critic-key-999",
            # The whole header value is the longest secret, so it is masked as one.
            "Gateway saw ***; use Berlin.",
        ),
        (
            {"base_url": "https://critic.test/v1?token=QUERYTOKEN99"},
            "My gateway token is QUERYTOKEN99; use Berlin.",
            "QUERYTOKEN99",
            None,
        ),
        (
            {"base_url": "https://critic.test/PATH_TOKEN_123/v1"},
            "Route PATH_TOKEN_123 says Berlin.",
            "PATH_TOKEN_123",
            None,
        ),
        (
            {"base_url": "https://critic.example/gateway/PATH_TOKEN_123/v1"},
            "Token PATH_TOKEN_123: Berlin.",
            "PATH_TOKEN_123",
            None,
        ),
        (
            {"base_url": "https://critic.example/gateway/Token%2Babc123/v1"},
            "Token+abc123 says Berlin.",
            "Token+abc123",
            None,
        ),
    ],
    ids=["key-and-header", "query-token", "path-token", "path-segment", "decoded-path"],
)
async def test_known_credentials_never_leave_the_reviewer_feedback(
    settings: dict[str, Any], feedback: str, secret: str, masked: str | None
) -> None:
    main = ScriptedAdapter([_result(calls=[PARIS]), _result(calls=[BERLIN]), _result("22C.")])
    reviewer = ScriptedAdapter([_verdict("revise", feedback), _verdict("approve")])

    result = await _run(main, reviewer, settings=_resolved(**settings))

    assert secret not in result.raw_log
    assert secret not in json.dumps(main.requests[1]["messages"])
    if masked is not None:
        assert masked in result.raw_log


@pytest.mark.asyncio
async def test_benchmark_credentials_never_reach_the_reviewer_host() -> None:
    reviewer_adapter = ScriptedAdapter([_verdict("approve")])
    reviewer = TurnReviewer(
        reviewer_adapter, _resolved(), benchmark_secrets=["sk-main-endpoint-key"]
    )
    messages: list[ChatMessage] = [
        {"role": "system", "content": "Key: sk-main-endpoint-key"},
        {"role": "user", "content": "hi"},
    ]

    await reviewer.review(messages=messages, tools=None, draft=_result("Echo sk-main-endpoint-key"))

    sent = json.dumps(reviewer_adapter.requests[0]["messages"])
    assert "sk-main-endpoint-key" not in sent
    assert sent.count("***") == 2


def test_the_service_hands_the_reviewer_every_known_benchmark_credential() -> None:
    secrets = benchmark_credentials(
        "https://user:pw-secret@gw.test/route/PATHTOKEN99/v1?key=QUERYKEY1",
        api_key="sk-main",
        headers={"Authorization": "Bearer hdr-secret-1"},
    )

    for value in ("sk-main", "pw-secret", "PATHTOKEN99", "QUERYKEY1", "hdr-secret-1"):
        assert value in secrets


def test_short_keys_and_truncated_tails_are_masked() -> None:
    assert credential_values("abcd", None) == ["abcd"]
    assert credential_values("abc", None) == []
    secrets = ["sk-reviewer-secret-123"]
    assert scrub("invalid token sk-reviewer-sec", secrets) == "invalid token ***"
    assert scrub("invalid token sk-revi", secrets) == "invalid token sk-revi"  # under 8 chars


def _real_reviewer(status: int, body: dict[str, Any] | str, **settings: Any) -> TurnReviewer:
    """A reviewer on the real OpenAI-compatible adapter, answering one fixed response."""
    from tool_eval_bench.adapters.openai_compat import OpenAICompatibleAdapter

    def respond(request: httpx.Request) -> httpx.Response:
        if isinstance(body, str):
            return httpx.Response(status, text=body)
        return httpx.Response(status, json=body)

    adapter = OpenAICompatibleAdapter()
    adapter._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    return TurnReviewer(adapter, _resolved(**{"base_url": "http://critic.test", **settings}))


@pytest.mark.asyncio
async def test_adapter_logs_mask_an_echoed_reviewer_key(caplog: pytest.LogCaptureFixture) -> None:
    reviewer = _real_reviewer(401, "invalid token sk-reviewer-secret-123")
    undo = mask_credentials_in_adapter_logs(
        _resolved(api_key="sk-reviewer-secret-123"), api_key=None, headers=None
    )
    try:
        with caplog.at_level("DEBUG"):
            await reviewer.review(messages=[], tools=None, draft=_result("x"))
    finally:
        undo()
        await reviewer.adapter.aclose()  # type: ignore[attr-defined]

    assert "401" in caplog.text
    assert "sk-reviewer-secret-123" not in caplog.text
    assert "***" in caplog.text


def test_httpx_request_logs_mask_a_reviewer_url_token(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    undo = mask_credentials_in_adapter_logs(
        _resolved(base_url="https://critic.example/v1?key=QUERYTOKEN77"), api_key=None, headers=None
    )
    try:
        with caplog.at_level("INFO"):
            logging.getLogger("httpx").info(
                'HTTP Request: %s %s "%s %d %s"',
                "POST",
                "https://critic.example/v1/chat/completions?key=QUERYTOKEN77",
                "HTTP/1.1",
                200,
                "OK",
            )
    finally:
        undo()

    assert "QUERYTOKEN77" not in caplog.text
    assert "?key=***" in caplog.text


def test_adapter_logs_mask_a_credential_path(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    undo = mask_credentials_in_adapter_logs(
        _resolved(base_url="https://critic.test/t/SECRET-PATH/v1"), api_key=None, headers=None
    )
    try:
        with caplog.at_level("WARNING"):
            logging.getLogger("tool_eval_bench.adapters.openai_compat").warning(
                "Stream request returned %d for %s", 401, "https://***/t/SECRET-PATH/v1/chat"
            )
    finally:
        undo()

    assert "SECRET-PATH" not in caplog.text
    assert "https://***/***/v1/chat" in caplog.text


# ---------------------------------------------------------------------------
# The startup check
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reviewer", "message"),
    [
        (
            lambda: _real_reviewer(401, {"error": {"message": "invalid api key"}}),
            "failed a test review .*HTTP 401",
        ),
        (
            lambda: _real_reviewer(404, {"error": {"message": "model critic not found"}}),
            "failed a test review .*HTTP 404",
        ),
        (lambda: TurnReviewer(ScriptedAdapter([_result("")]), _resolved()), "no reply"),
        (
            lambda: TurnReviewer(ScriptedAdapter([_result("[no content: SAFETY]")]), _resolved()),
            "no reply",
        ),
        (
            lambda: TurnReviewer(
                ScriptedAdapter([_result("[malformed response]", malformed=True)]), _resolved()
            ),
            "malformed response",
        ),
        (
            lambda: TurnReviewer(
                ScriptedAdapter([_result("upstream said no", transport_error_status=503)]),
                _resolved(),
            ),
            "HTTP 503",
        ),
        (
            lambda: TurnReviewer(
                ScriptedAdapter([_result("Looks fine."), _result("Still fine.")]), _resolved()
            ),
            "carry no verdict",
        ),
        # The error names the redacted origin only, never the path.
        (
            lambda: TurnReviewer(
                ScriptedAdapter([ConnectionError("down")]),
                _resolved(base_url="https://critic.test:8443/t/SECRET-PATH"),
            ),
            r"https://\*\*\*:8443 failed a test review",
        ),
    ],
    ids=[
        "http-401",
        "http-404",
        "empty",
        "gemini-placeholder",
        "malformed",
        "http-503-with-body",
        "never-a-verdict",
        "origin-only",
    ],
)
async def test_the_startup_check_refuses_a_reviewer_that_cannot_answer(
    reviewer: Any, message: str
) -> None:
    turn_reviewer = reviewer()
    try:
        with pytest.raises(ValueError, match=message) as exc_info:
            await check_reviewer(turn_reviewer)
    finally:
        await turn_reviewer.adapter.aclose()
    assert "SECRET-PATH" not in str(exc_info.value)


# ---------------------------------------------------------------------------
# The service
# ---------------------------------------------------------------------------


def _service(
    monkeypatch: pytest.MonkeyPatch, main: ScriptedAdapter, reviewer: ScriptedAdapter
) -> BenchmarkService:
    service = BenchmarkService(repo=None, reporter=None)
    monkeypatch.setattr(service, "_adapter_for", lambda *a, **k: main)
    built: list[dict[str, Any]] = []

    def build(base_url: str, **kwargs: Any) -> ScriptedAdapter:
        built.append({"base_url": base_url, **kwargs})
        return reviewer

    monkeypatch.setattr(service_module, "build_adapter", build)
    service.built_reviewers = built  # type: ignore[attr-defined]
    return service


async def _paired_run(service: BenchmarkService, **kwargs: Any) -> dict[str, Any]:
    return await service.run_benchmark(
        model="m",
        backend="vllm",
        base_url="http://bench.test:8000",
        scenarios=kwargs.pop("scenarios", [SCENARIO]),
        probe_engine=False,
        reviewer=kwargs.pop("reviewer", ReviewerSettings(model="critic")),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_the_service_runs_the_pair_and_stores_the_reviewer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    main = ScriptedAdapter([_result(calls=[BERLIN]), _result("22C.")])
    reviewer = ScriptedAdapter([_verdict("approve"), _verdict("approve"), _verdict("approve")])
    service = _service(monkeypatch, main, reviewer)

    run = await _paired_run(
        service,
        api_key="main-key",
        reviewer=ReviewerSettings(model="critic", base_url="http://other.test:8001"),
    )

    stored = run["config"]["reviewer"]
    assert stored["model"] == "critic"
    assert stored["endpoint_id"].startswith("endpoint:")
    assert run["scores"]["scenario_results"][0]["review"]["approved"] == 2
    # The startup review, then one review per turn.
    assert len(reviewer.requests) == 3
    assert reviewer.requests[0]["api_key"] is None
    assert service.built_reviewers[0]["base_url"] == "http://other.test:8001"  # type: ignore[attr-defined]
    assert main.closed and reviewer.closed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("startup", "message"),
    [
        (
            _result("", transport_error_status=404),
            r"failed a test review \(model 'critic'\): HTTP 404",
        ),
        (_result(""), "failed a test review"),
    ],
    ids=["http-404", "empty"],
)
async def test_a_failing_startup_review_stops_the_run_before_it_starts(
    monkeypatch: pytest.MonkeyPatch, startup: ChatCompletionResult, message: str
) -> None:
    main = ScriptedAdapter([])
    reviewer = ScriptedAdapter([startup])
    service = _service(monkeypatch, main, reviewer)

    with pytest.raises(ValueError, match=message):
        await _paired_run(service)

    assert main.requests == []
    assert main.closed and reviewer.closed


@pytest.mark.asyncio
async def test_a_rambling_startup_reply_retries_once_and_the_run_goes_on(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    main = ScriptedAdapter([_result("22C.")])
    reviewer = ScriptedAdapter([_result("Looks fine to me."), _verdict("approve")])
    service = _service(monkeypatch, main, reviewer)

    with caplog.at_level("WARNING"):
        run = await _paired_run(service)

    assert run["status"] == "completed"
    assert "without a verdict" in caplog.text
    # The startup review was retried once before the scenario's own request.
    assert len(reviewer.requests) >= 2


@pytest.mark.asyncio
async def test_a_resume_with_nothing_left_to_run_needs_no_reviewer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tool_eval_bench.domain.scenarios import ScenarioResult

    reviewer = ScriptedAdapter([ConnectionError("reviewer is down")])
    service = _service(monkeypatch, ScriptedAdapter([]), reviewer)
    done = ScenarioResult("REV-01", ScenarioStatus.PASS, 2, "ok", raw_log="trace").to_dict()

    run = await _paired_run(
        service,
        scenarios=[],
        resume_run_id="run-paired",
        resume_prior_results=[done],
        resume_scenarios=[SCENARIO],
    )

    assert run["status"] == "completed"
    assert run["config"]["reviewer"]["model"] == "critic"
    assert reviewer.requests == []


@pytest.mark.asyncio
async def test_the_service_refuses_a_reviewer_with_held_out_scenarios(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(monkeypatch, ScriptedAdapter([]), ScriptedAdapter([]))

    with pytest.raises(ValueError, match="held-out"):
        await _paired_run(service, scenarios=[HELD_OUT])


@pytest.mark.asyncio
async def test_the_markdown_report_summarizes_the_review(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    from tool_eval_bench.storage.reports import MarkdownReporter

    main = ScriptedAdapter([_result(calls=[PARIS]), _result(calls=[BERLIN]), _result("22C.")])
    reviewer = ScriptedAdapter(
        [
            _verdict("approve"),
            # A lone surrogate from a JSON escape must not break the UTF-8 report.
            _result('{"verdict": "revise", "feedback": "use Berlin \\ud800 now"}'),
            _verdict("approve"),
        ]
    )
    service = _service(monkeypatch, main, reviewer)
    service.reporter = MarkdownReporter(root=tmp_path)

    run = await _paired_run(service)

    assert run["status"] == "completed"
    report = next(tmp_path.rglob("*.md")).read_text(encoding="utf-8")
    assert "- **Reviewer**: `critic` (scope turns" in report
    assert "## Reviewer" in report
    assert "| REV-01 | pass | 2 | 1 | 1 | 1 | 0 |" in report
    assert "rejected_tool_call_1.1=get_weather" in report
    assert "review_1.1=revise: use Berlin ? now" in report


# ---------------------------------------------------------------------------
# Surfaces that label a paired run
# ---------------------------------------------------------------------------


def test_the_web_summary_names_the_reviewer() -> None:
    from tool_eval_bench.web.sanitize import sanitize_run_for_web

    paired = sanitize_run_for_web(
        {"run_id": "r", "model": "m", "config": {"reviewer": {"model": "critic"}}, "scores": {}}
    )
    solo = sanitize_run_for_web({"run_id": "r", "model": "m", "config": {}, "scores": {}})

    assert paired["reviewer_model"] == "critic"
    assert solo["reviewer_model"] is None


def test_every_card_wording_leads_with_the_reviewer_marker() -> None:
    from tool_eval_bench.domain.card import ToolEvalSummary
    from tool_eval_bench.storage.reports.card._data import paired_wordings, score_details
    from tool_eval_bench.storage.reports.card._svg import first_fit

    te = ToolEvalSummary(
        score=80.0,
        rating="Good",
        passed=3,
        partial=0,
        failed=1,
        total=4,
        suite_size=104,
        hardmode=True,
        reviewer="prism-ml/Ternary-Bonsai-2-27B-gguf:PQ2_0",
    )

    for wording in [*score_details(te), *paired_wordings(te, ["subset: 4 of 104"])]:
        assert wording.startswith("+ reviewer")
    # Even the end-truncated last resort keeps the marker.
    assert first_fit(score_details(te), 60, 14).startswith("+ rev")


@pytest.mark.asyncio
async def test_the_live_turn_payload_reports_the_drafts_ttft() -> None:
    payloads: list[dict[str, Any]] = []

    async def on_turn(scenario: Any, turn: int, payload: dict[str, Any]) -> None:
        payloads.append(payload)

    main = ScriptedAdapter(
        [
            _result(calls=[PARIS], ttft_ms=11.0),
            _result(calls=[BERLIN], ttft_ms=99.0),
            _result("22C."),
        ]
    )
    reviewer = ScriptedAdapter([_verdict("revise", "Berlin."), _verdict("approve")])

    result = await _run(main, reviewer, on_scenario_turn=on_turn)

    assert result.ttft_ms == 11.0
    assert payloads[0]["ttft_ms"] == 11.0


def test_a_reviewer_wire_format_typo_is_rejected_before_any_request() -> None:
    from tool_eval_bench.application.reviewer import validate_reviewer

    with pytest.raises(ValueError, match="reviewer wire_format must be one of"):
        validate_reviewer(ReviewerSettings(model="critic", wire_format="typo"))
