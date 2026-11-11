"""Tests for unparsed tool-call wire-hygiene detection, probes, display, and reports."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from tool_eval_bench.cli.display import (
    BenchmarkDisplay,
    _print_final_panel,
    print_final_report,
)
from tool_eval_bench.cli.probe import check_tool_call_wire_parsing
from tool_eval_bench.compare_reports.tool_eval import parse_md
from tool_eval_bench.domain.adapters import BackendAdapter, ChatCompletionResult, ProviderToolCall
from tool_eval_bench.domain.models import ChatMessage, RunContext
from tool_eval_bench.domain.scenarios import (
    INFRASTRUCTURE_FAILURE_KINDS,
    Category,
    CategoryScore,
    FailureKind,
    ModelScoreSummary,
    ScenarioDefinition,
    ScenarioEvaluation,
    ScenarioReportMetadata,
    ScenarioResult,
    ScenarioState,
    ScenarioStatus,
    ToolCallRecord,
    scenario_report_metadata,
)
from tool_eval_bench.domain.tools import UNIVERSAL_TOOLS
from tool_eval_bench.domain.wire_hygiene import (
    UnparsedToolCallMatch,
    advertised_tool_names,
    detect_unparsed_tool_call,
    dominant_parser_family,
    format_tool_parser_advice,
    infer_parser_family_from_text,
)
from tool_eval_bench.evals.scenarios import ALL_SCENARIOS
from tool_eval_bench.runner.orchestrator import (
    probe_tool_call_parsing,
    probe_tool_choice_required,
    run_scenario,
    score_results,
)
from tool_eval_bench.storage.reports import MarkdownReporter


class _ScriptedAdapter(BackendAdapter):
    """Deterministic scripted adapter for orchestrator and probe tests."""

    def __init__(self, responses: Sequence[ChatCompletionResult | Exception]) -> None:
        self._responses = list(responses)
        self.requests: list[dict[str, Any]] = []
        self.closed = False

    async def chat_completion(
        self,
        *,
        model: str,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = "auto",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout_seconds: float = 120.0,
        api_key: str | None = None,
        base_url: str = "",
        extra_params: dict[str, Any] | None = None,
        stream: bool = False,
        response_format: dict[str, Any] | None = None,
        parallel_tool_calls: bool | None = True,
        conversation_id: str | None = None,
    ) -> ChatCompletionResult:
        self.requests.append(
            {
                "model": model,
                "messages": messages,
                "tools": tools,
                "tool_choice": tool_choice,
                "max_tokens": max_tokens,
                "extra_params": extra_params,
                "stream": stream,
            }
        )
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def aclose(self) -> None:
        self.closed = True


def _tool_scenario(
    scenario_id: str = "WH-01",
    *,
    held_out: bool = False,
    tools_override: list[dict[str, Any]] | None = None,
    verdict: ScenarioEvaluation | None = None,
    follow_up_messages: list[str] | None = None,
) -> ScenarioDefinition:
    def _handle(state: ScenarioState, call: ToolCallRecord) -> dict[str, Any]:
        return {"ok": True}

    def _evaluate(state: ScenarioState) -> ScenarioEvaluation:
        if verdict is not None:
            return ScenarioEvaluation(
                status=verdict.status,
                points=verdict.points,
                summary=verdict.summary,
                note=verdict.note,
                failure_kind=verdict.failure_kind,
            )
        if not state.tool_calls:
            return ScenarioEvaluation(ScenarioStatus.FAIL, 0, "Expected a tool call.")
        return ScenarioEvaluation(ScenarioStatus.PASS, 2, "Called tool.")

    return ScenarioDefinition(
        id=scenario_id,
        title=f"Scenario {scenario_id}",
        category=Category.A,
        user_message="Check the weather in Tokyo.",
        description="Call get_weather for Tokyo.",
        handle_tool_call=_handle,
        evaluate=_evaluate,
        tools_override=tools_override,
        held_out=held_out,
        follow_up_messages=list(follow_up_messages or []),
    )


def _run_context(*, backend: str = "vllm", engine_name: str | None = "vLLM") -> RunContext:
    return RunContext(
        tool_version="0.1.0",
        git_sha="deadbeef",
        hostname="bench-host",
        platform_info="Linux",
        python_version="3.11.9",
        model="Qwen/Qwen3-32B",
        backend=backend,
        base_url="http://localhost:8000/v1",
        engine_name=engine_name,
    )


# ---------------------------------------------------------------------------
# Domain contract: FailureKind & scoring invariants
# ---------------------------------------------------------------------------


def test_unparsed_tool_call_is_not_an_infrastructure_failure() -> None:
    assert FailureKind.UNPARSED_TOOL_CALL == "unparsed_tool_call"
    assert FailureKind.UNPARSED_TOOL_CALL not in INFRASTRUCTURE_FAILURE_KINDS

    res = ScenarioResult(
        scenario_id="TC-01",
        status=ScenarioStatus.FAIL,
        points=0,
        summary="Expected get_weather call.",
        failure_kind=FailureKind.UNPARSED_TOOL_CALL,
    )
    assert res.is_infrastructure_failure is False


def test_unparsed_tool_call_remains_in_denominator_when_scoring() -> None:
    sc_no_tool = _tool_scenario("WH-PASS", tools_override=[])
    sc_broken_tool = _tool_scenario("WH-BROKEN")
    results = [
        ScenarioResult(
            scenario_id="WH-PASS",
            status=ScenarioStatus.PASS,
            points=2,
            summary="Answered directly.",
        ),
        ScenarioResult(
            scenario_id="WH-BROKEN",
            status=ScenarioStatus.FAIL,
            points=0,
            summary="Expected get_weather call.",
            failure_kind=FailureKind.UNPARSED_TOOL_CALL,
        ),
    ]
    summary = score_results(results, [sc_no_tool, sc_broken_tool])
    assert summary.excluded_scenarios == []
    assert summary.total_points == 2
    assert summary.max_points == 4
    assert summary.final_score == 50


# ---------------------------------------------------------------------------
# Pure detector: positive chat-template & raw-JSON cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("content", "expected_marker", "expected_family"),
    [
        (
            '<tool_call>\n{"name": "get_weather", "arguments": {"location": "Tokyo"}}\n</tool_call>',
            "<tool_call>",
            "hermes/qwen",
        ),
        (
            '<tool_call>\n{"name": "get_weather", "arguments": {"loc',
            "<tool_call>",
            "hermes/qwen",
        ),
        (
            '<tool_calls>\n[{"name": "get_weather", "arguments": {"location": "Tokyo"}}]\n</tool_calls>',
            "<tool_calls>",
            "hermes/qwen",
        ),
        (
            '<|tool_call|>{"name": "get_weather", "arguments": {"location": "Tokyo"}}',
            "<|tool_call|>",
            "hermes/qwen",
        ),
        (
            '```tool_call\n{"name": "get_weather", "arguments": {"location": "Tokyo"}}\n```',
            "```tool_call",
            "hermes/qwen",
        ),
        (
            "<tool_call>\n<function=get_weather>\n<parameter=location>Tokyo</parameter>\n</function>\n</tool_call>",
            "<function=...>",
            "qwen3_coder",
        ),
        (
            "<function=get_weather><parameter=location>Tokyo</parameter></function>",
            "<function=...>",
            "qwen3_coder",
        ),
        (
            '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"location": "Tokyo"}}]',
            "[TOOL_CALLS]",
            "mistral",
        ),
        (
            '<|python_tag|>{"name": "get_weather", "parameters": {"location": "Tokyo"}}',
            "<|python_tag|>",
            "llama3",
        ),
        (
            "<|tool_calls_begin|><|tool_call_begin|>function<|tool_sep|>get_weather\n"
            '{"location": "Tokyo"}<|tool_call_end|><|tool_calls_end|>',
            "<|tool_calls_begin|>",
            "deepseek",
        ),
        (
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
            '{"location": "Tokyo"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
            "<｜tool▁calls▁begin｜>",
            "deepseek",
        ),
    ],
)
def test_detect_unparsed_tool_call_template_markers(
    content: str,
    expected_marker: str,
    expected_family: str,
) -> None:
    match = detect_unparsed_tool_call(content, UNIVERSAL_TOOLS)
    assert match == UnparsedToolCallMatch(marker=expected_marker, parser_family=expected_family)


@pytest.mark.parametrize(
    "content",
    [
        '{"name": "get_weather", "arguments": {"location": "Tokyo"}}',
        '{"name": "get_weather", "parameters": {"location": "Tokyo"}}',
        '{"type": "function", "name": "get_weather", "arguments": "{\\"location\\": \\"Tokyo\\"}"}',
        '{"type": "function", "function": {"name": "get_weather", "arguments": {"location": "Tokyo"}}}',
        '{"function": {"name": "get_weather", "parameters": {"location": "Tokyo"}}}',
        '[{"name": "get_weather", "arguments": {"location": "Tokyo"}}]',
        '{"tool_call": {"name": "get_weather", "arguments": {"location": "Tokyo"}}}',
        '{"tool_calls": [{"name": "get_weather", "arguments": {"location": "Tokyo"}}]}',
        'Prefix {"broken_json": [} then {"name": "get_weather", "arguments": {"location": "Tokyo"}}',
        'Let me check that.\n```json\n{"name": "get_weather", "arguments": {"location": "Tokyo"}}\n```',
        '<think>I will call get_weather.</think>\n{"name": "get_weather", "arguments": {"location": "Tokyo"}}',
    ],
)
def test_detect_unparsed_tool_call_raw_json_envelopes(content: str) -> None:
    match = detect_unparsed_tool_call(content, UNIVERSAL_TOOLS)
    assert match == UnparsedToolCallMatch(
        marker='raw JSON call to "get_weather"',
        parser_family="raw_json",
        tool_name="get_weather",
    )
    assert match is not None
    assert match.display_marker(held_out=False) == 'raw JSON call to "get_weather"'
    assert match.display_marker(held_out=True) == "raw JSON tool call"


# ---------------------------------------------------------------------------
# Pure detector: negative cases & structured-output scenarios TC-64..TC-69
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "content",
    [
        None,
        "",
        "   ",
        "I cannot call get_weather because no city was specified.",
        "You can use `get_weather` or `web_search` to check live conditions.",
        "Calling `get_weather(location='Berlin')` is not needed here.",
        '{"name": "not_an_advertised_tool", "arguments": {"x": 1}}',
        '{"name": "get_weather", "role": "viewer"}',
        (
            '{"type": "function", "function": {"name": "get_weather", '
            '"description": "Get weather", '
            '"parameters": {"type": "object", "properties": {"location": {"type": "string"}}}}}'
        ),
        (
            '<think>Should I emit {"name": "get_weather", "arguments": {"location": "Tokyo"}}? '
            "No, the user asked a math question.</think>\nThe answer is 42."
        ),
    ],
)
def test_detect_unparsed_tool_call_rejects_prose_and_non_call_json(content: str | None) -> None:
    assert detect_unparsed_tool_call(content, UNIVERSAL_TOOLS) is None


def test_detect_unparsed_tool_call_returns_none_when_no_tools_are_advertised() -> None:
    raw = '<tool_call>{"name": "get_weather", "arguments": {"location": "Tokyo"}}</tool_call>'
    assert detect_unparsed_tool_call(raw, []) is None
    assert detect_unparsed_tool_call(raw, None) is None
    assert advertised_tool_names([]) == set()
    assert advertised_tool_names([{"name": "flat_tool"}]) == {"flat_tool"}


def test_detect_unparsed_tool_call_is_linear_on_repeated_unclosed_tags() -> None:
    for payload in (
        "<tool_call>abc " * 4000,
        "<tool_calls>abc " * 4000,
        "<think>abc " * 4000,
    ):
        t0 = time.perf_counter()
        match = detect_unparsed_tool_call(payload, UNIVERSAL_TOOLS)
        elapsed = time.perf_counter() - t0
        assert match is None
        assert elapsed < 0.1

    nested = '<tool_calls>abc <tool_calls>  <tool_call>{"name": "get_weather"}'
    assert detect_unparsed_tool_call(nested, UNIVERSAL_TOOLS) == UnparsedToolCallMatch(
        marker="<tool_call>",
        parser_family="hermes/qwen",
    )
    unclosed_plural = '<tool_calls>abc <tool_calls> [{"name": "get_weather"}]'
    assert detect_unparsed_tool_call(unclosed_plural, UNIVERSAL_TOOLS) == UnparsedToolCallMatch(
        marker="<tool_calls>",
        parser_family="hermes/qwen",
    )


@pytest.mark.parametrize(
    ("scenario_id", "sample_json_output"),
    [
        (
            "TC-64",
            json.dumps(
                {
                    "title": "The Matrix",
                    "year": 1999,
                    "rating": 9.2,
                    "genre": "sci-fi",
                    "summary": "A programmer discovers reality is a simulation.",
                }
            ),
        ),
        (
            "TC-65",
            json.dumps(
                {
                    "location": "Tokyo",
                    "temperature_celsius": 28,
                    "condition": "Sunny",
                    "recommendation": "Wear breathable clothing.",
                }
            ),
        ),
        (
            "TC-66",
            json.dumps(
                {
                    "query": "engineering",
                    "total": 2,
                    "contacts": [
                        {
                            "name": "Alice Zhang",
                            "email": "alice.zhang@company.com",
                            "department": "Engineering",
                        },
                        {
                            "name": "Carol Singh",
                            "email": "carol.singh@company.com",
                            "department": "Engineering",
                        },
                    ],
                }
            ),
        ),
        (
            "TC-67",
            json.dumps(
                {
                    "ticker": "NVDA",
                    "price": 892.5,
                    "currency": "USD",
                    "signal": "buy",
                    "reasoning": "Strong data center revenue growth supports the rating.",
                }
            ),
        ),
        (
            "TC-68",
            "```json\n"
            + json.dumps(
                {
                    "task_id": "PROJ-127",
                    "status": "in_progress",
                    "assignee": "me",
                }
            )
            + "\n```",
        ),
        (
            "TC-69",
            json.dumps(
                {
                    "date": "2025-03-15",
                    "weather": {
                        "location": "San Francisco",
                        "temperature": 16,
                        "condition": "Partly Cloudy",
                    },
                    "market": {
                        "ticker": "AAPL",
                        "price": 212.4,
                        "direction": "down",
                    },
                    "action_items": ["Pack a light jacket"],
                }
            ),
        ),
    ],
)
def test_structured_output_scenarios_tc64_to_tc69_never_false_positive(
    scenario_id: str,
    sample_json_output: str,
) -> None:
    by_id = {sc.id: sc for sc in ALL_SCENARIOS}
    scenario = by_id[scenario_id]
    tools = UNIVERSAL_TOOLS if scenario.tools_override is None else scenario.tools_override
    assert detect_unparsed_tool_call(sample_json_output, tools) is None


# ---------------------------------------------------------------------------
# Server flag advice & note parser-family inference
# ---------------------------------------------------------------------------


def test_format_tool_parser_advice_across_engines_and_families() -> None:
    vllm_qwen = format_tool_parser_advice(backend="vllm", parser_family="hermes/qwen")
    assert "--enable-auto-tool-choice --tool-call-parser hermes" in vllm_qwen
    assert "vLLM" in vllm_qwen

    sglang_coder = format_tool_parser_advice(engine_name="SGLang", parser_family="qwen3_coder")
    assert "--tool-call-parser qwen3_coder" in sglang_coder
    assert "SGLang" in sglang_coder

    llamacpp = format_tool_parser_advice(backend="llamacpp", parser_family="mistral")
    assert "--jinja" in llamacpp
    assert "llama.cpp" in llamacpp

    generic_deepseek = format_tool_parser_advice(parser_family="deepseek")
    assert "--enable-auto-tool-choice --tool-call-parser deepseek_v3" in generic_deepseek
    assert "--tool-call-parser deepseek_v3 on SGLang" in generic_deepseek
    assert "--jinja on llama.cpp" in generic_deepseek

    generic_unknown = format_tool_parser_advice(backend="unknown", parser_family="raw_json")
    assert "--tool-call-parser <parser>" in generic_unknown

    for text in (vllm_qwen, sglang_coder, llamacpp, generic_deepseek, generic_unknown):
        assert "—" not in text
        assert "–" not in text
        assert "(" not in text
        assert ")" not in text


def test_infer_and_dominant_parser_family_helpers() -> None:
    assert infer_parser_family_from_text(None) is None
    assert infer_parser_family_from_text("plain note") is None
    assert infer_parser_family_from_text("markup <tool_call> in content") == "hermes/qwen"
    assert infer_parser_family_from_text("markup <function=...> in content") == "qwen3_coder"
    assert infer_parser_family_from_text("markup [TOOL_CALLS] in content") == "mistral"
    assert infer_parser_family_from_text("markup <|python_tag|> in content") == "llama3"
    assert infer_parser_family_from_text("markup <|tool_calls_begin|> in content") == "deepseek"
    assert infer_parser_family_from_text('unparsed raw JSON call to "get_weather"') == "raw_json"
    assert infer_parser_family_from_text("unparsed raw JSON tool call in content") == "raw_json"

    assert (
        dominant_parser_family(
            [
                "Turn 1 emitted unparsed tool-call markup <tool_call> in content",
                'Turn 1 emitted unparsed raw JSON call to "get_weather" in content',
            ]
        )
        == "hermes/qwen"
    )
    assert dominant_parser_family(["unparsed raw JSON tool call in content"]) == "raw_json"
    assert (
        dominant_parser_family(
            [
                "Turn 1 emitted unparsed tool-call markup <tool_call> in content",
                "Turn 1 emitted unparsed tool-call markup [TOOL_CALLS] in content",
            ]
        )
        is None
    )


# ---------------------------------------------------------------------------
# Orchestrator integration: run_scenario, held-out safety, probes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_scenario_classifies_unparsed_tool_call_and_records_trace_and_note() -> None:
    scenario = _tool_scenario("WH-ORCH-1")
    adapter = _ScriptedAdapter(
        [
            ChatCompletionResult(
                content='<tool_call>\n{"name": "get_weather", "arguments": {"location": "Tokyo"}}\n</tool_call>',
                finish_reason="stop",
            )
        ]
    )
    result = await run_scenario(
        adapter,
        model="qwen-broken",
        base_url="http://localhost:8000/v1",
        api_key=None,
        scenario=scenario,
    )
    assert result.status == ScenarioStatus.FAIL
    assert result.failure_kind == FailureKind.UNPARSED_TOOL_CALL
    assert result.note is not None
    assert "Turn 1 emitted unparsed tool-call markup <tool_call> in content" in result.note
    assert (
        "unparsed_tool_call=Turn 1 emitted unparsed tool-call markup <tool_call>" in result.raw_log
    )


@pytest.mark.asyncio
async def test_run_scenario_redacts_held_out_tool_name_in_unparsed_marker() -> None:
    secret_tool_name = "secret_internal_ledger_lookup"
    held_out_tools = [
        {
            "type": "function",
            "function": {
                "name": secret_tool_name,
                "description": "Held-out tool",
                "parameters": {"type": "object", "properties": {"id": {"type": "string"}}},
            },
        }
    ]
    scenario = _tool_scenario("HO-99", held_out=True, tools_override=held_out_tools)
    adapter = _ScriptedAdapter(
        [
            ChatCompletionResult(
                content=json.dumps({"name": secret_tool_name, "arguments": {"id": "42"}}),
                finish_reason="stop",
            )
        ]
    )
    result = await run_scenario(
        adapter,
        model="raw-json-model",
        base_url="http://localhost:8000/v1",
        api_key=None,
        scenario=scenario,
    )
    assert result.failure_kind == FailureKind.UNPARSED_TOOL_CALL
    assert result.note is not None
    assert "raw JSON tool call" in result.note
    assert secret_tool_name not in result.note
    assert "unparsed_tool_call=Turn 1 emitted unparsed raw JSON tool call" in result.raw_log


@pytest.mark.asyncio
async def test_run_scenario_does_not_override_passing_result_with_unparsed_failure_kind() -> None:
    scenario = _tool_scenario(
        "WH-PASS-FOLLOWUP",
        verdict=ScenarioEvaluation(ScenarioStatus.PASS, 2, "Recovered on follow-up."),
        follow_up_messages=["Please call the tool properly."],
    )
    adapter = _ScriptedAdapter(
        [
            ChatCompletionResult(
                content='<tool_call>{"name": "get_weather", "arguments": {"location": "Tokyo"}}</tool_call>',
                finish_reason="stop",
            ),
            ChatCompletionResult(content="Done.", finish_reason="stop"),
        ]
    )
    result = await run_scenario(
        adapter,
        model="recovering-model",
        base_url="http://localhost:8000/v1",
        api_key=None,
        scenario=scenario,
    )
    assert result.status == ScenarioStatus.PASS
    assert result.failure_kind is None
    assert result.note is None
    assert "unparsed_tool_call=Turn 1" in result.raw_log


@pytest.mark.asyncio
async def test_run_scenario_skips_unparsed_detection_when_tool_choice_is_none() -> None:
    scenario = ScenarioDefinition(
        id="WH-NONE-01",
        title="Tool choice none restraint",
        category=Category.D,
        user_message="Do not call tools.",
        description="Answer without tools.",
        handle_tool_call=lambda state, call: {},
        evaluate=lambda state: ScenarioEvaluation(
            ScenarioStatus.FAIL,
            0,
            "Disobeyed tool_choice=none constraint.",
            failure_kind=FailureKind.FORBIDDEN_ACTION,
        ),
        tool_choice_override="none",
    )
    adapter = _ScriptedAdapter(
        [
            ChatCompletionResult(
                content='<tool_call>{"name": "get_weather", "arguments": {"location": "Tokyo"}}</tool_call>',
                finish_reason="stop",
            )
        ]
    )
    result = await run_scenario(
        adapter,
        model="disobedient-model",
        base_url="http://localhost:8000/v1",
        api_key=None,
        scenario=scenario,
    )
    assert result.failure_kind == FailureKind.FORBIDDEN_ACTION
    assert "unparsed_tool_call=" not in result.raw_log
    assert result.note is None


@pytest.mark.asyncio
async def test_probe_tool_choice_required_reports_unparsed_markup_instead_of_prose() -> None:
    adapter = _ScriptedAdapter(
        [
            ChatCompletionResult(
                content='<tool_call>\n{"name": "probe_ping", "arguments": {"value": "ok"}}\n</tool_call>',
                finish_reason="stop",
            )
        ]
    )
    probe = await probe_tool_choice_required(
        adapter,
        model="m",
        base_url="http://localhost:8000/v1",
        api_key=None,
        timeout_seconds=5.0,
    )
    assert probe.enforced is False
    assert probe.arguments_intact is False
    assert "<tool_call>" in probe.detail
    assert "--enable-auto-tool-choice" in probe.detail
    assert "--tool-call-parser hermes" in probe.detail
    assert "answered in prose" not in probe.detail

    raw_json_probe = await probe_tool_choice_required(
        _ScriptedAdapter(
            [
                ChatCompletionResult(
                    content='{"name": "probe_ping", "arguments": {"value": "ok"}}',
                    finish_reason="stop",
                )
            ]
        ),
        model="m",
        base_url="http://localhost:8000/v1",
        api_key=None,
        timeout_seconds=5.0,
    )
    assert raw_json_probe.enforced is False
    assert 'emitted unparsed raw JSON call to "probe_ping" in content' in raw_json_probe.detail
    assert "markup raw JSON" not in raw_json_probe.detail
    assert "raw raw" not in raw_json_probe.detail


@pytest.mark.asyncio
async def test_probe_tool_call_parsing_covers_parsed_unparsed_and_prose() -> None:
    ok_adapter = _ScriptedAdapter(
        [
            ChatCompletionResult(
                content="",
                tool_calls=[
                    ProviderToolCall(
                        id="call_1",
                        name="probe_ping",
                        arguments_str='{"value": "ok"}',
                    )
                ],
                finish_reason="tool_calls",
            )
        ]
    )
    ok_probe = await probe_tool_call_parsing(
        ok_adapter,
        model="m",
        base_url="http://localhost:8000/v1",
        api_key=None,
        timeout_seconds=5.0,
    )
    assert ok_probe.parsed is True
    assert ok_probe.enforced is True
    assert ok_probe.arguments_intact is True
    assert ok_adapter.requests[0]["tool_choice"] == "auto"

    raw_adapter = _ScriptedAdapter(
        [
            ChatCompletionResult(
                content='[TOOL_CALLS] [{"name": "probe_ping", "arguments": {"value": "ok"}}]',
                finish_reason="stop",
            )
        ]
    )
    raw_probe = await probe_tool_call_parsing(
        raw_adapter,
        model="m",
        base_url="http://localhost:8000/v1",
        api_key=None,
        timeout_seconds=5.0,
    )
    assert raw_probe.parsed is False
    assert raw_probe.enforced is False
    assert raw_probe.arguments_intact is False
    assert raw_probe.unparsed_marker == "[TOOL_CALLS]"
    assert raw_probe.parser_family == "mistral"
    assert "--tool-call-parser mistral" in raw_probe.detail

    raw_json_wire = await probe_tool_call_parsing(
        _ScriptedAdapter(
            [
                ChatCompletionResult(
                    content='{"name": "probe_ping", "arguments": {"value": "ok"}}',
                    finish_reason="stop",
                )
            ]
        ),
        model="m",
        base_url="http://localhost:8000/v1",
        api_key=None,
        timeout_seconds=5.0,
    )
    assert raw_json_wire.parsed is False
    assert raw_json_wire.parser_family == "raw_json"
    assert 'the endpoint emitted raw JSON call to "probe_ping" in content' in raw_json_wire.detail
    assert "raw raw" not in raw_json_wire.detail

    retry_adapter = _ScriptedAdapter(
        [
            ChatCompletionResult(content="", reasoning="thinking", finish_reason="length"),
            ChatCompletionResult(
                content="",
                tool_calls=[ProviderToolCall(id="c2", name="probe_ping", arguments_str="{}")],
                finish_reason="tool_calls",
            ),
        ]
    )
    retry_probe = await probe_tool_call_parsing(
        retry_adapter,
        model="m",
        base_url="http://localhost:8000/v1",
        api_key=None,
        timeout_seconds=5.0,
        extra_params={
            "max_tokens": 32,
            "generationConfig": {"maxOutputTokens": 32, "topK": 1},
        },
    )
    assert [req["max_tokens"] for req in retry_adapter.requests] == [256, 4096]
    assert retry_adapter.requests[0]["extra_params"] == {"generationConfig": {"topK": 1}}
    assert retry_probe.parsed is True
    assert retry_probe.arguments_intact is False
    assert "empty arguments" in retry_probe.detail

    for resp, expected_fragment in (
        (
            [ChatCompletionResult(content="bad", transport_error_status=400)],
            "rejected with HTTP 400",
        ),
        (
            [ChatCompletionResult(content="[malformed response]", malformed=True)],
            "not valid JSON",
        ),
        (
            [
                ChatCompletionResult(content="partial", finish_reason="length"),
                ChatCompletionResult(content="partial", finish_reason="length"),
            ],
            "leaving partial text",
        ),
        (
            [ChatCompletionResult(content="I decline to call tools.", finish_reason="stop")],
            "answered in prose",
        ),
        (
            [ChatCompletionResult(content="", finish_reason="stop")],
            "no tool call or answer",
        ),
        (
            [RuntimeError("boom")],
            "probe request failed with boom",
        ),
    ):
        edge_probe = await probe_tool_call_parsing(
            _ScriptedAdapter(resp),
            model="m",
            base_url="http://localhost:8000/v1",
            api_key=None,
            timeout_seconds=5.0,
        )
        assert edge_probe.parsed is False
        assert expected_fragment in edge_probe.detail


# ---------------------------------------------------------------------------
# CLI probe helper, live display banner, final panel, and Markdown report
# ---------------------------------------------------------------------------


def test_check_tool_call_wire_parsing_console_headless_and_exit(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buf = StringIO()
    console = Console(file=buf, width=200, no_color=True)
    broken_adapter = _ScriptedAdapter(
        [
            ChatCompletionResult(
                content="<function=probe_ping><parameter=value>ok</parameter></function>",
                finish_reason="stop",
            )
        ]
    )
    probe = check_tool_call_wire_parsing(
        console,
        "http://localhost:8000/v1",
        "qwen3-coder",
        None,
        backend="vllm",
        adapter=broken_adapter,
    )
    assert probe.parsed is False
    rendered = buf.getvalue()
    assert "<function=...>" in rendered
    assert "--enable-auto-tool-choice --tool-call-parser qwen3_coder" in rendered
    assert broken_adapter.closed is False

    owned_ok = _ScriptedAdapter(
        [
            ChatCompletionResult(
                content="",
                tool_calls=[
                    ProviderToolCall(id="c1", name="probe_ping", arguments_str='{"value": "ok"}')
                ],
                finish_reason="tool_calls",
            )
        ]
    )
    monkeypatch.setattr("tool_eval_bench.cli.probe.build_adapter", lambda *a, **kw: owned_ok)
    ok_buf = StringIO()
    ok_probe = check_tool_call_wire_parsing(
        Console(file=ok_buf, width=200, no_color=True),
        "http://localhost:8000/v1",
        "qwen3",
        None,
    )
    assert ok_probe.parsed is True
    assert owned_ok.closed is True
    assert "Tool-call wire parsing verified" in ok_buf.getvalue()

    prose_buf = StringIO()
    prose_adapter = _ScriptedAdapter(
        [
            ChatCompletionResult(
                content="Error from http://secret.internal:8000/v1",
                finish_reason="stop",
            )
        ]
    )
    prose_probe = check_tool_call_wire_parsing(
        Console(file=prose_buf, width=200, no_color=True),
        "http://secret.internal:8000/v1",
        "qwen3",
        None,
        backend="llamacpp",
        display_url="http://***:8000/v1",
        adapter=prose_adapter,
    )
    assert prose_probe.parsed is False
    assert "secret.internal" not in prose_buf.getvalue()
    assert "--jinja" in prose_buf.getvalue()

    headless_adapter = _ScriptedAdapter(
        [
            ChatCompletionResult(
                content='<|python_tag|>{"name": "probe_ping", "parameters": {"value": "ok"}}',
                finish_reason="stop",
            )
        ]
    )
    with pytest.raises(SystemExit) as exc_info:
        check_tool_call_wire_parsing(
            console,
            "http://secret-host.internal:8000/v1",
            "llama-3.1",
            None,
            backend="sglang",
            headless=True,
            adapter=headless_adapter,
            exit_on_failure=True,
        )
    assert exc_info.value.code == 1
    stderr_line = capsys.readouterr().err.strip()
    payload = json.loads(stderr_line)
    assert payload["event"] == "tool_call_probe"
    assert payload["status"] == "failed"
    assert payload["unparsed_marker"] == "<|python_tag|>"
    assert "--tool-call-parser llama3_json" in payload["advice"]


@pytest.mark.asyncio
async def test_display_prints_one_time_live_warning_and_final_panel_summary() -> None:
    sc1 = _tool_scenario("TC-01")
    sc2 = _tool_scenario("TC-02")
    ctx = _run_context(backend="vllm", engine_name="vLLM")
    display = BenchmarkDisplay(
        model="Qwen/Qwen3-32B",
        backend="vllm",
        base_url="http://localhost:8000/v1",
        scenarios=[sc1, sc2],
        run_context=ctx,
    )
    buf = StringIO()
    display.console = Console(file=buf, width=220, no_color=True)
    assert display._warned_unparsed_tool_call is False

    note = (
        "Turn 1 emitted unparsed tool-call markup [TOOL_CALLS] in content "
        "instead of structured tool_calls."
    )
    r1 = ScenarioResult(
        scenario_id="TC-01",
        status=ScenarioStatus.FAIL,
        points=0,
        summary="Expected tool call.",
        note=note,
        failure_kind=FailureKind.UNPARSED_TOOL_CALL,
    )
    r2 = ScenarioResult(
        scenario_id="TC-02",
        status=ScenarioStatus.FAIL,
        points=0,
        summary="Expected tool call.",
        note=note,
        failure_kind=FailureKind.UNPARSED_TOOL_CALL,
    )

    await display.on_scenario_result(sc1, r1, 0, 2)
    await display.on_scenario_result(sc2, r2, 1, 2)

    assert display._warned_unparsed_tool_call is True
    live_text = buf.getvalue()
    assert live_text.count("raw tool-call markup arrived in content instead of tool_calls") == 1
    assert "TC-01: raw tool-call markup arrived in content instead of tool_calls." in live_text
    assert "--enable-auto-tool-choice --tool-call-parser mistral" in live_text

    panel_buf = StringIO()
    panel_console = Console(file=panel_buf, width=220, no_color=True)
    summary = ModelScoreSummary(
        scenario_results=[r1, r2],
        category_scores=[CategoryScore(Category.A, "Tool Selection", 0, 4, 0.0, fail_count=2)],
        final_score=0,
        total_points=0,
        max_points=4,
        rating="★ Poor",
    )
    _print_final_panel(panel_console, "Qwen/Qwen3-32B", summary, 3.2, run_context=ctx)
    panel_text = panel_buf.getvalue()
    assert "2 scenarios failed with unparsed_tool_call" in panel_text
    assert "--enable-auto-tool-choice --tool-call-parser mistral" in panel_text

    report_buf = StringIO()
    report_console = Console(file=report_buf, width=220, no_color=True)
    print_final_report(report_console, "Qwen/Qwen3-32B", summary, 3.2, run_context=ctx)
    assert "2 scenarios failed with unparsed_tool_call" in report_buf.getvalue()


def test_markdown_report_renders_unparsed_warning_and_preserves_safety_parser(
    tmp_path: Path,
) -> None:
    sc1 = _tool_scenario("TC-01")
    ctx = _run_context(backend="sglang", engine_name="SGLang")
    r1 = ScenarioResult(
        scenario_id="TC-01",
        status=ScenarioStatus.FAIL,
        points=0,
        summary="Expected get_weather call.",
        note=(
            "Turn 1 emitted unparsed tool-call markup <tool_call> in content "
            "instead of structured tool_calls."
        ),
        raw_log="model=Qwen\nverdict=fail",
        failure_kind=FailureKind.UNPARSED_TOOL_CALL,
    )
    summary = ModelScoreSummary(
        scenario_results=[r1],
        category_scores=[CategoryScore(Category.A, "Tool Selection", 0, 2, 0.0, fail_count=1)],
        final_score=0,
        total_points=0,
        max_points=2,
        rating="★ Poor",
    )
    reporter = MarkdownReporter(root=str(tmp_path))
    report_path = reporter.write_scenario_report(
        "run_wh_01",
        "Qwen/Qwen3-32B",
        summary,
        run_context=ctx,
        scenario_metadata=scenario_report_metadata([sc1]),
    )
    content = report_path.read_text(encoding="utf-8")
    assert "> [!WARNING]" in content
    assert "1 `unparsed_tool_call` failure detected:" in content
    assert "--tool-call-parser hermes" in content
    parsed = parse_md(str(report_path))
    assert parsed["safety_critical_count"] == 0
    assert parsed["safety_critical"] == []

    summary_with_safety = ModelScoreSummary(
        scenario_results=[r1],
        category_scores=[CategoryScore(Category.A, "Tool Selection", 0, 2, 0.0, fail_count=1)],
        final_score=0,
        total_points=0,
        max_points=2,
        rating="★ Poor (safety-capped)",
        safety_warnings=["TC-34 (Prompt Injection): Obeyed injected instruction"],
    )
    report_path_2 = reporter.write_scenario_report(
        "run_wh_02",
        "Qwen/Qwen3-32B",
        summary_with_safety,
        run_context=ctx,
        scenario_metadata={
            "TC-01": ScenarioReportMetadata("Scenario TC-01", Category.A, difficulty=1)
        },
    )
    content_2 = report_path_2.read_text(encoding="utf-8")
    assert content_2.count("> [!WARNING]") == 1
    assert "1 `unparsed_tool_call` failure detected:" in content_2
    parsed_2 = parse_md(str(report_path_2))
    assert parsed_2["safety_critical_count"] == 1
    assert parsed_2["safety_critical"][0]["id"] == "TC-34"
