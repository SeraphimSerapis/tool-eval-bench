"""Tests for chat-template replay probes, special-token extraction, and leak detection."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable, Sequence
from io import StringIO
from pathlib import Path
from typing import Any

import httpx
import pytest
from rich.console import Console

from tool_eval_bench.cli.probe import check_chat_template_fidelity
from tool_eval_bench.domain.adapters import BackendAdapter, ChatCompletionResult, ProviderToolCall
from tool_eval_bench.domain.models import ChatMessage
from tool_eval_bench.domain.wire_hygiene import (
    BUILTIN_TEMPLATE_LEAK_TOKENS,
    UnparsedToolCallMatch,
    detect_parser_marker,
    detect_special_token_leak,
    is_control_special_token,
)
from tool_eval_bench.runner.template_probe import (
    ChatTemplateProbe,
    probe_chat_template_fidelity,
)
from tool_eval_bench.utils.tokenizers import TokenizerResolution, extract_special_tokens


class _ScriptedAdapter(BackendAdapter):
    """Deterministic adapter for chat-template probe tests."""

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
                "temperature": temperature,
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


def _ok_responses() -> list[ChatCompletionResult]:
    return [
        ChatCompletionResult(content="ALPHA-7942", finish_reason="stop"),
        ChatCompletionResult(content="ALPHA-7942", finish_reason="stop"),
        ChatCompletionResult(content="COBALT-41 and AMBER-88", finish_reason="stop"),
        ChatCompletionResult(content="COBALT-41, AMBER-88", finish_reason="stop"),
    ]


# ---------------------------------------------------------------------------
# Special-token filter and leak detection
# ---------------------------------------------------------------------------


def test_builtin_template_leak_tokens_all_pass_control_token_filter() -> None:
    assert BUILTIN_TEMPLATE_LEAK_TOKENS
    for token in BUILTIN_TEMPLATE_LEAK_TOKENS:
        assert is_control_special_token(token), f"expected control token: {token!r}"


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("<|im_start|>", True),
        ("<|im_end|>", True),
        ("<|eot_id|>", True),
        ("<|tool_call|>", True),
        ("<|tool_call>", True),
        ("<tool_call|>", True),
        ("<channel|>", True),
        ("<turn|>", True),
        ("<|turn>", True),
        ("<|channel>", True),
        ('<|"|>', True),
        ("<｜end▁of▁sentence｜>", True),
        ("<start_of_turn>", True),
        ("<end_of_turn>", True),
        ("<eos>", True),
        ("<bos>", True),
        ("[INST]", True),
        ("[/INST]", True),
        ("[TOOL_CALLS]", True),
        ("[AVAILABLE_TOOLS]", True),
        ("", False),
        ("   ", False),
        ("a", False),
        ("<|>", False),
        ("<s>", False),
        ("</s>", False),
        ("<p>", False),
        ("</p>", False),
        ("<b>", False),
        ("<div>", False),
        ("</div>", False),
        ("<span>", False),
        ("<code>", False),
        ("<table>", False),
        ("<details>", False),
        ("[1]", False),
        ("[x]", False),
        ("[OK]", False),
        ("[link]", False),
        ("assistant", False),
        ("system", False),
        ("<|bad token|>", False),
    ],
)
def test_is_control_special_token_classification(token: str, expected: bool) -> None:
    assert is_control_special_token(token) is expected


def test_detect_special_token_leak_filters_benign_tokens_and_closed_think() -> None:
    tokens = [
        "",
        "a",
        "system",
        "<div>",
        "</div>",
        "</s>",
        "[x]",
        "[link]",
        "<think>",
        "</think>",
        "<|im_end|>",
        "<turn|>",
        "<|eot_id|>",
    ]
    assert detect_special_token_leak(None, tokens) == []
    assert detect_special_token_leak("", tokens) == []
    assert (
        detect_special_token_leak(
            "<div>Hello system [x] [link] </s></div><think>internal</think>ALPHA-7942",
            tokens,
        )
        == []
    )
    assert detect_special_token_leak(
        "<think>reasoning</think>ALPHA-7942<turn|><|im_end|><turn|>",
        tokens,
    ) == ["<turn|>", "<|im_end|>"]
    # Stray unclosed </think> outside a closed pair is still flagged.
    assert detect_special_token_leak("reasoning</think>\nALPHA-7942", tokens) == ["</think>"]


def test_detect_parser_marker_works_without_advertised_tools() -> None:
    assert detect_parser_marker(None) is None
    assert detect_parser_marker("ALPHA-7942") is None
    assert detect_parser_marker("<think><|tool_call></think>ALPHA-7942") is None
    assert detect_parser_marker("<|tool_call>call:get_code{}<tool_call|>") == UnparsedToolCallMatch(
        marker="<|tool_call>",
        parser_family="gemma4",
    )
    assert detect_parser_marker("call:get_code{}") == UnparsedToolCallMatch(
        marker="call:<name>{...}",
        parser_family="gemma4",
    )
    assert detect_parser_marker("<channel|>thought") == UnparsedToolCallMatch(
        marker="<channel|>",
        parser_family="gemma4",
    )
    assert detect_parser_marker("<turn|>ALPHA-7942") == UnparsedToolCallMatch(
        marker="<turn|>",
        parser_family="gemma4",
    )


# ---------------------------------------------------------------------------
# Tokenizer special-token extraction (utils/tokenizers.py)
# ---------------------------------------------------------------------------


def test_extract_special_tokens_reads_tokenizer_and_special_map(tmp_path: Path) -> None:
    tok_json = {
        "added_tokens": [
            {"id": 0, "content": "<|im_start|>", "special": True},
            {"id": 1, "content": "<|im_end|>", "special": True},
            {"id": 2, "content": "<|tool_call>", "special": True},
            {"id": 3, "content": "<channel|>", "special": True},
            {"id": 4, "content": "<turn|>", "special": True},
            {"id": 5, "content": "<start_of_turn>", "special": True},
            {"id": 6, "content": "<end_of_turn>", "special": True},
            {"id": 7, "content": "[INST]", "special": True},
            {"id": 8, "content": "[/INST]", "special": True},
            {"id": 9, "content": "[TOOL_CALLS]", "special": True},
            # Non-special added token must be ignored.
            {"id": 10, "content": "<|non_special|>", "special": False},
            # Short / plain / HTML tokens marked special must be filtered out.
            {"id": 11, "content": "<s>", "special": True},
            {"id": 12, "content": "</s>", "special": True},
            {"id": 13, "content": "<div>", "special": True},
            {"id": 14, "content": "assistant", "special": True},
            {"id": 15, "content": "x", "special": True},
        ]
    }
    special_map = {
        "eos_token": "<|eot_id|>",
        "bos_token": {"content": "<|begin_of_text|>"},
        "pad_token": "<s>",
        "additional_special_tokens": [
            "<|custom_stop|>",
            {"content": "<|tool_call|>"},
            "plain_word",
            123,
        ],
    }
    tok_file = tmp_path / "tokenizer.json"
    tok_file.write_text(json.dumps(tok_json), encoding="utf-8")
    (tmp_path / "special_tokens_map.json").write_text(json.dumps(special_map), encoding="utf-8")

    expected = frozenset(
        {
            "<|im_start|>",
            "<|im_end|>",
            "<|tool_call>",
            "<channel|>",
            "<turn|>",
            "<start_of_turn>",
            "<end_of_turn>",
            "[INST]",
            "[/INST]",
            "[TOOL_CALLS]",
            "<|eot_id|>",
            "<|begin_of_text|>",
            "<|custom_stop|>",
            "<|tool_call|>",
        }
    )
    assert extract_special_tokens(tok_file) == expected
    assert extract_special_tokens(tmp_path) == expected


def test_extract_special_tokens_degrades_gracefully_on_missing_or_corrupt_files(
    tmp_path: Path,
) -> None:
    assert extract_special_tokens("") == frozenset()
    assert extract_special_tokens(tmp_path / "does_not_exist.json") == frozenset()

    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not valid json", encoding="utf-8")
    assert extract_special_tokens(corrupt) == frozenset()

    non_dict = tmp_path / "list.json"
    non_dict.write_text("[1, 2, 3]", encoding="utf-8")
    assert extract_special_tokens(non_dict) == frozenset()

    # Directory containing only special_tokens_map.json still extracts tokens.
    only_map_dir = tmp_path / "map_only"
    only_map_dir.mkdir()
    (only_map_dir / "special_tokens_map.json").write_text(
        json.dumps({"eos_token": "<|custom_eos|>"}),
        encoding="utf-8",
    )
    assert extract_special_tokens(only_map_dir) == frozenset({"<|custom_eos|>"})


# ---------------------------------------------------------------------------
# runner/template_probe.py
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_probe_chat_template_fidelity_passes_and_sends_four_expected_requests() -> None:
    adapter = _ScriptedAdapter(_ok_responses())
    probe = await probe_chat_template_fidelity(
        adapter,
        model="gemma-4-31b",
        base_url="http://localhost:8000/v1",
        api_key="k",
        timeout_seconds=10.0,
        extra_params={
            "max_tokens": 16,
            "generationConfig": {"maxOutputTokens": 16, "topK": 1},
            "chat_template_kwargs": {"enable_thinking": False},
        },
        special_tokens={"<|im_end|>", "<|custom_tok|>", "ignored_word"},
    )

    assert probe.passed is True
    assert probe.diagnosis == "ok"
    assert probe.diagnoses == ()
    assert probe.leaked_tokens == ()
    assert probe.tokenizer_tokens_checked == 2
    assert [check.name for check in probe.checks] == [
        "tool_replay",
        "tool_replay_twin",
        "multi_turn",
        "multi_turn_twin",
    ]
    assert all(check.passed and check.content_matched for check in probe.checks)
    assert probe.check_by_name("tool_replay") is probe.checks[0]
    assert probe.check_by_name("nonexistent") is None
    assert probe.to_dict()["diagnosis"] == "ok"

    assert len(adapter.requests) == 4
    # First request replays user -> assistant(tool_calls) -> tool(result) with get_code tool.
    tool_req = adapter.requests[0]
    assert [m["role"] for m in tool_req["messages"]] == ["user", "assistant", "tool"]
    assert tool_req["tools"] is not None
    assert tool_req["tool_choice"] == "auto"
    assert tool_req["temperature"] == 0.0
    assert tool_req["max_tokens"] == 128
    assert tool_req["stream"] is False
    assert tool_req["extra_params"] == {
        "generationConfig": {"topK": 1},
        "chat_template_kwargs": {"enable_thinking": False},
    }

    # Twin has 1 user message, no tools.
    assert [m["role"] for m in adapter.requests[1]["messages"]] == ["user"]
    assert adapter.requests[1]["tools"] is None

    # Multi-turn has 5 alternating user/assistant/user/assistant/user messages.
    assert [m["role"] for m in adapter.requests[2]["messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert [m["role"] for m in adapter.requests[3]["messages"]] == ["user"]


@pytest.mark.asyncio
async def test_probe_diagnoses_tool_replay_template_error() -> None:
    adapter = _ScriptedAdapter(
        [
            # Model re-calls get_code because server Jinja template dropped tool result.
            ChatCompletionResult(
                content="",
                tool_calls=[ProviderToolCall(id="c2", name="get_code", arguments_str="{}")],
                finish_reason="tool_calls",
            ),
            ChatCompletionResult(content="ALPHA-7942", finish_reason="stop"),
            ChatCompletionResult(content="COBALT-41, AMBER-88", finish_reason="stop"),
            ChatCompletionResult(content="COBALT-41, AMBER-88", finish_reason="stop"),
        ]
    )
    probe = await probe_chat_template_fidelity(
        adapter,
        model="m",
        base_url="http://localhost:8000/v1",
        api_key=None,
    )
    assert probe.passed is False
    assert probe.diagnosis == "tool_replay_template_error"
    assert probe.diagnoses == ("tool_replay_template_error",)
    tool_check = probe.check_by_name("tool_replay")
    assert tool_check is not None
    assert tool_check.passed is False
    assert "re-called tool (get_code)" in tool_check.detail


@pytest.mark.asyncio
async def test_probe_diagnoses_multi_turn_template_error() -> None:
    adapter = _ScriptedAdapter(
        [
            ChatCompletionResult(content="ALPHA-7942", finish_reason="stop"),
            ChatCompletionResult(content="ALPHA-7942", finish_reason="stop"),
            ChatCompletionResult(content="Only AMBER-88", finish_reason="stop"),
            ChatCompletionResult(content="COBALT-41 and AMBER-88", finish_reason="stop"),
        ]
    )
    probe = await probe_chat_template_fidelity(
        adapter,
        model="m",
        base_url="http://localhost:8000/v1",
        api_key=None,
    )
    assert probe.passed is False
    assert probe.diagnosis == "multi_turn_template_error"
    assert probe.diagnoses == ("multi_turn_template_error",)
    mt_check = probe.check_by_name("multi_turn")
    assert mt_check is not None
    assert "missing expected COBALT-41" in mt_check.detail


@pytest.mark.asyncio
async def test_probe_diagnoses_template_token_leak_from_tokenizer_and_gemma4_markers() -> None:
    # Dynamic tokenizer special token leaked on all replies while content matches.
    adapter = _ScriptedAdapter(
        [
            ChatCompletionResult(content="ALPHA-7942<|custom_eot|>", finish_reason="stop"),
            ChatCompletionResult(content="ALPHA-7942<|custom_eot|>", finish_reason="stop"),
            ChatCompletionResult(content="COBALT-41, AMBER-88", finish_reason="stop"),
            ChatCompletionResult(content="COBALT-41, AMBER-88", finish_reason="stop"),
        ]
    )
    probe = await probe_chat_template_fidelity(
        adapter,
        model="m",
        base_url="http://localhost:8000/v1",
        api_key=None,
        special_tokens={"<|custom_eot|>"},
    )
    assert probe.passed is False
    assert probe.diagnosis == "template_token_leak"
    assert probe.leaked_tokens == ("<|custom_eot|>",)
    assert probe.parser_family is None

    # Built-in Gemma 4 marker leak caught even without a local tokenizer.
    gemma_adapter = _ScriptedAdapter(
        [
            ChatCompletionResult(content="<channel|>ALPHA-7942<turn|>", finish_reason="stop"),
            ChatCompletionResult(content="ALPHA-7942", finish_reason="stop"),
            ChatCompletionResult(content="COBALT-41 AMBER-88", finish_reason="stop"),
            ChatCompletionResult(content="COBALT-41 AMBER-88", finish_reason="stop"),
        ]
    )
    gemma_probe = await probe_chat_template_fidelity(
        gemma_adapter,
        model="gemma4",
        base_url="http://localhost:8000/v1",
        api_key=None,
    )
    assert gemma_probe.passed is False
    assert gemma_probe.diagnosis == "template_token_leak"
    assert gemma_probe.leaked_tokens == ("<channel|>", "<turn|>")
    assert gemma_probe.parser_family == "gemma4"
    assert gemma_probe.unparsed_marker == "<channel|>"


@pytest.mark.asyncio
async def test_probe_diagnoses_model_or_endpoint_failure_and_handles_errors() -> None:
    adapter = _ScriptedAdapter(
        [
            ChatCompletionResult(content="bad", transport_error_status=500),
            ChatCompletionResult(content="bad", transport_error_status=500),
            ChatCompletionResult(content="[malformed]", malformed=True),
            RuntimeError("connection reset"),
        ]
    )
    probe = await probe_chat_template_fidelity(
        adapter,
        model="m",
        base_url="http://localhost:8000/v1",
        api_key=None,
    )
    assert probe.passed is False
    assert probe.diagnosis == "model_or_endpoint_failure"
    assert probe.checks[0].error == "request rejected with HTTP 500"
    assert probe.checks[2].error == "response was not valid JSON"
    assert probe.checks[3].error == "probe request failed: connection reset"
    serialized = probe.to_dict()
    assert serialized["checks"][0]["error"] == "request rejected with HTTP 500"


@pytest.mark.asyncio
async def test_probe_redacts_url_credentials_in_check_detail_and_error() -> None:
    adapter = _ScriptedAdapter(
        [
            RuntimeError(
                "Connection failed to https://user:secret@host.internal:8443/v1/chat?api_key=TOPSECRET"
            ),
            ChatCompletionResult(content="ALPHA-7942", finish_reason="stop"),
            ChatCompletionResult(content="COBALT-41 and AMBER-88", finish_reason="stop"),
            ChatCompletionResult(content="COBALT-41 and AMBER-88", finish_reason="stop"),
        ]
    )
    probe = await probe_chat_template_fidelity(
        adapter,
        model="m",
        base_url="https://user:secret@host.internal:8443/v1?api_key=TOPSECRET",
        api_key=None,
    )
    failed = probe.check_by_name("tool_replay")
    assert failed is not None
    assert failed.passed is False
    assert failed.error is not None
    for field_text in (failed.detail, failed.error):
        assert "probe request failed:" in field_text
        assert "secret" not in field_text
        assert "TOPSECRET" not in field_text
        assert "user:" not in field_text
        assert "host.internal" not in field_text
    dumped = json.dumps(probe.to_dict())
    assert "secret" not in dumped
    assert "TOPSECRET" not in dumped


@pytest.mark.asyncio
async def test_probe_retries_once_on_reasoning_truncation_and_allows_closed_think() -> None:
    adapter = _ScriptedAdapter(
        [
            # First check exhausts compact budget thinking, then succeeds on retry.
            ChatCompletionResult(content="", reasoning="thinking...", finish_reason="length"),
            ChatCompletionResult(
                content="<think>The tool returned ALPHA-7942.</think>\nALPHA-7942",
                finish_reason="stop",
            ),
            ChatCompletionResult(content="ALPHA-7942", finish_reason="stop"),
            ChatCompletionResult(content="COBALT-41 and AMBER-88", finish_reason="stop"),
            ChatCompletionResult(content="COBALT-41 and AMBER-88", finish_reason="stop"),
        ]
    )
    probe = await probe_chat_template_fidelity(
        adapter,
        model="thinking-model",
        base_url="http://localhost:8000/v1",
        api_key=None,
        special_tokens={"<think>", "</think>"},
    )
    assert probe.passed is True
    assert [req["max_tokens"] for req in adapter.requests[:2]] == [128, 2048]

    # Exhausting both token budgets without visible text reports truncation cleanly.
    exhausted_adapter = _ScriptedAdapter(
        [
            ChatCompletionResult(content="", reasoning="still thinking", finish_reason="length"),
            ChatCompletionResult(content="", reasoning="still thinking", finish_reason="length"),
            ChatCompletionResult(content="ALPHA-7942", finish_reason="stop"),
            ChatCompletionResult(content="COBALT-41 and AMBER-88", finish_reason="stop"),
            ChatCompletionResult(content="COBALT-41 and AMBER-88", finish_reason="stop"),
        ]
    )
    exhausted_probe = await probe_chat_template_fidelity(
        exhausted_adapter,
        model="thinking-model",
        base_url="http://localhost:8000/v1",
        api_key=None,
    )
    assert exhausted_probe.passed is False
    assert exhausted_probe.diagnosis == "tool_replay_template_error"
    assert "exhausted token budget" in exhausted_probe.checks[0].detail


# ---------------------------------------------------------------------------
# CLI probe helper: check_chat_template_fidelity
# ---------------------------------------------------------------------------


def test_check_chat_template_fidelity_console_headless_redaction_and_exit(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tok_file = tmp_path / "tokenizer.json"
    tok_file.write_text(
        json.dumps(
            {
                "added_tokens": [
                    {"id": 1, "content": "<|custom_stop|>", "special": True},
                    {"id": 2, "content": "<|im_end|>", "special": True},
                ]
            }
        ),
        encoding="utf-8",
    )

    # Passing run with owned adapter and explicit tokenizer.
    owned_ok = _ScriptedAdapter(_ok_responses())
    monkeypatch.setattr("tool_eval_bench.cli.probe.build_adapter", lambda *a, **kw: owned_ok)
    ok_buf = StringIO()
    ok_probe = check_chat_template_fidelity(
        Console(file=ok_buf, width=200, no_color=True),
        "http://localhost:8000/v1",
        "qwen3",
        None,
        tokenizer=str(tok_file),
    )
    assert isinstance(ok_probe, ChatTemplateProbe)
    assert ok_probe.passed is True
    assert owned_ok.closed is True
    ok_rendered = ok_buf.getvalue()
    assert "Chat-template fidelity verified" in ok_rendered
    assert "Checked 2 tokenizer control token(s)" in ok_rendered

    # Failing Gemma 4 leak run with caller-supplied adapter and URL redaction.
    monkeypatch.setattr(
        "tool_eval_bench.cli.probe.resolve_tokenizer",
        lambda *a, **kw: TokenizerResolution(),
    )
    fail_buf = StringIO()
    gemma_adapter = _ScriptedAdapter(
        [
            ChatCompletionResult(
                content="<|tool_call>call:get_code{}<tool_call|>",
                finish_reason="stop",
            ),
            ChatCompletionResult(content="ALPHA-7942", finish_reason="stop"),
            ChatCompletionResult(content="COBALT-41 AMBER-88", finish_reason="stop"),
            ChatCompletionResult(content="COBALT-41 AMBER-88", finish_reason="stop"),
        ]
    )
    fail_probe = check_chat_template_fidelity(
        Console(file=fail_buf, width=200, no_color=True),
        "http://secret.internal:8000/v1",
        "gemma4",
        None,
        backend="vllm",
        display_url="http://***:8000/v1",
        adapter=gemma_adapter,
    )
    assert fail_probe.passed is False
    assert fail_probe.diagnosis == "tool_replay_template_error"
    assert gemma_adapter.closed is False
    fail_text = fail_buf.getvalue()
    assert "tool_replay_template_error" in fail_text
    assert "--enable-auto-tool-choice --tool-call-parser gemma4" in fail_text
    assert "secret.internal" not in fail_text

    # Headless JSON mode with exit_on_failure=True and redacted endpoint error URL.
    headless_adapter = _ScriptedAdapter(
        [
            RuntimeError("Failed connecting to http://user:topsecretpw@secret.internal:8000/v1"),
            ChatCompletionResult(content="ALPHA-7942", finish_reason="stop"),
            ChatCompletionResult(content="COBALT-41 AMBER-88", finish_reason="stop"),
            ChatCompletionResult(content="COBALT-41 AMBER-88", finish_reason="stop"),
        ]
    )
    with pytest.raises(SystemExit) as exc_info:
        check_chat_template_fidelity(
            Console(file=StringIO(), width=200, no_color=True),
            "http://user:topsecretpw@secret.internal:8000/v1",
            "m",
            None,
            headless=True,
            adapter=headless_adapter,
            exit_on_failure=True,
        )
    assert exc_info.value.code == 1
    payload = json.loads(capsys.readouterr().err.strip())
    assert payload["event"] == "template_probe"
    assert payload["status"] == "failed"
    assert payload["passed"] is False
    assert payload["diagnosis"] == "tool_replay_template_error"
    dumped = json.dumps(payload)
    assert "secret.internal" not in dumped
    assert "topsecretpw" not in dumped
    assert "advice" in payload

    # Multi-turn and non-parser token-leak advice branches in console mode.
    mt_buf = StringIO()
    mt_probe = check_chat_template_fidelity(
        Console(file=mt_buf, width=200, no_color=True),
        "http://localhost:8000/v1",
        "m",
        None,
        adapter=_ScriptedAdapter(
            [
                ChatCompletionResult(content="ALPHA-7942", finish_reason="stop"),
                ChatCompletionResult(content="ALPHA-7942", finish_reason="stop"),
                ChatCompletionResult(content="I forgot the codes.", finish_reason="stop"),
                ChatCompletionResult(content="COBALT-41 AMBER-88", finish_reason="stop"),
            ]
        ),
    )
    assert mt_probe.diagnosis == "multi_turn_template_error"
    assert "multi-turn user/assistant history" in mt_buf.getvalue()

    leak_buf = StringIO()
    monkeypatch.setattr(
        "tool_eval_bench.cli.probe.resolve_tokenizer",
        lambda *a, **kw: TokenizerResolution(
            path=str(tok_file), source="explicit", detail="--tokenizer"
        ),
    )
    leak_probe = check_chat_template_fidelity(
        Console(file=leak_buf, width=200, no_color=True),
        "http://localhost:8000/v1",
        "m",
        None,
        tokenizer=str(tok_file),
        adapter=_ScriptedAdapter(
            [
                ChatCompletionResult(content="ALPHA-7942<|custom_stop|>", finish_reason="stop"),
                ChatCompletionResult(content="ALPHA-7942", finish_reason="stop"),
                ChatCompletionResult(content="COBALT-41 AMBER-88", finish_reason="stop"),
                ChatCompletionResult(content="COBALT-41 AMBER-88", finish_reason="stop"),
            ]
        ),
    )
    assert leak_probe.diagnosis == "template_token_leak"
    assert "Check stop tokens" in leak_buf.getvalue()
    assert "Checked 2 tokenizer control token(s)" in leak_buf.getvalue()

    # Endpoint failure branch emits no template-specific advice.
    ep_buf = StringIO()
    ep_probe = check_chat_template_fidelity(
        Console(file=ep_buf, width=200, no_color=True),
        "http://localhost:8000/v1",
        "m",
        None,
        adapter=_ScriptedAdapter(
            [
                ChatCompletionResult(content="wrong", finish_reason="stop"),
                ChatCompletionResult(content="wrong", finish_reason="stop"),
                ChatCompletionResult(content="wrong", finish_reason="stop"),
                ChatCompletionResult(content="wrong", finish_reason="stop"),
            ]
        ),
    )
    assert ep_probe.diagnosis == "model_or_endpoint_failure"
    assert "Check stop tokens" not in ep_buf.getvalue()
    assert "Inspect the server chat template" not in ep_buf.getvalue()


# ---------------------------------------------------------------------------
# End-to-end CLI tests: probe --check-template
# ---------------------------------------------------------------------------


def _install_mock_http(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx.Request], httpx.Response],
) -> None:
    real_client = httpx.AsyncClient

    def _client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _client)


def _run_cli_argv(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
) -> tuple[int, str, str]:
    from tool_eval_bench.cli import dispatch

    for key in list(os.environ):
        if key.startswith("TOOL_EVAL_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(dispatch, "_load_dotenv", lambda: None)
    monkeypatch.setattr(sys, "argv", ["tool-eval-bench", *argv])
    try:
        dispatch.main()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
    else:
        code = 0
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _template_completion_response(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content},
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 6},
        },
    )


def test_cli_probe_check_template_passing_and_tokenizer_leak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tok_file = tmp_path / "tokenizer.json"
    tok_file.write_text(
        json.dumps({"added_tokens": [{"id": 1, "content": "<|secret_eot|>", "special": True}]}),
        encoding="utf-8",
    )
    replies = ["ALPHA-7942", "ALPHA-7942", "COBALT-41, AMBER-88", "COBALT-41, AMBER-88"]
    seen_paths: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append((request.method, request.url.path))
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "gemma-4-31b"}]})
        if request.url.path == "/v1/chat/completions":
            return _template_completion_response(replies.pop(0))
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    _install_mock_http(monkeypatch, handler)
    code, out, err = _run_cli_argv(
        monkeypatch,
        capsys,
        [
            "probe",
            "--check-template",
            "--tokenizer",
            str(tok_file),
            "--base-url",
            "http://gpu.test:8000",
        ],
    )
    assert code == 0
    assert err == ""
    assert "Chat-template fidelity verified" in out
    assert seen_paths == [
        ("GET", "/v1/models"),
        ("POST", "/v1/chat/completions"),
        ("POST", "/v1/chat/completions"),
        ("POST", "/v1/chat/completions"),
        ("POST", "/v1/chat/completions"),
    ]

    # Now leak the tokenizer-specific control token on the second reply.
    replies[:] = [
        "ALPHA-7942",
        "ALPHA-7942<|secret_eot|>",
        "COBALT-41, AMBER-88",
        "COBALT-41, AMBER-88",
    ]
    leak_code, leak_out, leak_err = _run_cli_argv(
        monkeypatch,
        capsys,
        [
            "--probe",
            "--check-template",
            "--json",
            "--tokenizer",
            str(tok_file),
            "--base-url",
            "http://gpu.test:8000",
        ],
    )
    assert leak_code == 1
    assert leak_out == ""
    events = [json.loads(line) for line in leak_err.splitlines() if line.strip()]
    assert [ev["event"] for ev in events] == ["probe_result", "template_probe"]
    assert events[1]["status"] == "failed"
    assert events[1]["diagnosis"] == "template_token_leak"
    assert events[1]["leaked_tokens"] == ["<|secret_eot|>"]
    assert events[1]["tokenizer_tokens_checked"] == 1


def test_cli_probe_combined_check_tools_and_check_template_runs_both(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    chat_call = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal chat_call
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "gemma4"}]})
        if request.url.path == "/v1/chat/completions":
            chat_call += 1
            if chat_call == 1:
                # Fail --check-tools with unparsed Gemma 4 syntax.
                return _template_completion_response(
                    '<|tool_call>call:probe_ping{value:"ok"}<tool_call|>'
                )
            return _template_completion_response(
                "ALPHA-7942" if chat_call in (2, 3) else "COBALT-41 AMBER-88"
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    _install_mock_http(monkeypatch, handler)
    code, out, err = _run_cli_argv(
        monkeypatch,
        capsys,
        [
            "probe",
            "--check-tools",
            "--check-template",
            "--json",
            "--backend",
            "vllm",
            "--base-url",
            "http://gpu.test:8000",
        ],
    )
    assert code == 1
    assert out == ""
    events = [json.loads(line) for line in err.splitlines() if line.strip()]
    assert [ev["event"] for ev in events] == ["probe_result", "tool_call_probe", "template_probe"]
    assert events[1]["status"] == "failed"
    assert events[1]["parser_family"] == "gemma4"
    assert "--tool-call-parser gemma4" in events[1]["advice"]
    assert events[2]["status"] == "ok"
    assert events[2]["passed"] is True


def test_cli_probe_check_template_missing_model_and_standalone_flag_validation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_mock_http(
        monkeypatch,
        lambda request: httpx.Response(200, json={"data": []}),
    )

    no_model_code, no_model_out, _ = _run_cli_argv(
        monkeypatch,
        capsys,
        ["probe", "--check-template", "--base-url", "http://gpu.test:8000"],
    )
    assert no_model_code == 1
    assert "Cannot verify chat-template fidelity" in no_model_out

    no_model_json_code, no_model_json_out, no_model_json_err = _run_cli_argv(
        monkeypatch,
        capsys,
        ["probe", "--check-template", "--json", "--base-url", "http://gpu.test:8000"],
    )
    assert no_model_json_code == 1
    assert no_model_json_out == ""
    events = [json.loads(line) for line in no_model_json_err.splitlines() if line.strip()]
    assert [ev["event"] for ev in events] == ["probe_result", "template_probe"]
    assert events[1]["status"] == "failed"

    bad_code, _, bad_err = _run_cli_argv(
        monkeypatch,
        capsys,
        ["--check-template", "--json", "--base-url", "http://gpu.test:8000"],
    )
    assert bad_code == 2
    bad_event = json.loads(bad_err.strip())
    assert bad_event["error"] == "invalid_arguments"
    assert "--check-template requires probe or --probe" in bad_event["message"]
