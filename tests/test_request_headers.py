"""User-supplied request headers and the per-conversation session header."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from tool_eval_bench.adapters.anthropic import AnthropicAdapter
from tool_eval_bench.adapters.factory import build_adapter
from tool_eval_bench.adapters.gemini import GeminiAdapter
from tool_eval_bench.adapters.openai_compat import OpenAICompatibleAdapter
from tool_eval_bench.adapters.requests import minimal_request
from tool_eval_bench.cli.helpers import adapter_options
from tool_eval_bench.domain.adapters import BackendAdapter, ChatCompletionResult
from tool_eval_bench.domain.scenarios import ScenarioDefinition
from tool_eval_bench.evals.scenarios import SCENARIOS
from tool_eval_bench.runner.orchestrator import run_scenario
from tool_eval_bench.utils.headers import (
    USER_AGENT,
    attach_session_id,
    parse_header_env,
    parse_header_pair,
    parse_header_pairs,
)


class TestParsing:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("User-Agent=my-agent/1.0", ("User-Agent", "my-agent/1.0")),
            ("X-Trace: abc=def", ("X-Trace", "abc=def")),
            ("  X-Pad =  padded  ", ("X-Pad", "padded")),
            ("X-Empty=", ("X-Empty", "")),
        ],
    )
    def test_pair_spellings(self, raw: str, expected: tuple[str, str]) -> None:
        assert parse_header_pair(raw) == expected

    @pytest.mark.parametrize("raw", ["no-separator", "=value", "Bad Name=1", ": x"])
    def test_invalid_pairs_are_rejected(self, raw: str) -> None:
        with pytest.raises(ValueError):
            parse_header_pair(raw)

    def test_later_pairs_win(self) -> None:
        assert parse_header_pairs(["A=1", "B=2", "A=3"]) == {"A": "3", "B": "2"}
        assert parse_header_pairs(None) == {}

    def test_env_form_splits_on_semicolons(self) -> None:
        assert parse_header_env("A=1; B: two ;") == {"A": "1", "B": "two"}
        assert parse_header_env(None) == {}
        assert parse_header_env("   ") == {}

    def test_attach_session_id(self) -> None:
        assert attach_session_id({"A": "1"}, None) == {"A": "1"}
        assert attach_session_id({"A": "1"}, "x-s", "conv") == {"A": "1", "x-s": "conv"}
        minted = attach_session_id({}, "x-s")["x-s"]
        assert len(minted) == 32
        assert attach_session_id({}, "x-s")["x-s"] != minted


def _capture(adapter: BackendAdapter, response: dict[str, Any], **kwargs: Any) -> httpx.Request:
    seen: list[httpx.Request] = []

    def _handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=response)

    adapter._client = httpx.AsyncClient(transport=httpx.MockTransport(_handle))  # type: ignore[attr-defined]  # noqa: SLF001
    asyncio.run(
        adapter.chat_completion(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            api_key="k",
            **kwargs,
        )
    )
    return seen[0]


_OPENAI_RESPONSE = {"choices": [{"message": {"content": "ok"}}]}
_GEMINI_RESPONSE = {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}
_ANTHROPIC_RESPONSE = {"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn"}


class TestAdapters:
    @pytest.mark.parametrize(
        ("adapter_cls", "response", "base_url"),
        [
            (OpenAICompatibleAdapter, _OPENAI_RESPONSE, "http://x/v1"),
            (GeminiAdapter, _GEMINI_RESPONSE, "https://generativelanguage.googleapis.com"),
            (AnthropicAdapter, _ANTHROPIC_RESPONSE, "https://opencode.ai/zen/go/v1/messages"),
        ],
    )
    def test_every_adapter_sends_user_headers_and_the_session_id(
        self, adapter_cls: type, response: dict[str, Any], base_url: str
    ) -> None:
        adapter = adapter_cls(
            default_headers={"X-Extra": "1", "User-Agent": "mine/2"},
            session_header="x-opencode-session",
        )
        request = _capture(adapter, response, base_url=base_url, conversation_id="conv-1")
        assert request.headers["x-extra"] == "1"
        assert request.headers["x-opencode-session"] == "conv-1"
        # A user header beats the adapter's own.
        assert request.headers["user-agent"] == "mine/2"

    def test_default_user_agent_names_the_benchmark(self) -> None:
        request = _capture(OpenAICompatibleAdapter(), _OPENAI_RESPONSE, base_url="http://x/v1")
        assert request.headers["user-agent"] == USER_AGENT

    def test_no_session_header_means_no_session_id(self) -> None:
        request = _capture(
            OpenAICompatibleAdapter(), _OPENAI_RESPONSE, base_url="http://x/v1", conversation_id="c"
        )
        assert "x-opencode-session" not in request.headers

    def test_single_shot_request_gets_a_fresh_id(self) -> None:
        adapter = OpenAICompatibleAdapter(session_header="x-s")
        first = _capture(adapter, _OPENAI_RESPONSE, base_url="http://x/v1").headers["x-s"]
        second = _capture(adapter, _OPENAI_RESPONSE, base_url="http://x/v1").headers["x-s"]
        assert first != second

    def test_user_headers_override_wire_format_headers(self) -> None:
        adapter = AnthropicAdapter(default_headers={"anthropic-version": "2099-01-01"})
        request = _capture(adapter, _ANTHROPIC_RESPONSE, base_url="https://api.anthropic.com")
        assert request.headers["anthropic-version"] == "2099-01-01"


class TestFactoryAndOptions:
    def test_build_adapter_forwards_header_options(self) -> None:
        adapter = build_adapter("http://x/v1", default_headers={"A": "1"}, session_header="x-s")
        assert isinstance(adapter, OpenAICompatibleAdapter)
        headers = adapter._request_headers({}, "conv")  # noqa: SLF001
        assert headers == {"User-Agent": USER_AGENT, "A": "1", "x-s": "conv"}

    def test_adapter_options_read_the_resolved_namespace(self) -> None:
        class _Args:
            format = "anthropic"
            _request_headers = {"A": "1"}
            _session_header = "x-s"

        assert adapter_options(_Args()) == {
            "wire_format": "anthropic",
            "default_headers": {"A": "1"},
            "session_header": "x-s",
        }
        assert adapter_options(object()) == {
            "wire_format": None,
            "default_headers": None,
            "session_header": None,
        }


class TestPreflightRequests:
    def test_minimal_request_applies_user_headers_last(self) -> None:
        _, _, headers = minimal_request(
            "http://x", "m", "k", headers={"Authorization": "Bearer other", "X-A": "1"}
        )
        assert headers["Authorization"] == "Bearer other"
        assert headers["X-A"] == "1"
        assert headers["User-Agent"] == USER_AGENT


class _RecordingAdapter(BackendAdapter):
    def __init__(self) -> None:
        self.conversation_ids: list[str | None] = []

    async def chat_completion(self, **kwargs: Any) -> ChatCompletionResult:
        self.conversation_ids.append(kwargs.get("conversation_id"))
        turn = len(self.conversation_ids)
        if turn == 1:
            return ChatCompletionResult(
                content="",
                tool_calls=[],
                raw_response={},
            )
        return ChatCompletionResult(content="done", raw_response={})


def _scenario(scenario_id: str) -> ScenarioDefinition:
    return next(s for s in SCENARIOS if s.id == scenario_id)


def test_every_turn_of_a_scenario_shares_one_conversation_id() -> None:
    adapter = _RecordingAdapter()
    scenario = _scenario("TC-01")
    asyncio.run(
        run_scenario(
            adapter,
            model="m",
            base_url="http://x",
            api_key=None,
            scenario=scenario,
            max_turns=3,
        )
    )
    assert adapter.conversation_ids
    assert all(isinstance(cid, str) and len(cid) == 32 for cid in adapter.conversation_ids)
    assert len(set(adapter.conversation_ids)) == 1


def test_two_scenarios_get_different_conversation_ids() -> None:
    ids: list[str | None] = []
    for _ in range(2):
        adapter = _RecordingAdapter()
        asyncio.run(
            run_scenario(
                adapter,
                model="m",
                base_url="http://x",
                api_key=None,
                scenario=_scenario("TC-01"),
                max_turns=1,
            )
        )
        ids.append(adapter.conversation_ids[0])
    assert ids[0] != ids[1]


def test_request_body_is_untouched_by_headers() -> None:
    adapter = OpenAICompatibleAdapter(default_headers={"X-A": "1"}, session_header="x-s")
    request = _capture(adapter, _OPENAI_RESPONSE, base_url="http://x/v1", conversation_id="c")
    body = json.loads(request.content)
    assert "X-A" not in body and "x-s" not in body
