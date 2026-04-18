"""Tests for the context pressure feature.

Covers:
  - Fill budget calculation with various context sizes and ratios
  - Filler message building (structure, token budget, edge cases)
  - Context size detection from mock /v1/models responses
  - ContextPressureConfig summary string
  - Integration with the orchestrator (_initial_messages)
"""

from __future__ import annotations


from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from tool_eval_bench.runner.context_pressure import (
    ContextPressureConfig,
    build_pressure_messages,
    compute_fill_budget,
    detect_context_size,
    prepare_context_pressure,
    _RESERVED_FOR_OUTPUT,
    _RESERVED_FOR_SCENARIO,
)


# ---------------------------------------------------------------------------
# compute_fill_budget
# ---------------------------------------------------------------------------


class TestComputeFillBudget:
    def test_standard_ratio(self) -> None:
        """75% pressure on a 32K context should fill most of the available space."""
        fill = compute_fill_budget(32768, 0.75)
        available = 32768 - _RESERVED_FOR_OUTPUT - _RESERVED_FOR_SCENARIO
        expected = int(available * 0.75)
        assert fill == expected

    def test_zero_ratio(self) -> None:
        """0% pressure means no fill."""
        fill = compute_fill_budget(32768, 0.0)
        assert fill == 0

    def test_full_ratio(self) -> None:
        """100% pressure fills all available space."""
        fill = compute_fill_budget(32768, 1.0)
        available = 32768 - _RESERVED_FOR_OUTPUT - _RESERVED_FOR_SCENARIO
        assert fill == available

    def test_tiny_context_returns_zero(self) -> None:
        """Context too small for any fill (smaller than reserved overhead)."""
        fill = compute_fill_budget(4000, 0.75)
        assert fill == 0

    def test_ratio_clamped_above_one(self) -> None:
        """Ratio > 1.0 is clamped to 1.0."""
        fill = compute_fill_budget(32768, 1.5)
        available = 32768 - _RESERVED_FOR_OUTPUT - _RESERVED_FOR_SCENARIO
        assert fill == available

    def test_ratio_clamped_below_zero(self) -> None:
        """Negative ratio is clamped to 0.0."""
        fill = compute_fill_budget(32768, -0.5)
        assert fill == 0

    def test_large_context(self) -> None:
        """128K context should produce a large fill."""
        fill = compute_fill_budget(131072, 0.75)
        assert fill > 80000  # Sanity: should be substantial


# ---------------------------------------------------------------------------
# build_pressure_messages
# ---------------------------------------------------------------------------


class TestBuildPressureMessages:
    def test_zero_fill_returns_empty(self) -> None:
        """No fill tokens → no messages."""
        cfg = ContextPressureConfig(ratio=0.0, fill_tokens=0, detected_context=32768)
        msgs = build_pressure_messages(cfg)
        assert msgs == []

    def test_alternating_roles(self) -> None:
        """Messages should alternate user/assistant."""
        cfg = ContextPressureConfig(ratio=0.75, fill_tokens=5000, detected_context=32768)
        msgs = build_pressure_messages(cfg)
        assert len(msgs) > 0
        for i, msg in enumerate(msgs):
            expected_role = "user" if i % 2 == 0 else "assistant"
            assert msg["role"] == expected_role, f"Message {i} should be {expected_role}"

    def test_even_count(self) -> None:
        """Should always produce an even number of messages (user/assistant pairs)."""
        cfg = ContextPressureConfig(ratio=0.75, fill_tokens=10000, detected_context=32768)
        msgs = build_pressure_messages(cfg)
        assert len(msgs) % 2 == 0

    def test_first_message_has_framing(self) -> None:
        """First user message should include the framing prefix."""
        cfg = ContextPressureConfig(ratio=0.75, fill_tokens=5000, detected_context=32768)
        msgs = build_pressure_messages(cfg)
        assert "background context" in msgs[0]["content"].lower()

    def test_small_fill_produces_single_pair(self) -> None:
        """A small fill budget produces at least one user/assistant pair."""
        cfg = ContextPressureConfig(ratio=0.1, fill_tokens=200, detected_context=32768)
        msgs = build_pressure_messages(cfg)
        assert len(msgs) == 2  # one user + one assistant

    def test_very_small_fill_skipped(self) -> None:
        """Fill budget < 50 tokens is too small for a meaningful chunk."""
        cfg = ContextPressureConfig(ratio=0.01, fill_tokens=30, detected_context=32768)
        msgs = build_pressure_messages(cfg)
        assert msgs == []

    def test_user_messages_are_substantial(self) -> None:
        """User filler messages should be significantly longer than assistant acks."""
        cfg = ContextPressureConfig(ratio=0.75, fill_tokens=10000, detected_context=32768)
        msgs = build_pressure_messages(cfg)
        user_msgs = [m for m in msgs if m["role"] == "user"]
        assistant_msgs = [m for m in msgs if m["role"] == "assistant"]
        for um in user_msgs:
            assert len(um["content"]) > 200
        for am in assistant_msgs:
            assert len(am["content"]) < 200

    def test_on_chunk_callback_fires(self) -> None:
        """on_chunk should fire after each pair with monotonically increasing tokens."""
        cfg = ContextPressureConfig(ratio=0.75, fill_tokens=10000, detected_context=32768)
        reported: list[int] = []
        build_pressure_messages(cfg, on_chunk=lambda t: reported.append(t))
        assert len(reported) > 0
        # Should be monotonically increasing
        for i in range(1, len(reported)):
            assert reported[i] > reported[i - 1]
        # Final value should be close to (but not exceed) fill_tokens + overhead
        assert reported[-1] <= cfg.fill_tokens + 500

    def test_adjacent_chunks_are_diverse(self) -> None:
        """Adjacent user messages should have different content (not repeated)."""
        cfg = ContextPressureConfig(ratio=0.75, fill_tokens=10000, detected_context=32768)
        msgs = build_pressure_messages(cfg)
        user_msgs = [m["content"] for m in msgs if m["role"] == "user"]
        assert len(user_msgs) >= 3, "Need at least 3 user chunks to test diversity"
        # Skip the first (has framing prefix) and compare adjacent pairs
        for i in range(2, len(user_msgs)):
            assert user_msgs[i][:200] != user_msgs[i - 1][:200], (
                f"User chunks {i - 1} and {i} should have different content"
            )

    def test_consecutive_builds_produce_unique_content(self) -> None:
        """Two consecutive builds should produce different content (noise + shuffle)."""
        cfg = ContextPressureConfig(ratio=0.5, fill_tokens=5000, detected_context=32768)
        msgs_a = build_pressure_messages(cfg)
        msgs_b = build_pressure_messages(cfg)
        user_a = [m["content"] for m in msgs_a if m["role"] == "user"]
        user_b = [m["content"] for m in msgs_b if m["role"] == "user"]
        # At least some chunks should differ (noise makes them unique)
        differences = sum(1 for a, b in zip(user_a, user_b) if a != b)
        assert differences > 0, "Consecutive builds should produce different content"

    def test_noise_tokens_present(self) -> None:
        """Filler messages should contain injected noise markers."""
        cfg = ContextPressureConfig(ratio=0.75, fill_tokens=10000, detected_context=32768)
        msgs = build_pressure_messages(cfg)
        all_content = " ".join(m["content"] for m in msgs if m["role"] == "user")
        # Should contain at least one noise pattern
        noise_patterns = ["ref #", "ticket SRE-", "[v", "node ", "batch ", "[id:"]
        found = any(p in all_content for p in noise_patterns)
        assert found, "Filler text should contain injected noise tokens"

# ---------------------------------------------------------------------------
# ContextPressureConfig.summary
# ---------------------------------------------------------------------------


class TestContextPressureConfigSummary:
    def test_summary_format(self) -> None:
        cfg = ContextPressureConfig(
            ratio=0.75, fill_tokens=20000, detected_context=32768,
        )
        s = cfg.summary()
        assert "75%" in s
        assert "20K" in s
        assert "33K" in s or "32K" in s  # 32768 / 1000 ≈ 33

    def test_summary_zero(self) -> None:
        cfg = ContextPressureConfig(ratio=0.0, fill_tokens=0, detected_context=8192)
        s = cfg.summary()
        assert "0%" in s


# ---------------------------------------------------------------------------
# detect_context_size (mock HTTP)
# ---------------------------------------------------------------------------


class TestDetectContextSize:
    @pytest.mark.asyncio
    async def test_vllm_max_model_len(self) -> None:
        """Should detect context size from vLLM's max_model_len field."""
        from unittest.mock import MagicMock

        mock_response = {
            "data": [
                {"id": "test-model", "max_model_len": 32768, "root": "test-model"}
            ]
        }
        with patch("tool_eval_bench.runner.context_pressure.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = mock_response
            instance.get = AsyncMock(return_value=resp)
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await detect_context_size("http://localhost:8080", "test-model")
            assert result == 32768

    @pytest.mark.asyncio
    async def test_litellm_context_window(self) -> None:
        """Should detect from LiteLLM's context_window field."""
        from unittest.mock import MagicMock

        mock_response = {
            "data": [
                {"id": "gpt-4o", "context_window": 128000}
            ]
        }
        with patch("tool_eval_bench.runner.context_pressure.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = mock_response
            instance.get = AsyncMock(return_value=resp)
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await detect_context_size("http://localhost:4000", "gpt-4o")
            assert result == 128000

    @pytest.mark.asyncio
    async def test_no_context_field_returns_none(self) -> None:
        """If model metadata has no context size fields, return None."""
        from unittest.mock import MagicMock

        mock_response = {
            "data": [{"id": "test-model"}]
        }
        with patch("tool_eval_bench.runner.context_pressure.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = mock_response
            instance.get = AsyncMock(return_value=resp)
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await detect_context_size("http://localhost:8080", "test-model")
            assert result is None

    @pytest.mark.asyncio
    async def test_connection_error_returns_none(self) -> None:
        """Network errors should return None, not raise."""
        with patch("tool_eval_bench.runner.context_pressure.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get = AsyncMock(side_effect=ConnectionError("refused"))
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await detect_context_size("http://localhost:8080", "test-model")
            assert result is None


# ---------------------------------------------------------------------------
# prepare_context_pressure
# ---------------------------------------------------------------------------


class TestPrepareContextPressure:
    @pytest.mark.asyncio
    async def test_with_override(self) -> None:
        """Should use context_size_override without querying the server."""
        cfg = await prepare_context_pressure(
            "http://localhost:8080", "test-model", None,
            ratio=0.75, context_size_override=32768,
        )
        assert cfg.detected_context == 32768
        assert cfg.ratio == 0.75
        assert cfg.fill_tokens > 0

    @pytest.mark.asyncio
    async def test_raises_when_no_detection_and_no_override(self) -> None:
        """Should raise ValueError if auto-detect fails and no override given."""
        with patch(
            "tool_eval_bench.runner.context_pressure.detect_context_size",
            return_value=None,
        ):
            with pytest.raises(ValueError, match="auto-detect"):
                await prepare_context_pressure(
                    "http://localhost:8080", "test-model", None,
                    ratio=0.75,
                )


# ---------------------------------------------------------------------------
# Integration: pressure messages in _initial_messages
# ---------------------------------------------------------------------------


class TestOrchestratorIntegration:
    def test_initial_messages_without_pressure(self) -> None:
        """Without pressure, messages are just system + user."""
        from tool_eval_bench.runner.orchestrator import _initial_messages

        msgs = _initial_messages("What's the weather?")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_initial_messages_with_pressure(self) -> None:
        """With pressure messages, they appear between system and user."""
        from tool_eval_bench.runner.orchestrator import _initial_messages

        pressure = [
            {"role": "user", "content": "Background filler text..."},
            {"role": "assistant", "content": "Understood."},
        ]

        msgs = _initial_messages(
            "What's the weather?",
            context_pressure_messages=pressure,
        )

        assert len(msgs) == 4  # system + 2 pressure + user
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "Background filler text..."
        assert msgs[2]["role"] == "assistant"
        assert msgs[3]["role"] == "user"
        assert msgs[3]["content"] == "What's the weather?"

    def test_pressure_messages_order_preserved(self) -> None:
        """Multiple pressure pairs should maintain their order."""
        from tool_eval_bench.runner.orchestrator import _initial_messages

        pressure = [
            {"role": "user", "content": "chunk_1"},
            {"role": "assistant", "content": "ack_1"},
            {"role": "user", "content": "chunk_2"},
            {"role": "assistant", "content": "ack_2"},
        ]

        msgs = _initial_messages(
            "Real question",
            context_pressure_messages=pressure,
        )

        # system, chunk_1, ack_1, chunk_2, ack_2, real question
        assert len(msgs) == 6
        assert msgs[1]["content"] == "chunk_1"
        assert msgs[2]["content"] == "ack_1"
        assert msgs[3]["content"] == "chunk_2"
        assert msgs[4]["content"] == "ack_2"
        assert msgs[5]["content"] == "Real question"


# ---------------------------------------------------------------------------
# Integration: run_scenario with pressure messages
# ---------------------------------------------------------------------------


class TestRunScenarioWithPressure:
    @pytest.mark.asyncio
    async def test_pressure_messages_reach_adapter(self) -> None:
        """Pressure messages should be present in the adapter's first call."""
        from tool_eval_bench.adapters.base import BackendAdapter, ChatCompletionResult
        from tool_eval_bench.domain.scenarios import (
            Category,
            ScenarioDefinition,
            ScenarioEvaluation,
            ScenarioStatus,
        )
        from tool_eval_bench.runner.orchestrator import run_scenario

        class CapturingAdapter(BackendAdapter):
            def __init__(self) -> None:
                self.captured: list[list[dict]] = []

            async def chat_completion(self, **kwargs: Any) -> ChatCompletionResult:
                import copy
                self.captured.append(copy.deepcopy(kwargs.get("messages", [])))
                return ChatCompletionResult(content="It's 22C in Berlin.")

        def handler(state, call):
            return {"result": "ok"}

        def evaluator(state):
            return ScenarioEvaluation(
                status=ScenarioStatus.PASS, points=2, summary="ok"
            )

        scenario = ScenarioDefinition(
            id="CP-01", title="Pressure test", category=Category.A,
            user_message="What's the weather?",
            description="Test with pressure",
            handle_tool_call=handler,
            evaluate=evaluator,
        )

        pressure = [
            {"role": "user", "content": "Background filler " * 100},
            {"role": "assistant", "content": "Understood."},
        ]

        adapter = CapturingAdapter()
        result = await run_scenario(
            adapter,
            model="test",
            base_url="http://localhost:8080",
            api_key=None,
            scenario=scenario,
            context_pressure_messages=pressure,
        )

        assert result.status == ScenarioStatus.PASS
        # The first (and only) call should have system + pressure + user
        msgs = adapter.captured[0]
        roles = [m["role"] for m in msgs]
        assert roles == ["system", "user", "assistant", "user"]
        # The filler should be in the second message
        assert "Background filler" in msgs[1]["content"]
        # The actual scenario message should be last
        assert msgs[-1]["content"] == "What's the weather?"
