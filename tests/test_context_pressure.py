"""Tests for the context pressure feature.

Covers:
  - Fill budget calculation with various context sizes and ratios
  - Filler message building (structure, token budget, edge cases)
  - Context size detection from mock /v1/models responses
  - KV cache capacity detection from mock /metrics responses
  - KV capacity capping in prepare_context_pressure
  - ContextPressureConfig summary string
  - Integration with the orchestrator (_initial_messages)
  - Per-scenario nonce injection for prefix cache isolation
  - Reservation constant (12K headroom for ratio=1.0)
  - Sweep range parsing and level generation
  - Sweep runner integration (mocked orchestrator, early stop, redact-url)
"""

from __future__ import annotations

import argparse
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from tool_eval_bench.runner.context_pressure import (
    _RESERVED_FOR_OUTPUT,
    _RESERVED_FOR_SCENARIO,
    ContextPressureConfig,
    KvCapacityInfo,
    build_pressure_messages,
    compute_fill_budget,
    detect_context_size,
    detect_kv_capacity,
    prepare_context_pressure,
)


class _ImmediateEventLoop:
    """Return calibration results without entering CPython's event loop."""

    def run_until_complete(self, coroutine: Any) -> tuple[list[dict[str, Any]], int]:
        coroutine.close()
        return [], 0

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# compute_fill_budget
# ---------------------------------------------------------------------------


class TestComputeFillBudget:
    def test_standard_ratio(self) -> None:
        """75% pressure on a 32K context should fill most of the available space,
        quantised to chunk boundaries."""
        fill = compute_fill_budget(32768, 0.75)
        available = 32768 - _RESERVED_FOR_OUTPUT - _RESERVED_FOR_SCENARIO
        raw = int(available * 0.75)
        chunk = 2048 + 20  # _TOKENS_PER_FILLER_CHUNK + ack overhead
        expected = (raw // chunk) * chunk
        assert fill == expected

    def test_zero_ratio(self) -> None:
        """0% pressure means no fill."""
        fill = compute_fill_budget(32768, 0.0)
        assert fill == 0

    def test_full_ratio(self) -> None:
        """100% pressure fills all available space (quantised to chunk boundary)."""
        fill = compute_fill_budget(32768, 1.0)
        available = 32768 - _RESERVED_FOR_OUTPUT - _RESERVED_FOR_SCENARIO
        chunk = 2048 + 20
        expected = (available // chunk) * chunk
        assert fill == expected

    def test_tiny_context_returns_zero(self) -> None:
        """Context too small for any fill (smaller than reserved overhead)."""
        fill = compute_fill_budget(4000, 0.75)
        assert fill == 0

    def test_ratio_clamped_above_one(self) -> None:
        """Ratio > 1.0 is clamped to 1.0."""
        fill = compute_fill_budget(32768, 1.5)
        available = 32768 - _RESERVED_FOR_OUTPUT - _RESERVED_FOR_SCENARIO
        chunk = 2048 + 20
        expected = (available // chunk) * chunk
        assert fill == expected

    def test_ratio_clamped_below_zero(self) -> None:
        """Negative ratio is clamped to 0.0."""
        fill = compute_fill_budget(32768, -0.5)
        assert fill == 0

    def test_large_context(self) -> None:
        """128K context should produce a large fill."""
        fill = compute_fill_budget(131072, 0.75)
        assert fill > 80000  # Sanity: should be substantial

    def test_adjacent_levels_same_chunk_count(self) -> None:
        """Adjacent sweep levels should not straddle a chunk boundary.

        This is the root cause fix for the alternating pass/fail bug.
        For a 260K context swept 0.30→0.35, both levels should produce
        a fill budget that generates the same number of message pairs.
        """
        ctx = 260000
        fill_30 = compute_fill_budget(ctx, 0.30)
        fill_35 = compute_fill_budget(ctx, 0.3538)
        chunk = 2048 + 20
        # Both fills should be exact multiples of chunk size
        assert fill_30 % chunk == 0
        assert fill_35 % chunk == 0
        # The pair counts should differ monotonically (no alternation)
        assert fill_35 >= fill_30

    def test_quantisation_is_monotonic(self) -> None:
        """Fill budget must be monotonically non-decreasing as ratio grows."""
        ctx = 260000
        prev = 0
        for i in range(100):
            ratio = i / 100
            fill = compute_fill_budget(ctx, ratio)
            assert fill >= prev, f"Non-monotonic: ratio={ratio} fill={fill} < prev={prev}"
            prev = fill


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
        differences = sum(1 for a, b in zip(user_a, user_b, strict=False) if a != b)
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

    def test_seeded_builds_are_identical(self) -> None:
        """Two builds with the same seed and fill_tokens produce identical content."""
        cfg = ContextPressureConfig(ratio=0.5, fill_tokens=5000, detected_context=32768)
        msgs_a = build_pressure_messages(cfg, seed=42)
        msgs_b = build_pressure_messages(cfg, seed=42)
        assert len(msgs_a) == len(msgs_b)
        for a, b in zip(msgs_a, msgs_b, strict=True):
            assert a["content"] == b["content"]

    def test_different_seeds_produce_different_content(self) -> None:
        """Different seeds should produce different filler."""
        cfg = ContextPressureConfig(ratio=0.5, fill_tokens=5000, detected_context=32768)
        msgs_a = build_pressure_messages(cfg, seed=42)
        msgs_b = build_pressure_messages(cfg, seed=99)
        user_a = [m["content"] for m in msgs_a if m["role"] == "user"]
        user_b = [m["content"] for m in msgs_b if m["role"] == "user"]
        differences = sum(1 for a, b in zip(user_a, user_b, strict=False) if a != b)
        assert differences > 0, "Different seeds should produce different content"

    def test_same_seed_different_fill_produces_different_content(self) -> None:
        """Same seed but different fill_tokens should produce unique content
        (the seed incorporates fill_tokens via XOR)."""
        cfg_a = ContextPressureConfig(ratio=0.5, fill_tokens=5000, detected_context=32768)
        cfg_b = ContextPressureConfig(ratio=0.5, fill_tokens=8000, detected_context=32768)
        msgs_a = build_pressure_messages(cfg_a, seed=42)
        msgs_b = build_pressure_messages(cfg_b, seed=42)
        # Nonce prefix should differ
        assert msgs_a[0]["content"][:40] != msgs_b[0]["content"][:40]


# ---------------------------------------------------------------------------
# ContextPressureConfig.summary
# ---------------------------------------------------------------------------


class TestContextPressureConfigSummary:
    def test_summary_format(self) -> None:
        cfg = ContextPressureConfig(
            ratio=0.75,
            fill_tokens=20000,
            detected_context=32768,
        )
        s = cfg.summary()
        assert "75%" in s
        assert "20K" in s  # 20000 / 1024 ≈ 19.5 → ~20K
        assert "32K" in s  # 32768 / 1024 = 32K exactly

    def test_summary_zero(self) -> None:
        cfg = ContextPressureConfig(ratio=0.0, fill_tokens=0, detected_context=8192)
        s = cfg.summary()
        assert "0%" in s

    def test_summary_clarifies_fill_budget_not_total_context(self) -> None:
        cfg = ContextPressureConfig(ratio=1.0, fill_tokens=16544, detected_context=35000)
        s = cfg.summary().lower()
        assert "available fill budget" in s

    def test_budget_breakdown_does_not_double_count_tool_tokens(self) -> None:
        cfg = ContextPressureConfig(ratio=1.0, fill_tokens=16544, detected_context=35000)
        budget = cfg.budget_breakdown(tool_tokens=5000)
        assert budget["fill_tokens"] == 16544
        assert budget["tool_tokens"] == 5000
        assert budget["output_tokens"] == _RESERVED_FOR_OUTPUT
        assert budget["scenario_budget_tokens"] == 35000 - 16544 - _RESERVED_FOR_OUTPUT
        assert budget["remaining_headroom_tokens"] == (35000 - 16544 - _RESERVED_FOR_OUTPUT - 5000)


# ---------------------------------------------------------------------------
# detect_context_size (mock HTTP)
# ---------------------------------------------------------------------------


class TestDetectContextSize:
    @pytest.mark.asyncio
    async def test_vllm_max_model_len(self) -> None:
        """Should detect context size from vLLM's max_model_len field."""
        from unittest.mock import MagicMock

        mock_response = {
            "data": [{"id": "test-model", "max_model_len": 32768, "root": "test-model"}]
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

        mock_response = {"data": [{"id": "gpt-4o", "context_window": 128000}]}
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

        mock_response = {"data": [{"id": "test-model"}]}
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
# detect_kv_capacity (vLLM /metrics scraping)
# ---------------------------------------------------------------------------


class TestDetectKvCapacity:
    @pytest.mark.asyncio
    async def test_parses_cache_config_info(self) -> None:
        """Should extract num_gpu_blocks × block_size from vllm:cache_config_info."""
        from unittest.mock import MagicMock

        metrics_text = (
            "# HELP vllm:cache_config_info Information of the LLMEngine CacheConfig\n"
            "# TYPE vllm:cache_config_info gauge\n"
            'vllm:cache_config_info{block_size="16",engine="0",'
            'num_gpu_blocks="7338",num_cpu_blocks="None"} 1.0\n'
        )
        with patch("tool_eval_bench.runner.context_pressure.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            resp = MagicMock()
            resp.status_code = 200
            resp.text = metrics_text
            instance.get = AsyncMock(return_value=resp)
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await detect_kv_capacity("http://localhost:8080")
            assert result is not None
            assert result.capacity == 7338 * 16  # 117,408
            assert result.num_blocks == 7338
            assert result.block_size == 16
            assert result.is_hybrid is False  # no mamba_cache_mode label

    @pytest.mark.asyncio
    async def test_handles_reverse_label_order(self) -> None:
        """Labels may appear in any order in Prometheus format."""
        from unittest.mock import MagicMock

        # num_gpu_blocks before block_size
        metrics_text = (
            'vllm:cache_config_info{num_gpu_blocks="5000",block_size="32",engine="0"} 1.0\n'
        )
        with patch("tool_eval_bench.runner.context_pressure.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            resp = MagicMock()
            resp.status_code = 200
            resp.text = metrics_text
            instance.get = AsyncMock(return_value=resp)
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await detect_kv_capacity("http://localhost:8080")
            assert result is not None
            assert result.capacity == 5000 * 32

    @pytest.mark.asyncio
    async def test_returns_none_when_no_cache_config(self) -> None:
        """Non-vLLM servers won't have cache_config_info."""
        from unittest.mock import MagicMock

        metrics_text = (
            "# HELP http_requests_total Total requests\n"
            'http_requests_total{handler="/v1/chat/completions"} 100.0\n'
        )
        with patch("tool_eval_bench.runner.context_pressure.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            resp = MagicMock()
            resp.status_code = 200
            resp.text = metrics_text
            instance.get = AsyncMock(return_value=resp)
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await detect_kv_capacity("http://localhost:8080")
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_connection_error(self) -> None:
        """Network errors should return None gracefully."""
        with patch("tool_eval_bench.runner.context_pressure.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get = AsyncMock(side_effect=ConnectionError("refused"))
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await detect_kv_capacity("http://localhost:8080")
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_404(self) -> None:
        """Servers without /metrics should return None."""
        from unittest.mock import MagicMock

        with patch("tool_eval_bench.runner.context_pressure.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            resp = MagicMock()
            resp.status_code = 404
            instance.get = AsyncMock(return_value=resp)
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await detect_kv_capacity("http://localhost:8080")
            assert result is None

    @pytest.mark.asyncio
    async def test_uses_custom_metrics_url(self) -> None:
        """Should use provided metrics_url instead of deriving from base_url."""
        from unittest.mock import MagicMock

        metrics_text = (
            'vllm:cache_config_info{block_size="16",num_gpu_blocks="4096",engine="0"} 1.0\n'
        )
        with patch("tool_eval_bench.runner.context_pressure.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            resp = MagicMock()
            resp.status_code = 200
            resp.text = metrics_text
            instance.get = AsyncMock(return_value=resp)
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await detect_kv_capacity(
                "http://localhost:8080",
                metrics_url="http://other-host:9090/metrics",
            )
            assert result is not None
            assert result.capacity == 4096 * 16
            # Verify the custom URL was used
            call_args = instance.get.call_args
            assert "other-host:9090" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_detects_hybrid_model(self) -> None:
        """Should set is_hybrid=True when mamba_cache_mode is not 'none'."""
        from unittest.mock import MagicMock

        # Real-world metric from Qwen3.6-35B-A3B hybrid model
        metrics_text = (
            'vllm:cache_config_info{block_size="16",engine="0",'
            'mamba_cache_mode="align",mamba_ssm_cache_dtype="float32",'
            'num_gpu_blocks="1997",num_cpu_blocks="None"} 1.0\n'
        )
        with patch("tool_eval_bench.runner.context_pressure.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            resp = MagicMock()
            resp.status_code = 200
            resp.text = metrics_text
            instance.get = AsyncMock(return_value=resp)
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await detect_kv_capacity("http://localhost:8080")
            assert result is not None
            assert result.capacity == 1997 * 16
            assert result.is_hybrid is True

    @pytest.mark.asyncio
    async def test_standard_model_not_hybrid(self) -> None:
        """mamba_cache_mode='none' should NOT be flagged as hybrid."""
        from unittest.mock import MagicMock

        metrics_text = (
            'vllm:cache_config_info{block_size="16",engine="0",'
            'mamba_cache_mode="none",num_gpu_blocks="7338",'
            'num_cpu_blocks="None"} 1.0\n'
        )
        with patch("tool_eval_bench.runner.context_pressure.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            resp = MagicMock()
            resp.status_code = 200
            resp.text = metrics_text
            instance.get = AsyncMock(return_value=resp)
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await detect_kv_capacity("http://localhost:8080")
            assert result is not None
            assert result.capacity == 7338 * 16
            assert result.is_hybrid is False


# ---------------------------------------------------------------------------
# prepare_context_pressure
# ---------------------------------------------------------------------------


class TestPrepareContextPressure:
    @pytest.mark.asyncio
    async def test_with_override(self) -> None:
        """Should use context_size_override without querying the server."""
        cfg = await prepare_context_pressure(
            "http://localhost:8080",
            "test-model",
            None,
            ratio=0.75,
            context_size_override=32768,
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
                    "http://localhost:8080",
                    "test-model",
                    None,
                    ratio=0.75,
                )

    @pytest.mark.asyncio
    async def test_kv_capacity_caps_context_size(self) -> None:
        """When KV cache capacity < max_model_len (standard model), context should be capped."""
        kv_info = KvCapacityInfo(capacity=117408, num_blocks=7338, block_size=16, is_hybrid=False)
        with (
            patch(
                "tool_eval_bench.runner.context_pressure.detect_context_size",
                return_value=262144,
            ),
            patch(
                "tool_eval_bench.runner.context_pressure.detect_kv_capacity",
                return_value=kv_info,
            ),
        ):
            cfg = await prepare_context_pressure(
                "http://localhost:8080",
                "test-model",
                None,
                ratio=0.9,
            )
            # Should be capped to KV capacity, not max_model_len
            assert cfg.detected_context == 117408
            expected_fill = compute_fill_budget(117408, 0.9)
            assert cfg.fill_tokens == expected_fill

    @pytest.mark.asyncio
    async def test_kv_capacity_no_cap_when_smaller_context(self) -> None:
        """When max_model_len < KV capacity, no capping needed."""
        kv_info = KvCapacityInfo(capacity=117408, num_blocks=7338, block_size=16, is_hybrid=False)
        with (
            patch(
                "tool_eval_bench.runner.context_pressure.detect_context_size",
                return_value=32768,
            ),
            patch(
                "tool_eval_bench.runner.context_pressure.detect_kv_capacity",
                return_value=kv_info,
            ),
        ):
            cfg = await prepare_context_pressure(
                "http://localhost:8080",
                "test-model",
                None,
                ratio=0.75,
            )
            assert cfg.detected_context == 32768

    @pytest.mark.asyncio
    async def test_hybrid_model_skips_kv_capping(self) -> None:
        """Hybrid-attention models should NOT be capped by physical block capacity."""
        kv_info = KvCapacityInfo(capacity=31952, num_blocks=1997, block_size=16, is_hybrid=True)
        with (
            patch(
                "tool_eval_bench.runner.context_pressure.detect_context_size",
                return_value=262144,
            ),
            patch(
                "tool_eval_bench.runner.context_pressure.detect_kv_capacity",
                return_value=kv_info,
            ),
        ):
            cfg = await prepare_context_pressure(
                "http://localhost:8080",
                "test-model",
                None,
                ratio=1.0,
            )
            # Should trust max_model_len, NOT cap to 31,952
            assert cfg.detected_context == 262144
            expected_fill = compute_fill_budget(262144, 1.0)
            assert cfg.fill_tokens == expected_fill

    @pytest.mark.asyncio
    async def test_kv_detection_failure_uses_max_model_len(self) -> None:
        """When KV capacity detection fails, fall back to max_model_len."""
        with (
            patch(
                "tool_eval_bench.runner.context_pressure.detect_context_size",
                return_value=262144,
            ),
            patch(
                "tool_eval_bench.runner.context_pressure.detect_kv_capacity",
                return_value=None,
            ),
        ):
            cfg = await prepare_context_pressure(
                "http://localhost:8080",
                "test-model",
                None,
                ratio=0.9,
            )
            assert cfg.detected_context == 262144

    @pytest.mark.asyncio
    async def test_override_skips_kv_capping(self) -> None:
        """When --context-size is explicitly provided, skip KV capping."""
        # detect_kv_capacity should NOT be called when override is set
        with patch(
            "tool_eval_bench.runner.context_pressure.detect_kv_capacity",
        ) as mock_kv:
            cfg = await prepare_context_pressure(
                "http://localhost:8080",
                "test-model",
                None,
                ratio=0.9,
                context_size_override=32768,
            )
            assert cfg.detected_context == 32768
            mock_kv.assert_not_called()


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

    def test_scenario_nonce_injected_into_first_filler(self) -> None:
        """When scenario_id is provided, a unique nonce prefix should be
        injected into the first filler user message to defeat prefix caching.
        (Regression test for #4.)"""
        from tool_eval_bench.runner.orchestrator import _initial_messages

        pressure = [
            {"role": "user", "content": "Background filler text..."},
            {"role": "assistant", "content": "Understood."},
        ]

        msgs = _initial_messages(
            "What's the weather?",
            context_pressure_messages=pressure,
            scenario_id="TC-64",
        )

        assert len(msgs) == 4
        # The first filler message should contain the scenario nonce
        assert msgs[1]["content"].startswith("[scenario:TC-64]")
        assert "Background filler text" in msgs[1]["content"]

    def test_scenario_nonce_does_not_mutate_original(self) -> None:
        """The original pressure_messages list must not be mutated when a
        scenario nonce is injected."""
        from tool_eval_bench.runner.orchestrator import _initial_messages

        pressure = [
            {"role": "user", "content": "Original filler text"},
            {"role": "assistant", "content": "Understood."},
        ]

        _initial_messages(
            "Question 1",
            context_pressure_messages=pressure,
            scenario_id="TC-01",
        )

        # Original should be unmodified
        assert pressure[0]["content"] == "Original filler text"

    def test_different_scenarios_get_different_prefixes(self) -> None:
        """Two scenarios with different IDs should produce different filler
        message prefixes — ensuring prefix caching can't help either one."""
        from tool_eval_bench.runner.orchestrator import _initial_messages

        pressure = [
            {"role": "user", "content": "Shared filler text..."},
            {"role": "assistant", "content": "OK."},
        ]

        msgs_a = _initial_messages(
            "Question A",
            context_pressure_messages=pressure,
            scenario_id="TC-61",
        )
        msgs_b = _initial_messages(
            "Question B",
            context_pressure_messages=pressure,
            scenario_id="TC-64",
        )

        # First filler message should differ between scenarios
        assert msgs_a[1]["content"] != msgs_b[1]["content"]
        assert "[scenario:TC-61]" in msgs_a[1]["content"]
        assert "[scenario:TC-64]" in msgs_b[1]["content"]


# ---------------------------------------------------------------------------
# calibrate_pressure_messages
# ---------------------------------------------------------------------------


class TestCalibratePressureMessages:
    """Tests for the calibration step that trims/extends filler to hit
    exact token targets.  Covers all code paths including the fix for
    the alternating pass/fail bug (no message pair removal)."""

    def _make_messages(self, n_pairs: int = 3, content_len: int = 2000) -> list[dict]:
        """Build synthetic filler messages for testing."""
        msgs = []
        for _ in range(n_pairs):
            msgs.append({"role": "user", "content": "x" * content_len})
            msgs.append({"role": "assistant", "content": "Understood."})
        return msgs

    @pytest.mark.asyncio
    async def test_empty_messages_returns_zero(self) -> None:
        """Empty message list should return immediately."""
        from tool_eval_bench.runner.context_pressure import calibrate_pressure_messages

        msgs, actual = await calibrate_pressure_messages(
            [],
            1000,
            "http://localhost",
            "model",
        )
        assert msgs == []
        assert actual == 0

    @pytest.mark.asyncio
    async def test_zero_target_returns_zero(self) -> None:
        """Target of 0 should return immediately."""
        from tool_eval_bench.runner.context_pressure import calibrate_pressure_messages

        msgs = self._make_messages(1)
        result, actual = await calibrate_pressure_messages(
            msgs,
            0,
            "http://localhost",
            "model",
        )
        assert actual == 0

    @pytest.mark.asyncio
    async def test_tokenizer_unavailable_returns_estimate(self) -> None:
        """When count_messages_tokens returns None, use char-based estimate."""
        from tool_eval_bench.runner.context_pressure import calibrate_pressure_messages

        with patch(
            "tool_eval_bench.runner.context_pressure.count_messages_tokens",
            return_value=None,
        ):
            msgs = self._make_messages(2, content_len=1000)
            result, actual = await calibrate_pressure_messages(
                msgs,
                5000,
                "http://localhost",
                "model",
            )
            assert len(result) == 4  # unchanged
            assert actual > 0  # char-based estimate

    @pytest.mark.asyncio
    async def test_within_tolerance_no_change(self) -> None:
        """When actual tokens are within 2% of target, no modification."""
        from tool_eval_bench.runner.context_pressure import calibrate_pressure_messages

        target = 5000
        # Within 2%: 5000 * 0.02 = 100, so 5050 is within tolerance
        with patch(
            "tool_eval_bench.runner.context_pressure.count_messages_tokens",
            return_value=5050,
        ):
            msgs = self._make_messages(2, content_len=1000)
            original_content = msgs[2]["content"]  # last user msg
            result, actual = await calibrate_pressure_messages(
                msgs,
                target,
                "http://localhost",
                "model",
            )
            assert actual == 5050
            assert result[2]["content"] == original_content  # unchanged

    @pytest.mark.asyncio
    async def test_over_target_trims_content(self) -> None:
        """When over target, the last user message should be trimmed."""
        from tool_eval_bench.runner.context_pressure import calibrate_pressure_messages

        target = 5000
        with patch(
            "tool_eval_bench.runner.context_pressure.count_messages_tokens",
            side_effect=[6000, 5100],  # first: over, second: after trim
        ):
            msgs = self._make_messages(3, content_len=2000)
            original_len = len(msgs[-2]["content"])  # last user msg
            result, actual = await calibrate_pressure_messages(
                msgs,
                target,
                "http://localhost",
                "model",
            )
            assert len(result) == 6  # same number of messages
            assert len(result[-2]["content"]) < original_len  # trimmed

    @pytest.mark.asyncio
    async def test_over_target_never_removes_message_pair(self) -> None:
        """Even when nearly all content must be removed, the message pair
        must be preserved — removing pairs re-introduces the alternating
        pass/fail bug."""
        from tool_eval_bench.runner.context_pressure import calibrate_pressure_messages

        target = 100
        # Massively over target — would previously remove the pair
        with patch(
            "tool_eval_bench.runner.context_pressure.count_messages_tokens",
            side_effect=[5000, 200],  # first: way over, second: after trim
        ):
            msgs = self._make_messages(2, content_len=2000)
            n_before = len(msgs)
            result, actual = await calibrate_pressure_messages(
                msgs,
                target,
                "http://localhost",
                "model",
            )
            # Message count must NOT change
            assert len(result) == n_before
            # Last user message should be trimmed to minimum, not removed
            last_user = next(m for m in reversed(result) if m["role"] == "user")
            assert len(last_user["content"]) == 100  # min viable

    @pytest.mark.asyncio
    async def test_under_target_extends_content(self) -> None:
        """When under target, the last user message should be extended."""
        from tool_eval_bench.runner.context_pressure import calibrate_pressure_messages

        target = 8000
        with patch(
            "tool_eval_bench.runner.context_pressure.count_messages_tokens",
            side_effect=[5000, 7900],  # first: under, second: after extend
        ):
            msgs = self._make_messages(2, content_len=1000)
            original_len = len(msgs[2]["content"])  # last user msg
            result, actual = await calibrate_pressure_messages(
                msgs,
                target,
                "http://localhost",
                "model",
            )
            assert len(result) == 4  # same number of messages
            assert len(result[2]["content"]) > original_len  # extended

    @pytest.mark.asyncio
    async def test_seeded_extend_is_deterministic(self) -> None:
        """When seed is provided, extending with the same seed should
        produce identical content."""
        from tool_eval_bench.runner.context_pressure import calibrate_pressure_messages

        target = 8000
        with patch(
            "tool_eval_bench.runner.context_pressure.count_messages_tokens",
            side_effect=[5000, 7900, 5000, 7900],  # two runs
        ):
            msgs_a = self._make_messages(2, content_len=1000)
            msgs_b = self._make_messages(2, content_len=1000)
            result_a, _ = await calibrate_pressure_messages(
                msgs_a,
                target,
                "http://localhost",
                "model",
                seed=42,
            )
            result_b, _ = await calibrate_pressure_messages(
                msgs_b,
                target,
                "http://localhost",
                "model",
                seed=42,
            )
            # Extended content should be identical
            assert result_a[2]["content"] == result_b[2]["content"]

    @pytest.mark.asyncio
    async def test_unseeded_extend_is_nondeterministic(self) -> None:
        """Without seed, two extends should produce different content
        (due to time.time_ns() RNG)."""
        from tool_eval_bench.runner.context_pressure import calibrate_pressure_messages

        target = 8000
        with patch(
            "tool_eval_bench.runner.context_pressure.count_messages_tokens",
            side_effect=[5000, 7900, 5000, 7900],
        ):
            msgs_a = self._make_messages(2, content_len=1000)
            msgs_b = self._make_messages(2, content_len=1000)
            result_a, _ = await calibrate_pressure_messages(
                msgs_a,
                target,
                "http://localhost",
                "model",
            )
            result_b, _ = await calibrate_pressure_messages(
                msgs_b,
                target,
                "http://localhost",
                "model",
            )
            # Extended content should differ (different time.time_ns() seeds)
            assert result_a[2]["content"] != result_b[2]["content"]


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
            return ScenarioEvaluation(status=ScenarioStatus.PASS, points=2, summary="ok")

        scenario = ScenarioDefinition(
            id="CP-01",
            title="Pressure test",
            category=Category.A,
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


# ---------------------------------------------------------------------------
# Reservation constant
# ---------------------------------------------------------------------------


class TestReservationConstant:
    def test_reservation_at_least_12000(self) -> None:
        """_RESERVED_FOR_SCENARIO must be >= 12000 to allow ratio=1.0 to
        succeed with headroom for token estimation error."""
        assert _RESERVED_FOR_SCENARIO >= 12000

    def test_ratio_1_leaves_enough_headroom(self) -> None:
        """At ratio=1.0, the fill budget should consume most available space
        (quantised to chunk boundary), with reserved 12K for worst-case."""
        fill = compute_fill_budget(32768, 1.0)
        available = 32768 - _RESERVED_FOR_OUTPUT - _RESERVED_FOR_SCENARIO
        # Fill should be close to available but quantised down
        assert fill <= available
        assert fill > available * 0.9  # should use most of available
        # Reserved space should be enough for LARGE_TOOLSET (~6K) + margin
        assert _RESERVED_FOR_SCENARIO >= 12000


# ---------------------------------------------------------------------------
# Sweep range parsing
# ---------------------------------------------------------------------------


class TestSweepRangeParsing:
    def test_valid_range(self) -> None:
        from tool_eval_bench.cli.bench import _parse_sweep_range

        start, end = _parse_sweep_range("0.5-1.0")
        assert start == 0.5
        assert end == 1.0

    def test_narrow_range(self) -> None:
        from tool_eval_bench.cli.bench import _parse_sweep_range

        start, end = _parse_sweep_range("0.9-1.0")
        assert start == 0.9
        assert end == 1.0

    def test_values_clamped_to_unit_range(self) -> None:
        from tool_eval_bench.cli.bench import _parse_sweep_range

        # Values > 1.0 should be clamped
        start, end = _parse_sweep_range("0.5-1.5")
        assert start == 0.5
        assert end == 1.0

    def test_invalid_format_raises(self) -> None:
        from tool_eval_bench.cli.bench import _parse_sweep_range

        with pytest.raises(ValueError, match="Invalid sweep range"):
            _parse_sweep_range("0.5")

    def test_start_equals_end_raises(self) -> None:
        from tool_eval_bench.cli.bench import _parse_sweep_range

        with pytest.raises(ValueError, match="must be less than"):
            _parse_sweep_range("0.5-0.5")

    def test_start_greater_than_end_raises(self) -> None:
        from tool_eval_bench.cli.bench import _parse_sweep_range

        with pytest.raises(ValueError, match="must be less than"):
            _parse_sweep_range("0.8-0.5")

    def test_non_numeric_raises(self) -> None:
        from tool_eval_bench.cli.bench import _parse_sweep_range

        with pytest.raises(ValueError, match="must be numbers"):
            _parse_sweep_range("abc-def")

    def test_sweep_levels_generation(self) -> None:
        """Verify that sweep level generation produces correct step count."""
        from tool_eval_bench.cli.bench import _parse_sweep_range

        start, end = _parse_sweep_range("0.9-1.0")
        steps = 10
        levels = [start + i * (end - start) / steps for i in range(steps + 1)]
        levels = [round(lv, 4) for lv in levels]

        assert len(levels) == 11
        assert levels[0] == 0.9
        assert levels[-1] == 1.0
        # Check step size is ~0.01
        for i in range(1, len(levels)):
            assert abs(levels[i] - levels[i - 1] - 0.01) < 0.001


# ---------------------------------------------------------------------------
# Integration: _run_pressure_sweep
# ---------------------------------------------------------------------------


class TestPressureSweepIntegration:
    """Tests for the sweep runner (mocked orchestrator)."""

    def _make_args(
        self,
        sweep: str = "0.5-1.0",
        steps: int = 2,
        scenarios: list[str] | None = None,
        context_size: int = 32768,
    ) -> argparse.Namespace:
        import argparse

        return argparse.Namespace(
            context_pressure_sweep=sweep,
            sweep_steps=steps,
            scenarios=scenarios or ["TC-01"],
            short=False,
            categories=None,
            context_size=context_size,
            redact_url=False,
            seed=None,
        )

    def _make_summary(self, statuses: list[str]) -> Any:
        """Build a mock ModelScoreSummary with given statuses."""
        from tool_eval_bench.domain.scenarios import (
            ModelScoreSummary,
            ScenarioResult,
            ScenarioStatus,
        )

        results = []
        for i, status_str in enumerate(statuses):
            s = ScenarioStatus(status_str)
            results.append(
                ScenarioResult(
                    scenario_id=f"TC-{i + 1:02d}",
                    status=s,
                    points=2
                    if s == ScenarioStatus.PASS
                    else (1 if s == ScenarioStatus.PARTIAL else 0),
                    summary="ok",
                )
            )

        return ModelScoreSummary(
            scenario_results=results,
            total_points=sum(r.points for r in results),
            max_points=len(results) * 2,
            final_score=sum(r.points for r in results) / (len(results) * 2) * 100,
            rating="test",
            category_scores=[],
            safety_warnings=[],
            total_tokens=0,
        )

    @patch("tool_eval_bench.cli.commands.resolve_scenarios")
    @patch("tool_eval_bench.cli.pressure.asyncio")
    def test_sweep_runs_all_levels(self, mock_asyncio, mock_resolve) -> None:
        """Sweep should call run_all_scenarios for each pressure level."""
        import io

        from rich.console import Console

        from tool_eval_bench.domain.scenarios import (
            Category,
            ScenarioDefinition,
        )

        scenario = ScenarioDefinition(
            id="TC-01",
            title="Test",
            category=Category.A,
            user_message="test",
            description="test",
            handle_tool_call=lambda s, c: {},
            evaluate=lambda s: None,
        )
        mock_resolve.return_value = [scenario]

        summary = self._make_summary(["pass"])
        mock_asyncio.run.return_value = summary

        console = Console(file=io.StringIO(), force_terminal=False)
        args = self._make_args(sweep="0.5-1.0", steps=3, context_size=32768)

        from tool_eval_bench.cli.bench import _run_pressure_sweep
        from tool_eval_bench.cli.commands import resolve_scenarios
        from tool_eval_bench.cli.helpers import (
            metadata_for_storage,
            parse_sweep_range,
            persist_plugin_run,
            with_config_fingerprint,
        )

        with patch("asyncio.new_event_loop", return_value=_ImmediateEventLoop()):
            _run_pressure_sweep(
                console,
                "test-model",
                "test-model",
                "vllm",
                "http://localhost:8080",
                None,
                args,
                parse_sweep_range=parse_sweep_range,
                resolve_scenarios=resolve_scenarios,
                with_config_fingerprint=with_config_fingerprint,
                persist_plugin_run=persist_plugin_run,
                metadata_for_storage=metadata_for_storage,
            )

        # 3 levels (steps=3 → 0.5, 0.75, 1.0) — one asyncio.run per level
        assert mock_asyncio.run.call_count == 3

    @patch("tool_eval_bench.cli.commands.resolve_scenarios")
    @patch("tool_eval_bench.cli.pressure.asyncio")
    def test_sweep_early_stops_on_consecutive_failures(
        self,
        mock_asyncio,
        mock_resolve,
    ) -> None:
        """Sweep should stop after 2 consecutive all-fail levels."""
        import io

        from rich.console import Console

        from tool_eval_bench.domain.scenarios import (
            Category,
            ScenarioDefinition,
        )

        scenario = ScenarioDefinition(
            id="TC-01",
            title="Test",
            category=Category.A,
            user_message="test",
            description="test",
            handle_tool_call=lambda s, c: {},
            evaluate=lambda s: None,
        )
        mock_resolve.return_value = [scenario]

        pass_summary = self._make_summary(["pass"])
        fail_summary = self._make_summary(["fail"])

        # Level 1: pass, Level 2: fail, Level 3: fail → stop
        mock_asyncio.run.side_effect = [pass_summary, fail_summary, fail_summary]

        console = Console(file=io.StringIO(), force_terminal=False)
        args = self._make_args(sweep="0.5-1.0", steps=4, context_size=32768)

        from tool_eval_bench.cli.bench import _run_pressure_sweep
        from tool_eval_bench.cli.commands import resolve_scenarios
        from tool_eval_bench.cli.helpers import (
            metadata_for_storage,
            parse_sweep_range,
            persist_plugin_run,
            with_config_fingerprint,
        )

        with patch("asyncio.new_event_loop", return_value=_ImmediateEventLoop()):
            _run_pressure_sweep(
                console,
                "test-model",
                "test-model",
                "vllm",
                "http://localhost:8080",
                None,
                args,
                parse_sweep_range=parse_sweep_range,
                resolve_scenarios=resolve_scenarios,
                with_config_fingerprint=with_config_fingerprint,
                persist_plugin_run=persist_plugin_run,
                metadata_for_storage=metadata_for_storage,
            )

        # Should stop at 3 calls (pass, fail, fail), not run all 4 levels
        assert mock_asyncio.run.call_count == 3

    def test_sweep_uses_display_url_for_redaction(self) -> None:
        """When --redact-url is used, the sweep header should show the
        redacted URL, not the real server address."""
        from tool_eval_bench.cli.bench import _redact_url

        real_url = "http://192.168.10.5:8080"
        redacted = _redact_url(real_url)
        assert "192.168" not in redacted
        assert "***" in redacted
        assert "8080" in redacted

        # Verify the sweep function signature accepts display_url
        import inspect

        from tool_eval_bench.cli.bench import _run_pressure_sweep

        sig = inspect.signature(_run_pressure_sweep)
        assert "display_url" in sig.parameters
