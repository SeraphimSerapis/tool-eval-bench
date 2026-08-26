"""Tests for the throughput measurement module.

TEST-02: Tests TokenizerConfig, filler text generation, binary search setup,
calibration logic, and latency estimation — all without a real server.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from tests.conftest import MeasurementTestClient
from tool_eval_bench.runner.throughput import (
    _DEFAULT_CHARS_PER_TOKEN,
    _FILLER_PARAGRAPH,
    ThroughputSample,
    TokenizerConfig,
    _build_filler_heuristic,
)

# ---------------------------------------------------------------------------
# TokenizerConfig
# ---------------------------------------------------------------------------


class TestTokenizerConfig:
    def test_defaults(self) -> None:
        cfg = TokenizerConfig()
        assert cfg.chars_per_token == _DEFAULT_CHARS_PER_TOKEN
        assert cfg.has_tokenize_endpoint is False

    def test_filler_pool_caching(self) -> None:
        cfg = TokenizerConfig()
        pool1 = cfg.get_filler_pool(100)
        pool2 = cfg.get_filler_pool(50)  # smaller request → returns cached
        assert pool1 is pool2  # same object

    def test_filler_pool_grows(self) -> None:
        cfg = TokenizerConfig()
        small = cfg.get_filler_pool(100)
        big = cfg.get_filler_pool(100_000)
        assert len(big) >= 100_000
        assert len(big) > len(small)

    def test_filler_pool_min_length(self) -> None:
        cfg = TokenizerConfig()
        pool = cfg.get_filler_pool(10_000)
        assert len(pool) >= 10_000


# ---------------------------------------------------------------------------
# Heuristic filler builder
# ---------------------------------------------------------------------------


class TestBuildFillerHeuristic:
    def test_basic_length(self) -> None:
        """Heuristic should produce approximately target_tokens * chars_per_token chars."""
        result = _build_filler_heuristic(100)
        expected_chars = int(100 * _DEFAULT_CHARS_PER_TOKEN)
        assert len(result) == expected_chars

    def test_uses_config_chars_per_token(self) -> None:
        cfg = TokenizerConfig(chars_per_token=2.5)
        result = _build_filler_heuristic(200, cfg)
        assert len(result) == int(200 * 2.5)

    def test_large_target(self) -> None:
        """Even very large targets should generate valid text."""
        result = _build_filler_heuristic(50_000)
        assert len(result) == int(50_000 * _DEFAULT_CHARS_PER_TOKEN)
        # Should be repetitions of the filler paragraph
        assert _FILLER_PARAGRAPH[:50] in result

    def test_zero_tokens(self) -> None:
        result = _build_filler_heuristic(0)
        assert result == ""


# ---------------------------------------------------------------------------
# ThroughputSample
# ---------------------------------------------------------------------------


class TestThroughputSample:
    def test_defaults(self) -> None:
        s = ThroughputSample()
        assert s.pp_tokens == 0
        assert s.tg_tokens == 0
        assert s.error is None
        assert s.concurrency == 1
        assert s.token_timestamps == []
        assert s.mtp_chunks_detected is False

    def test_error_sample(self) -> None:
        s = ThroughputSample(error="Connection refused")
        assert s.error == "Connection refused"

    def test_peak_tg_tps_empty(self) -> None:
        """No timestamps → 0.0."""
        s = ThroughputSample()
        assert s.peak_tg_tps == 0.0

    def test_peak_tg_tps_single_timestamp(self) -> None:
        """Only one timestamp → 0.0 (need at least 2)."""
        s = ThroughputSample(token_timestamps=[100.0])
        assert s.peak_tg_tps == 0.0

    def test_peak_tg_tps_short_burst(self) -> None:
        """When all tokens fit within 1 second, use actual duration."""
        # 10 tokens in 0.5 seconds = 20 t/s
        ts = [100.0 + i * 0.05 for i in range(10)]
        s = ThroughputSample(token_timestamps=ts)
        peak = s.peak_tg_tps
        # duration = 0.45s, 10 tokens → 10/0.45 ≈ 22.2
        assert peak > 20.0

    def test_peak_tg_tps_sliding_window(self) -> None:
        """With >1s of data, sliding window picks the densest second."""
        # 5 tokens in first second, then gap, then 2 tokens
        ts = [
            100.0,
            100.2,
            100.4,
            100.6,
            100.8,  # 5 tokens in 0.8s
            102.0,
            102.5,  # 2 tokens much later
        ]
        s = ThroughputSample(token_timestamps=ts)
        peak = s.peak_tg_tps
        # Best 1-second window should contain the first 5 tokens
        assert peak == 5.0

    def test_mtp_chunks_detected_flag(self) -> None:
        """Verify the MTP detection flag can be set."""
        s = ThroughputSample(mtp_chunks_detected=True)
        assert s.mtp_chunks_detected is True


# ---------------------------------------------------------------------------
# MTP-aware token counting
# ---------------------------------------------------------------------------


class TestCountChunkTokens:
    """Tests for _count_chunk_tokens helper."""

    def test_with_token_ids(self) -> None:
        from tool_eval_bench.runner.throughput import _count_chunk_tokens

        choices = [{"token_ids": [101, 202, 303]}]
        assert _count_chunk_tokens(choices, "hello") == 3

    def test_single_token_id(self) -> None:
        from tool_eval_bench.runner.throughput import _count_chunk_tokens

        choices = [{"token_ids": [42]}]
        assert _count_chunk_tokens(choices, "x") == 1

    def test_no_token_ids(self) -> None:
        from tool_eval_bench.runner.throughput import _count_chunk_tokens

        choices = [{"delta": {"content": "hello"}}]
        assert _count_chunk_tokens(choices, "hello") == 1

    def test_empty_token_ids(self) -> None:
        from tool_eval_bench.runner.throughput import _count_chunk_tokens

        choices = [{"token_ids": []}]
        assert _count_chunk_tokens(choices, "hello") == 1

    def test_empty_choices(self) -> None:
        from tool_eval_bench.runner.throughput import _count_chunk_tokens

        assert _count_chunk_tokens([], "hello") == 1

    def test_token_ids_not_list(self) -> None:
        from tool_eval_bench.runner.throughput import _count_chunk_tokens

        choices = [{"token_ids": "not_a_list"}]
        assert _count_chunk_tokens(choices, "hello") == 1


# ---------------------------------------------------------------------------
# _stream_one with MTP mock server
# ---------------------------------------------------------------------------


def _make_sse_line(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _build_mtp_mock(tokens_per_chunk: int = 2, num_chunks: int = 5):
    """Build a mock HTTP handler that simulates an MTP server.

    Each SSE chunk contains ``tokens_per_chunk`` token_ids.
    """
    total_tokens = tokens_per_chunk * num_chunks

    def handler(request: httpx.Request) -> httpx.Response:
        if "/chat/completions" not in str(request.url):
            return httpx.Response(404)

        lines = []
        for i in range(num_chunks):
            ids = list(range(i * tokens_per_chunk, (i + 1) * tokens_per_chunk))
            chunk = {
                "choices": [
                    {
                        "delta": {"content": f"tok{i}"},
                        "token_ids": ids,
                    }
                ],
            }
            lines.append(_make_sse_line(chunk))

        # Final usage chunk
        usage_chunk = {
            "choices": [],
            "usage": {"prompt_tokens": 100, "completion_tokens": total_tokens},
        }
        lines.append(_make_sse_line(usage_chunk))
        lines.append("data: [DONE]\n\n")

        body = "".join(lines)
        return httpx.Response(
            200,
            content=body.encode(),
            headers={"content-type": "text/event-stream"},
        )

    return handler, total_tokens


@pytest.mark.asyncio
async def test_stream_one_mtp_token_counting() -> None:
    """MTP server sends 3 tokens per chunk — verify correct counting."""
    from tool_eval_bench.runner.throughput import _stream_one

    handler, expected_tokens = _build_mtp_mock(tokens_per_chunk=3, num_chunks=4)

    async with MeasurementTestClient(transport=httpx.MockTransport(handler)) as client:
        sample = await _stream_one(
            client,
            "http://localhost:8000/v1",
            "test-model",
            [{"role": "user", "content": "hi"}],
            12,
            None,
        )

    assert sample.error is None
    assert sample.tg_tokens == expected_tokens  # 12
    assert sample.mtp_chunks_detected is True
    # Should have per-token timestamps (3 per chunk × 4 chunks = 12)
    assert len(sample.token_timestamps) == expected_tokens
    assert sample.tg_tps > 0


@pytest.mark.asyncio
async def test_stream_one_standard_ar() -> None:
    """Standard AR server (1 token per chunk) — verify backward compat."""
    from tool_eval_bench.runner.throughput import _stream_one

    handler, expected_tokens = _build_mtp_mock(tokens_per_chunk=1, num_chunks=8)

    async with MeasurementTestClient(transport=httpx.MockTransport(handler)) as client:
        sample = await _stream_one(
            client,
            "http://localhost:8000/v1",
            "test-model",
            [{"role": "user", "content": "hi"}],
            8,
            None,
        )

    assert sample.error is None
    assert sample.tg_tokens == expected_tokens  # 8
    assert sample.mtp_chunks_detected is False
    assert len(sample.token_timestamps) == expected_tokens


@pytest.mark.asyncio
async def test_stream_one_retries_with_max_completion_tokens() -> None:
    from tool_eval_bench.runner.throughput import _stream_one

    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if "max_tokens" in body:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "Unsupported parameter: max_tokens. Use max_completion_tokens instead."
                    }
                },
            )
        return httpx.Response(
            200,
            content=(
                _make_sse_line({"choices": [{"delta": {"content": "ok"}, "token_ids": [1]}]})
                + "data: [DONE]\n\n"
            ).encode(),
            headers={"content-type": "text/event-stream"},
        )

    cfg = TokenizerConfig()
    async with MeasurementTestClient(transport=httpx.MockTransport(handler)) as client:
        sample = await _stream_one(
            client,
            "http://localhost:8000/v1",
            "test-model",
            [{"role": "user", "content": "hi"}],
            9,
            None,
            cfg,
        )

    assert sample.error is None
    assert cfg.output_token_field == "max_completion_tokens"
    assert len(bodies) == 2
    assert bodies[1]["max_completion_tokens"] == 9
    assert "max_tokens" not in bodies[1]


@pytest.mark.asyncio
async def test_stream_one_no_token_ids() -> None:
    """Server without token_ids support — fallback to 1 per chunk."""
    from tool_eval_bench.runner.throughput import _stream_one

    def handler(request: httpx.Request) -> httpx.Response:
        if "/chat/completions" not in str(request.url):
            return httpx.Response(404)

        lines = []
        for i in range(5):
            chunk = {"choices": [{"delta": {"content": f"word{i} "}}]}
            lines.append(_make_sse_line(chunk))

        usage = {"choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": 5}}
        lines.append(_make_sse_line(usage))
        lines.append("data: [DONE]\n\n")

        return httpx.Response(
            200,
            content="".join(lines).encode(),
            headers={"content-type": "text/event-stream"},
        )

    async with MeasurementTestClient(transport=httpx.MockTransport(handler)) as client:
        sample = await _stream_one(
            client,
            "http://localhost:8000/v1",
            "test-model",
            [{"role": "user", "content": "hi"}],
            5,
            None,
        )

    assert sample.error is None
    assert sample.tg_tokens == 5
    assert sample.mtp_chunks_detected is False
    assert len(sample.token_timestamps) == 5


@pytest.mark.asyncio
async def test_stream_one_retries_without_return_token_ids_on_strict_endpoint() -> None:
    """Strict OpenAI endpoints may reject the vLLM-only token_ids extension."""
    from tool_eval_bench.runner.throughput import _stream_one

    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if "return_token_ids" in body:
            return httpx.Response(400, json={"error": "unknown field"})
        return httpx.Response(
            200,
            content=(
                _make_sse_line({"choices": [{"delta": {"content": "ok"}}]}) + "data: [DONE]\n\n"
            ).encode(),
            headers={"content-type": "text/event-stream"},
        )

    cfg = TokenizerConfig()
    async with MeasurementTestClient(transport=httpx.MockTransport(handler)) as client:
        sample = await _stream_one(
            client,
            "http://localhost:8000/v1",
            "test-model",
            [{"role": "user", "content": "hi"}],
            5,
            None,
            cfg,
        )

    assert sample.error is None
    assert cfg.supports_return_token_ids is False
    assert "return_token_ids" in bodies[0]
    assert "return_token_ids" not in bodies[1]


class _DelayedSSEStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes], delay_seconds: float) -> None:
        self._chunks = chunks
        self._delay_seconds = delay_seconds

    async def __aiter__(self):
        for chunk in self._chunks:
            await asyncio.sleep(self._delay_seconds)
            yield chunk


@pytest.mark.asyncio
async def test_stream_one_ttft_waits_for_first_content_not_headers() -> None:
    """TTFT must include a delayed first usable SSE event after headers."""
    from tool_eval_bench.runner.throughput import _stream_one

    first = _make_sse_line({"choices": [{"delta": {"content": "ok"}}]}).encode()
    done = b"data: [DONE]\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            stream=_DelayedSSEStream([first, done], 0.025),
            headers={"content-type": "text/event-stream"},
        )

    async with MeasurementTestClient(transport=httpx.MockTransport(handler)) as client:
        sample = await _stream_one(
            client,
            "http://localhost:8000/v1",
            "test-model",
            [{"role": "user", "content": "hi"}],
            1,
            None,
        )

    assert sample.error is None
    assert sample.ttft_ms >= 20.0


@pytest.mark.parametrize("reasoning_field", ["reasoning", "reasoning_content"])
@pytest.mark.asyncio
async def test_stream_one_counts_reasoning_as_generated_output(reasoning_field: str) -> None:
    """Reasoning-model TTFT starts at the first generated reasoning token."""
    from tool_eval_bench.runner.throughput import _stream_one

    first = _make_sse_line(
        {
            "choices": [
                {
                    "delta": {reasoning_field: "thinking"},
                    "token_ids": [42],
                }
            ]
        }
    ).encode()
    done = b"data: [DONE]\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            stream=_DelayedSSEStream([first, done], 0.025),
            headers={"content-type": "text/event-stream"},
        )

    async with MeasurementTestClient(transport=httpx.MockTransport(handler)) as client:
        sample = await _stream_one(
            client,
            "http://localhost:8000/v1",
            "test-model",
            [{"role": "user", "content": "hi"}],
            1,
            None,
        )

    assert sample.error is None
    assert sample.ttft_ms >= 20.0
    assert sample.ttft_ms < sample.total_ms
    assert sample.tg_tokens == 1
    assert len(sample.token_timestamps) == 1


# ---------------------------------------------------------------------------
# Calibration via mock transport
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_calibrate_via_tokenize() -> None:
    """When /tokenize is available, calibration should use it."""
    from tool_eval_bench.runner.throughput import calibrate

    def mock_handler(request: httpx.Request) -> httpx.Response:
        if "/tokenize" in str(request.url):
            body = json.loads(request.content)
            # Simulate: every 4 characters = 1 token
            text = body.get("prompt", "")
            count = len(text) // 4
            return httpx.Response(200, json={"count": count})
        return httpx.Response(404)

    async with MeasurementTestClient(transport=httpx.MockTransport(mock_handler)) as client:
        cfg = await calibrate(client, "http://localhost:8000", "test-model")

    assert cfg.has_tokenize_endpoint is True
    assert cfg.chars_per_token > 0


@pytest.mark.asyncio
async def test_calibrate_fallback_to_probe() -> None:
    """When /tokenize returns 404, calibration falls back to probe request."""
    from tool_eval_bench.runner.throughput import calibrate

    def mock_handler(request: httpx.Request) -> httpx.Response:
        if "/tokenize" in str(request.url):
            return httpx.Response(404)
        if "/chat/completions" in str(request.url):
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "hi"}}],
                    "usage": {"prompt_tokens": 200, "completion_tokens": 1},
                },
            )
        return httpx.Response(404)

    async with MeasurementTestClient(transport=httpx.MockTransport(mock_handler)) as client:
        cfg = await calibrate(client, "http://localhost:8000", "test-model")

    assert cfg.has_tokenize_endpoint is False
    assert cfg.chars_per_token > 0


@pytest.mark.asyncio
async def test_calibrate_total_failure() -> None:
    """When everything fails, default config is returned."""
    from tool_eval_bench.runner.throughput import calibrate

    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    async with MeasurementTestClient(transport=httpx.MockTransport(mock_handler)) as client:
        cfg = await calibrate(client, "http://localhost:8000", "test-model")

    assert cfg.chars_per_token == _DEFAULT_CHARS_PER_TOKEN
    assert cfg.has_tokenize_endpoint is False


# ---------------------------------------------------------------------------
# Latency estimation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_estimate_latency() -> None:
    from tool_eval_bench.runner.throughput import estimate_latency

    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": []})

    async with MeasurementTestClient(transport=httpx.MockTransport(mock_handler)) as client:
        latency = await estimate_latency(client, "http://localhost:8000", rounds=3)

    assert latency >= 0.0


@pytest.mark.asyncio
async def test_estimate_latency_failure() -> None:
    """When all requests fail, latency should be 0.0."""
    from tool_eval_bench.runner.throughput import estimate_latency

    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    async with MeasurementTestClient(transport=httpx.MockTransport(mock_handler)) as client:
        latency = await estimate_latency(client, "http://localhost:8000", rounds=3)

    assert latency == 0.0
