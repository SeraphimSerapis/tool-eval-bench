"""Tests for the throughput measurement module.

TEST-02: Tests TokenizerConfig, filler text generation, binary search setup,
calibration logic, and latency estimation — all without a real server.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import httpx
import pytest

from tests.conftest import MeasurementTestClient
from tests.coverage_helpers import MagicAsyncClient
from tests.coverage_helpers import throughput_sample as _cov_sample
from tool_eval_bench.runner.throughput import (
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
        assert cfg.chars_per_token > 0
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
    def test_scales_proportionally_and_respects_chars_per_token(self) -> None:
        """Heuristic text scales with token target and TokenizerConfig.chars_per_token."""
        small = _build_filler_heuristic(100)
        large = _build_filler_heuristic(200)
        assert len(large) == len(small) * 2
        assert len(small.strip()) > 0

        cfg = TokenizerConfig(chars_per_token=2.5)
        result = _build_filler_heuristic(200, cfg)
        assert len(result) == int(200 * 2.5)

    def test_large_target_repeats_filler_content(self) -> None:
        """Very large targets scale cleanly past one base paragraph."""
        cfg = TokenizerConfig(chars_per_token=3.5)
        small = _build_filler_heuristic(100, cfg)
        result = _build_filler_heuristic(50_000, cfg)
        assert len(result) == int(50_000 * 3.5)
        assert small[:50] in result

    def test_zero_tokens(self) -> None:
        result = _build_filler_heuristic(0)
        assert result == ""


# ---------------------------------------------------------------------------
# ThroughputSample
# ---------------------------------------------------------------------------


class TestThroughputSample:
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


# Windows timers can wake asyncio.sleep several milliseconds early, so the TTFT
# assertions check half the delay: enough to tell a delayed first event from
# header arrival (about 0 ms) without depending on timer precision.
_FIRST_EVENT_DELAY_S = 0.05


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
            stream=_DelayedSSEStream([first, done], _FIRST_EVENT_DELAY_S),
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
    assert sample.ttft_ms >= _FIRST_EVENT_DELAY_S * 1000 / 2


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
            stream=_DelayedSSEStream([first, done], _FIRST_EVENT_DELAY_S),
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
    assert sample.ttft_ms >= _FIRST_EVENT_DELAY_S * 1000 / 2
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

    assert cfg.chars_per_token == TokenizerConfig().chars_per_token
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


# ---------------------------------------------------------------------------
# HTTP error statuses on a streamed body
# ---------------------------------------------------------------------------


class _StreamedBody(httpx.AsyncByteStream):
    """A body httpx has not buffered, as a real socket delivers it.

    A ``Response(content=...)`` is already in memory, so reading it after the
    stream context closed works in a test and fails in production.
    """

    def __init__(self, data: bytes) -> None:
        self._data = data

    async def __aiter__(self):  # type: ignore[override]
        yield self._data


def _sse_ok() -> httpx.Response:
    return httpx.Response(
        200,
        stream=_StreamedBody(
            (
                _make_sse_line({"choices": [{"delta": {"content": "ok"}}]}) + "data: [DONE]\n\n"
            ).encode()
        ),
        headers={"content-type": "text/event-stream"},
    )


async def _stream_against(handler, tok_cfg: TokenizerConfig) -> ThroughputSample:
    from tool_eval_bench.adapters.measurement import HTTPMeasurementClient
    from tool_eval_bench.runner.throughput import _stream_one

    async with HTTPMeasurementClient(
        base_url="http://gpu-box.internal:8000/v1",
        timeout=5,
        transport=httpx.MockTransport(handler),
    ) as client:
        return await _stream_one(
            client,
            "http://gpu-box.internal:8000/v1",
            "test-model",
            [{"role": "user", "content": "hi"}],
            5,
            None,
            tok_cfg,
            temperature=0.7,
        )


async def test_stream_one_retries_a_streamed_400_without_token_ids_at_the_same_temperature() -> (
    None
):
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if "return_token_ids" in body:
            return httpx.Response(400, stream=_StreamedBody(b'{"error": "unknown field"}'))
        return _sse_ok()

    cfg = TokenizerConfig()
    sample = await _stream_against(handler, cfg)

    assert sample.error is None
    assert sample.tg_tokens == 1
    assert cfg.supports_return_token_ids is False
    assert [("return_token_ids" in body, body["temperature"]) for body in bodies] == [
        (True, 0.7),
        (False, 0.7),
    ]


async def test_stream_one_returns_a_streamed_server_error_as_a_failed_sample() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(503, stream=_StreamedBody(b"overloaded"))

    cfg = TokenizerConfig()
    sample = await _stream_against(handler, cfg)

    assert len(requests) == 1
    assert sample.error is not None and "503" in sample.error
    assert "gpu-box" not in sample.error
    # A 5xx says nothing about the token_ids extension.
    assert cfg.supports_return_token_ids is None


class _ByteChunks(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


def _async_return(value: object):
    async def inner(*args: object, **kwargs: object):
        return value

    return inner


@pytest.mark.asyncio
async def test_throughput_matrix_sweep_and_exact_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    from tool_eval_bench.runner import speculative, throughput

    cfg = TokenizerConfig(has_tokenize_endpoint=True, chars_per_token=2.0)
    counts = iter([5, 15, 10])

    async def tokenize(*args: object, **kwargs: object):
        return next(counts)

    monkeypatch.setattr(throughput, "_tokenize_text", tokenize)
    exact = await throughput._build_exact_prompt(MagicAsyncClient(), "url", "model", 10, None, cfg)
    assert exact

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args: object):
            return None

    async def measure(*args: object, depth: int = 0, concurrency: int = 1, **kwargs: object):
        return _cov_sample(depth=depth, concurrency=concurrency)

    seen: list[int] = []

    async def callback(sample: ThroughputSample, idx: int, total: int) -> None:
        seen.append(idx)

    monkeypatch.setattr(throughput, "warmup", _async_return(1.0))
    monkeypatch.setattr(throughput, "calibrate", _async_return(cfg))
    monkeypatch.setattr(throughput, "estimate_latency", _async_return(2.0))
    monkeypatch.setattr(
        throughput,
        "detect_spec_decoding",
        _async_return(speculative.SpecDecodeInfo(active=True, method="mtp")),
    )
    monkeypatch.setattr(throughput, "measure_concurrent", measure)
    result = await throughput.run_throughput_matrix(
        "url",
        "model",
        client_factory=lambda **kwargs: Client(),
        depths=[0, 10],
        concurrency_levels=[2, 1],
        on_sample=callback,
    )

    assert result.spec_decoding_detected is True
    assert [s.concurrency for s in result.samples] == [1, 2, 1, 2]
    assert seen == [0, 1, 2, 3]


@pytest.mark.asyncio
async def test_throughput_single_and_concurrent_aggregation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tool_eval_bench.runner import throughput

    async def stream(*args: object, **kwargs: object):
        return _cov_sample(pp_tokens=10, tg_tokens=5, ttft_ms=2, pp_tps=100)

    monkeypatch.setattr(throughput, "_stream_one", stream)
    one = await throughput.measure_single(MagicAsyncClient(), "url", "m", pp=10, tg=5, depth=2)
    assert one.requested_depth == 2
    aggregate = await throughput.measure_concurrent(
        MagicAsyncClient(), "url", "m", pp=10, tg=5, depth=2, concurrency=2
    )
    assert aggregate.tg_tokens == 10 and aggregate.concurrency == 2

    calls = iter([_cov_sample(error="bad"), RuntimeError("boom")])

    async def failing(*args: object, **kwargs: object):
        value = next(calls)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(throughput, "_stream_one", failing)
    failed = await throughput.measure_concurrent(MagicAsyncClient(), "url", "m", concurrency=2)
    assert failed.error and "bad" in failed.error


@pytest.mark.asyncio
async def test_measure_concurrent_treats_partial_failure_as_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tool_eval_bench.runner import throughput

    calls = iter([_cov_sample(pp_tokens=10, tg_tokens=5), _cov_sample(error="503 overloaded")])

    async def mixed(*args: object, **kwargs: object):
        return next(calls)

    monkeypatch.setattr(throughput, "_stream_one", mixed)
    sample = await throughput.measure_concurrent(
        MagicAsyncClient(), "url", "m", pp=10, tg=5, depth=64, concurrency=2
    )
    assert sample.error is not None
    assert "1/2 requests failed" in sample.error
    assert "503 overloaded" in sample.error
    assert sample.concurrency == 2
    assert sample.requested_pp == 10
    assert sample.requested_depth == 64


@pytest.mark.asyncio
async def test_throughput_streamed_json_null_fields_and_cancellation() -> None:
    from tool_eval_bench.adapters.measurement import HTTPMeasurementClient
    from tool_eval_bench.runner.throughput import _stream_one, calibrate, measure_concurrent

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/tokenize"):
            return httpx.Response(404)
        mode = request.headers.get("x-tp-mode", "json-stream")
        if mode == "probe-null":
            return httpx.Response(200, json={"usage": {"prompt_tokens": None}})
        if mode == "json-stream":
            payload = json.dumps(
                {
                    "choices": [{"message": {"content": "hello world"}}],
                    "usage": {"prompt_tokens": 14, "completion_tokens": 2},
                }
            ).encode()
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                stream=_ByteChunks([payload]),
            )
        sse_chunks = (
            b"data: 5\r\n\r\n"
            b'data: {"choices": [{"delta": null}]}\r\n\r\n'
            b"data: {\r\n"
            b'data:   "choices": [{"delta": {"content": "tok"}}],\r\n'
            b'data:   "usage": {"prompt_tokens": 10, "completion_tokens": 3}\r\n'
            b"data: }\r\n\r\n"
            b'data: {"choices": [], "usage": {"prompt_tokens": null, "completion_tokens": null}}\r\n\r\n'
            b"data: [DONE]\r\n\r\n"
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/plain"},
            stream=_ByteChunks([sse_chunks]),
        )

    async with HTTPMeasurementClient(
        base_url="http://localhost:8000",
        timeout=5.0,
        completion_headers={"x-tp-mode": "probe-null"},
        transport=httpx.MockTransport(handler),
    ) as probe_client:
        tok_cfg = await calibrate(probe_client, "http://localhost:8000", "m")
        assert tok_cfg.calibration_confidence == "heuristic"

    async with HTTPMeasurementClient(
        base_url="http://localhost:8000",
        timeout=5.0,
        completion_headers={"x-tp-mode": "json-stream"},
        transport=httpx.MockTransport(handler),
    ) as json_client:
        sample = await _stream_one(
            json_client,
            "http://localhost:8000",
            "m",
            [{"role": "user", "content": "hi"}],
            4,
            None,
            TokenizerConfig(supports_return_token_ids=False),
        )
        assert sample.error is None
        assert sample.pp_tokens == 14
        assert sample.tg_tokens == 2

    async with HTTPMeasurementClient(
        base_url="http://localhost:8000",
        timeout=5.0,
        completion_headers={"x-tp-mode": "sse-nulls"},
        transport=httpx.MockTransport(handler),
    ) as sse_client:
        sample_sse = await _stream_one(
            sse_client,
            "http://localhost:8000",
            "m",
            [{"role": "user", "content": "hi"}],
            4,
            None,
            TokenizerConfig(supports_return_token_ids=False),
        )
        assert sample_sse.error is None
        assert sample_sse.pp_tokens == 10
        assert sample_sse.tg_tokens == 3

    def cancel_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/chat/completions"):
            raise asyncio.CancelledError("abort concurrent sweep")
        return httpx.Response(404)

    async with HTTPMeasurementClient(
        base_url="http://localhost:8000",
        timeout=5.0,
        transport=httpx.MockTransport(cancel_handler),
    ) as cancel_client:
        with pytest.raises(asyncio.CancelledError):
            await measure_concurrent(
                cancel_client,
                "http://localhost:8000",
                "m",
                pp=32,
                tg=8,
                concurrency=2,
                tok_cfg=TokenizerConfig(supports_return_token_ids=False),
            )


@pytest.mark.asyncio
async def test_warmup_applies_all_three_rewrites_across_four_attempts() -> None:
    from tool_eval_bench.adapters.measurement import HTTPMeasurementClient
    from tool_eval_bench.runner.throughput import warmup

    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if "max_tokens" in body:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "Unsupported parameter: max_tokens. Use max_completion_tokens."
                    }
                },
            )
        if "temperature" in body:
            return httpx.Response(400, json={"error": {"message": "temperature must be default"}})
        if "chat_template_kwargs" in body:
            return httpx.Response(400, text="unknown extra field")
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    tok_cfg = TokenizerConfig()
    await warmup(
        "http://localhost:8000/v1",
        "m",
        client_factory=lambda **kw: HTTPMeasurementClient(
            **kw, transport=httpx.MockTransport(handler)
        ),
        tok_cfg=tok_cfg,
    )
    assert len(seen) == 4
    assert tok_cfg.output_token_field == "max_completion_tokens"
    assert "max_tokens" not in seen[-1]
    assert "temperature" not in seen[-1]
    assert "chat_template_kwargs" not in seen[-1]
    assert seen[-1]["max_completion_tokens"] > 0


@pytest.mark.asyncio
async def test_stream_one_max_tokens_retry_preserves_include_token_ids_and_none_cfg() -> None:
    from tool_eval_bench.adapters.measurement import HTTPMeasurementClient
    from tool_eval_bench.runner.throughput import _stream_one

    seen_explicit: list[dict] = []

    def explicit_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen_explicit.append(body)
        if "max_tokens" in body:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "Unsupported parameter: max_tokens. Use max_completion_tokens instead."
                    }
                },
            )
        return _sse_ok()

    async with HTTPMeasurementClient(
        base_url="http://localhost:8000/v1",
        timeout=5.0,
        transport=httpx.MockTransport(explicit_handler),
    ) as client:
        cfg = TokenizerConfig()
        sample = await _stream_one(
            client,
            "http://localhost:8000/v1",
            "m",
            [{"role": "user", "content": "hi"}],
            4,
            None,
            cfg,
            _include_token_ids=False,
        )
    assert sample.error is None
    assert all("return_token_ids" not in b for b in seen_explicit)
    assert ["max_completion_tokens" in b for b in seen_explicit] == [False, True]

    seen_none: list[dict] = []

    def none_cfg_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen_none.append(body)
        if "max_tokens" in body:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "Unsupported parameter: max_tokens. Use max_completion_tokens instead."
                    }
                },
            )
        if "return_token_ids" in body:
            return httpx.Response(400, text="unknown field return_token_ids")
        return _sse_ok()

    async with HTTPMeasurementClient(
        base_url="http://localhost:8000/v1",
        timeout=5.0,
        transport=httpx.MockTransport(none_cfg_handler),
    ) as client:
        sample = await _stream_one(
            client,
            "http://localhost:8000/v1",
            "m",
            [{"role": "user", "content": "hi"}],
            4,
            None,
            tok_cfg=None,
        )
    assert sample.error is None
    assert len(seen_none) == 3
    assert "max_completion_tokens" in seen_none[-1]
    assert "return_token_ids" not in seen_none[-1]


@pytest.mark.asyncio
async def test_calibrate_switches_to_max_completion_tokens_on_400_retry() -> None:
    from tool_eval_bench.runner.throughput import calibrate

    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/tokenize"):
            return httpx.Response(404)
        body = json.loads(request.content)
        seen.append(body)
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
            json={
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 220, "completion_tokens": 1},
            },
        )

    async with MeasurementTestClient(transport=httpx.MockTransport(handler)) as client:
        cfg = await calibrate(client, "http://localhost:8000", "test-model")

    assert len(seen) == 2
    assert "max_tokens" in seen[0]
    assert "max_completion_tokens" in seen[1] and "max_tokens" not in seen[1]
    assert cfg.output_token_field == "max_completion_tokens"
    assert cfg.calibration_confidence == "probe"


@pytest.mark.asyncio
async def test_stream_one_returns_error_sample_on_connection_failure_before_response() -> None:
    from tool_eval_bench.runner.throughput import _stream_one

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    async with MeasurementTestClient(transport=httpx.MockTransport(handler)) as client:
        sample = await _stream_one(
            client,
            "http://localhost:8000/v1",
            "test-model",
            [{"role": "user", "content": "hi"}],
            8,
            None,
        )

    assert sample.error is not None
    assert "connection refused" in sample.error
    assert sample.total_ms >= 0.0


@pytest.mark.asyncio
async def test_stream_one_logs_mtp_warning_once_per_tokenizer_config(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from tool_eval_bench.runner.throughput import _stream_one

    handler, _ = _build_mtp_mock(tokens_per_chunk=3, num_chunks=2)
    cfg = TokenizerConfig()

    with caplog.at_level("INFO", logger="tool_eval_bench.runner.throughput"):
        async with MeasurementTestClient(transport=httpx.MockTransport(handler)) as client:
            first = await _stream_one(
                client,
                "http://localhost:8000/v1",
                "test-model",
                [{"role": "user", "content": "hi"}],
                6,
                None,
                cfg,
            )
            second = await _stream_one(
                client,
                "http://localhost:8000/v1",
                "test-model",
                [{"role": "user", "content": "hi"}],
                6,
                None,
                cfg,
            )

    assert first.mtp_chunks_detected is True
    assert second.mtp_chunks_detected is True
    assert cfg.mtp_warned is True
    mtp_logs = [
        rec for rec in caplog.records if "Multi-token prediction (MTP) detected" in rec.message
    ]
    assert len(mtp_logs) == 1


@pytest.mark.asyncio
async def test_tokenize_failure_falls_back_to_heuristic_filler_for_user_and_depth_prompts() -> None:
    from tool_eval_bench.runner.throughput import _build_exact_prompt, _build_messages

    # 1. When has_tokenize_endpoint is False, _build_exact_prompt uses heuristic directly.
    heuristic_cfg = TokenizerConfig(has_tokenize_endpoint=False, chars_per_token=4.0)
    direct = await _build_exact_prompt(
        MagicAsyncClient(), "http://localhost:8000", "m", 25, None, heuristic_cfg
    )
    assert len(direct) == 100

    # 2. When /tokenize raises a transport error, _tokenize_text returns None and both
    #    depth system prompt and user prompt fall back to heuristic filler cleanly.
    def failing_tokenize(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("tokenize endpoint unreachable")

    tok_cfg = TokenizerConfig(has_tokenize_endpoint=True, chars_per_token=3.0)
    async with MeasurementTestClient(transport=httpx.MockTransport(failing_tokenize)) as client:
        messages = await _build_messages(
            client,
            "http://localhost:8000",
            "m",
            pp=40,
            depth=20,
            api_key=None,
            tok_cfg=tok_cfg,
        )

    assert [m["role"] for m in messages] == ["system", "user"]
    assert len(messages[0]["content"]) == 60
    assert len(messages[1]["content"]) == 120
