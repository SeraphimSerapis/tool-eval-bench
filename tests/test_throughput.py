"""Tests for the throughput measurement module.

TEST-02: Tests TokenizerConfig, filler text generation, binary search setup,
calibration logic, and latency estimation — all without a real server.
"""

from __future__ import annotations

import json

import httpx
import pytest

from tool_eval_bench.runner.throughput import (
    TokenizerConfig,
    ThroughputSample,
    _build_filler_heuristic,
    _DEFAULT_CHARS_PER_TOKEN,
    _FILLER_PARAGRAPH,
    _headers,
    _tokenize_url,
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
# URL helpers
# ---------------------------------------------------------------------------


class TestUrlHelpers:
    def test_tokenize_url_strips_v1(self) -> None:
        assert _tokenize_url("http://localhost:8000/v1") == "http://localhost:8000/tokenize"

    def test_tokenize_url_no_v1(self) -> None:
        assert _tokenize_url("http://localhost:8000") == "http://localhost:8000/tokenize"

    def test_tokenize_url_trailing_slash(self) -> None:
        assert _tokenize_url("http://localhost:8000/") == "http://localhost:8000/tokenize"

    def test_headers_with_key(self) -> None:
        h = _headers("mykey")
        assert h["Authorization"] == "Bearer mykey"
        assert "Content-Type" in h

    def test_headers_without_key(self) -> None:
        h = _headers(None)
        assert "Authorization" not in h
        assert "Content-Type" in h


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

    def test_error_sample(self) -> None:
        s = ThroughputSample(error="Connection refused")
        assert s.error == "Connection refused"


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

    async with httpx.AsyncClient(transport=httpx.MockTransport(mock_handler)) as client:
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
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "hi"}}],
                "usage": {"prompt_tokens": 200, "completion_tokens": 1},
            })
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(mock_handler)) as client:
        cfg = await calibrate(client, "http://localhost:8000", "test-model")

    assert cfg.has_tokenize_endpoint is False
    assert cfg.chars_per_token > 0


@pytest.mark.asyncio
async def test_calibrate_total_failure() -> None:
    """When everything fails, default config is returned."""
    from tool_eval_bench.runner.throughput import calibrate

    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    async with httpx.AsyncClient(transport=httpx.MockTransport(mock_handler)) as client:
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

    async with httpx.AsyncClient(transport=httpx.MockTransport(mock_handler)) as client:
        latency = await estimate_latency(client, "http://localhost:8000", rounds=3)

    assert latency >= 0.0


@pytest.mark.asyncio
async def test_estimate_latency_failure() -> None:
    """When all requests fail, latency should be 0.0."""
    from tool_eval_bench.runner.throughput import estimate_latency

    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    async with httpx.AsyncClient(transport=httpx.MockTransport(mock_handler)) as client:
        latency = await estimate_latency(client, "http://localhost:8000", rounds=3)

    assert latency == 0.0
