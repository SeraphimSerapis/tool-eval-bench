"""Non-duplicate metrics_url forwarding regression tests from v1.2.2."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


class TestMetricsUrlOverride:
    @pytest.mark.asyncio
    async def test_scrape_uses_override(self) -> None:
        """scrape_spec_metrics should use metrics_url when provided."""
        from tool_eval_bench.runner.speculative import scrape_spec_metrics

        client = AsyncMock()
        resp = MagicMock()
        resp.status_code = 200
        resp.text = (
            "spec_decode_num_accepted_tokens_total 100\nspec_decode_num_draft_tokens_total 200\n"
        )
        client.metrics = AsyncMock(return_value=resp)

        result = await scrape_spec_metrics(
            client,
            "http://litellm:4000",
            metrics_url="http://vllm:8080/metrics",
        )
        # The semantic port receives the override, not a runner-built URL.
        client.metrics.assert_awaited_once_with(metrics_url="http://vllm:8080/metrics")
        assert result is not None
        assert result.accepted_tokens == 100.0

    @pytest.mark.asyncio
    async def test_detect_uses_override(self) -> None:
        """detect_spec_decoding should use metrics_url when provided."""
        from tool_eval_bench.runner.speculative import detect_spec_decoding

        client = AsyncMock()
        resp = MagicMock()
        resp.status_code = 200
        resp.text = "spec_decode_num_accepted_tokens_total 50\n"
        client.metrics = AsyncMock(return_value=resp)

        info = await detect_spec_decoding(
            client,
            "http://litellm:4000",
            metrics_url="http://vllm:8080/metrics",
        )
        assert info.active is True
        assert info.has_prometheus is True
        client.metrics.assert_awaited_once_with(metrics_url="http://vllm:8080/metrics")
