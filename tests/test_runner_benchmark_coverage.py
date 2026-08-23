"""High-value deterministic coverage for reporting, performance CLI, and datasets."""

from __future__ import annotations

import pytest

from tests.coverage_helpers import MagicAsyncClient
from tests.coverage_helpers import throughput_sample as _sample
from tool_eval_bench.runner.speculative import SpecDecodeSample
from tool_eval_bench.runner.throughput import ThroughputSample


@pytest.mark.asyncio
async def test_speculative_measurement_prometheus_and_llamacpp_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tool_eval_bench.runner import speculative

    counters = iter(
        [
            speculative.SpecDecodeCounters(accepted_tokens=10, draft_tokens=20, num_drafts=5),
            speculative.SpecDecodeCounters(accepted_tokens=24, draft_tokens=40, num_drafts=9),
        ]
    )

    async def scrape(*args: object, **kwargs: object):
        return next(counters)

    async def stream(*args: object, **kwargs: object):
        return _sample(draft_n=10, draft_n_accepted=6)

    monkeypatch.setattr(speculative, "scrape_spec_metrics", scrape)
    monkeypatch.setattr(speculative, "_stream_one", stream)
    sample = await speculative.measure_spec_single(
        MagicAsyncClient(),
        "url",
        "model",
        prompt_type="code",
        spec_info=speculative.SpecDecodeInfo(has_prometheus=True, method="mtp"),
    )
    assert sample.acceptance_rate == 0.7
    assert sample.acceptance_length == 4.5

    monkeypatch.setattr(speculative, "scrape_spec_metrics", lambda *args, **kwargs: None)
    fallback = await speculative.measure_spec_single(
        MagicAsyncClient(),
        "url",
        "model",
        prompt_type="structured",
        spec_info=speculative.SpecDecodeInfo(),
    )
    assert fallback.acceptance_rate == 0.6


@pytest.mark.asyncio
async def test_speculative_full_sweep_invokes_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    from tool_eval_bench.runner import speculative

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args: object):
            return None

    def client_factory(**kwargs: object) -> Client:
        return Client()

    monkeypatch.setattr(
        speculative,
        "calibrate",
        async_return(speculative.TokenizerConfig()),
    )
    monkeypatch.setattr(
        speculative,
        "detect_spec_decoding",
        async_return(speculative.SpecDecodeInfo(has_per_request_timings=True)),
    )

    async def measure(*args: object, depth=0, prompt_type="filler", **kwargs: object):
        return SpecDecodeSample(depth=depth, prompt_type=prompt_type)

    seen: list[tuple[int, int]] = []

    async def callback(sample: SpecDecodeSample, idx: int, total: int):
        seen.append((idx, total))

    monkeypatch.setattr(speculative, "measure_spec_single", measure)
    samples = await speculative.run_spec_bench(
        "url",
        "model",
        client_factory=client_factory,
        depths=[0, 1024],
        prompt_types=["filler", "code"],
        on_sample=callback,
    )

    assert len(samples) == 4
    assert seen == [(0, 4), (1, 4), (2, 4), (3, 4)]


def async_return(value: object):
    async def inner(*args: object, **kwargs: object):
        return value

    return inner


@pytest.mark.asyncio
async def test_throughput_matrix_sweep_and_exact_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    from tool_eval_bench.runner import speculative, throughput

    cfg = throughput.TokenizerConfig(has_tokenize_endpoint=True, chars_per_token=2.0)
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

    def client_factory(**kwargs: object) -> Client:
        return Client()

    monkeypatch.setattr(throughput, "warmup", async_return(1.0))
    monkeypatch.setattr(throughput, "calibrate", async_return(cfg))
    monkeypatch.setattr(throughput, "estimate_latency", async_return(2.0))
    monkeypatch.setattr(
        speculative,
        "detect_spec_decoding",
        async_return(speculative.SpecDecodeInfo(active=True, method="mtp")),
    )

    async def measure(*args: object, depth=0, concurrency=1, **kwargs: object):
        return _sample(depth=depth, concurrency=concurrency)

    seen: list[int] = []

    async def callback(sample: ThroughputSample, idx: int, total: int):
        seen.append(idx)

    monkeypatch.setattr(throughput, "measure_concurrent", measure)
    result = await throughput.run_throughput_matrix(
        "url",
        "model",
        client_factory=client_factory,
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

    async def stream(*args, **kwargs):
        return _sample(pp_tokens=10, tg_tokens=5, ttft_ms=2, pp_tps=100)

    monkeypatch.setattr(throughput, "_stream_one", stream)
    one = await throughput.measure_single(MagicAsyncClient(), "url", "m", pp=10, tg=5, depth=2)
    assert one.requested_depth == 2
    aggregate = await throughput.measure_concurrent(
        MagicAsyncClient(), "url", "m", pp=10, tg=5, depth=2, concurrency=2
    )
    assert aggregate.tg_tokens == 10 and aggregate.concurrency == 2

    calls = iter([_sample(error="bad"), RuntimeError("boom")])

    async def failing(*args, **kwargs):
        value = next(calls)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(throughput, "_stream_one", failing)
    failed = await throughput.measure_concurrent(MagicAsyncClient(), "url", "m", concurrency=2)
    assert failed.error and "bad" in failed.error
