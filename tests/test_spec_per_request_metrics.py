"""vLLM ``--per-request-spec-decode-metrics`` support in spec-bench.

The response-body field is optional: a server without the flag never sends
it and every pre-existing source (Prometheus deltas, llama.cpp timings) has
to keep working unchanged. The fixture below is the exact shape a vLLM
server in ``detailed`` mode returned on the final usage chunk.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from rich.console import Console

from tests.conftest import MeasurementTestClient
from tool_eval_bench.cli.spec_bench import run_spec_bench as run_spec_bench_cli
from tool_eval_bench.domain.spec_decode import per_position_acceptance
from tool_eval_bench.runner import speculative
from tool_eval_bench.runner.speculative import (
    SpecDecodeCounters,
    SpecDecodeInfo,
    SpecDecodeSample,
    parse_per_request_spec_metrics,
)
from tool_eval_bench.runner.throughput import ThroughputSample, _stream_one
from tool_eval_bench.storage.reports.spec_decode import write_spec_decode_report

DETAILED = {
    "mean_acceptance_length": 6.583333333333333,
    "draft_acceptance_rate": 0.7976190476190477,
    "acceptance_histogram": [0, 1, 0, 1, 1, 2, 0, 7],
    "num_spec_steps": 12,
    "num_accepted_draft_tokens": 67,
    "num_draft_tokens": 84,
    "num_spec_tokens": 7,
    "per_step_accepted": [7, 5, 5, 4, 1, 7, 7, 7, 7, 7, 7, 3],
    "per_step_drafted": [7] * 12,
}
SUMMARY = {k: v for k, v in DETAILED.items() if not k.startswith("per_step")}


# ---------------------------------------------------------------------------
# domain.spec_decode.per_position_acceptance
# ---------------------------------------------------------------------------


def test_per_position_counts_prefix_acceptance() -> None:
    rates = per_position_acceptance([(DETAILED["per_step_accepted"], DETAILED["per_step_drafted"])])
    # 12 steps drafted every position; accepted-at-p is the number of steps
    # whose accepted prefix reaches past p.
    assert rates == pytest.approx([12 / 12, 11 / 12, 11 / 12, 10 / 12, 9 / 12, 7 / 12, 7 / 12])


def test_per_position_handles_variable_draft_lengths_and_aggregates() -> None:
    # Two requests: an ngram-style drafter with uneven draft lengths, and a
    # fixed-k one. Position 2 is only drafted in three of the four steps.
    rates = per_position_acceptance(
        [
            ([2, 1], [3, 1]),
            ([3, 0], [3, 3]),
        ]
    )
    assert rates == pytest.approx([3 / 4, 2 / 3, 1 / 3])


def test_per_position_skips_mismatched_arrays_and_handles_empty() -> None:
    assert per_position_acceptance([]) == []
    assert per_position_acceptance([([1, 2], [3])]) == []
    # A step that "accepted" more than it drafted cannot inflate a position.
    assert per_position_acceptance([([5], [2])]) == [1.0, 1.0]


# ---------------------------------------------------------------------------
# parse_per_request_spec_metrics
# ---------------------------------------------------------------------------


def test_parse_detailed_mode() -> None:
    parsed = parse_per_request_spec_metrics(DETAILED)
    assert parsed is not None
    assert (parsed.num_draft_tokens, parsed.num_accepted_draft_tokens) == (84, 67)
    assert parsed.num_spec_steps == 12
    assert parsed.num_spec_tokens == 7
    assert parsed.per_step_accepted == DETAILED["per_step_accepted"]
    assert parsed.per_step_drafted == [7] * 12


def test_parse_summary_mode_has_no_step_arrays() -> None:
    parsed = parse_per_request_spec_metrics(SUMMARY)
    assert parsed is not None
    assert parsed.num_spec_tokens == 7
    assert parsed.per_step_accepted is None
    assert parsed.per_step_drafted is None


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "not a dict",
        {},
        {**SUMMARY, "num_draft_tokens": "84"},
        {**SUMMARY, "num_spec_steps": -1},
        {**SUMMARY, "num_accepted_draft_tokens": True},
        {k: v for k, v in SUMMARY.items() if k != "num_spec_steps"},
    ],
)
def test_parse_rejects_missing_or_malformed_counters(raw: object) -> None:
    assert parse_per_request_spec_metrics(raw) is None


def test_parse_drops_step_arrays_that_cannot_be_paired() -> None:
    parsed = parse_per_request_spec_metrics({**DETAILED, "per_step_drafted": [7] * 11})
    assert parsed is not None
    assert parsed.per_step_accepted is None
    assert parsed.per_step_drafted is None
    parsed = parse_per_request_spec_metrics({**DETAILED, "per_step_accepted": [7.0] * 12})
    assert parsed is not None
    assert parsed.per_step_accepted is None
    # A bad num_spec_tokens is dropped, not fatal.
    parsed = parse_per_request_spec_metrics({**SUMMARY, "num_spec_tokens": "7"})
    assert parsed is not None
    assert parsed.num_spec_tokens is None


# ---------------------------------------------------------------------------
# SpecDecodeSample
# ---------------------------------------------------------------------------


def test_apply_per_request_metrics_fills_rates_and_source() -> None:
    sample = SpecDecodeSample(tg_tokens=80, total_ms=1100, ttft_ms=240)
    parsed = parse_per_request_spec_metrics(DETAILED)
    assert parsed is not None
    sample.apply_per_request_metrics(parsed)
    assert sample.acceptance_source == "response"
    assert sample.acceptance_rate == pytest.approx(67 / 84)
    # Matches vLLM's own mean_acceptance_length, which counts the bonus token.
    assert sample.acceptance_length == pytest.approx(DETAILED["mean_acceptance_length"])
    assert sample.draft_window == 7.0
    assert sample.num_spec_tokens == 7
    per_pos = sample.per_position_acceptance
    assert per_pos is not None and len(per_pos) == 7


def test_apply_per_request_metrics_with_zero_steps_leaves_rates_unset() -> None:
    sample = SpecDecodeSample()
    sample.apply_per_request_metrics(
        speculative.PerRequestSpecMetrics(
            num_draft_tokens=0, num_accepted_draft_tokens=0, num_spec_steps=0
        )
    )
    assert sample.acceptance_source == "response"
    assert sample.acceptance_rate is None
    assert sample.acceptance_length is None
    assert sample.per_position_acceptance is None


# ---------------------------------------------------------------------------
# _stream_one captures the field from the final usage chunk
# ---------------------------------------------------------------------------


def _sse_handler(final_extra: dict) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        chunks = [
            {"choices": [{"delta": {"content": "one"}, "token_ids": [1]}]},
            {"choices": [{"delta": {"content": "two"}, "token_ids": [2]}]},
            {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2}, **final_extra},
        ]
        body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        return httpx.Response(
            200, content=body.encode(), headers={"content-type": "text/event-stream"}
        )

    return httpx.MockTransport(handler)


async def _run_stream(transport: httpx.MockTransport) -> ThroughputSample:
    async with MeasurementTestClient(transport=transport) as client:
        return await _stream_one(
            client, "http://localhost:8000/v1", "m", [{"role": "user", "content": "hi"}], 2, None
        )


@pytest.mark.asyncio
async def test_stream_one_captures_spec_metrics_from_usage_chunk() -> None:
    sample = await _run_stream(_sse_handler({"metrics": {"speculative_decoding": DETAILED}}))
    assert sample.error is None
    assert sample.tg_tokens == 2
    assert sample.spec_decode_metrics == DETAILED


@pytest.mark.asyncio
async def test_stream_one_without_the_field_leaves_metrics_none() -> None:
    plain = await _run_stream(_sse_handler({}))
    assert plain.spec_decode_metrics is None
    # vLLM timing metrics without the spec block, as with the flag set to none.
    timing_only = await _run_stream(_sse_handler({"metrics": {"time_to_first_token_ms": 3.0}}))
    assert timing_only.spec_decode_metrics is None


@pytest.mark.asyncio
async def test_stream_one_flags_timings_without_draft_counts() -> None:
    # llama.cpp's shape when the request drafted nothing.
    sample = await _run_stream(_sse_handler({"timings": {"prompt_n": 5, "predicted_n": 2}}))
    assert sample.timings_present is True
    assert sample.draft_n is None


@pytest.mark.asyncio
async def test_stream_one_reads_draft_counts_from_timings() -> None:
    sample = await _run_stream(
        _sse_handler({"timings": {"predicted_n": 2, "draft_n": 4, "draft_n_accepted": 3}})
    )
    assert sample.timings_present is True
    assert (sample.draft_n, sample.draft_n_accepted) == (4, 3)


@pytest.mark.asyncio
async def test_stream_one_without_timings_leaves_the_flag_unset() -> None:
    assert (await _run_stream(_sse_handler({}))).timings_present is False
    assert (await _run_stream(_sse_handler({"timings": None}))).timings_present is False


# ---------------------------------------------------------------------------
# measure_spec_single source selection
# ---------------------------------------------------------------------------


def _patch_stream(monkeypatch: pytest.MonkeyPatch, sample: ThroughputSample) -> None:
    async def stream(*args: object, **kwargs: object) -> ThroughputSample:
        return sample

    monkeypatch.setattr(speculative, "_stream_one", stream)


def _patch_scrape(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []

    async def scrape(*args: object, **kwargs: object) -> SpecDecodeCounters:
        calls.append(1)
        return SpecDecodeCounters(accepted_tokens=1000, draft_tokens=2000, num_drafts=100)

    monkeypatch.setattr(speculative, "scrape_spec_metrics", scrape)
    return calls


@pytest.mark.asyncio
async def test_response_metrics_win_over_prometheus_delta(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_stream(monkeypatch, ThroughputSample(tg_tokens=80, spec_decode_metrics=DETAILED))
    calls = _patch_scrape(monkeypatch)
    sample = await speculative.measure_spec_single(
        None,  # type: ignore[arg-type]
        "url",
        "m",
        prompt_type="code",
        spec_info=SpecDecodeInfo(has_prometheus=True),
    )
    assert sample.acceptance_source == "response"
    assert sample.accepted_tokens_delta == 67
    # The before-scrape ran (the server had not proven itself yet); the
    # after-scrape was skipped because the response already had the answer.
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_known_per_request_server_skips_scrapes_entirely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_stream(monkeypatch, ThroughputSample(tg_tokens=80, spec_decode_metrics=SUMMARY))
    calls = _patch_scrape(monkeypatch)
    sample = await speculative.measure_spec_single(
        None,  # type: ignore[arg-type]
        "url",
        "m",
        prompt_type="code",
        spec_info=SpecDecodeInfo(has_prometheus=True, has_per_request_metrics=True),
    )
    assert sample.acceptance_source == "response"
    assert sample.per_position_acceptance is None
    assert calls == []


@pytest.mark.asyncio
async def test_server_without_the_flag_still_uses_prometheus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_stream(monkeypatch, ThroughputSample(tg_tokens=80))
    calls = _patch_scrape(monkeypatch)
    sample = await speculative.measure_spec_single(
        None,  # type: ignore[arg-type]
        "url",
        "m",
        prompt_type="code",
        spec_info=SpecDecodeInfo(has_prometheus=True),
    )
    assert sample.acceptance_source == "prometheus"
    assert len(calls) == 2
    assert sample.per_position_acceptance is None


@pytest.mark.asyncio
async def test_llamacpp_timings_are_labelled(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_stream(monkeypatch, ThroughputSample(tg_tokens=80, draft_n=10, draft_n_accepted=6))
    sample = await speculative.measure_spec_single(
        None,  # type: ignore[arg-type]
        "url",
        "m",
        prompt_type="code",
        spec_info=SpecDecodeInfo(has_per_request_timings=True),
    )
    assert sample.acceptance_source == "timings"
    assert sample.acceptance_rate == 0.6


def _patch_scrape_sequence(
    monkeypatch: pytest.MonkeyPatch, *counters: SpecDecodeCounters
) -> list[int]:
    """Return *counters* in order, one per scrape: before, then after."""
    calls: list[int] = []

    async def scrape(*args: object, **kwargs: object) -> SpecDecodeCounters:
        calls.append(1)
        return counters[len(calls) - 1]

    monkeypatch.setattr(speculative, "scrape_spec_metrics", scrape)
    return calls


# Counters from a live llama-server around one 96-token request that drafted
# 98 tokens, accepted 70, in 25 verification steps.
_LLAMACPP_BEFORE = SpecDecodeCounters(accepted_tokens=1787, draft_tokens=5017, num_drafts=1261)
_LLAMACPP_AFTER = SpecDecodeCounters(accepted_tokens=1857, draft_tokens=5115, num_drafts=1286)


async def _measure_llamacpp() -> SpecDecodeSample:
    return await speculative.measure_spec_single(
        None,  # type: ignore[arg-type]
        "url",
        "m",
        prompt_type="code",
        spec_info=SpecDecodeInfo(has_prometheus=True, has_per_request_timings=True),
    )


@pytest.mark.asyncio
async def test_timings_win_over_a_delta_polluted_by_other_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_stream(monkeypatch, ThroughputSample(tg_tokens=96, draft_n=98, draft_n_accepted=70))
    # Another request drafted 40 tokens (12 accepted, 9 steps) in the same window.
    polluted = SpecDecodeCounters(accepted_tokens=1869, draft_tokens=5155, num_drafts=1295)
    calls = _patch_scrape_sequence(monkeypatch, _LLAMACPP_BEFORE, polluted)

    sample = await _measure_llamacpp()

    assert len(calls) == 2
    assert sample.acceptance_source == "timings"
    assert (sample.draft_tokens_delta, sample.accepted_tokens_delta) == (98, 70)
    assert sample.acceptance_rate == pytest.approx(70 / 98)
    # The step delta includes the other request's steps, so it is not used.
    assert sample.num_drafts_delta is None
    assert sample.acceptance_length is None


@pytest.mark.asyncio
async def test_timings_take_the_step_count_when_the_delta_is_this_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_stream(monkeypatch, ThroughputSample(tg_tokens=96, draft_n=98, draft_n_accepted=70))
    _patch_scrape_sequence(monkeypatch, _LLAMACPP_BEFORE, _LLAMACPP_AFTER)

    sample = await _measure_llamacpp()

    assert sample.acceptance_source == "timings"
    assert sample.acceptance_rate == pytest.approx(70 / 98)
    assert sample.num_drafts_delta == 25
    assert sample.acceptance_length == pytest.approx(1 + 70 / 25)
    assert sample.draft_window == pytest.approx(98 / 25)


@pytest.mark.asyncio
async def test_timings_reject_a_step_count_when_only_the_draft_total_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_stream(monkeypatch, ThroughputSample(tg_tokens=96, draft_n=98, draft_n_accepted=70))
    off_by_one = SpecDecodeCounters(accepted_tokens=1858, draft_tokens=5115, num_drafts=1286)
    _patch_scrape_sequence(monkeypatch, _LLAMACPP_BEFORE, off_by_one)

    sample = await _measure_llamacpp()

    assert sample.acceptance_source == "timings"
    assert sample.accepted_tokens_delta == 70
    assert sample.num_drafts_delta is None


@pytest.mark.asyncio
async def test_strata_timings_win_over_its_prometheus_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strata puts draft_n in timings and exports no step counter."""
    _patch_stream(monkeypatch, ThroughputSample(tg_tokens=64, draft_n=48, draft_n_accepted=30))
    before = SpecDecodeCounters(accepted_tokens=90, draft_tokens=120, has_num_drafts=False)
    after = SpecDecodeCounters(accepted_tokens=150, draft_tokens=200, has_num_drafts=False)
    _patch_scrape_sequence(monkeypatch, before, after)

    sample = await speculative.measure_spec_single(
        None,  # type: ignore[arg-type]
        "url",
        "m",
        prompt_type="code",
        spec_info=SpecDecodeInfo(has_prometheus=True, method="mtp"),
    )

    assert sample.acceptance_source == "timings"
    assert (sample.draft_tokens_delta, sample.accepted_tokens_delta) == (48, 30)
    assert sample.acceptance_rate == pytest.approx(30 / 48)
    assert sample.acceptance_length is None


@pytest.mark.asyncio
async def test_llamacpp_timings_without_draft_n_mean_zero_drafts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """llama.cpp omits draft_n exactly when it drafted nothing."""
    _patch_stream(monkeypatch, ThroughputSample(tg_tokens=96, timings_present=True))
    # Another request drafted 40 tokens in the same window.
    polluted = SpecDecodeCounters(accepted_tokens=1799, draft_tokens=5057, num_drafts=1270)
    _patch_scrape_sequence(monkeypatch, _LLAMACPP_BEFORE, polluted)

    sample = await _measure_llamacpp()

    assert sample.acceptance_source == "timings"
    assert (sample.draft_tokens_delta, sample.accepted_tokens_delta) == (0, 0)
    assert sample.acceptance_rate is None
    assert sample.num_drafts_delta is None


@pytest.mark.asyncio
async def test_llamacpp_zero_drafts_take_the_step_count_of_an_idle_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_stream(monkeypatch, ThroughputSample(tg_tokens=96, timings_present=True))
    _patch_scrape_sequence(monkeypatch, _LLAMACPP_BEFORE, _LLAMACPP_BEFORE)

    sample = await _measure_llamacpp()

    assert sample.acceptance_source == "timings"
    assert (sample.draft_tokens_delta, sample.num_drafts_delta) == (0, 0)
    assert sample.acceptance_length is None


@pytest.mark.asyncio
async def test_strata_timings_without_draft_n_fall_back_to_prometheus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strata omits draft_n when its engine reported nothing, not only at zero."""
    _patch_stream(monkeypatch, ThroughputSample(tg_tokens=64, timings_present=True))
    before = SpecDecodeCounters(accepted_tokens=90, draft_tokens=120, has_num_drafts=False)
    after = SpecDecodeCounters(accepted_tokens=150, draft_tokens=200, has_num_drafts=False)
    _patch_scrape_sequence(monkeypatch, before, after)

    sample = await speculative.measure_spec_single(
        None,  # type: ignore[arg-type]
        "url",
        "m",
        prompt_type="code",
        spec_info=SpecDecodeInfo(has_prometheus=True, method="mtp"),
    )

    assert sample.acceptance_source == "prometheus"
    assert (sample.draft_tokens_delta, sample.accepted_tokens_delta) == (80, 60)


@pytest.mark.asyncio
async def test_llamacpp_in_a_label_value_does_not_zero_missing_draft_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only llama.cpp's own namespace makes a missing draft_n an exact zero.

    A server with no speculative counters whose /metrics merely mentions
    ``llamacpp:`` inside a label value used to be detected as llama.cpp, so a
    response ``timings`` object without ``draft_n`` was reported as an exact
    zero-draft measurement from response timings.
    """
    metrics = 'vllm:num_requests_running{model_name="acme/llamacpp:7b"} 1\n'
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text=metrics))
    async with MeasurementTestClient(transport=transport) as client:
        spec_info = await speculative.detect_spec_decoding(
            client, "http://test", backend_hint="mtp"
        )
    _patch_stream(monkeypatch, ThroughputSample(tg_tokens=64, timings_present=True))

    sample = await speculative.measure_spec_single(
        None,  # type: ignore[arg-type]
        "url",
        "m",
        prompt_type="code",
        spec_info=spec_info,
    )

    assert spec_info.has_per_request_timings is False
    assert sample.acceptance_source is None
    assert sample.draft_tokens_delta is None


@pytest.mark.asyncio
async def test_llamacpp_response_without_timings_falls_back_to_prometheus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_stream(monkeypatch, ThroughputSample(tg_tokens=96))
    _patch_scrape_sequence(monkeypatch, _LLAMACPP_BEFORE, _LLAMACPP_AFTER)

    sample = await _measure_llamacpp()

    assert sample.acceptance_source == "prometheus"
    assert (sample.draft_tokens_delta, sample.accepted_tokens_delta) == (98, 70)


@pytest.mark.asyncio
async def test_llamacpp_response_without_timings_or_prometheus_has_no_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_stream(monkeypatch, ThroughputSample(tg_tokens=96))
    sample = await speculative.measure_spec_single(
        None,  # type: ignore[arg-type]
        "url",
        "m",
        prompt_type="code",
        spec_info=SpecDecodeInfo(has_per_request_timings=True),
    )
    assert sample.acceptance_source is None
    assert sample.draft_tokens_delta is None


@pytest.mark.asyncio
async def test_sweep_with_timings_does_not_warn_about_cross_talk(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    class Client:
        async def __aenter__(self) -> Client:
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

    async def calibrate(*args: object, **kwargs: object) -> speculative.TokenizerConfig:
        return speculative.TokenizerConfig()

    async def detect(*args: object, **kwargs: object) -> SpecDecodeInfo:
        return SpecDecodeInfo(has_prometheus=True, has_per_request_timings=True)

    monkeypatch.setattr(speculative, "calibrate", calibrate)
    monkeypatch.setattr(speculative, "detect_spec_decoding", detect)
    _patch_stream(monkeypatch, ThroughputSample(tg_tokens=96, draft_n=98, draft_n_accepted=70))
    polluted = SpecDecodeCounters(accepted_tokens=1869, draft_tokens=5155, num_drafts=1295)
    _patch_scrape_sequence(monkeypatch, _LLAMACPP_BEFORE, polluted)

    with caplog.at_level("WARNING", logger=speculative.logger.name):
        samples = await speculative.run_spec_bench(
            "url", "m", client_factory=lambda **kwargs: Client(), prompt_types=["code"]
        )

    assert [s.acceptance_source for s in samples] == ["timings"]
    assert samples[0].acceptance_rate == pytest.approx(70 / 98)
    assert not [r for r in caplog.records if "server-wide aggregates" in r.getMessage()]


@pytest.mark.asyncio
async def test_no_source_at_all_leaves_source_none(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_stream(monkeypatch, ThroughputSample(tg_tokens=80))
    sample = await speculative.measure_spec_single(
        None,  # type: ignore[arg-type]
        "url",
        "m",
        prompt_type="code",
        spec_info=SpecDecodeInfo(),
    )
    assert sample.acceptance_source is None
    assert sample.acceptance_rate is None


@pytest.mark.asyncio
async def test_sweep_learns_per_request_support_from_first_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Client:
        async def __aenter__(self) -> Client:
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

    info = SpecDecodeInfo(has_prometheus=True)

    async def calibrate(*args: object, **kwargs: object) -> speculative.TokenizerConfig:
        return speculative.TokenizerConfig()

    async def detect(*args: object, **kwargs: object) -> SpecDecodeInfo:
        return info

    seen_flags: list[bool] = []

    async def measure(
        *args: object, spec_info: SpecDecodeInfo, prompt_type: str = "filler", **kwargs: object
    ) -> SpecDecodeSample:
        seen_flags.append(spec_info.has_per_request_metrics)
        sample = SpecDecodeSample(prompt_type=prompt_type)
        parsed = parse_per_request_spec_metrics(SUMMARY)
        assert parsed is not None
        sample.apply_per_request_metrics(parsed)
        return sample

    monkeypatch.setattr(speculative, "calibrate", calibrate)
    monkeypatch.setattr(speculative, "detect_spec_decoding", detect)
    monkeypatch.setattr(speculative, "measure_spec_single", measure)
    samples = await speculative.run_spec_bench(
        "url", "m", client_factory=lambda **kwargs: Client(), prompt_types=["filler", "code"]
    )
    assert len(samples) == 2
    # First sample runs with the flag off, every later one with it on.
    assert seen_flags == [False, True]
    assert info.has_per_request_metrics is True


@pytest.mark.asyncio
async def test_sweep_warns_about_cross_talk_only_when_prometheus_was_used(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    class Client:
        async def __aenter__(self) -> Client:
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

    async def calibrate(*args: object, **kwargs: object) -> speculative.TokenizerConfig:
        return speculative.TokenizerConfig()

    async def detect(*args: object, **kwargs: object) -> SpecDecodeInfo:
        return SpecDecodeInfo(has_prometheus=True)

    source = "prometheus"

    async def measure(*args: object, prompt_type: str = "filler", **kwargs: object):
        return SpecDecodeSample(prompt_type=prompt_type, acceptance_source=source)

    monkeypatch.setattr(speculative, "calibrate", calibrate)
    monkeypatch.setattr(speculative, "detect_spec_decoding", detect)
    monkeypatch.setattr(speculative, "measure_spec_single", measure)

    with caplog.at_level("WARNING", logger=speculative.logger.name):
        await speculative.run_spec_bench(
            "url", "m", client_factory=lambda **kwargs: Client(), prompt_types=["filler", "code"]
        )
    warnings = [r for r in caplog.records if "server-wide aggregates" in r.getMessage()]
    assert len(warnings) == 1
    assert "--per-request-spec-decode-metrics" in warnings[0].getMessage()

    caplog.clear()
    source = "response"
    with caplog.at_level("WARNING", logger=speculative.logger.name):
        await speculative.run_spec_bench(
            "url", "m", client_factory=lambda **kwargs: Client(), prompt_types=["filler"]
        )
    assert not [r for r in caplog.records if "server-wide aggregates" in r.getMessage()]


# ---------------------------------------------------------------------------
# CLI summary and Markdown report
# ---------------------------------------------------------------------------


def _response_sample(prompt_type: str) -> SpecDecodeSample:
    sample = SpecDecodeSample(
        tg_tokens=80, total_ms=1100, ttft_ms=240, tg_tps=70, prompt_type=prompt_type
    )
    parsed = parse_per_request_spec_metrics(DETAILED)
    assert parsed is not None
    sample.apply_per_request_metrics(parsed)
    return sample


def _cli_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, samples: list) -> str:
    from tool_eval_bench.storage import reports

    async def fake_run(*args: object, on_sample=None, **kwargs: object):
        for idx, sample in enumerate(samples):
            await on_sample(sample, idx, len(samples))
        return samples

    monkeypatch.setattr(speculative, "run_spec_bench", fake_run)
    monkeypatch.setattr(
        reports.MarkdownReporter,
        "write_spec_decode_report",
        lambda *args, **kwargs: tmp_path / "spec.md",
    )
    console = Console(record=True, width=200)
    run_spec_bench_cli(
        console,
        "m",
        "Display",
        "http://test/v1",
        None,
        pp=100,
        tg=80,
        depths=[0],
        output_dir=str(tmp_path),
        metadata_for_storage=lambda _: {},
        with_config_fingerprint=lambda config: {**config, "config_fingerprint": "fp"},
        persist_plugin_run=lambda _: None,
    )
    return console.export_text()


def test_cli_reports_response_source_and_per_position(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = _cli_output(monkeypatch, tmp_path, [_response_sample("code")])
    assert "per-request response metrics" in output
    assert "Per-position acceptance:" in output
    assert "0:100%" in output
    assert "1:92%" in output
    assert "6:58%" in output
    assert "Prometheus counter deltas" not in output


def test_cli_points_prometheus_users_at_the_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sample = SpecDecodeSample(
        tg_tokens=80,
        total_ms=1100,
        acceptance_rate=0.7,
        draft_tokens_delta=40,
        accepted_tokens_delta=28,
        num_drafts_delta=5,
        acceptance_source="prometheus",
        prompt_type="code",
    )
    output = _cli_output(monkeypatch, tmp_path, [sample])
    assert "Prometheus counter deltas" in output
    assert "--per-request-spec-decode-metrics" in output
    assert "Per-position acceptance" not in output


def test_report_has_source_line_and_per_position_section(tmp_path: Path) -> None:
    path = write_spec_decode_report(
        tmp_path, "run", "m", [_response_sample("code"), _response_sample("filler")]
    )
    text = path.read_text(encoding="utf-8")
    assert "- **Acceptance Source**: per-request response metrics (exact)" in text
    assert "## Per-Position Acceptance" in text
    assert "| 0 | 100.0% |" in text
    assert "| 1 | 91.7% |" in text
    assert "| 6 | 58.3% |" in text
    assert "Prometheus" not in text.split("## Interpretation Guide")[0]


def test_report_flags_prometheus_source_without_per_position(tmp_path: Path) -> None:
    sample = SpecDecodeSample(
        tg_tokens=80,
        total_ms=1100,
        acceptance_rate=0.7,
        draft_tokens_delta=40,
        accepted_tokens_delta=28,
        num_drafts_delta=5,
        acceptance_source="prometheus",
        prompt_type="code",
    )
    text = write_spec_decode_report(tmp_path, "run", "m", [sample]).read_text(encoding="utf-8")
    assert "- **Acceptance Source**: prometheus" in text
    assert "--per-request-spec-decode-metrics" in text
    assert "## Per-Position Acceptance" not in text


def test_report_without_any_source_is_unchanged(tmp_path: Path) -> None:
    sample = SpecDecodeSample(tg_tokens=80, total_ms=1100, prompt_type="code")
    text = write_spec_decode_report(tmp_path, "run", "m", [sample]).read_text(encoding="utf-8")
    assert "Acceptance Source" not in text
    assert "Acceptance rate metrics were not available" in text


def _timings_sample() -> SpecDecodeSample:
    return SpecDecodeSample(
        tg_tokens=96,
        total_ms=1100,
        acceptance_rate=70 / 98,
        draft_tokens_delta=98,
        accepted_tokens_delta=70,
        acceptance_source="timings",
        prompt_type="code",
    )


def test_cli_names_response_timings_as_exact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = _cli_output(monkeypatch, tmp_path, [_timings_sample()])
    assert "per-request response timings" in output
    assert "Prometheus counter deltas" not in output


def test_report_names_response_timings_as_exact(tmp_path: Path) -> None:
    text = write_spec_decode_report(tmp_path, "run", "m", [_timings_sample()]).read_text(
        encoding="utf-8"
    )
    assert "- **Acceptance Source**: per-request response timings (exact)" in text
    assert "server-wide Prometheus deltas" not in text


def test_report_names_no_source_when_every_request_drafted_nothing(tmp_path: Path) -> None:
    """llama.cpp without a draft model: timings 0/0 on every request, no α."""
    zero = SpecDecodeSample(
        tg_tokens=96,
        total_ms=1100,
        draft_tokens_delta=0,
        accepted_tokens_delta=0,
        acceptance_source="timings",
        prompt_type="code",
    )
    text = write_spec_decode_report(tmp_path, "run", "m", [zero]).read_text(encoding="utf-8")
    assert "Acceptance Source" not in text
    assert "Acceptance rate metrics were not available" in text
