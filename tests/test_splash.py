"""Splash (incoai/splash) backend detection, metadata, and speculative metrics.

The fixtures are trimmed captures from Splash 1.7.2 serving
``unsloth/Inkling-2-Mini-GGUF:UD-Q4_K_M``: its ``/v1/models`` listing, the
``frontend`` block of ``/status``, ``/metrics`` lines before and after one
120-token request, and the last two SSE chunks of a streamed tool call. Only
the scheduler counts in ``SPLASH_METRICS_BUSY`` are constructed, because the
captured scrapes were idle.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
import pytest

from tests.conftest import MeasurementTestClient
from tool_eval_bench.adapters.openai_compat import OpenAICompatibleAdapter
from tool_eval_bench.application.service import BenchmarkService
from tool_eval_bench.cli.server import detect_backend_from_response
from tool_eval_bench.domain.engines import SPLASH, engine_profile
from tool_eval_bench.runner import speculative, throughput
from tool_eval_bench.runner.context_pressure import reported_context_window
from tool_eval_bench.runner.spec_detection import (
    SpecDecodeInfo,
    detect_spec_decoding,
    parse_prometheus_spec_metrics,
    scrape_spec_metrics,
)
from tool_eval_bench.runner.spec_live import _parse_snapshot, compute_delta
from tool_eval_bench.runner.speculative import parse_splash_spec_metrics
from tool_eval_bench.runner.throughput import ThroughputSample, TokenizerConfig, _stream_one
from tool_eval_bench.utils import metadata

MODEL = "unsloth/Inkling-2-Mini-GGUF:UD-Q4_K_M"

SPLASH_MODELS: dict[str, Any] = {
    "object": "list",
    "data": [
        {
            "id": MODEL,
            "object": "model",
            "created": 0,
            "owned_by": "splash",
            "max_model_len": 1010000,
            "context_length": 1010000,
            "vision": False,
            "input_modalities": ["text"],
            "limits": {
                "request_bytes": 134217728,
                "images": 64,
                "image_bytes": 33554432,
                "image_aspect": 200,
                "pdf_bytes": 67108864,
                "pdf_pages": 64,
            },
            "reasoning": {
                "thinks": True,
                "efforts": ["minimal", "low", "medium", "high", "xhigh"],
                "default": "high",
            },
        }
    ],
    "models": [{"name": MODEL, "description": "Splash resident model", "release_date": ""}],
}
SPLASH_STATUS: dict[str, Any] = {
    "schema_version": 8,
    "ready": True,
    "maximum_context_tokens": 1010000,
    "capabilities": {"max_context_tokens": 1010000},
    "frontend": {"preparation_capacity": 4, "active": 0, "waiting": 0},
}

_SCHEDULER_IDLE = """\
splash_scheduler_queued 0
splash_scheduler_waiting_resources 0
splash_scheduler_waiting_prefix 0
splash_scheduler_prefilling 0
splash_scheduler_decoding 0
splash_scheduler_waiting_mask 0
"""
SPLASH_METRICS_T1 = (
    """\
# HELP splash_info Splash runtime metrics.
# TYPE splash_info gauge
splash_info{runtime="native"} 1
splash_ready 1
splash_requests_submitted_total 842
"""
    + _SCHEDULER_IDLE
    + """\
splash_prefill_input_tokens_total 203880
splash_decode_output_tokens_total 357474
splash_decode_tokens_per_second 110.0006969
splash_drafted_tokens_total 583695
splash_accepted_draft_tokens_total 269850
splash_draft_acceptance_ratio 0.4622561755
# TYPE splash_http_ttft_seconds histogram
splash_http_ttft_seconds_bucket{le="0.1"} 272
splash_http_ttft_seconds_count 833
splash_http_ttft_seconds_sum 213.12744368694257
"""
)
SPLASH_METRICS_T2 = (
    SPLASH_METRICS_T1.replace("submitted_total 842", "submitted_total 843")
    .replace("input_tokens_total 203880", "input_tokens_total 203913")
    .replace("output_tokens_total 357474", "output_tokens_total 357594")
    .replace("drafted_tokens_total 583695", "drafted_tokens_total 583927")
    .replace("accepted_draft_tokens_total 269850", "accepted_draft_tokens_total 269948")
)
# Constructed: two lanes decode, one prefills, one waits for a grammar mask;
# one request is queued, one waits for memory, one waits for a shared prefix.
SPLASH_METRICS_BUSY = SPLASH_METRICS_T2.replace(
    _SCHEDULER_IDLE,
    "splash_scheduler_queued 1\n"
    "splash_scheduler_waiting_resources 1\n"
    "splash_scheduler_waiting_prefix 1\n"
    "splash_scheduler_prefilling 1\n"
    "splash_scheduler_decoding 2\n"
    "splash_scheduler_waiting_mask 1\n",
)
SPLASH_METRICS_IDLE_ZERO = (
    'splash_info{runtime="native"} 1\n'
    "splash_drafted_tokens_total 0\n"
    "splash_accepted_draft_tokens_total 0\n"
    "splash_draft_acceptance_ratio 0\n"
)
SPLASH_LOAD_ONLY = 'splash_info{runtime="native"} 1\nsplash_ready 1\n' + _SCHEDULER_IDLE

# The last two chunks of a streamed tool call, as Splash sent them.
SPLASH_TIMINGS_CHUNK = {
    "id": "chatcmpl-48e9bb3a566dffb85d1a8e217d610386",
    "object": "chat.completion.chunk",
    "created": 1795603504,
    "model": MODEL,
    "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
    "timings": {
        "prompt_n": 110,
        "prompt_ms": 51.268,
        "prompt_per_second": 273.07482250136536,
        "predicted_n": 75,
        "predicted_ms": 469.389,
        "predicted_per_second": 157.65175579316943,
        "cache_n": 96,
        "draft_n": 135,
        "draft_n_accepted": 58,
    },
}
SPLASH_SPECULATION = {"drafted_tokens": 135, "accepted_draft_tokens": 58, "cycles": 16}
SPLASH_USAGE_CHUNK = {
    "id": "chatcmpl-48e9bb3a566dffb85d1a8e217d610386",
    "object": "chat.completion.chunk",
    "created": 1795603504,
    "model": MODEL,
    "choices": [],
    "usage": {
        "prompt_tokens": 110,
        "completion_tokens": 75,
        "total_tokens": 185,
        "prompt_tokens_details": {
            "cached_tokens": 96,
            "prefix_tokens": 96,
            "state_source": "memory",
            "lost_state": False,
        },
        "completion_tokens_details": {"reasoning_tokens": 40},
    },
    "metrics": {
        "prefill": {"tokens": 14},
        "decode": {"tokens": 74},
        "request_latency": {
            "start_to_first_token_ms": 51.268,
            "first_token_to_done_ms": 469.389,
            "wall_ms": 521.633,
            "ttft_ms": 52.24400000000003,
            "queue_to_start_ms": 0.9760000000000275,
            "stream_tokens_per_second": 157.65175579316943,
        },
        "cache": {"status": "hit", "matched_tokens": 96, "deduplicated_tokens": 0, "lane": 0},
        "speculation": SPLASH_SPECULATION,
    },
}


def _install_transport(monkeypatch: pytest.MonkeyPatch, respond) -> None:
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        metadata.httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(respond), **kw),
    )


class _MetricsClient:
    def __init__(self, text: str | None) -> None:
        self._text = text

    async def metrics(self, *, metrics_url: str | None = None) -> httpx.Response:
        if self._text is None:
            return httpx.Response(404)
        return httpx.Response(200, text=self._text)


# ---------------------------------------------------------------------------
# Identity and auto-detection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_probe_backend_hint_reads_the_metrics_namespace_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path == "/metrics":
            return httpx.Response(200, text=SPLASH_METRICS_T1)
        return httpx.Response(404)

    _install_transport(monkeypatch, respond)
    assert await metadata.probe_backend_hint("http://splash.test:8000") == ("splash", "Splash")
    assert seen == ["/metrics"]


@pytest.mark.asyncio
async def test_probe_backend_hint_sends_the_key_to_a_protected_metrics_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # With an API key configured, Splash answers /metrics only with it.
    seen: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.headers.get("Authorization") != "Bearer k":
            return httpx.Response(401)
        if request.url.path == "/metrics":
            return httpx.Response(200, text=SPLASH_METRICS_T1)
        return httpx.Response(404)

    _install_transport(monkeypatch, respond)
    assert await metadata.probe_backend_hint("http://splash.test:8000/v1", "k") == (
        "splash",
        "Splash",
    )
    assert seen == ["/metrics"]


@pytest.mark.asyncio
async def test_probe_backend_hint_falls_back_to_owned_by_when_metrics_are_hidden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A proxy hides /metrics; owned_by is the only identity signal left.
    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json=SPLASH_MODELS)
        return httpx.Response(404)

    _install_transport(monkeypatch, respond)
    assert await metadata.probe_backend_hint("http://splash.test:8000/v1") == (
        "splash",
        "Splash",
    )


def test_owned_by_and_server_header_identify_splash() -> None:
    assert metadata.backend_from_models(SPLASH_MODELS) == ("splash", "Splash")
    by_owner = httpx.Response(200, json=SPLASH_MODELS)
    assert detect_backend_from_response(by_owner, 8000) == ("splash", "Splash")
    by_header = httpx.Response(200, headers={"Server": "Splash"}, json={"data": [{"id": "m"}]})
    assert metadata.backend_from_response(by_header) == ("splash", "Splash")


@pytest.mark.parametrize("label", ["splash", "SPLASH", "Splash"])
def test_splash_labels_and_adapter_dispatch(label: str) -> None:
    assert engine_profile(label) is SPLASH
    service = BenchmarkService(repo=None, reporter=None)
    assert isinstance(service._adapter_for(label), OpenAICompatibleAdapter)


# ---------------------------------------------------------------------------
# Engine metadata
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("base_url", ["http://splash.test:8000", "http://splash.test:8000/v1"])
@pytest.mark.parametrize("backend", ["splash", "unknown"])
async def test_collect_run_context_reads_models_and_status(
    monkeypatch: pytest.MonkeyPatch, base_url: str, backend: str
) -> None:
    calls: list[tuple[str, str | None]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append((request.url.path, request.headers.get("Authorization")))
        if request.url.path == "/v1/models":
            return httpx.Response(200, headers={"Server": "Splash"}, json=SPLASH_MODELS)
        if request.url.path == "/status":
            return httpx.Response(200, headers={"Server": "Splash"}, json=SPLASH_STATUS)
        if request.url.path == "/health":
            return httpx.Response(200, headers={"Server": "Splash"}, json={"status": "ok"})
        return httpx.Response(404, headers={"Server": "Splash"})

    _install_transport(monkeypatch, respond)
    ctx = await metadata.collect_run_context(
        model=MODEL, backend=backend, base_url=base_url, api_key="k"
    )

    assert ctx.engine_name == "Splash"
    assert ctx.engine_version is None
    assert ctx.max_model_len == 1010000
    assert ctx.slot_count == 4
    assert ctx.server_model_id == MODEL
    assert reported_context_window(ctx) == 1010000
    paths = [path for path, _ in calls]
    assert paths.count("/v1/models") == 1
    assert paths.count("/status") == 1
    assert ("/status", "Bearer k") in calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status",
    [
        {**SPLASH_STATUS, "frontend": {"preparation_capacity": 0}},
        {**SPLASH_STATUS, "frontend": {"preparation_capacity": -1}},
        {**SPLASH_STATUS, "frontend": {"preparation_capacity": True}},
        {**SPLASH_STATUS, "frontend": {"preparation_capacity": "4"}},
        {**SPLASH_STATUS, "frontend": {}},
        {**SPLASH_STATUS, "frontend": [4]},
        {"schema_version": 8},
        [SPLASH_STATUS],
    ],
)
async def test_probe_splash_ignores_unusable_capacity(
    monkeypatch: pytest.MonkeyPatch, status: object
) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json=SPLASH_MODELS)
        if request.url.path == "/status":
            return httpx.Response(200, json=status)
        return httpx.Response(404)

    _install_transport(monkeypatch, respond)
    info = await metadata._probe_engine("http://splash.test:8000", None, "splash")
    assert info["engine_name"] == "Splash"
    assert info["max_model_len"] == 1010000
    assert "slot_count" not in info
    assert "engine_version" not in info


@pytest.mark.asyncio
async def test_explicit_splash_backend_survives_a_refused_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_transport(monkeypatch, lambda _req: httpx.Response(401))
    info = await metadata._probe_engine("http://proxy.test:8000/v1", None, "splash")
    assert info == {"engine_name": "Splash"}


# ---------------------------------------------------------------------------
# Prometheus counters and spec detection
# ---------------------------------------------------------------------------


def test_parse_prometheus_spec_metrics_reads_splash_counters_without_steps() -> None:
    counters = parse_prometheus_spec_metrics(SPLASH_METRICS_T2)
    assert counters.draft_tokens == 583927.0
    assert counters.accepted_tokens == 269948.0
    assert counters.num_drafts == 0.0
    assert counters.has_num_drafts is False
    assert counters.acceptance_rate == pytest.approx(269948 / 583927)
    assert counters.acceptance_length is None


def test_parse_prometheus_spec_metrics_prefers_a_spec_decode_family() -> None:
    mixed = (
        "vllm:spec_decode_num_accepted_tokens_total 6\n"
        "vllm:spec_decode_num_draft_tokens_total 10\n"
        "vllm:spec_decode_num_drafts_total 4\n" + SPLASH_METRICS_T2
    )
    counters = parse_prometheus_spec_metrics(mixed)
    assert (counters.accepted_tokens, counters.draft_tokens, counters.num_drafts) == (6, 10, 4)


def test_splash_counter_names_inside_a_label_are_not_read() -> None:
    text = 'other_metric{name="splash_drafted_tokens_total 99"} 1\n'
    counters = parse_prometheus_spec_metrics(text)
    assert counters.draft_tokens == 0.0


@pytest.mark.asyncio
async def test_scrape_spec_metrics_active_idle_and_load_only() -> None:
    active = await scrape_spec_metrics(_MetricsClient(SPLASH_METRICS_T1), "http://s")  # type: ignore[arg-type]
    assert active is not None
    assert (active.draft_tokens, active.accepted_tokens) == (583695.0, 269850.0)

    idle = await scrape_spec_metrics(_MetricsClient(SPLASH_METRICS_IDLE_ZERO), "http://s")  # type: ignore[arg-type]
    assert idle is not None
    assert (idle.draft_tokens, idle.accepted_tokens) == (0.0, 0.0)

    assert await scrape_spec_metrics(_MetricsClient(SPLASH_LOAD_ONLY), "http://s") is None  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_detect_spec_decoding_needs_drafted_tokens_and_names_splash() -> None:
    idle = await detect_spec_decoding(_MetricsClient(SPLASH_METRICS_IDLE_ZERO), "http://s")  # type: ignore[arg-type]
    assert idle.active is False
    assert idle.has_prometheus is True
    assert idle.has_per_request_timings is True
    assert idle.method == "unknown"
    assert idle.engine_name == "Splash"
    assert idle.detail == SPLASH.spec_counter_detail

    active = await detect_spec_decoding(_MetricsClient(SPLASH_METRICS_T1), "http://s")  # type: ignore[arg-type]
    assert active.active is True
    assert active.engine_name == "Splash"

    load_only = await detect_spec_decoding(_MetricsClient(SPLASH_LOAD_ONLY), "http://s")  # type: ignore[arg-type]
    assert (load_only.active, load_only.has_prometheus, load_only.engine_name) == (
        False,
        False,
        None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "other_namespace",
    [
        'strata:live_state{state="generating"} 1\n',
        "llamacpp:prompt_tokens_total 212548\n",
        "tensorfold:requests_total 1\n",
    ],
    ids=["strata", "llamacpp", "tensorfold"],
)
async def test_a_load_only_namespace_does_not_claim_splash_counters(
    other_namespace: str,
) -> None:
    # The counters parsed are Splash's, so Splash's rules apply, and
    # detection agrees with spec-live on the engine and the method.
    text = other_namespace + SPLASH_METRICS_T1
    info = await detect_spec_decoding(_MetricsClient(text), "http://s")  # type: ignore[arg-type]
    assert info.engine_name == "Splash"
    assert info.detail == SPLASH.spec_counter_detail
    assert info.has_per_request_timings is True
    assert info.active is True
    snap = _parse_snapshot(text)
    assert snap.spec_backend == "splash"
    assert info.method == snap.spec_method == "unknown"


@pytest.mark.asyncio
async def test_a_non_finite_flare_sample_does_not_claim_splash_counters() -> None:
    # The parser discards the infinite Flare sample and reads Splash's
    # counters, so detection applies Splash's rules, not Flare's.
    text = "flare_verification_draft_tokens_total 1e309\n" + SPLASH_METRICS_T1
    assert parse_prometheus_spec_metrics(text).draft_tokens == 583695.0
    info = await detect_spec_decoding(_MetricsClient(text), "http://s")  # type: ignore[arg-type]
    assert info.engine_name == "Splash"
    assert info.detail == SPLASH.spec_counter_detail
    assert _parse_snapshot(text).spec_backend == "splash"


# ---------------------------------------------------------------------------
# Per-request metrics.speculation
# ---------------------------------------------------------------------------


def test_parse_splash_spec_metrics_reads_counts_and_cycles() -> None:
    parsed = parse_splash_spec_metrics(SPLASH_SPECULATION)
    assert parsed is not None
    assert (parsed.num_draft_tokens, parsed.num_accepted_draft_tokens) == (135, 58)
    assert parsed.num_spec_steps == 16


@pytest.mark.parametrize(
    "raw",
    [
        None,
        [],
        {},
        {"drafted_tokens": 10},
        {"drafted_tokens": 10, "accepted_draft_tokens": 11, "cycles": 2},
        {"drafted_tokens": -1, "accepted_draft_tokens": 0, "cycles": 1},
        {"drafted_tokens": True, "accepted_draft_tokens": 0, "cycles": 1},
        {"drafted_tokens": "10", "accepted_draft_tokens": 5, "cycles": 2},
    ],
)
def test_parse_splash_spec_metrics_rejects_untrusted_counts(raw: object) -> None:
    assert parse_splash_spec_metrics(raw) is None


@pytest.mark.parametrize(
    "cycles",
    [None, 0, 11, -1, True, "2"],
)
def test_parse_splash_spec_metrics_drops_only_a_bad_cycle_count(cycles: object) -> None:
    parsed = parse_splash_spec_metrics(
        {"drafted_tokens": 10, "accepted_draft_tokens": 4, "cycles": cycles}
    )
    assert parsed is not None
    assert (parsed.num_draft_tokens, parsed.num_accepted_draft_tokens) == (10, 4)
    assert parsed.num_spec_steps is None


def _sse(chunks: list[dict[str, Any]]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        return httpx.Response(
            200,
            content=body.encode(),
            headers={"content-type": "text/event-stream", "server": "Splash"},
        )

    return httpx.MockTransport(handler)


async def _stream(chunks: list[dict[str, Any]]) -> ThroughputSample:
    async with MeasurementTestClient(transport=_sse(chunks)) as client:
        return await _stream_one(
            client, "http://localhost:8000", MODEL, [{"role": "user", "content": "hi"}], 75, None
        )


_CONTENT = {"choices": [{"index": 0, "delta": {"content": "Berlin"}, "finish_reason": None}]}


@pytest.mark.asyncio
async def test_stream_one_captures_splash_speculation_and_timings() -> None:
    sample = await _stream([_CONTENT, SPLASH_TIMINGS_CHUNK, SPLASH_USAGE_CHUNK])
    assert sample.error is None
    assert sample.splash_spec_metrics == SPLASH_SPECULATION
    assert sample.spec_decode_metrics is None
    assert (sample.draft_n, sample.draft_n_accepted) == (135, 58)
    assert sample.server_header == "Splash"


@pytest.mark.asyncio
async def test_stream_one_without_speculation_leaves_the_field_unset() -> None:
    timings = {**SPLASH_TIMINGS_CHUNK, "timings": {"prompt_n": 110, "predicted_n": 75}}
    metrics = dict(SPLASH_USAGE_CHUNK["metrics"])
    metrics.pop("speculation")
    sample = await _stream([_CONTENT, timings, {**SPLASH_USAGE_CHUNK, "metrics": metrics}])
    assert sample.splash_spec_metrics is None
    assert sample.timings_present is True
    assert sample.draft_n is None


def _patch_stream(monkeypatch: pytest.MonkeyPatch, sample: ThroughputSample) -> None:
    async def stream(*args: object, **kwargs: object) -> ThroughputSample:
        return sample

    monkeypatch.setattr(speculative, "_stream_one", stream)


def _patch_scrape(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []

    async def scrape(*args: object, **kwargs: object) -> speculative.SpecDecodeCounters:
        calls.append(1)
        return speculative.SpecDecodeCounters(accepted_tokens=1000, draft_tokens=2000)

    monkeypatch.setattr(speculative, "scrape_spec_metrics", scrape)
    return calls


@pytest.mark.asyncio
async def test_measure_spec_single_uses_splash_cycles_for_acceptance_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_stream(
        monkeypatch,
        ThroughputSample(
            tg_tokens=75,
            ttft_ms=52.2,
            total_ms=521.6,
            draft_n=135,
            draft_n_accepted=58,
            timings_present=True,
            splash_spec_metrics=SPLASH_SPECULATION,
        ),
    )
    calls = _patch_scrape(monkeypatch)
    sample = await speculative.measure_spec_single(
        None,  # type: ignore[arg-type]
        "url",
        MODEL,
        prompt_type="code",
        spec_info=SpecDecodeInfo(has_prometheus=True, has_per_request_timings=True),
    )
    assert sample.acceptance_source == "response"
    assert (sample.draft_tokens_delta, sample.accepted_tokens_delta) == (135, 58)
    assert sample.num_drafts_delta == 16
    assert sample.acceptance_rate == pytest.approx(58 / 135)
    assert sample.acceptance_length == pytest.approx(1 + 58 / 16)
    assert sample.draft_window == pytest.approx(135 / 16)
    # The cycles cover Splash's first emission too, so no Steps/s.
    assert sample.steps_cover_first_token is True
    assert sample.verify_steps_per_s is None
    assert sample.to_result()["verify_steps_per_s"] is None
    # Before the server proved itself, spec-bench scraped once; the response
    # answered, so the after-scrape was skipped.
    assert len(calls) == 1


def test_pooled_splash_samples_keep_steps_per_second_unreported() -> None:
    def splash_run(drafted: int, accepted: int, cycles: int) -> speculative.SpecDecodeSample:
        sample = speculative.SpecDecodeSample(tg_tokens=75, ttft_ms=50.0, total_ms=500.0)
        parsed = parse_splash_spec_metrics(
            {"drafted_tokens": drafted, "accepted_draft_tokens": accepted, "cycles": cycles}
        )
        assert parsed is not None
        sample.apply_per_request_metrics(parsed)
        return sample

    pooled = speculative.pool_spec_samples([splash_run(135, 58, 16), splash_run(132, 59, 15)])
    assert pooled.runs == 2
    assert pooled.acceptance_length == pytest.approx(1 + (58 + 59) / (16 + 15))
    assert pooled.steps_cover_first_token is True
    assert pooled.verify_steps_per_s is None

    # A cell whose other run drafted nothing (an exact zero from timings,
    # unflagged and without steps) still reports no Steps/s.
    zero = speculative.SpecDecodeSample(tg_tokens=75, ttft_ms=50.0, total_ms=500.0)
    zero.apply_per_request_metrics(speculative.PerRequestSpecMetrics(0, 0, None))
    assert zero.steps_cover_first_token is False
    mixed = speculative.pool_spec_samples([splash_run(135, 58, 16), zero])
    assert mixed.steps_cover_first_token is True
    assert mixed.num_drafts_delta is not None
    assert mixed.verify_steps_per_s is None

    # A vLLM-style sample with the same counts still reports Steps/s.
    vllm = speculative.SpecDecodeSample(tg_tokens=75, ttft_ms=50.0, total_ms=500.0)
    vllm.apply_per_request_metrics(speculative.PerRequestSpecMetrics(135, 58, 16))
    assert vllm.verify_steps_per_s == pytest.approx(16 / 0.45)


def test_report_explains_a_missing_steps_per_second_only_for_splash_rows() -> None:
    from tool_eval_bench.storage.reports.spec_decode import spec_decode_report

    def report_with(metrics: speculative.PerRequestSpecMetrics) -> str:
        sample = speculative.SpecDecodeSample(
            prompt_type="code", tg_tokens=64, ttft_ms=50.0, total_ms=500.0
        )
        sample.apply_per_request_metrics(metrics)
        report = spec_decode_report(MODEL, [sample], label=None, temperature=0.0)
        return "\n".join(report.body)

    splash = parse_splash_spec_metrics(SPLASH_SPECULATION)
    assert splash is not None
    caveat = "as Splash's cycles do, shows no Steps/s"
    assert caveat in report_with(splash)
    assert caveat not in report_with(speculative.PerRequestSpecMetrics(135, 58, 16))


def test_cli_spec_bench_prints_no_speedup_ceiling_for_splash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    from io import StringIO

    from rich.console import Console

    from tool_eval_bench.cli import spec_bench as cli_spec_bench

    def run_with(metrics: speculative.PerRequestSpecMetrics) -> str:
        async def fake_run(*args: object, on_sample=None, **kwargs: object):
            sample = speculative.SpecDecodeSample(
                prompt_type="code", tg_tokens=64, ttft_ms=50.0, total_ms=500.0
            )
            sample.apply_per_request_metrics(metrics)
            if on_sample is not None:
                await on_sample(sample, 0, 1)
            return [sample]

        monkeypatch.setattr("tool_eval_bench.runner.speculative.run_spec_bench", fake_run)
        buf = StringIO()
        cli_spec_bench.run_spec_bench(
            Console(file=buf, width=200, no_color=True),
            MODEL,
            MODEL,
            "http://localhost:8001",
            None,
            pp=128,
            tg=64,
            depths=[0],
            prompt_types=["code"],
            output_dir=str(tmp_path / "runs"),
        )
        return buf.getvalue()

    monkeypatch.setattr(
        "tool_eval_bench.storage.db._default_db_path", lambda: str(tmp_path / "b.sqlite")
    )
    splash = parse_splash_spec_metrics(SPLASH_SPECULATION)
    assert splash is not None
    assert "Speedup ceiling" not in run_with(splash)
    # The same counts without the flag still print the ceiling.
    assert "Speedup ceiling" in run_with(speculative.PerRequestSpecMetrics(135, 58, 16))


@pytest.mark.asyncio
async def test_a_splash_response_that_drafted_nothing_records_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Once a response has proven per-request metrics, spec-bench stops
    # scraping. A later response without metrics.speculation or draft_n still
    # has an exact answer: zero drafts.
    _patch_stream(monkeypatch, ThroughputSample(tg_tokens=75, timings_present=True))
    calls = _patch_scrape(monkeypatch)
    sample = await speculative.measure_spec_single(
        None,  # type: ignore[arg-type]
        "url",
        MODEL,
        prompt_type="code",
        spec_info=SpecDecodeInfo(
            has_prometheus=True, has_per_request_timings=True, has_per_request_metrics=True
        ),
    )
    assert sample.acceptance_source == "timings"
    assert (sample.draft_tokens_delta, sample.accepted_tokens_delta) == (0, 0)
    assert sample.acceptance_rate is None
    assert calls == []


def _returns(value: object):
    async def inner(*args: object, **kwargs: object) -> object:
        return value

    return inner


@pytest.mark.asyncio
async def test_spec_bench_log_names_splash_not_llamacpp(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    class Client:
        async def __aenter__(self) -> Client:
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

    info = SpecDecodeInfo(
        active=True, has_prometheus=True, has_per_request_timings=True, engine_name="Splash"
    )

    monkeypatch.setattr(speculative, "calibrate", _returns(TokenizerConfig()))
    monkeypatch.setattr(speculative, "detect_spec_decoding", _returns(info))
    monkeypatch.setattr(
        speculative, "measure_spec_single", _returns(speculative.SpecDecodeSample())
    )
    with caplog.at_level(logging.INFO, logger="tool_eval_bench.runner.speculative"):
        await speculative.run_spec_bench(
            "url", MODEL, client_factory=lambda **kwargs: Client(), prompt_types=["code"]
        )
    assert "Splash backend detected" in caplog.text
    assert "llama.cpp" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("server_header", "timings_present", "zero"),
    [
        # /metrics identified nothing; the response's own Server header names
        # an engine that writes draft_n only for a request that drafted.
        ("Splash", True, True),
        ("llama.cpp", True, True),
        # A timings object is still required: without it nothing is proven.
        ("Splash", False, False),
        # Strata may omit draft_n for "not reported", and its Server header
        # names nothing; other or missing headers prove nothing either.
        ("strata", True, False),
        ("uvicorn", True, False),
        (None, True, False),
    ],
)
async def test_zero_drafts_need_the_response_to_name_its_engine(
    monkeypatch: pytest.MonkeyPatch,
    server_header: str | None,
    timings_present: bool,
    zero: bool,
) -> None:
    _patch_stream(
        monkeypatch,
        ThroughputSample(
            tg_tokens=75, timings_present=timings_present, server_header=server_header
        ),
    )
    calls = _patch_scrape(monkeypatch)
    sample = await speculative.measure_spec_single(
        None,  # type: ignore[arg-type]
        "url",
        MODEL,
        prompt_type="code",
        spec_info=SpecDecodeInfo(),
    )
    if zero:
        assert sample.acceptance_source == "timings"
        assert (sample.draft_tokens_delta, sample.accepted_tokens_delta) == (0, 0)
    else:
        assert sample.acceptance_source is None
        assert sample.draft_tokens_delta is None
    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("info", "expected", "absent"),
    [
        (
            SpecDecodeInfo(has_prometheus=True, has_per_request_timings=True, engine_name="Splash"),
            "Splash backend detected; its draft counters are zero",
            "llama.cpp backend detected",
        ),
        (
            SpecDecodeInfo(has_per_request_timings=True, engine_name="llama.cpp"),
            "llama.cpp backend detected",
            "Splash",
        ),
        (
            SpecDecodeInfo(has_per_request_timings=True),
            "llama.cpp backend detected",
            "Splash",
        ),
    ],
)
async def test_throughput_hint_names_the_detected_engine(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    info: SpecDecodeInfo,
    expected: str,
    absent: str,
) -> None:
    class Client:
        async def __aenter__(self) -> Client:
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(throughput, "warmup", _returns(1.0))
    monkeypatch.setattr(throughput, "calibrate", _returns(TokenizerConfig()))
    monkeypatch.setattr(throughput, "estimate_latency", _returns(2.0))
    monkeypatch.setattr(throughput, "detect_spec_decoding", _returns(info))
    monkeypatch.setattr(throughput, "measure_concurrent", _returns(ThroughputSample()))
    with caplog.at_level(logging.INFO, logger="tool_eval_bench.runner.throughput"):
        await throughput.run_throughput_matrix(
            "url",
            MODEL,
            client_factory=lambda **kwargs: Client(),
            depths=[0],
            concurrency_levels=[1],
        )
    assert expected in caplog.text
    assert absent not in caplog.text


# ---------------------------------------------------------------------------
# Spec-live
# ---------------------------------------------------------------------------


def test_spec_live_splash_snapshot_and_delta() -> None:
    prev = _parse_snapshot(SPLASH_METRICS_T1)
    curr = _parse_snapshot(SPLASH_METRICS_T2)
    prev.timestamp = 100.0
    curr.timestamp = 102.0

    assert curr.spec_backend == "splash"
    assert curr.spec_method == "unknown"
    assert curr.has_spec_decode is True
    assert curr.has_counter_spec_decode is True
    assert (curr.draft_tokens, curr.accepted_tokens, curr.num_drafts) == (583927, 269948, 0)
    assert curr.per_position_rates == {}

    delta = compute_delta(prev, curr)
    assert delta.had_activity is True
    assert delta.counter_metrics_available is True
    assert (delta.delta_drafted, delta.delta_accepted, delta.delta_drafts) == (232, 98, 0)
    assert delta.acceptance_rate == pytest.approx(98 / 232)
    assert delta.waste_ratio == pytest.approx(1 - 98 / 232)
    assert delta.cumulative_acceptance_rate == pytest.approx(269948 / 583927)
    # No cycle counter: no step-based figure.
    assert delta.acceptance_length is None
    assert delta.cumulative_acceptance_length is None
    assert delta.draft_window is None
    assert delta.num_spec_tokens is None
    assert delta.accepted_tps == pytest.approx(98 / 2)
    assert delta.drafted_tps == pytest.approx(232 / 2)
    assert delta.prompt_tps == pytest.approx(33 / 2)
    assert delta.generation_tps == pytest.approx(120 / 2)
    assert delta.prompt_tps_is_gauge is False
    assert delta.generation_tps_is_gauge is False
    assert (delta.running_reqs, delta.waiting_reqs) == (0, 0)
    # Splash maps no KV or prefix cache gauge.
    assert delta.gpu_cache_pct == 0.0
    assert delta.prefix_cache_hit_pct == 0.0


def test_spec_live_splash_counts_scheduler_phases() -> None:
    snap = _parse_snapshot(SPLASH_METRICS_BUSY)
    assert snap.running_reqs == 4  # prefilling 1 + decoding 2 + waiting_mask 1
    assert snap.waiting_reqs == 3  # queued, waiting_resources, waiting_prefix


def test_spec_live_splash_load_only_and_idle() -> None:
    load_only = _parse_snapshot(SPLASH_LOAD_ONLY)
    assert load_only.spec_backend == "splash"
    assert load_only.has_spec_decode is False

    idle = _parse_snapshot(SPLASH_METRICS_IDLE_ZERO)
    assert idle.spec_backend == "splash"
    assert idle.has_spec_decode is True
    delta = compute_delta(idle, idle)
    assert delta.had_activity is False
    assert delta.cumulative_acceptance_rate is None


def test_spec_live_does_not_difference_vllm_and_splash_token_counters() -> None:
    # vLLM's counter resets to 0 between polls; Splash's stays at 1000. The
    # rate must follow the vLLM counter, not jump to Splash's.
    prev = _parse_snapshot("vllm:generation_tokens_total 3\n" + SPLASH_METRICS_T1)
    curr = _parse_snapshot("vllm:generation_tokens_total 0\n" + SPLASH_METRICS_T1)
    prev.timestamp, curr.timestamp = 100.0, 101.0
    assert (prev.generation_tokens_total, curr.generation_tokens_total) == (3, 0)
    assert compute_delta(prev, curr).generation_tps == 0.0


def test_spec_live_keeps_flare_values_over_splash_values() -> None:
    # Flare's counters are read before Splash's, and so are its token totals
    # and request gauges, a valid zero included.
    text = (
        "flare_verification_draft_tokens_total 10\n"
        "flare_accepted_draft_tokens_total 5\n"
        "flare_draft_request_rounds_total 4\n"
        "flare_generated_tokens_total 0\n"
        "flare_active_requests 0\n" + SPLASH_METRICS_BUSY
    )
    snap = _parse_snapshot(text)
    assert snap.spec_backend == "flare"
    assert (snap.draft_tokens, snap.accepted_tokens, snap.num_drafts) == (10, 5, 4)
    assert snap.generation_tokens_total == 0
    assert snap.running_reqs == 0
    # Fields Flare does not supply still come from Splash.
    assert snap.prompt_tokens_total == 203913
    assert snap.waiting_reqs == 3


def test_spec_live_a_non_finite_vllm_sample_does_not_block_splash() -> None:
    snap = _parse_snapshot("vllm:generation_tokens_total 1e309\n" + SPLASH_METRICS_T2)
    assert snap.generation_tokens_total == 357594


def test_spec_live_mixed_vllm_and_splash_reads_vllm() -> None:
    mixed = (
        "vllm:spec_decode_num_draft_tokens_total 10\n"
        "vllm:spec_decode_num_accepted_tokens_total 5\n"
        "vllm:spec_decode_num_drafts_total 4\n" + SPLASH_METRICS_T2
    )
    snap = _parse_snapshot(mixed)
    assert snap.spec_backend == "vllm"
    assert (snap.draft_tokens, snap.accepted_tokens, snap.num_drafts) == (10, 5, 4)
