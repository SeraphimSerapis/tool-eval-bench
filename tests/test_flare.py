"""Flare (NVIDIA/hyperflare-inference) backend detection, metadata, and metrics."""

from __future__ import annotations

import httpx
import pytest

from tool_eval_bench.adapters.openai_compat import OpenAICompatibleAdapter
from tool_eval_bench.application.service import BenchmarkService
from tool_eval_bench.cli.server import detect_backend_from_response
from tool_eval_bench.domain.engines import FLARE, engine_profile
from tool_eval_bench.runner.context_pressure import reported_context_window
from tool_eval_bench.runner.spec_detection import (
    detect_spec_decoding,
    parse_prometheus_spec_metrics,
    scrape_spec_metrics,
)
from tool_eval_bench.runner.spec_live import _parse_snapshot, compute_delta
from tool_eval_bench.utils import metadata

FLARE_MODEL_CARD = {
    "id": "nvidia/Nemotron-3-Nano-30B-A3B",
    "object": "model",
    "created": 1700000000,
    "owned_by": "local",
    "root": "nvidia/Nemotron-3-Nano-30B-A3B",
    "parent": None,
    "max_model_len": 131072,
    "flare": {
        "vocab_size": 131072,
        "max_concurrency": 64,
        "prefill_chunk_size": 8192,
        "default_max_tokens": 4096,
        "input_modalities": ["text"],
        "output_modalities": ["text"],
    },
}
FLARE_MODELS = {"object": "list", "data": [FLARE_MODEL_CARD]}
FLARE_HEALTH = {"status": "ok"}

FLARE_METRICS_IDLE = """\
# HELP flare_prefill_tokens_total Prompt tokens consumed by prefill.
# TYPE flare_prefill_tokens_total counter
flare_prefill_tokens_total 0
# TYPE flare_generated_tokens_total counter
flare_generated_tokens_total 0
# TYPE flare_accepted_draft_tokens_total counter
flare_accepted_draft_tokens_total 0
# TYPE flare_verification_draft_tokens_total counter
flare_verification_draft_tokens_total 0
# TYPE flare_draft_request_rounds_total counter
flare_draft_request_rounds_total 0
# TYPE flare_active_requests gauge
flare_active_requests 0
# TYPE flare_pending_requests gauge
flare_pending_requests 0
# TYPE flare_cache_lookups_total counter
flare_cache_lookups_total 0
# TYPE flare_cache_hits_total counter
flare_cache_hits_total 0
# TYPE flare_total_kv_pages gauge
flare_total_kv_pages 200
# TYPE flare_free_kv_pages gauge
flare_free_kv_pages 200
flare_speculative_drafts_total{depth="1",source="mtp",outcome="offered"} 0
flare_speculative_drafts_total{depth="1",source="mtp",outcome="accepted"} 0
flare_speculative_drafts_total{depth="1",source="copy",outcome="offered"} 0
flare_speculative_drafts_total{depth="1",source="copy",outcome="accepted"} 0
"""

FLARE_METRICS_T1 = """\
flare_prefill_tokens_total 100
flare_generated_tokens_total 50
flare_accepted_draft_tokens_total 10
flare_verification_draft_tokens_total 20
flare_draft_request_rounds_total 5
flare_active_requests 1
flare_pending_requests 0
flare_cache_lookups_total 10
flare_cache_hits_total 4
flare_total_kv_pages 100
flare_free_kv_pages 75
flare_speculative_drafts_total{depth="1",source="mtp",outcome="offered"} 4
flare_speculative_drafts_total{depth="1",source="mtp",outcome="accepted"} 3
flare_speculative_drafts_total{depth="1",source="copy",outcome="offered"} 1
flare_speculative_drafts_total{depth="1",source="copy",outcome="accepted"} 1
flare_speculative_drafts_total{depth="2",source="mtp",outcome="offered"} 5
flare_speculative_drafts_total{depth="2",source="mtp",outcome="accepted"} 2
"""

FLARE_METRICS_T2 = """\
flare_prefill_tokens_total 500
flare_generated_tokens_total 150
flare_accepted_draft_tokens_total 25
flare_verification_draft_tokens_total 40
flare_draft_request_rounds_total 10
flare_active_requests 2
flare_pending_requests 3
flare_cache_lookups_total 20
flare_cache_hits_total 15
flare_total_kv_pages 100
flare_free_kv_pages 20
flare_speculative_drafts_total{depth="1",source="mtp",outcome="offered"} 6
flare_speculative_drafts_total{depth="1",source="mtp",outcome="accepted"} 5
flare_speculative_drafts_total{depth="1",source="copy",outcome="offered"} 4
flare_speculative_drafts_total{depth="1",source="copy",outcome="accepted"} 3
flare_speculative_drafts_total{depth="2",source="mtp",outcome="offered"} 8
flare_speculative_drafts_total{depth="2",source="mtp",outcome="accepted"} 4
flare_speculative_drafts_total{depth="2",source="copy",outcome="offered"} 2
flare_speculative_drafts_total{depth="2",source="copy",outcome="accepted"} 0
flare_speculative_drafts_total{depth="3",source="mtp",outcome="offered"} 0
flare_speculative_drafts_total{depth="3",source="mtp",outcome="accepted"} 0
"""


def _install_transport(monkeypatch: pytest.MonkeyPatch, respond) -> None:
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        metadata.httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(respond), **kw),
    )


class _MetricsClient:
    def __init__(self, text: str | None, status: int = 200) -> None:
        self._text = text
        self._status = status

    async def metrics(self, *, metrics_url: str | None = None) -> httpx.Response:
        if self._text is None:
            return httpx.Response(404)
        return httpx.Response(self._status, text=self._text)


# ---------------------------------------------------------------------------
# Identity and auto-detection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("base_url", ["http://flare.test:8000", "http://flare.test:8000/v1"])
async def test_probe_backend_hint_detects_unauthenticated_metrics_first(
    monkeypatch: pytest.MonkeyPatch, base_url: str
) -> None:
    seen_paths: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.path)
        if request.url.path == "/metrics":
            assert "Authorization" not in request.headers
            return httpx.Response(200, text=FLARE_METRICS_IDLE)
        return httpx.Response(401)

    _install_transport(monkeypatch, respond)
    assert await metadata.probe_backend_hint(base_url) == ("flare", "Flare")
    assert seen_paths == ["/metrics"]


@pytest.mark.asyncio
@pytest.mark.parametrize("base_url", ["http://flare.test:8000", "http://flare.test:8000/v1"])
async def test_probe_backend_hint_falls_back_to_models_flare_object(
    monkeypatch: pytest.MonkeyPatch, base_url: str
) -> None:
    seen_paths: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.path)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json=FLARE_MODELS)
        if request.url.path == "/health":
            return httpx.Response(200, json=FLARE_HEALTH)
        return httpx.Response(404)

    _install_transport(monkeypatch, respond)
    assert await metadata.probe_backend_hint(base_url, "secret") == ("flare", "Flare")
    assert seen_paths == ["/metrics", "/version", "/v1/models"]


def test_backend_from_models_and_server_discovery() -> None:
    assert metadata.backend_from_models(FLARE_MODELS) == ("flare", "Flare")
    resp = httpx.Response(200, json=FLARE_MODELS)
    assert metadata.backend_from_response(resp) == ("flare", "Flare")
    assert detect_backend_from_response(resp, 8000) == ("flare", "Flare")


@pytest.mark.parametrize("bad_flare", [None, "", "flare", True, False, 1, ["a"]])
def test_backend_from_models_rejects_non_dict_flare_field(bad_flare: object) -> None:
    body = {"object": "list", "data": [{"id": "m", "owned_by": "local", "flare": bad_flare}]}
    assert metadata.backend_from_models(body) is None


def test_backend_from_models_rejects_generic_local_without_flare_key() -> None:
    body = {"object": "list", "data": [{"id": "m", "owned_by": "local"}]}
    assert metadata.backend_from_models(body) is None


# ---------------------------------------------------------------------------
# Aliases, adapter dispatch, and engine metadata probing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label", ["flare", "FLARE", "hyperflare", "HyperFlare"])
def test_flare_labels_and_adapter_dispatch(label: str) -> None:
    assert engine_profile(label) is FLARE
    service = BenchmarkService(repo=None, reporter=None)
    assert isinstance(service._adapter_for(label), OpenAICompatibleAdapter)


@pytest.mark.asyncio
@pytest.mark.parametrize("base_url", ["http://flare.test:8000", "http://flare.test:8000/v1"])
@pytest.mark.parametrize("backend", ["flare", "hyperflare", "HyperFlare", "unknown"])
async def test_collect_run_context_extracts_flare_metadata_without_duplicate_models_call(
    monkeypatch: pytest.MonkeyPatch, base_url: str, backend: str
) -> None:
    paths: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json=FLARE_MODELS)
        if request.url.path == "/health":
            return httpx.Response(200, json=FLARE_HEALTH)
        return httpx.Response(404)

    _install_transport(monkeypatch, respond)
    ctx = await metadata.collect_run_context(
        model="nvidia/Nemotron-3-Nano-30B-A3B",
        backend=backend,
        base_url=base_url,
        api_key="k",
    )

    assert ctx.engine_name == "Flare"
    assert ctx.engine_version is None
    assert ctx.max_model_len == 131072
    assert ctx.slot_count == 64
    assert ctx.server_model_id == "nvidia/Nemotron-3-Nano-30B-A3B"
    assert ctx.server_model_root == "nvidia/Nemotron-3-Nano-30B-A3B"
    assert reported_context_window(ctx) == 131072
    assert paths.count("/v1/models") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["flare", "hyperflare"])
async def test_explicit_flare_backend_trusted_behind_anonymous_proxy(
    monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    _install_transport(monkeypatch, lambda _req: httpx.Response(404))
    info = await metadata._probe_engine("http://proxy.test:8000/v1", None, backend)
    assert info == {"engine_name": "Flare"}


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_concurrency", [0, -1, True, "64", None])
async def test_probe_flare_ignores_invalid_max_concurrency(
    monkeypatch: pytest.MonkeyPatch, bad_concurrency: object
) -> None:
    card = {
        **FLARE_MODEL_CARD,
        "flare": {**FLARE_MODEL_CARD["flare"], "max_concurrency": bad_concurrency},
    }
    _install_transport(
        monkeypatch,
        lambda req: (
            httpx.Response(200, json={"data": [card]})
            if req.url.path == "/v1/models"
            else httpx.Response(404)
        ),
    )
    info = await metadata._probe_engine("http://flare.test:8000", None, "flare")
    assert info["engine_name"] == "Flare"
    assert info["max_model_len"] == 131072
    assert "slot_count" not in info
    assert "engine_version" not in info


@pytest.mark.asyncio
async def test_probe_flare_selects_matching_model_in_multi_entry_listing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listing = {
        "data": [
            {
                "id": "other-model",
                "owned_by": "local",
                "max_model_len": 4096,
                "flare": {"max_concurrency": 4},
            },
            {
                "id": "target-model",
                "owned_by": "local",
                "max_model_len": 65536,
                "flare": {"max_concurrency": 32},
            },
        ]
    }
    _install_transport(
        monkeypatch,
        lambda req: (
            httpx.Response(200, json=listing)
            if req.url.path == "/v1/models"
            else httpx.Response(404)
        ),
    )
    info = await metadata._probe_engine(
        "http://flare.test:8000", None, "flare", model="target-model"
    )
    assert info["max_model_len"] == 65536
    assert info["slot_count"] == 32


# ---------------------------------------------------------------------------
# Speculative decoding detection & scraping
# ---------------------------------------------------------------------------


def test_parse_prometheus_spec_metrics_flare_counters() -> None:
    counters = parse_prometheus_spec_metrics(FLARE_METRICS_T2)
    assert counters.accepted_tokens == 25.0
    assert counters.draft_tokens == 40.0
    assert counters.num_drafts == 10.0
    assert counters.has_num_drafts is True
    assert counters.acceptance_rate == pytest.approx(0.625)
    assert counters.acceptance_length == pytest.approx(3.5)


def test_parse_prometheus_spec_metrics_does_not_double_count_mixed_prefixes() -> None:
    mixed = (
        "vllm:spec_decode_num_accepted_tokens_total 6\n"
        "vllm:spec_decode_num_draft_tokens_total 10\n"
        "vllm:spec_decode_num_drafts_total 4\n" + FLARE_METRICS_T2
    )
    counters = parse_prometheus_spec_metrics(mixed)
    assert counters.accepted_tokens == 6.0
    assert counters.draft_tokens == 10.0
    assert counters.num_drafts == 4.0


@pytest.mark.asyncio
async def test_scrape_spec_metrics_flare_active_idle_and_load_only() -> None:
    active = await scrape_spec_metrics(_MetricsClient(FLARE_METRICS_T2), "http://flare.test")  # type: ignore[arg-type]
    assert active is not None
    assert active.accepted_tokens == 25.0
    assert active.draft_tokens == 40.0
    assert active.num_drafts == 10.0

    idle = await scrape_spec_metrics(_MetricsClient(FLARE_METRICS_IDLE), "http://flare.test")  # type: ignore[arg-type]
    assert idle is not None
    assert idle.accepted_tokens == 0.0
    assert idle.draft_tokens == 0.0
    assert idle.has_num_drafts is True

    load_only = await scrape_spec_metrics(
        _MetricsClient("flare_active_requests 2\nflare_generated_tokens_total 99\n"),  # type: ignore[arg-type]
        "http://flare.test",
    )
    assert load_only is None


@pytest.mark.asyncio
async def test_detect_spec_decoding_flare_zero_active_and_hint() -> None:
    idle_info = await detect_spec_decoding(_MetricsClient(FLARE_METRICS_IDLE), "http://flare.test")  # type: ignore[arg-type]
    assert idle_info.active is False
    assert idle_info.has_prometheus is True
    assert idle_info.has_per_request_timings is False
    assert idle_info.method == "unknown"
    assert idle_info.detail == "Flare speculative draft counters available"

    active_info = await detect_spec_decoding(_MetricsClient(FLARE_METRICS_T2), "http://flare.test")  # type: ignore[arg-type]
    assert active_info.active is True
    assert active_info.has_prometheus is True
    assert active_info.has_per_request_timings is False
    assert active_info.method == "unknown"
    assert active_info.detail == "Flare speculative draft counters available"

    hinted_info = await detect_spec_decoding(
        _MetricsClient(FLARE_METRICS_IDLE),  # type: ignore[arg-type]
        "http://flare.test",
        backend_hint="mtp",
    )
    assert hinted_info.active is True
    assert hinted_info.method == "mtp"
    assert hinted_info.has_prometheus is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "flare_extra",
    [
        "flare_active_requests 1\n",
        FLARE_METRICS_T2,
    ],
)
async def test_detect_spec_decoding_mixed_vllm_and_flare_prefers_vllm_counters(
    flare_extra: str,
) -> None:
    mixed = (
        "vllm:spec_decode_num_draft_tokens_total 0\n"
        "vllm:spec_decode_num_accepted_tokens_total 0\n"
        "vllm:spec_decode_num_drafts_total 0\n" + flare_extra
    )
    info = await detect_spec_decoding(_MetricsClient(mixed), "http://test")  # type: ignore[arg-type]
    assert info.active is True
    assert info.has_prometheus is True
    assert info.detail == "Detected via Prometheus /metrics (spec_decode counters present)"
    assert _parse_snapshot(mixed).spec_backend == "vllm"


# ---------------------------------------------------------------------------
# Spec-live parsing and delta computation
# ---------------------------------------------------------------------------


def test_spec_live_flare_snapshot_and_delta() -> None:
    prev = _parse_snapshot(FLARE_METRICS_T1)
    curr = _parse_snapshot(FLARE_METRICS_T2)
    prev.timestamp = 100.0
    curr.timestamp = 102.0

    assert prev.spec_backend == "flare"
    assert curr.spec_backend == "flare"
    assert curr.has_spec_decode is True
    assert curr.has_counter_spec_decode is True
    assert curr.per_position_counters == {0: 8.0, 1: 4.0, 2: 0.0}
    assert curr.per_position_drafted_counters == {0: 10.0, 1: 10.0, 2: 0.0}
    assert curr.per_position_rates == pytest.approx({0: 0.8, 1: 0.4})

    delta = compute_delta(prev, curr)
    assert delta.had_activity is True
    assert delta.counter_metrics_available is True
    assert delta.cumulative_acceptance_rate == pytest.approx(25 / 40)
    assert delta.cumulative_acceptance_length == pytest.approx(1.0 + 25 / 10)
    assert delta.cumulative_draft_window == pytest.approx(4.0)
    assert delta.num_spec_tokens == 4
    assert delta.acceptance_rate == pytest.approx(15 / 20)
    assert delta.waste_ratio == pytest.approx(0.25)
    assert delta.acceptance_length == pytest.approx(1.0 + 15 / 5)
    assert delta.draft_window == pytest.approx(20 / 5)
    assert delta.accepted_tps == pytest.approx(7.5)
    assert delta.drafted_tps == pytest.approx(10.0)
    assert delta.prompt_tps == pytest.approx(200.0)
    assert delta.generation_tps == pytest.approx(50.0)
    assert delta.prompt_tps_is_gauge is False
    assert delta.generation_tps_is_gauge is False
    assert delta.gpu_cache_pct == pytest.approx(80.0)
    assert delta.running_reqs == 2
    assert delta.waiting_reqs == 3
    assert delta.prefix_cache_hit_pct == pytest.approx(75.0)
    assert delta.per_position_rates == pytest.approx({0: 0.8, 1: 0.4})
    assert delta.total_accepted == 25
    assert delta.total_drafted == 40
    assert delta.total_drafts == 10
    assert delta.spec_method == "unknown"


def test_spec_live_flare_idle_zero_counters_and_boundary_pages() -> None:
    snap = _parse_snapshot(FLARE_METRICS_IDLE)
    snap.timestamp = 10.0
    later = _parse_snapshot(FLARE_METRICS_IDLE)
    later.timestamp = 11.0

    assert snap.spec_backend == "flare"
    assert snap.has_spec_decode is True
    assert snap.has_counter_spec_decode is True
    assert snap.per_position_rates == {}

    delta = compute_delta(snap, later)
    assert delta.had_activity is False
    assert delta.cumulative_acceptance_rate is None
    assert delta.cumulative_acceptance_length is None
    assert delta.acceptance_rate is None
    assert delta.gpu_cache_pct == pytest.approx(0.0)
    assert delta.prefix_cache_hit_pct == pytest.approx(0.0)

    # Zero total pages avoids division by zero; free > total clamps to 0%.
    zero_total = _parse_snapshot("flare_total_kv_pages 0\nflare_free_kv_pages 0\n")
    assert zero_total.spec_backend == "flare"
    assert zero_total.kv_cache_usage is None

    skewed_pages = _parse_snapshot("flare_total_kv_pages 100\nflare_free_kv_pages 120\n")
    assert skewed_pages.kv_cache_usage == pytest.approx(0.0)


def test_spec_live_flare_ignores_invalid_depth_labels() -> None:
    text = (
        'flare_speculative_drafts_total{depth="0",source="mtp",outcome="offered"} 10\n'
        'flare_speculative_drafts_total{depth="0",source="mtp",outcome="accepted"} 10\n'
        'flare_speculative_drafts_total{depth="bad",source="mtp",outcome="offered"} 10\n'
        'flare_speculative_drafts_total{depth="1",source="mtp",outcome="offered"} 4\n'
        'flare_speculative_drafts_total{depth="1",source="mtp",outcome="accepted"} 3\n'
    )
    snap = _parse_snapshot(text)
    assert snap.per_position_counters == {0: 3.0}
    assert snap.per_position_drafted_counters == {0: 4.0}
    assert snap.per_position_rates == pytest.approx({0: 0.75})
