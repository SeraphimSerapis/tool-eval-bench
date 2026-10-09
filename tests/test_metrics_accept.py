"""Every /metrics reader asks for Prometheus text, so negotiating servers send it.

Strata serves JSON from /metrics unless the request's Accept names text/plain
or OpenMetrics (serve/prometheus.py ``wants_prometheus`` at
Niko1221/Strata@fb58e0d). httpx's default ``Accept: */*`` therefore got JSON,
and every parser below silently read nothing.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from tests.test_spec_live_shutdown import install_spec_live_harness
from tool_eval_bench.adapters.measurement import HTTPMeasurementClient
from tool_eval_bench.cli import spec_live_display as display
from tool_eval_bench.runner.spec_detection import detect_spec_decoding, scrape_spec_metrics
from tool_eval_bench.runner.spec_live import _parse_snapshot, compute_delta, scrape_snapshot
from tool_eval_bench.utils import metadata
from tool_eval_bench.utils.urls import metrics_request_target

ACCEPT = "text/plain; version=0.0.4, */*;q=0.1"


def _strata_prometheus(*, drafted: int, accepted: int) -> str:
    """Lines in the shape of Strata's render(): vLLM names, model_name label.

    Trimmed to the families our parsers read plus one ``strata:`` family.
    Upstream emits the spec-decode counters on every scrape, even at zero.
    """
    lab = 'model_name="Qwen3-8B"'
    return (
        "# HELP vllm:num_requests_running Requests reading their prompt or generating.\n"
        "# TYPE vllm:num_requests_running gauge\n"
        f"vllm:num_requests_running{{{lab}}} 2\n"
        "# HELP vllm:num_requests_waiting Requests waiting for their turn.\n"
        "# TYPE vllm:num_requests_waiting gauge\n"
        f"vllm:num_requests_waiting{{{lab}}} 1\n"
        "# HELP vllm:kv_cache_usage_perc The running request's share of the context (1 = full).\n"
        "# TYPE vllm:kv_cache_usage_perc gauge\n"
        f"vllm:kv_cache_usage_perc{{{lab}}} 0.25\n"
        "# HELP vllm:generation_tokens_total Generated tokens.\n"
        "# TYPE vllm:generation_tokens_total counter\n"
        f"vllm:generation_tokens_total{{{lab}}} 300\n"
        "# HELP vllm:spec_decode_num_draft_tokens_total MTP draft tokens offered.\n"
        "# TYPE vllm:spec_decode_num_draft_tokens_total counter\n"
        f"vllm:spec_decode_num_draft_tokens_total{{{lab}}} {drafted}\n"
        "# HELP vllm:spec_decode_num_accepted_tokens_total MTP draft tokens accepted.\n"
        "# TYPE vllm:spec_decode_num_accepted_tokens_total counter\n"
        f"vllm:spec_decode_num_accepted_tokens_total{{{lab}}} {accepted}\n"
        "# HELP strata:live_state 1 for what the engine is doing (live.state).\n"
        "# TYPE strata:live_state gauge\n"
        f'strata:live_state{{{lab},state="generating"}} 1\n'
    )


def _strata_server(*, drafted: int = 120, accepted: int = 90):
    """A /metrics handler that negotiates the way Strata's server does."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        accept = request.headers.get("accept", "")
        query = str(request.url.query)
        if "format=prometheus" in query or "openmetrics" in accept or "text/plain" in accept:
            return httpx.Response(200, text=_strata_prometheus(drafted=drafted, accepted=accepted))
        return httpx.Response(200, json={"live": {"state": "generating", "running": 2}})

    return handler, seen


def _client(handler: Any, **kwargs: Any) -> HTTPMeasurementClient:
    return HTTPMeasurementClient(
        base_url="http://strata.test:8080/v1",
        timeout=1.0,
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("api_key", "metrics_url", "authorization"),
    [
        (None, None, None),
        ("sk-test", None, "Bearer sk-test"),
        ("sk-test", "http://sidecar.test:9090/metrics", None),
    ],
)
async def test_measurement_scrape_asks_for_prometheus_text(
    api_key: str | None, metrics_url: str | None, authorization: str | None
) -> None:
    handler, seen = _strata_server()
    async with _client(handler, api_key=api_key) as client:
        await client.metrics(metrics_url=metrics_url)

    assert seen[0].headers["accept"] == ACCEPT
    assert seen[0].headers.get("authorization") == authorization


@pytest.mark.asyncio
async def test_strata_spec_counters_read_through_the_measurement_client() -> None:
    handler, _ = _strata_server(drafted=120, accepted=90)
    async with _client(handler) as client:
        counters = await scrape_spec_metrics(client, "http://strata.test:8080/v1")
        info = await detect_spec_decoding(client, "http://strata.test:8080/v1")

    assert counters is not None
    assert (counters.draft_tokens, counters.accepted_tokens) == (120, 90)
    assert counters.acceptance_rate == pytest.approx(0.75)
    # Strata exports no num_drafts counter, so acceptance length stays unknown.
    assert counters.has_num_drafts is False
    assert counters.acceptance_length is None
    assert info.active is True
    assert info.has_prometheus is True
    assert info.method == "mtp"


@pytest.mark.asyncio
async def test_strata_zero_drafts_is_not_active_spec_decoding() -> None:
    """Strata renders the counters unconditionally; presence proves nothing."""
    handler, _ = _strata_server(drafted=0, accepted=0)
    async with _client(handler) as client:
        info = await detect_spec_decoding(client, "http://strata.test:8080/v1")

    assert info.active is False
    assert info.has_prometheus is True


@pytest.mark.asyncio
async def test_strata_zero_drafts_with_method_hint_is_active() -> None:
    handler, _ = _strata_server(drafted=0, accepted=0)
    async with _client(handler) as client:
        info = await detect_spec_decoding(client, "http://strata.test:8080/v1", backend_hint="mtp")

    assert info.active is True
    assert info.method == "mtp"


@pytest.mark.asyncio
async def test_spec_live_snapshot_reads_strata_load_and_counters() -> None:
    handler, seen = _strata_server(drafted=120, accepted=90)
    url, headers = metrics_request_target("http://strata.test:8080/v1", None, None)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        snap = await scrape_snapshot(client, url, headers)

    assert seen[0].headers["accept"] == ACCEPT
    assert snap is not None
    assert (snap.running_reqs, snap.waiting_reqs) == (2, 1)
    assert snap.kv_cache_usage == pytest.approx(0.25)
    assert snap.generation_tokens_total == 300
    assert (snap.draft_tokens, snap.accepted_tokens) == (120, 90)


def test_spec_live_labels_strata_counters_as_strata() -> None:
    previous = _parse_snapshot(_strata_prometheus(drafted=100, accepted=70))
    current = _parse_snapshot(_strata_prometheus(drafted=120, accepted=85))

    delta = compute_delta(previous, current)

    assert current.spec_backend == "strata"
    assert current.spec_method == "mtp"
    assert delta.spec_metrics_source == "strata"
    assert delta.spec_method == "mtp"
    assert delta.acceptance_rate == pytest.approx(15 / 20)
    assert delta.counter_metrics_available is True


def test_spec_live_keeps_vllm_label_when_strata_is_only_a_label_value() -> None:
    text = (
        'vllm:spec_decode_num_draft_tokens_total{model_name="acme/strata:7b"} 120\n'
        'vllm:spec_decode_num_accepted_tokens_total{model_name="acme/strata:7b"} 90\n'
        'proxy:requests_info{model_name="x",note="\\nstrata:live_state"} 1\n'
    )
    snap = _parse_snapshot(text)

    assert snap.spec_backend == "vllm"
    assert snap.spec_method == "unknown"
    assert compute_delta(snap, snap).spec_metrics_source == "vllm"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("metrics_url", "expected"),
    [
        (None, {"Accept": ACCEPT, "Authorization": "Bearer sk-test"}),
        ("http://sidecar.test:9090/metrics", {"Accept": ACCEPT}),
    ],
)
async def test_spec_live_polls_with_scoped_prometheus_headers(
    monkeypatch: pytest.MonkeyPatch, metrics_url: str | None, expected: dict[str, str]
) -> None:
    """The bearer token still stops at a cross-origin --metrics-url."""
    handlers, _ = install_spec_live_harness(monkeypatch)
    polled: list[Any] = []

    async def scrape(client: Any, url: str, headers: Any) -> None:
        polled.append(dict(headers))
        next(iter(handlers.values()))()

    monkeypatch.setattr(display, "scrape_snapshot", scrape)
    await asyncio.wait_for(
        display.run_spec_live(
            "http://api.test:8000/v1",
            api_key="sk-test",
            metrics_url=metrics_url,
            poll_interval=0.01,
        ),
        timeout=2.0,
    )

    assert polled[0] == expected


@pytest.mark.asyncio
async def test_strata_is_identified_from_its_metrics_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Asking for text lets the strata: prefix answer before any fallback probe."""
    handler, seen = _strata_server()
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        metadata.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )

    assert await metadata.probe_backend_hint("http://strata.test:8080/v1") == ("strata", "Strata")
    assert [request.url.path for request in seen] == ["/metrics"]
    assert seen[0].headers["accept"] == ACCEPT


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["acme/strata:7b", "acme/tensorfold:7b"])
async def test_engine_namespace_inside_a_vllm_label_names_no_engine(model: str) -> None:
    """Only a metric name at line start is a namespace, not a label value."""
    lab = f'engine="0",model_name="{model}"'
    text = (
        f"vllm:spec_decode_num_draft_tokens_total{{{lab}}} 0\n"
        f"vllm:spec_decode_num_accepted_tokens_total{{{lab}}} 0\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=text)

    async with _client(handler) as client:
        info = await detect_spec_decoding(client, "http://vllm.test:8000/v1")

    assert info.active is True
    assert info.method == "unknown"
    assert info.detail == "Detected via Prometheus /metrics (spec_decode counters present)"
