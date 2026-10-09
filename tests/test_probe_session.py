"""Probing an endpoint must cost one connection pool, and two timeouts at worst.

Each probe used to open its own `AsyncClient` and the ladder ran to the end
regardless, so a wrong `--base-url` spent `_PROBE_TIMEOUT` per rung before the
run could start. A refused connection ends the ladder at once; a server that
accepts connections and never answers ends it after two timeouts in a row.
"""

from __future__ import annotations

import asyncio
import logging
import time
from unittest.mock import AsyncMock

import httpx
import pytest

from tool_eval_bench.api import run_benchmark
from tool_eval_bench.application import service as service_module
from tool_eval_bench.application.service import BenchmarkService
from tool_eval_bench.domain.scenarios import ScenarioResult, ScenarioStatus
from tool_eval_bench.evals.scenarios import ALL_SCENARIOS
from tool_eval_bench.runner.orchestrator import score_results
from tool_eval_bench.utils import metadata

_REAL_CLIENT = httpx.AsyncClient


class _CountingClient:
    """Records every GET, and fails the way a dead host does."""

    def __init__(self, *, connect_error: bool = False) -> None:
        self.urls: list[str] = []
        self._connect_error = connect_error

    async def get(self, url: str, headers: dict[str, str] | None = None) -> httpx.Response:
        self.urls.append(url)
        if self._connect_error:
            raise httpx.ConnectError("connection refused")
        return httpx.Response(404, request=httpx.Request("GET", url))

    async def __aenter__(self) -> _CountingClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    """Install one client for the whole probe sequence and hand it back."""

    def install(**kwargs) -> _CountingClient:
        made = _CountingClient(**kwargs)
        monkeypatch.setattr(metadata.httpx, "AsyncClient", lambda **_: made)
        return made

    return install


@pytest.mark.asyncio
async def test_an_unreachable_endpoint_stops_the_ladder_after_one_attempt(client) -> None:
    dead = client(connect_error=True)

    assert await metadata.probe_backend_hint("http://127.0.0.1:9") is None
    assert dead.urls == ["http://127.0.0.1:9/metrics"], (
        f"probed {len(dead.urls)} endpoints on a host that is not answering"
    )


@pytest.mark.asyncio
async def test_an_unreachable_endpoint_stops_the_engine_probes_too(client) -> None:
    dead = client(connect_error=True)

    result = await metadata._probe_engine("http://127.0.0.1:9", None, "unknown")

    assert result == {}
    assert len(dead.urls) == 1


LADDER = {"/metrics", "/version", "/v1/models", "/health", "/.well-known/serviceinfo", "/props"}


def _paths(client: _CountingClient) -> list[str]:
    return [url.removeprefix("http://localhost:8000") for url in client.urls]


@pytest.mark.asyncio
async def test_a_responding_endpoint_still_walks_every_rung(client) -> None:
    """A 404 says something about the server; it must not end the sequence.

    Each endpoint is asked once: later rungs reuse answers earlier ones fetched.
    """
    answering = client()

    assert await metadata.probe_backend_hint("http://localhost:8000") is None
    paths = _paths(answering)
    assert len(paths) == len(set(paths)), f"fetched an endpoint twice: {paths}"
    assert set(paths) == LADDER


@pytest.mark.asyncio
async def test_unlabelled_engine_probe_asks_each_endpoint_once(client) -> None:
    answering = client()

    assert await metadata._probe_engine("http://localhost:8000", None, "unknown") == {}
    paths = _paths(answering)
    assert len(paths) == len(set(paths)), f"fetched an endpoint twice: {paths}"
    assert set(paths) == LADDER - {"/metrics"}


@pytest.mark.asyncio
async def test_separate_probe_calls_do_not_share_answers(client) -> None:
    answering = client()

    await metadata.probe_backend_hint("http://localhost:8000")
    await metadata.probe_backend_hint("http://localhost:8000")

    paths = _paths(answering)
    assert sorted(paths) == sorted([*LADDER, *LADDER])


class _KeyedClient:
    """Refuses requests without the key, and fails to connect on /dead."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    async def get(self, url: str, headers: dict[str, str] | None = None) -> httpx.Response:
        auth = (headers or {}).get("Authorization")
        self.calls.append((url, auth))
        if url.endswith("/dead"):
            raise httpx.ConnectError("connection refused")
        status = 200 if auth == "Bearer k" else 401
        return httpx.Response(status, request=httpx.Request("GET", url))


@pytest.mark.asyncio
async def test_a_keyless_refusal_does_not_answer_a_keyed_request() -> None:
    keyed = _KeyedClient()
    session = metadata._ProbeSession(keyed)
    url = "http://test/props"

    refused = await metadata._probe_get(session, url, headers={}, what="t")
    accepted = await metadata._probe_get(
        session, url, headers={"Authorization": "Bearer k"}, what="t"
    )

    assert refused is not None and refused.status_code == 401
    assert accepted is not None and accepted.status_code == 200
    assert keyed.calls == [(url, None), (url, "Bearer k")]


@pytest.mark.asyncio
async def test_an_answer_survives_a_later_connect_failure() -> None:
    keyed = _KeyedClient()
    session = metadata._ProbeSession(keyed)
    headers = {"Authorization": "Bearer k"}

    first = await metadata._probe_get(session, "http://test/props", headers=headers, what="t")
    assert await metadata._probe_get(session, "http://test/dead", headers=headers, what="t") is None
    assert session.unreachable

    again = await metadata._probe_get(session, "http://test/props", headers=headers, what="t")
    assert again is first
    assert (
        await metadata._probe_get(session, "http://test/other", headers=headers, what="t") is None
    )
    assert [url for url, _ in keyed.calls] == ["http://test/props", "http://test/dead"]


@pytest.mark.asyncio
async def test_the_whole_ladder_shares_one_connection_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built = 0

    def count(**kwargs) -> _CountingClient:
        nonlocal built
        built += 1
        return _CountingClient()

    monkeypatch.setattr(metadata.httpx, "AsyncClient", count)

    await metadata.probe_backend_hint("http://localhost:8000")

    assert built == 1, f"opened {built} clients for one endpoint"


# -- A server that accepts connections and never answers ---------------------

TIMEOUTS = [httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout]


def _serve(monkeypatch: pytest.MonkeyPatch, handler) -> list[str]:
    """Route every probe through *handler* and return the paths it was asked for."""
    paths: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return handler(request)

    monkeypatch.setattr(
        metadata.httpx,
        "AsyncClient",
        lambda **kw: _REAL_CLIENT(transport=httpx.MockTransport(respond), **kw),
    )
    return paths


def _hang(timeout: type[httpx.TimeoutException]):
    def handler(request: httpx.Request) -> httpx.Response:
        raise timeout("timed out", request=request)

    return handler


def _latch_records(caplog: pytest.LogCaptureFixture) -> list[tuple[int, str]]:
    return [
        (r.levelno, r.message)
        for r in caplog.records
        if r.name == metadata.__name__ and r.levelno >= logging.INFO
    ]


def _latch_message(purpose: str) -> str:
    return (
        "The server did not answer 2 probes in a row within 5 s; "
        f"skipping the remaining {purpose} probes"
    )


@pytest.mark.parametrize("timeout", TIMEOUTS)
async def test_a_silent_server_stops_the_ladder_after_two_timeouts(
    monkeypatch, caplog, timeout
) -> None:
    paths = _serve(monkeypatch, _hang(timeout))

    with caplog.at_level(logging.INFO, logger=metadata.__name__):
        assert await metadata.probe_backend_hint("http://test") is None

    assert paths == ["/metrics", "/version"]
    # INFO, never WARNING: the CLI has no logging config, so a warning would
    # print plain text on stderr, which --json reserves for JSON lines.
    assert _latch_records(caplog) == [(logging.INFO, _latch_message("backend detection"))]


@pytest.mark.parametrize("timeout", TIMEOUTS)
async def test_a_silent_server_stops_the_engine_probes_after_two_timeouts(
    monkeypatch, caplog, timeout
) -> None:
    paths = _serve(monkeypatch, _hang(timeout))

    with caplog.at_level(logging.INFO, logger=metadata.__name__):
        context = await metadata.collect_run_context(
            model="m", backend="unknown", base_url="http://test"
        )

    assert paths == ["/v1/models", "/version"]
    assert _latch_records(caplog) == [(logging.INFO, _latch_message("engine metadata"))]
    assert context.model == "m"
    assert context.engine_name is None
    assert context.server_model_id is None
    assert context.max_model_len is None


async def test_one_slow_endpoint_does_not_end_the_ladder(monkeypatch) -> None:
    """llama-server answers /metrics from its task queue, behind a decode step."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/metrics":
            raise httpx.ReadTimeout("busy", request=request)
        body = {
            "/v1/models": {"data": [{"id": "m", "owned_by": "llamacpp"}]},
            "/props": {"build_info": "b1-abc", "total_slots": 1},
        }.get(request.url.path)
        return httpx.Response(200, json=body) if body else httpx.Response(404)

    _serve(monkeypatch, handler)

    assert await metadata.probe_backend_hint("http://test") == ("llamacpp", "llama.cpp")


async def test_an_answer_between_timeouts_resets_the_count() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/slow"):
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(404)

    async with _REAL_CLIENT(transport=httpx.MockTransport(handler)) as client:
        session = metadata._ProbeSession(client)

        async def get(path: str) -> httpx.Response | None:
            return await metadata._probe_get(session, f"http://test{path}", headers={}, what=path)

        await get("/slow1")
        await get("/fine")
        await get("/slow2")
        assert not session.unreachable, "two timeouts with an answer between them latched"

        await get("/slow3")
        assert session.unreachable
        assert await get("/fine2") is None


async def test_a_silent_server_still_lets_an_api_run_start(monkeypatch) -> None:
    scenario = next(s for s in ALL_SCENARIOS if s.id == "TC-01")
    summary = score_results([ScenarioResult("TC-01", ScenarioStatus.PASS, 2, "ok")], [scenario])
    run_all = AsyncMock(return_value=summary)
    monkeypatch.setattr(service_module, "run_all_scenarios", run_all)
    monkeypatch.setattr(BenchmarkService, "_adapter_for", lambda *a, **k: object())
    paths = _serve(monkeypatch, _hang(httpx.ReadTimeout))

    result = await run_benchmark(
        model="m", base_url="http://test", scenarios=[scenario], persist=False
    )

    run_all.assert_awaited_once()
    assert result["status"] == "completed"
    assert result["metadata"]["backend"] == "unknown"
    # Detection and metadata are separate sessions: two timeouts each, not one per rung.
    assert paths == ["/metrics", "/version", "/v1/models", "/version"]


async def test_a_real_socket_that_never_answers_costs_two_timeouts(monkeypatch) -> None:
    """End to end over TCP: httpx raises ReadTimeout and the ladder stops."""
    connections = 0

    async def accept_and_ignore(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        nonlocal connections
        connections += 1
        await reader.read()  # until the client gives up and closes
        writer.close()

    server = await asyncio.start_server(accept_and_ignore, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    monkeypatch.setattr(metadata, "_PROBE_TIMEOUT", 0.25)
    try:
        started = time.monotonic()
        assert await metadata.probe_backend_hint(f"http://127.0.0.1:{port}") is None
        elapsed = time.monotonic() - started
    finally:
        server.close()
        await server.wait_closed()

    assert connections == 2
    # The full ladder is six rungs, 1.5 s at this timeout.
    assert elapsed < 1.5


async def test_a_body_that_trickles_forever_is_cut_off_at_the_probe_timeout(monkeypatch) -> None:
    """httpx's timeout is per read, so a server that drips its body never trips it.

    Each probe must still end at ``_PROBE_TIMEOUT``, and the overrun must count
    as a timeout: two in a row end the ladder, not one (the builtin
    ``TimeoutError`` is an ``OSError``, which would latch it at once).
    """
    connections = 0

    async def drip(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        nonlocal connections
        connections += 1
        try:
            await reader.readuntil(b"\r\n\r\n")
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 1000000\r\n\r\n")
            while True:
                writer.write(b"x")
                await writer.drain()
                await asyncio.sleep(0.02)
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()

    server = await asyncio.start_server(drip, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    monkeypatch.setattr(metadata, "_PROBE_TIMEOUT", 0.25)
    try:
        started = time.monotonic()
        # The outer bound only keeps a regression from hanging the suite.
        hint = await asyncio.wait_for(metadata.probe_backend_hint(f"http://127.0.0.1:{port}"), 10)
        elapsed = time.monotonic() - started
    finally:
        server.close()
        await server.wait_closed()

    assert hint is None
    assert connections == 2
    assert elapsed < 1.5
