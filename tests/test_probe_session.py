"""Probing an endpoint must cost one connection, and one timeout at worst.

Each probe used to open its own `AsyncClient` and the ladder ran to the end
regardless, so a wrong `--base-url` spent `_PROBE_TIMEOUT` per rung before the
run could start.
"""

from __future__ import annotations

import httpx
import pytest

from tool_eval_bench.utils import metadata


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
