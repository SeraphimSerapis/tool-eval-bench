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


@pytest.mark.asyncio
async def test_a_responding_endpoint_still_walks_every_rung(client) -> None:
    """A 404 says something about the server; it must not end the sequence."""
    answering = client()

    assert await metadata.probe_backend_hint("http://localhost:8000") is None
    assert [url.rsplit("/", 1)[-1] for url in answering.urls] == [
        "metrics",
        "version",
        "models",
        "props",
        "health",
    ]


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
