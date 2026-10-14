"""Tests for starting the web UI server: binding, port fallback, and the sign-in link.

Every test binds on 127.0.0.1 only. uvicorn is replaced by a stub that records
the socket it was handed instead of serving.
"""

from __future__ import annotations

import re
import socket
from collections.abc import Iterator
from typing import Any

import pytest
from rich.console import Console

import tool_eval_bench.web as web
from tool_eval_bench.web import run_web_server


class _StubUvicorn:
    """Stands in for the uvicorn module; ``Server.run`` records instead of serving."""

    started: list[dict[str, Any]] = []

    class Config:
        def __init__(self, app: Any, *, log_level: str) -> None:
            self.app = app
            self.log_level = log_level

    class Server:
        def __init__(self, config: Any) -> None:
            self.config = config

        def run(self, sockets: list[socket.socket]) -> None:
            _StubUvicorn.started.append(
                {
                    "port": sockets[0].getsockname()[1],
                    "token": self.config.app.state.token,
                }
            )


@pytest.fixture()
def stub_uvicorn(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[dict[str, Any]]]:
    _StubUvicorn.started = []
    monkeypatch.setattr(web, "_require_web_dependencies", lambda: (object(), _StubUvicorn))
    yield _StubUvicorn.started


@pytest.fixture()
def busy_port() -> Iterator[int]:
    """A loopback port held open by another listener for the test's duration."""
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("127.0.0.1", 0))
    holder.listen(1)
    try:
        yield holder.getsockname()[1]
    finally:
        holder.close()


def _link_port(output: str) -> int:
    match = re.search(r"http://localhost:(\d+)/\?token=", output)
    assert match, output
    return int(match.group(1))


def test_default_port_falls_back_to_the_next_free_one(
    monkeypatch: pytest.MonkeyPatch, stub_uvicorn: list[dict[str, Any]], busy_port: int
) -> None:
    monkeypatch.setattr(web, "DEFAULT_WEB_PORT", busy_port)
    console = Console(record=True, width=300)
    run_web_server(console, host="127.0.0.1", port=None)
    output = console.export_text()

    [started] = stub_uvicorn
    assert started["port"] != busy_port
    assert busy_port < started["port"] < busy_port + 10
    assert f"Port {busy_port} is busy, using {started['port']}." in output
    # The printed link names the port that is actually serving.
    assert _link_port(output) == started["port"]
    assert f"token={started['token']}" in output


def test_explicit_busy_port_fails_without_printing_a_link(
    stub_uvicorn: list[dict[str, Any]], busy_port: int
) -> None:
    console = Console(record=True, width=300)
    with pytest.raises(SystemExit) as exc_info:
        run_web_server(console, host="127.0.0.1", port=busy_port)
    output = console.export_text()
    assert exc_info.value.code == 1
    assert f"Could not listen on 127.0.0.1 port {busy_port}" in output
    assert "--serve-port 0" in output
    assert "token=" not in output
    assert stub_uvicorn == []


def test_every_fallback_port_busy_reports_the_range(
    monkeypatch: pytest.MonkeyPatch, stub_uvicorn: list[dict[str, Any]]
) -> None:
    def _always_busy(host: str, port: int) -> socket.socket:
        raise OSError(98, "Address already in use")

    monkeypatch.setattr(web, "_bind", _always_busy)
    console = Console(record=True, width=300)
    with pytest.raises(SystemExit):
        run_web_server(console, host="127.0.0.1", port=None)
    output = console.export_text()
    assert f"any port from {web.DEFAULT_WEB_PORT} to {web.DEFAULT_WEB_PORT + 9}" in output
    assert "Address already in use" in output
    assert stub_uvicorn == []


def test_port_zero_lets_the_os_choose(stub_uvicorn: list[dict[str, Any]]) -> None:
    console = Console(record=True, width=300)
    run_web_server(console, host="127.0.0.1", port=0)
    output = console.export_text()
    [started] = stub_uvicorn
    assert started["port"] > 0
    assert _link_port(output) == started["port"]
    assert "is busy" not in output
    assert "non-loopback" not in output


def test_non_loopback_host_prints_the_exposure_warning(
    monkeypatch: pytest.MonkeyPatch, stub_uvicorn: list[dict[str, Any]]
) -> None:
    # 192.0.2.10 is a reserved documentation address (TEST-NET-1). The bind is
    # redirected to loopback, so the test never listens beyond this machine.
    real_bind = web.bind_web_socket
    monkeypatch.setattr(web, "bind_web_socket", lambda host, port: real_bind("127.0.0.1", 0))
    console = Console(record=True, width=400)
    run_web_server(console, host="192.0.2.10", port=9000)
    output = console.export_text()
    assert "non-loopback address 192.0.2.10" in output
    assert "API keys" in output
    assert re.search(r"http://192\.0\.2\.10:\d+/\?token=", output)


@pytest.mark.parametrize(
    ("host", "shown"),
    [
        ("127.0.0.1", "localhost"),
        ("::1", "[::1]"),
        ("[::1]", "[::1]"),
        ("192.0.2.10", "192.0.2.10"),
    ],
)
def test_display_host_brackets_ipv6(host: str, shown: str) -> None:
    assert web._display_host(host) == shown


def test_default_port_avoids_inference_discovery_ports() -> None:
    from tool_eval_bench.cli.server import DISCOVERY_PORTS

    discovery = {port for port, _hint, _label in DISCOVERY_PORTS}
    fallback = set(range(web.DEFAULT_WEB_PORT, web.DEFAULT_WEB_PORT + 10))
    assert not fallback & discovery
