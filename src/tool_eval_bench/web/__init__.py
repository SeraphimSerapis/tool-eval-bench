"""Optional local web UI for tool-eval-bench."""

from __future__ import annotations

import os
import socket
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rich.console import Console

WEB_INSTALL_HINT = (
    "The web UI requires the [web] optional dependencies.\n"
    "Install them with: pip install 'tool-eval-bench[web]'"
)

#: Not 8080: that is llama.cpp's default and one of the ports server
#: discovery probes, so the UI would collide with the servers it benchmarks.
DEFAULT_WEB_PORT = 8765
#: Ports tried from DEFAULT_WEB_PORT upward when no port was requested.
_PORT_FALLBACK_ATTEMPTS = 10

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def _require_web_dependencies() -> tuple[Any, Any]:
    """Import and return ``(starlette, uvicorn)``, or raise RuntimeError with install hint."""
    try:
        import starlette
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(WEB_INSTALL_HINT) from exc
    return starlette, uvicorn


def create_app(
    *,
    output_dir: str | None = None,
    redact_url: bool = False,
    token: str | None = None,
) -> Any:
    """Create and return the Starlette ASGI application.

    ``token`` is the session secret every request must prove; a fresh one is
    generated when omitted and exposed as ``app.state.token``.
    """
    _require_web_dependencies()
    from tool_eval_bench.web.app import build_starlette_app
    from tool_eval_bench.web.security import generate_token

    session_token = token or generate_token()
    app = build_starlette_app(token=session_token, output_dir=output_dir, redact_url=redact_url)
    app.state.token = session_token
    return app


def _display_host(host: str) -> str:
    if host == "127.0.0.1":
        return "localhost"
    return f"[{host}]" if ":" in host and not host.startswith("[") else host


def _bind(host: str, port: int) -> socket.socket:
    """Open a listening socket on ``host``:``port``; port 0 lets the OS choose."""
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        # Lets a restart reuse a port still in TIME_WAIT. Windows gives the flag
        # a different meaning (binding over a live listener), so skip it there.
        if os.name != "nt":
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host.strip("[]"), port))
        sock.listen(128)
    except OSError:
        sock.close()
        raise
    return sock


def bind_web_socket(host: str, port: int | None) -> socket.socket:
    """Bind the UI's socket, falling back to nearby ports only when none was requested.

    An explicit ``port`` is honoured exactly so bookmarks and SSH tunnels keep
    working; when it is busy this raises instead of moving somewhere else.
    """
    if port is not None:
        return _bind(host, port)
    last = DEFAULT_WEB_PORT + _PORT_FALLBACK_ATTEMPTS - 1
    for candidate in range(DEFAULT_WEB_PORT, last):
        try:
            return _bind(host, candidate)
        except OSError:
            continue
    # The last attempt's error propagates, naming why the whole range failed.
    return _bind(host, last)


def run_web_server(
    console: Console,
    *,
    host: str = "127.0.0.1",
    port: int | None = None,
    output_dir: str | None = None,
    redact_url: bool = False,
) -> None:
    """Start the local web UI server and block until interrupted.

    ``port`` None means the default port, or the next free one after it.
    The sign-in link is printed only after the socket is bound, so it always
    names a port that is actually serving.
    """
    from rich.markup import escape

    try:
        _, uvicorn = _require_web_dependencies()
    except RuntimeError as exc:
        console.print(f"\n[bold red]Error:[/] {escape(str(exc))}\n")
        raise SystemExit(1) from None

    try:
        sock = bind_web_socket(host, port)
    except OSError as exc:
        if port is None:
            last = DEFAULT_WEB_PORT + _PORT_FALLBACK_ATTEMPTS - 1
            where = f"any port from {DEFAULT_WEB_PORT} to {last}"
        else:
            where = f"port {port}"
        console.print(
            f"\n[bold red]Error:[/] Could not listen on {escape(host)} {where}: "
            f"{escape(exc.strerror or str(exc))}.\n"
            "  [dim]Pick another with --serve-port PORT, or --serve-port 0 for any free port.[/]\n"
        )
        raise SystemExit(1) from None

    bound_port = sock.getsockname()[1]
    app = create_app(output_dir=output_dir, redact_url=redact_url)
    if host not in _LOOPBACK_HOSTS:
        console.print(
            "\n  [bold yellow]Warning:[/] Binding web UI to non-loopback address "
            f"[bold]{escape(host)}[/]. Anyone who gets the link below can read stored traces "
            "and start runs that use this machine's API keys, and the connection is plain "
            "HTTP. Prefer 127.0.0.1 with SSH port forwarding."
        )
    if port is None and bound_port != DEFAULT_WEB_PORT:
        console.print(f"\n  [dim]Port {DEFAULT_WEB_PORT} is busy, using {bound_port}.[/]")
    # The token is this process's session secret, printed once for the user
    # who started it; it changes on every start.
    console.print(
        f"\n  [bold green]tool-eval-bench web UI[/] listening on "
        f"[bold cyan]http://{_display_host(host)}:{bound_port}/?token={app.state.token}[/]"
        "\n  [dim]Open that link to sign in. Press Ctrl+C to stop.[/]\n",
        soft_wrap=True,
    )
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    try:
        server.run(sockets=[sock])
    finally:
        sock.close()


__all__ = [
    "DEFAULT_WEB_PORT",
    "WEB_INSTALL_HINT",
    "bind_web_socket",
    "create_app",
    "run_web_server",
]
