"""Optional local web UI for tool-eval-bench."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rich.console import Console

WEB_INSTALL_HINT = (
    "The web UI requires the [web] optional dependencies.\n"
    "Install them with: pip install 'tool-eval-bench[web]'"
)

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


def run_web_server(
    console: Console,
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    output_dir: str | None = None,
    redact_url: bool = False,
) -> None:
    """Start the local web UI server and block until interrupted."""
    try:
        _, uvicorn = _require_web_dependencies()
    except RuntimeError as exc:
        from rich.markup import escape

        console.print(f"\n[bold red]Error:[/] {escape(str(exc))}\n")
        raise SystemExit(1) from None

    app = create_app(output_dir=output_dir, redact_url=redact_url)
    if host not in _LOOPBACK_HOSTS:
        console.print(
            "\n  [bold yellow]Warning:[/] Binding web UI to non-loopback address "
            f"[bold]{host}[/]. Anyone who gets the link below can read stored traces and "
            "start runs that use this machine's API keys, and the connection is plain HTTP. "
            "Prefer 127.0.0.1 with SSH port forwarding."
        )
    # The token is this process's session secret, printed once for the user
    # who started it; it changes on every start.
    console.print(
        f"\n  [bold green]tool-eval-bench web UI[/] listening on "
        f"[bold cyan]http://{_display_host(host)}:{port}/?token={app.state.token}[/]"
        "\n  [dim]Open that link to sign in. Press Ctrl+C to stop.[/]\n",
        soft_wrap=True,
    )
    uvicorn.run(app, host=host, port=port, log_level="warning")


__all__ = ["WEB_INSTALL_HINT", "create_app", "run_web_server"]
