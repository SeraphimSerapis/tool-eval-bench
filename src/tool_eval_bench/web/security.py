"""Request gate for the local web UI.

The server holds credentials (``TOOL_EVAL_API_KEY`` from the environment or a
``.env``) and can start runs that spend them, so every request except static
assets must prove it came from someone who saw the startup link:

* ``GET /?token=...`` with the per-process token sets an ``HttpOnly``,
  ``SameSite=Strict`` session cookie and redirects to ``/``. Browsers never
  attach a ``SameSite=Strict`` cookie to a cross-site request, and a DNS
  rebinding page runs on the attacker's origin, which never received it.
* Scripts may send ``Authorization: Bearer <token>`` instead.
* Requests that change state must also come from this origin (when the
  browser says where it came from) and carry ``application/json``, which a
  cross-site form or ``text/plain`` fetch cannot send without a CORS
  preflight that this server never answers.
"""

from __future__ import annotations

import hashlib
import secrets
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any
from urllib.parse import urlsplit

from starlette.datastructures import Headers, QueryParams
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
_PUBLIC_PREFIXES = ("/static/",)

_LOGIN_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>tool-eval-bench</title></head>
<body style="font-family: system-ui, sans-serif; max-width: 40rem; margin: 4rem auto">
<h1>Session link required</h1>
<p>Open the link that <code>tool-eval-bench serve</code> printed in your terminal.
It carries a token that changes every time the server starts.</p>
</body></html>
"""


def generate_token() -> str:
    """Return a fresh session token for one server process."""
    return secrets.token_urlsafe(32)


def cookie_name(token: str) -> str:
    """Name the session cookie after the token.

    Cookies are not scoped by port, so two servers on one host would otherwise
    overwrite each other's session.
    """
    return "teb_session_" + hashlib.sha256(token.encode()).hexdigest()[:12]


class SessionGate:
    """Pure ASGI middleware enforcing the token, origin, and content-type rules."""

    def __init__(self, app: ASGIApp, *, token: str) -> None:
        if not token:
            raise ValueError("The web UI session token must not be empty")
        self.app = app
        self._token = token
        self._cookie = cookie_name(token)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        response = self._check(scope)
        if response is not None:
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)

    def _matches(self, candidate: str | None) -> bool:
        return bool(candidate) and secrets.compare_digest(
            str(candidate).encode(), self._token.encode()
        )

    def _authenticated(self, headers: Headers, cookies: dict[str, str]) -> bool:
        if self._matches(cookies.get(self._cookie)):
            return True
        scheme, _, credential = headers.get("authorization", "").partition(" ")
        return scheme.lower() == "bearer" and self._matches(credential.strip())

    def _check(self, scope: Scope) -> Response | None:
        path: str = scope.get("path", "")
        if path.startswith(_PUBLIC_PREFIXES):
            return None
        headers = Headers(scope=scope)
        method = str(scope.get("method", "GET")).upper()

        if path == "/" and method == "GET":
            offered = QueryParams(scope.get("query_string", b"").decode("latin-1")).get("token")
            if offered is not None:
                if not self._matches(offered):
                    return HTMLResponse(_LOGIN_PAGE, status_code=401)
                # Redirect so the token leaves the address bar and is not
                # re-sent by a reload.
                redirect = RedirectResponse("/", status_code=303)
                redirect.set_cookie(
                    self._cookie, self._token, httponly=True, samesite="strict", path="/"
                )
                return redirect

        if not self._authenticated(headers, _parse_cookies(headers)):
            if path == "/":
                return HTMLResponse(_LOGIN_PAGE, status_code=401)
            return JSONResponse({"error": "Authentication required"}, status_code=401)

        if method in _SAFE_METHODS:
            return None
        if headers.get("sec-fetch-site", "").lower() == "cross-site":
            return JSONResponse({"error": "Cross-site request refused"}, status_code=403)
        origin = headers.get("origin")
        if origin is not None and not _same_origin(origin, scope, headers):
            return JSONResponse({"error": "Cross-origin request refused"}, status_code=403)
        content_type = headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            return JSONResponse(
                {"error": "Requests that change state must send application/json"},
                status_code=415,
            )
        return None


def _parse_cookies(headers: Headers) -> dict[str, str]:
    cookies: dict[str, str] = {}
    for chunk in headers.get("cookie", "").split(";"):
        name, sep, value = chunk.strip().partition("=")
        if sep:
            cookies[name] = value
    return cookies


def _same_origin(origin: str, scope: Scope, headers: Headers) -> bool:
    """Whether ``origin`` names this server as the request reached it."""
    if origin == "null":
        return False
    parsed = urlsplit(origin)
    scheme = str(scope.get("scheme", "http"))
    host = headers.get("host", "")
    return parsed.scheme == scheme and parsed.netloc.lower() == host.lower()
