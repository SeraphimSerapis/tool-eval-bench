"""Tests for the web UI request gate: session token, cookie, origin, and content type."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from tests.web_support import AUTH_HEADERS, BASE_URL, TEST_TOKEN, web_client
from tool_eval_bench.storage.db import RunRepository
from tool_eval_bench.web import create_app
from tool_eval_bench.web.sanitize import is_sensitive_key, sanitize_mapping
from tool_eval_bench.web.security import SessionGate, cookie_name


@pytest.fixture()
def empty_db(tmp_path, monkeypatch: pytest.MonkeyPatch) -> str:
    db_path = str(tmp_path / "benchmarks.sqlite")
    monkeypatch.setattr(
        "tool_eval_bench.application.run_queries._repository",
        lambda: RunRepository(db_path=db_path),
    )
    return db_path


@pytest.mark.asyncio
async def test_requests_without_the_token_are_refused(empty_db: str) -> None:
    app = create_app(token=TEST_TOKEN)
    async with web_client(app, authenticated=False) as client:
        api = await client.get("/api/runs")
        assert api.status_code == 401
        assert api.json() == {"error": "Authentication required"}

        index = await client.get("/")
        assert index.status_code == 401
        assert "tool-eval-bench serve" in index.text

        wrong = await client.get("/api/runs", headers={"Authorization": "Bearer nope"})
        assert wrong.status_code == 401

        # Static assets hold no data and must load for the sign-in page.
        assert (await client.get("/static/styles.css")).status_code == 200


@pytest.mark.asyncio
async def test_token_link_sets_a_strict_cookie_and_strips_the_token(empty_db: str) -> None:
    app = create_app(token=TEST_TOKEN)
    async with web_client(app, authenticated=False) as client:
        assert (await client.get("/", params={"token": "wrong"})).status_code == 401

        login = await client.get("/", params={"token": TEST_TOKEN})
        assert login.status_code == 303
        assert login.headers["location"] == "/"
        set_cookie = login.headers["set-cookie"].lower()
        assert cookie_name(TEST_TOKEN).lower() in set_cookie
        assert "httponly" in set_cookie
        assert "samesite=strict" in set_cookie

        # httpx keeps the cookie, as a browser would.
        assert (await client.get("/")).status_code == 200
        assert (await client.get("/api/runs")).status_code == 200


def test_cookie_names_differ_per_token() -> None:
    assert cookie_name("a") != cookie_name("b")


def test_empty_token_is_rejected() -> None:
    with pytest.raises(ValueError):
        SessionGate(lambda *a: None, token="")  # type: ignore[arg-type,return-value]


@pytest.mark.asyncio
async def test_state_changing_requests_need_same_origin_json(empty_db: str) -> None:
    app = create_app(token=TEST_TOKEN)
    async with web_client(app) as client:
        cross_origin = await client.post(
            "/api/runs", json={}, headers={"Origin": "https://evil.example"}
        )
        assert cross_origin.status_code == 403

        cross_site = await client.post(
            "/api/runs", json={}, headers={"Sec-Fetch-Site": "cross-site"}
        )
        assert cross_site.status_code == 403

        # text/plain is a CORS "simple" type: a browser sends it cross-site
        # without asking, so the server refuses it.
        plain = await client.post(
            "/api/runs", content=b"{}", headers={"Content-Type": "text/plain"}
        )
        assert plain.status_code == 415

        same_origin = await client.post("/api/runs", json={}, headers={"Origin": BASE_URL})
        assert same_origin.status_code == 400  # reached the handler; body is invalid


@pytest.mark.asyncio
async def test_cross_origin_page_cannot_make_the_server_send_the_env_key(
    empty_db: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test for the probe that forwarded TOOL_EVAL_API_KEY anywhere."""
    monkeypatch.setenv("TOOL_EVAL_API_KEY", "sk-env-sentinel")
    sent: list[dict[str, Any]] = []

    async def _record(url: str, endpoint: str, headers: Any, fmt: str) -> Any:
        sent.append(dict(headers))
        return {"data": []}

    monkeypatch.setattr("tool_eval_bench.web.launcher._fetch_models_json", _record)
    app = create_app(token=TEST_TOKEN)
    rebinding = {"Origin": "http://attacker.example:8080", "Host": "attacker.example:8080"}
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as client:
        probe = await client.post(
            "/api/probe", json={"base_url": "https://evil.example/v1"}, headers=rebinding
        )
        launch = await client.post(
            "/api/runs",
            content=b'{"base_url": "https://evil.example/v1", "model": "x"}',
            headers={**rebinding, "Content-Type": "text/plain"},
        )
        traces = await client.get("/api/runs", headers=rebinding)
    assert (probe.status_code, launch.status_code, traces.status_code) == (401, 401, 401)
    assert sent == []


@pytest.mark.parametrize(
    ("key", "sensitive"),
    [
        ("api_key", True),
        ("hf_token", True),
        ("token", True),
        ("auth_token", True),
        ("client_secret", True),
        ("Authorization", True),
        ("max_tokens", False),
        ("fill_tokens", False),
        ("fill_tokens_target", False),
        ("max_completion_tokens", False),
        ("token_efficiency", False),
        ("tokenizer", False),
    ],
)
def test_sensitive_key_rule(key: str, sensitive: bool) -> None:
    assert is_sensitive_key(key) is sensitive


def test_sanitize_keeps_benchmark_token_settings() -> None:
    cleaned = sanitize_mapping(
        {
            "context_pressure": {"ratio": 0.5, "fill_tokens": 1200, "fill_tokens_target": 1250},
            "extra_params": {"max_completion_tokens": 512, "hf_token": "hf_secret"},
        }
    )
    assert cleaned["context_pressure"]["fill_tokens"] == 1200
    assert cleaned["extra_params"] == {"max_completion_tokens": 512}


def test_auth_headers_match_token() -> None:
    assert AUTH_HEADERS["Authorization"].endswith(TEST_TOKEN)


@pytest.mark.asyncio
async def test_every_response_carries_the_hardening_headers(empty_db: str) -> None:
    from tool_eval_bench.web.app import _STATIC_MIME_TYPES
    from tool_eval_bench.web.security import CONTENT_SECURITY_POLICY

    js_assets = [name for name in _STATIC_MIME_TYPES if name.endswith(".js")]
    assert js_assets

    app = create_app(token=TEST_TOKEN)
    async with web_client(app) as client:
        responses: dict[str, httpx.Response] = {
            "index": await client.get("/"),
            "api": await client.get("/api/runs"),
        }
        for name in js_assets:
            responses[f"static:{name}"] = await client.get(f"/static/{name}")
    async with web_client(app, authenticated=False) as anonymous:
        responses["denied_api"] = await anonymous.get("/api/runs")
        responses["signin"] = await anonymous.get("/")

    for name, resp in responses.items():
        headers = resp.headers
        assert headers["content-security-policy"] == CONTENT_SECURITY_POLICY, name
        assert headers["x-content-type-options"] == "nosniff", name
        assert headers["x-frame-options"] == "DENY", name
        assert headers["referrer-policy"] == "no-referrer", name
    assert responses["api"].headers["cache-control"] == "no-store"
    assert responses["denied_api"].headers["cache-control"] == "no-store"
    for name in js_assets:
        assert "cache-control" not in responses[f"static:{name}"].headers, name


def test_policy_forbids_inline_code_and_framing() -> None:
    from tool_eval_bench.web.security import CONTENT_SECURITY_POLICY

    directives = dict(part.split(" ", 1) for part in CONTENT_SECURITY_POLICY.split("; "))
    assert directives["script-src"] == "'self'"
    assert directives["style-src"] == "'self'"
    assert directives["frame-ancestors"] == "'none'"
    assert "unsafe" not in CONTENT_SECURITY_POLICY


@pytest.mark.asyncio
async def test_ui_assets_need_no_inline_code(empty_db: str) -> None:
    """The CSP forbids inline scripts and styles, so the shipped pages must not use them."""
    from tool_eval_bench.web.app import _STATIC_MIME_TYPES

    js_assets = [name for name in _STATIC_MIME_TYPES if name.endswith(".js")]
    assert js_assets

    app = create_app(token=TEST_TOKEN)
    scripts: dict[str, str] = {}
    async with web_client(app) as client:
        index = (await client.get("/")).text
        for name in js_assets:
            resp = await client.get(f"/static/{name}")
            assert resp.status_code == 200, name
            scripts[name] = resp.text
    async with web_client(app, authenticated=False) as anonymous:
        signin = (await anonymous.get("/")).text
    for page in (index, signin):
        assert "style=" not in page
        assert "<script>" not in page
        assert " on" + "click=" not in page
    assert '<link rel="stylesheet" href="/static/styles.css">' in signin
    # The frontend builds DOM nodes; it never parses HTML strings.
    for name, script in scripts.items():
        for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write"):
            assert sink not in script, f"{name}: {sink}"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_origin",
    [
        "null",
        f"{BASE_URL}/extra-path",
        f"{BASE_URL}?query=1",
        f"{BASE_URL}#fragment",
        "http://user@127.0.0.1:8765",
        "http://127.0.0.1:notaport",
        "http://127.0.0.1:99999",
        "http://",
        "https://127.0.0.1:8765",
    ],
)
async def test_origin_check_rejects_null_components_and_invalid_ports(
    empty_db: str, bad_origin: str
) -> None:
    app = create_app(token=TEST_TOKEN)
    async with web_client(app) as client:
        resp = await client.post("/api/runs", json={}, headers={"Origin": bad_origin})
    assert resp.status_code == 403
    assert resp.json() == {"error": "Cross-origin request refused"}


def test_same_origin_rejects_duplicate_and_invalid_host_headers() -> None:
    from starlette.datastructures import Headers

    from tool_eval_bench.web.security import _same_origin

    scope: dict[str, Any] = {"type": "http", "scheme": "http"}
    duplicate_host = Headers(raw=[(b"host", b"127.0.0.1:8765"), (b"host", b"127.0.0.1:8765")])
    assert _same_origin(BASE_URL, scope, duplicate_host) is False

    disallowed_host = Headers(raw=[(b"host", b"evil.example:8765")])
    assert _same_origin("http://evil.example:8765", scope, disallowed_host) is False

    invalid_port_host = Headers(raw=[(b"host", b"127.0.0.1:notaport")])
    assert _same_origin(BASE_URL, scope, invalid_port_host) is False


@pytest.mark.asyncio
async def test_non_http_asgi_scopes_pass_through_security_middlewares() -> None:
    from tool_eval_bench.web.security import SecurityHeaders

    seen: list[str] = []

    async def inner(scope: Any, receive: Any, send: Any) -> None:
        seen.append(scope["type"])

    wrapped = SecurityHeaders(SessionGate(inner, token=TEST_TOKEN))

    async def _noop_receive() -> dict[str, Any]:
        return {}

    async def _noop_send(message: Any) -> None:
        return None

    await wrapped({"type": "lifespan"}, _noop_receive, _noop_send)
    await wrapped({"type": "websocket", "path": "/ws"}, _noop_receive, _noop_send)
    assert seen == ["lifespan", "websocket"]


@pytest.mark.asyncio
async def test_session_gate_blocks_dns_rebinding_hosts(empty_db: str) -> None:
    app = create_app(token=TEST_TOKEN)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://attacker.example:8765"
    ) as rebound:
        assert (await rebound.get("/static/styles.css")).status_code == 403
        assert (await rebound.get("/", params={"token": TEST_TOKEN})).status_code == 403
        assert (await rebound.get("/api/runs")).status_code == 401
        authed_rebound = await rebound.get("/api/runs", headers=AUTH_HEADERS)
        assert authed_rebound.status_code == 403
        assert "Invalid Host header" in authed_rebound.json()["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "good_base_url",
    [
        "http://127.0.0.1:8765",
        "http://localhost:8765",
        "http://localhost.:8765",
        "http://[::1]:8765",
    ],
)
async def test_session_gate_accepts_loopback_hosts(empty_db: str, good_base_url: str) -> None:
    app = create_app(token=TEST_TOKEN)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url=good_base_url, headers=AUTH_HEADERS
    ) as client:
        assert (await client.get("/static/styles.css")).status_code == 200
        assert (await client.get("/api/runs")).status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_host",
    [
        "",
        ".",
        "..",
        "127.0.0.1.",
        "localhost..",
        "127.0.0.1:0",
        "127.0.0.1:70000",
        "127.0.0.1:notaport",
        "user@127.0.0.1:8765",
        "[::1",
        "evil.localhost:8765",
    ],
)
async def test_session_gate_rejects_malformed_or_non_loopback_host_headers(
    empty_db: str, bad_host: str
) -> None:
    app = create_app(token=TEST_TOKEN)
    async with web_client(app) as client:
        resp = await client.get("/api/runs", headers={"Host": bad_host})
        assert resp.status_code == 403


@pytest.mark.asyncio
async def test_session_gate_rejects_duplicate_host_malformed_origin_and_sets_secure_cookie(
    empty_db: str,
) -> None:
    app = create_app(token=TEST_TOKEN)
    transport = httpx.ASGITransport(app=app)

    # Duplicate Host headers are rejected as 403 rather than trusting the first.
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8765") as client:
        dup_resp = await client.get(
            "/api/runs",
            headers=[
                ("Authorization", f"Bearer {TEST_TOKEN}"),
                ("Host", "127.0.0.1:8765"),
                ("Host", "attacker.example:8765"),
            ],
        )
        assert dup_resp.status_code == 403

        # Malformed IPv6 Origin returns 403 rather than raising ValueError (500).
        bad_origin = await client.post(
            "/api/runs/delete",
            json={"run_ids": ["r1"]},
            headers={**AUTH_HEADERS, "Origin": "http://[::1"},
        )
        assert bad_origin.status_code == 403

    # HTTPS login sets the Secure attribute on the session cookie.
    async with httpx.AsyncClient(
        transport=transport, base_url="https://127.0.0.1:8765"
    ) as https_client:
        login = await https_client.get("/", params={"token": TEST_TOKEN})
        assert login.status_code == 303
        assert "secure" in login.headers["set-cookie"].lower()


@pytest.mark.asyncio
async def test_session_gate_custom_and_wildcard_allowed_hosts(empty_db: str) -> None:
    lan_app = create_app(token=TEST_TOKEN, allowed_hosts=("127.0.0.1", "192.168.1.50"))
    transport = httpx.ASGITransport(app=lan_app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://192.168.1.50:8765", headers=AUTH_HEADERS
    ) as lan_client:
        assert (await lan_client.get("/api/runs")).status_code == 200

    async with httpx.AsyncClient(
        transport=transport, base_url="http://192.168.1.99:8765", headers=AUTH_HEADERS
    ) as other_client:
        assert (await other_client.get("/api/runs")).status_code == 403

    wildcard_app = create_app(token=TEST_TOKEN, allowed_hosts=("*",))
    wildcard_transport = httpx.ASGITransport(app=wildcard_app)
    async with httpx.AsyncClient(
        transport=wildcard_transport,
        base_url="http://bench-box.internal:8765",
        headers=AUTH_HEADERS,
    ) as wild_client:
        assert (await wild_client.get("/api/runs")).status_code == 200


@pytest.mark.asyncio
async def test_session_gate_rejects_same_site_fetch_metadata_across_ports(
    empty_db: str,
) -> None:
    """Cookies are port-agnostic, so a sibling localhost port sends Sec-Fetch-Site: same-site."""
    app = create_app(token=TEST_TOKEN)
    async with web_client(app, authenticated=False) as client:
        login = await client.get("/", params={"token": TEST_TOKEN})
        assert login.status_code == 303
        assert cookie_name(TEST_TOKEN) in client.cookies

        ok_get = await client.get("/api/runs", headers={"Sec-Fetch-Site": "same-origin"})
        assert ok_get.status_code == 200
        ok_nav = await client.get("/api/runs", headers={"Sec-Fetch-Site": "none"})
        assert ok_nav.status_code == 200

        cross_port_get = await client.get("/api/runs", headers={"Sec-Fetch-Site": "same-site"})
        assert cross_port_get.status_code == 403

        cross_port_post = await client.post(
            "/api/runs/delete",
            json={"run_ids": ["r1"]},
            headers={"Sec-Fetch-Site": "same-site"},
        )
        assert cross_port_post.status_code == 403


def test_sanitize_mapping_scrubs_secrets_nested_in_lists() -> None:
    dirty = {
        "model": "qwen",
        "providers": [
            {"name": "primary", "api_key": "sk-secret-1", "max_tokens": 256},
            [{"Authorization": "Bearer secret-2", "prompt_tokens": 64}],
        ],
    }
    cleaned = sanitize_mapping(dirty)
    assert cleaned == {
        "model": "qwen",
        "providers": [
            {"name": "primary", "max_tokens": 256},
            [{"prompt_tokens": 64}],
        ],
    }
