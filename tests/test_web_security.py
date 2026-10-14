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
