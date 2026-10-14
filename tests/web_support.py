"""Shared helpers for web UI tests: a fixed session token and an authenticated client."""

from __future__ import annotations

from typing import Any

import httpx

TEST_TOKEN = "test-session-token"
AUTH_HEADERS = {"Authorization": f"Bearer {TEST_TOKEN}"}
BASE_URL = "http://127.0.0.1:8080"


def web_client(app: Any, *, authenticated: bool = True) -> httpx.AsyncClient:
    """Return an in-process client for *app*, signed in with the test token by default."""
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url=BASE_URL,
        headers=AUTH_HEADERS if authenticated else None,
    )
