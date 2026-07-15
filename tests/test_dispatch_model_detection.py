"""Behavioral tests for model discovery and its user-facing failures."""

from __future__ import annotations

import httpx
import pytest
from rich.console import Console

from tool_eval_bench.cli import dispatch


class _Client:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requested: list[tuple[str, dict[str, str]]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return None

    async def get(self, url: str, *, headers: dict[str, str]):
        self.requested.append((url, headers))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def _response(status: int, url: str, **kwargs) -> httpx.Response:
    return httpx.Response(status, request=httpx.Request("GET", url), **kwargs)


def test_detect_model_falls_back_and_reprompts_for_valid_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _Client(
        [
            _response(404, "https://secret.example/v1/models"),
            _response(
                200,
                "https://secret.example/models",
                json={
                    "data": [
                        {"id": "alias-a", "root": "org/model-a"},
                        {"id": "model-b"},
                        {"root": "missing-id-is-ignored"},
                    ]
                },
            ),
        ]
    )
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client)
    choices = iter(["invalid", "9", "2"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(choices))
    console = Console(record=True, width=120)

    selected = dispatch._detect_model(
        "https://user:password@secret.example/v1",
        "api-key",
        console,
        display_url="https://secret.example/v1",
    )

    assert selected == ("model-b", "model-b")
    assert client.requested == [
        ("https://user:password@secret.example/v1/models", {"Authorization": "Bearer api-key"}),
        ("https://user:password@secret.example/v1/models", {"Authorization": "Bearer api-key"}),
    ]
    output = console.export_text()
    assert "used /models fallback" in output
    assert "Please enter a number between 1 and 2" in output
    assert "Selected: model-b" in output


@pytest.mark.parametrize(
    ("responses", "message"),
    [
        (
            [
                httpx.ConnectError(
                    "refused",
                    request=httpx.Request("GET", "http://localhost:8000/v1/models"),
                )
            ],
            "Could not connect",
        ),
        (
            [_response(401, "http://localhost:8000/v1/models")],
            "Server returned 401",
        ),
        (
            [
                _response(
                    200,
                    "http://localhost:8000/v1/models",
                    content=b"not-json",
                    headers={"content-type": "text/plain"},
                )
            ],
            "invalid JSON",
        ),
        (
            [_response(200, "http://localhost:8000/v1/models", json={"data": []})],
            "empty model list",
        ),
    ],
)
def test_detect_model_exits_with_actionable_errors(
    monkeypatch: pytest.MonkeyPatch,
    responses,
    message: str,
) -> None:
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: _Client(responses))
    console = Console(record=True, width=120)

    with pytest.raises(SystemExit) as exc_info:
        dispatch._detect_model("http://localhost:8000", None, console)

    assert exc_info.value.code == 1
    assert message in console.export_text()
