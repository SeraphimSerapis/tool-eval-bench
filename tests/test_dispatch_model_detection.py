"""Behavioral tests for model discovery and its user-facing failures."""

from __future__ import annotations

import json

import httpx
import pytest
from rich.console import Console

from tool_eval_bench.cli import dispatch
from tool_eval_bench.utils.headers import USER_AGENT


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
    expected_headers = {"User-Agent": USER_AGENT, "Authorization": "Bearer api-key"}
    assert client.requested == [
        ("https://user:password@secret.example/v1/models", expected_headers),
        ("https://user:password@secret.example/v1/models", expected_headers),
    ]
    output = console.export_text()
    assert "used /models fallback" in output
    assert "Please enter a number between 1 and 2" in output
    assert "Selected: model-b" in output


@pytest.mark.parametrize(
    ("responses", "message", "code"),
    [
        (
            [
                httpx.ConnectError(
                    "refused",
                    request=httpx.Request("GET", "http://localhost:8000/v1/models"),
                )
            ],
            "Could not connect",
            2,
        ),
        (
            [_response(401, "http://localhost:8000/v1/models")],
            "Server returned 401",
            2,
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
            2,
        ),
        (
            [_response(200, "http://localhost:8000/v1/models", json={"data": []})],
            "empty model list",
            3,
        ),
    ],
)
def test_detect_model_exits_with_actionable_errors(
    monkeypatch: pytest.MonkeyPatch,
    responses,
    message: str,
    code: int,
) -> None:
    # Console mode uses the same exit codes as --json, so scripts that skip
    # --json can still tell an unreachable server from an empty one.
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: _Client(responses))
    console = Console(record=True, width=120)

    with pytest.raises(SystemExit) as exc_info:
        dispatch._detect_model("http://localhost:8000", None, console)

    assert exc_info.value.code == code
    assert message in console.export_text()


@pytest.mark.parametrize(
    "body",
    [
        {"data": ["llama-3"]},
        {"data": {"id": "m"}},
        {"data": [{"id": "ok"}, "stray"]},
        ["llama-3"],
    ],
)
@pytest.mark.parametrize("headless", [False, True])
def test_detect_model_rejects_a_model_list_that_is_not_objects(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    body: object,
    headless: bool,
) -> None:
    responses = [_response(200, "http://localhost:8000/v1/models", json=body)]
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: _Client(responses))
    console = Console(record=True, width=120)

    with pytest.raises(SystemExit) as exc_info:
        dispatch._detect_model("http://localhost:8000", None, console, headless=headless)

    assert exc_info.value.code == 2
    if headless:
        event = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
        assert event["error"] == "invalid_response"
        assert "not a list of objects" in event["message"]
    else:
        assert "not a list of objects" in console.export_text()


def test_detect_model_accepts_the_native_gemini_list(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [
        _response(200, "http://localhost:8000/v1/models", json={"models": [{"id": "gemini-x"}]})
    ]
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: _Client(responses))

    selected = dispatch._detect_model("http://localhost:8000", None, Console(record=True))

    assert selected[0] == "gemini-x"
