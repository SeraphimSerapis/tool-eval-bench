"""Warm-up must survive endpoints that reject optional speed hints.

Gemini's OpenAI-compatibility layer answers HTTP 400 for unknown request
fields instead of ignoring them, which used to fail warm-up outright.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from tool_eval_bench.runner import throughput


class _Response:
    def __init__(self, status_code: int, text: str = "unknown field") -> None:
        self.status_code = status_code
        self.text = text

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _RecordingClient:
    """Fake httpx.AsyncClient that fails every request carrying a given key."""

    def __init__(self, reject_key: str | None, status_code: int = 400) -> None:
        self._reject_key = reject_key
        self._status_code = status_code
        self.calls: list[tuple[str, dict[str, Any], dict[str, str]]] = []

    async def __aenter__(self) -> _RecordingClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def post(self, url: str, *, json: dict[str, Any], headers: dict[str, str]) -> _Response:
        self.calls.append((url, json, headers))
        if self._reject_key is not None and self._reject_key in json:
            text = (
                "Unsupported parameter: max_tokens. Use max_completion_tokens instead."
                if self._reject_key == "max_tokens"
                else "unknown field"
            )
            return _Response(self._status_code, text)
        return _Response(200)

    async def completion(self, payload: dict[str, Any]) -> _Response:
        return await self.post(
            getattr(self, "completion_url", "http://server/v1/chat/completions"),
            json=payload,
            headers=getattr(self, "completion_headers", {}),
        )


@pytest.fixture
def client_factory():
    def install(client: _RecordingClient):
        def factory(**kwargs: Any) -> _RecordingClient:
            client.completion_url = (
                kwargs.get("completion_url") or "http://server/v1/chat/completions"
            )
            client.completion_headers = kwargs.get("completion_headers") or {}
            return client

        return client, factory

    return install


class TestWarmupHintFallback:
    def test_retries_without_chat_template_kwargs_on_400(self, client_factory):
        client, factory = client_factory(_RecordingClient("chat_template_kwargs"))

        elapsed = asyncio.run(
            throughput.warmup("http://server/v1", "m", "key", client_factory=factory)
        )

        assert elapsed >= 0
        assert len(client.calls) == 2
        assert "chat_template_kwargs" in client.calls[0][1]
        assert "chat_template_kwargs" not in client.calls[1][1]
        # The rest of the request is preserved.
        assert client.calls[1][1]["model"] == "m"
        assert client.calls[1][1]["max_tokens"] == throughput.WARMUP_MAX_TOKENS

    def test_retries_on_422(self, client_factory):
        client, factory = client_factory(_RecordingClient("chat_template_kwargs", status_code=422))

        asyncio.run(throughput.warmup("http://server/v1", "m", client_factory=factory))

        assert len(client.calls) == 2

    def test_retries_with_max_completion_tokens(self, client_factory):
        client, factory = client_factory(_RecordingClient("max_tokens"))
        tok_cfg = throughput.TokenizerConfig()

        asyncio.run(
            throughput.warmup("http://server/v1", "m", tok_cfg=tok_cfg, client_factory=factory)
        )

        assert len(client.calls) == 2
        assert client.calls[0][1]["max_tokens"] == throughput.WARMUP_MAX_TOKENS
        assert "max_tokens" not in client.calls[1][1]
        assert client.calls[1][1]["max_completion_tokens"] == throughput.WARMUP_MAX_TOKENS
        assert tok_cfg.output_token_field == "max_completion_tokens"

    def test_handles_hint_and_token_field_rejections(self, client_factory):
        class RejectBoth(_RecordingClient):
            async def post(self, url, *, json, headers):
                self.calls.append((url, json, headers))
                if "chat_template_kwargs" in json:
                    return _Response(400)
                if "max_tokens" in json:
                    return _Response(
                        400,
                        "Unsupported parameter: max_tokens. Use max_completion_tokens instead.",
                    )
                return _Response(200)

        client, factory = client_factory(RejectBoth(None))

        asyncio.run(throughput.warmup("http://server/v1", "m", client_factory=factory))

        assert len(client.calls) == 3
        assert "chat_template_kwargs" not in client.calls[1][1]
        assert "max_tokens" not in client.calls[2][1]
        assert client.calls[2][1]["max_completion_tokens"] == throughput.WARMUP_MAX_TOKENS

    def test_output_limit_after_fallbacks_counts_as_success(self, client_factory):
        class ExhaustedReasoningBudget(_RecordingClient):
            async def post(self, url, *, json, headers):
                self.calls.append((url, json, headers))
                if "max_tokens" in json:
                    return _Response(
                        400,
                        "Unsupported parameter: max_tokens. Use max_completion_tokens instead.",
                    )
                if "chat_template_kwargs" in json:
                    return _Response(400)
                return _Response(
                    400,
                    "Could not finish the message because max_tokens or model output limit "
                    "was reached. Please try again with higher max_tokens.",
                )

        client, factory = client_factory(ExhaustedReasoningBudget(None))

        elapsed = asyncio.run(
            throughput.warmup("http://server/v1", "reasoning-model", client_factory=factory)
        )

        assert elapsed >= 0
        assert len(client.calls) == 3

    def test_failure_still_raises_after_the_retry(self, client_factory):
        """A 400 the hints did not cause is a real failure and must surface."""
        client, factory = client_factory(_RecordingClient("model"))

        with pytest.raises(RuntimeError, match="HTTP 400"):
            asyncio.run(throughput.warmup("http://server/v1", "m", client_factory=factory))

        assert len(client.calls) == 2  # first attempt, then the stripped retry

    def test_no_second_attempt_when_nothing_can_be_stripped(self, client_factory):
        client, factory = client_factory(_RecordingClient("contents"))
        request = ("http://gemini/models/m:generateContent", {"contents": []}, {})

        with pytest.raises(RuntimeError, match="HTTP 400"):
            asyncio.run(throughput.warmup("ignored", "m", request=request, client_factory=factory))

        assert len(client.calls) == 1

    def test_success_sends_one_request(self, client_factory):
        client, factory = client_factory(_RecordingClient(None))

        asyncio.run(throughput.warmup("http://server/v1", "m", client_factory=factory))

        assert len(client.calls) == 1

    def test_other_error_statuses_are_not_retried(self, client_factory):
        client, factory = client_factory(_RecordingClient("chat_template_kwargs", status_code=500))

        with pytest.raises(RuntimeError, match="HTTP 500"):
            asyncio.run(throughput.warmup("http://server/v1", "m", client_factory=factory))

        assert len(client.calls) == 1


class TestWarmupRequestOverride:
    def test_probe_builds_warmup_with_request_configuration(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from rich.console import Console

        from tool_eval_bench.cli import probe

        observed: dict[str, Any] = {}

        async def record(*args, **kwargs):
            observed.update(kwargs)
            return 1.0

        monkeypatch.setattr(throughput, "warmup", record)

        probe.warmup_server(
            Console(),
            "http://server/v1",
            "reasoning-model",
            None,
            temperature=1.0,
            extra_params={
                "reasoning_effort": "low",
                "chat_template_kwargs": {"thinking": True},
            },
        )

        _, payload, _ = observed["request"]
        assert payload["temperature"] == 1.0
        assert payload["reasoning_effort"] == "low"
        assert payload["chat_template_kwargs"] == {"thinking": True}

    def test_supplied_request_is_used_verbatim(self, client_factory):
        client, factory = client_factory(_RecordingClient(None))
        request = (
            "http://gemini/v1beta/models/m:generateContent",
            {"contents": [{"role": "user", "parts": [{"text": "Say hello."}]}]},
            {"x-goog-api-key": "k"},
        )

        asyncio.run(
            throughput.warmup("ignored", "m", "ignored", request=request, client_factory=factory)
        )

        url, payload, headers = client.calls[0]
        assert url == request[0]
        assert payload == request[1]
        assert headers == request[2]

    def test_gemini_thinking_config_is_stripped_on_rejection(self):
        payload = {
            "contents": [{"role": "user", "parts": [{"text": "Say hello."}]}],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 4,
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }

        stripped = throughput._without_optional_hints(payload)

        assert stripped is not None
        assert "thinkingConfig" not in stripped["generationConfig"]
        assert stripped["generationConfig"]["maxOutputTokens"] == 4
        assert stripped["contents"] == payload["contents"]

    def test_nothing_to_strip_returns_none(self):
        assert throughput._without_optional_hints({"model": "m"}) is None


class TestPreflightAndWarmupCliOutcomes:
    """User-visible preflight and warmup outcomes in ``cli/probe.py``."""

    def test_preflight_and_warmup_user_outcomes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import httpx
        from rich.console import Console

        from tests.coverage_helpers import async_return
        from tool_eval_bench.cli import probe

        class Client:
            mode = "ok"

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

            async def post(self, url, **kwargs):
                if self.mode == "connect":
                    raise httpx.ConnectError("down")
                if self.mode == "error":
                    raise RuntimeError("unexpected")
                if self.mode == "blank_error":
                    raise RuntimeError()
                request = httpx.Request("POST", url)
                if self.mode == "http":
                    return httpx.Response(500, text="bad", request=request)
                return httpx.Response(200, json={"choices": []}, request=request)

        client = Client()
        monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client)
        console = Console(record=True)
        probe.preflight_model_check(console, "url", "m", "key")
        for mode, code in (("http", 3), ("connect", 2), ("error", 3)):
            client.mode = mode
            with pytest.raises(SystemExit) as exc:
                probe.preflight_model_check(console, "url", "m", None)
            assert exc.value.code == code
        client.mode = "blank_error"
        with pytest.raises(SystemExit) as exc:
            probe.preflight_model_check(console, "url", "m", None)
        assert exc.value.code == 3
        assert "RuntimeError" in console.export_text()

        monkeypatch.setattr(throughput, "warmup", async_return(20_000))
        probe.warmup_server(console, "url", "m", None)
        monkeypatch.setattr(throughput, "warmup", async_return(10))
        probe.warmup_server(console, "url", "m", None)

        async def fail(*args, **kwargs):
            raise RuntimeError()

        monkeypatch.setattr(throughput, "warmup", fail)
        probe.warmup_server(console, "url", "m", None)
        assert "Warm-up failed" in console.export_text()

        async def fail_with_response(*args, **kwargs):
            request = httpx.Request("POST", "http://server/v1/chat/completions")
            response = httpx.Response(400, text="provider detail", request=request)
            raise httpx.HTTPStatusError("bad request", request=request, response=response)

        monkeypatch.setattr(throughput, "warmup", fail_with_response)
        probe.warmup_server(console, "url", "m", None)
        assert "provider detail" in console.export_text()

    def test_preflight_forwards_request_configuration(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import httpx
        from rich.console import Console

        from tool_eval_bench.cli import probe

        observed: dict[str, object] = {}

        class Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

            async def post(self, url, **kwargs):
                observed["url"] = url
                observed["post"] = kwargs
                return httpx.Response(
                    200,
                    json={"choices": [{}]},
                    request=httpx.Request("POST", url),
                )

        def make_client(**kwargs):
            observed["client_kwargs"] = kwargs
            return Client()

        monkeypatch.setattr(httpx, "AsyncClient", make_client)
        probe.preflight_model_check(
            Console(),
            "http://server/v1",
            "m",
            None,
            timeout_seconds=17.5,
            temperature=0.4,
            extra_params={"reasoning_effort": "low", "top_p": 0.9},
        )

        assert observed["client_kwargs"] == {"timeout": 17.5}
        post = observed["post"]
        assert isinstance(post, dict)
        payload = post["json"]
        assert isinstance(payload, dict)
        assert payload["reasoning_effort"] == "low"
        assert payload["top_p"] == 0.9
        assert payload["temperature"] == 0.4
        assert payload["model"] == "m"

    def test_preflight_retries_with_max_completion_tokens(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import httpx
        from rich.console import Console

        from tool_eval_bench.cli import probe

        bodies: list[dict] = []

        class Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

            async def post(self, url, *, json, headers):
                bodies.append(json)
                if "max_tokens" in json:
                    return httpx.Response(
                        400,
                        json={
                            "error": {
                                "message": "Unsupported parameter: max_tokens. Use max_completion_tokens instead."
                            }
                        },
                        request=httpx.Request("POST", url),
                    )
                return httpx.Response(200, json={"choices": [{}]})

        monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: Client())
        probe.preflight_model_check(Console(), "http://server/v1", "m", None)

        assert len(bodies) == 2
        assert bodies[1]["max_completion_tokens"] == 1
        assert "max_tokens" not in bodies[1]

    def test_preflight_accepts_output_limit_as_model_availability(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import httpx
        from rich.console import Console

        from tool_eval_bench.cli import probe

        bodies: list[dict] = []

        class Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

            async def post(self, url, *, json, headers):
                bodies.append(json)
                request = httpx.Request("POST", url)
                if "max_tokens" in json:
                    return httpx.Response(
                        400,
                        json={
                            "error": {
                                "message": "Unsupported parameter: max_tokens. Use max_completion_tokens instead."
                            }
                        },
                        request=request,
                    )
                return httpx.Response(
                    400,
                    json={
                        "error": {
                            "message": "Could not finish the message because max_tokens or model output limit was reached. Please try again with higher max_tokens."
                        }
                    },
                    request=request,
                )

        monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: Client())
        probe.preflight_model_check(Console(), "http://server/v1", "reasoning-model", None)

        assert len(bodies) == 2
        assert bodies[1]["max_completion_tokens"] == 1
