"""HTTP adapters for the low-level measurement port.

Throughput, speculative decoding, and context pressure need exact arrival
timing, so they talk to a server through ``domain.measurement`` rather than
through ``BackendAdapter``: the measurement port hands back the raw response
and the unparsed SSE lines, which is what makes inter-token gaps measurable.

Two classes, one implementation.  ``_ConfiguredMeasurementClient`` holds the
request building and takes any client that satisfies the HTTP shape, so a test
can inject a transport double.  ``HTTPMeasurementClient`` adds ownership of a
real ``httpx.AsyncClient`` and its lifetime.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from typing import Any, Protocol, cast

import httpx

from tool_eval_bench.domain.measurement import MeasurementResponse
from tool_eval_bench.utils.urls import (
    chat_completions_url,
    metrics_request_target,
    models_url,
)


class _HTTPClient(Protocol):
    """The slice of ``httpx.AsyncClient`` this module actually uses.

    Narrow on purpose: a test double has three methods to implement, not the
    whole client surface.
    """

    async def get(self, url: str, **kwargs: Any) -> httpx.Response: ...

    async def post(self, url: str, **kwargs: Any) -> httpx.Response: ...

    def stream(
        self, method: str, url: str, **kwargs: Any
    ) -> AbstractAsyncContextManager[httpx.Response]: ...


class _ConfiguredMeasurementClient:
    """Request building for the measurement port, over any HTTP client.

    Knows the endpoint conventions (where ``/tokenize`` lives relative to a
    ``/v1`` base, which header carries the key, where metrics are scraped from)
    so callers state what they want measured rather than how to address it.
    """

    def __init__(
        self,
        client: _HTTPClient,
        *,
        base_url: str,
        api_key: str | None,
        completion_url: str | None = None,
        completion_headers: Mapping[str, str] | None = None,
    ) -> None:
        """Configure request building against *client*.

        *completion_url* and *completion_headers* override the defaults derived
        from *base_url*, which is how a non-OpenAI wire format reuses this
        client without a second implementation.
        """
        self._client = client
        self._base_url = base_url
        self._api_key = api_key
        self._completion_url = completion_url or chat_completions_url(base_url)
        self._completion_headers = (
            dict(self._json_headers(api_key))
            if completion_headers is None
            else dict(completion_headers)
        )

    @staticmethod
    def _json_headers(api_key: str | None) -> dict[str, str]:
        """Return JSON headers, adding bearer auth only when a key is present."""
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    async def tokenize(self, *, model: str, text: str) -> MeasurementResponse:
        """Count tokens server-side, for prompt sizing that matches the model.

        ``/tokenize`` sits at the server root rather than under ``/v1``, so a
        base URL ending in ``/v1`` has it stripped.
        """
        root = self._base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[:-3]
        return cast(
            MeasurementResponse,
            await self._client.post(
                f"{root}/tokenize",
                json={"model": model, "prompt": text},
                headers=self._json_headers(self._api_key),
            ),
        )

    async def models(self) -> MeasurementResponse:
        """List the endpoint's models, used for readiness and model detection."""
        return cast(
            MeasurementResponse,
            await self._client.get(
                models_url(self._base_url), headers=self._json_headers(self._api_key)
            ),
        )

    async def metrics(self, *, metrics_url: str | None = None) -> MeasurementResponse:
        """Scrape Prometheus metrics, for KV capacity and acceptance rates.

        Kept to a short timeout: metrics are supporting detail, and a server
        that does not export them should not delay the run.
        """
        url, headers = metrics_request_target(self._base_url, metrics_url, self._api_key)
        return cast(MeasurementResponse, await self._client.get(url, headers=headers, timeout=5.0))

    async def completion(self, payload: dict[str, Any]) -> MeasurementResponse:
        """Send one non-streaming completion and return the raw response."""
        return cast(
            MeasurementResponse,
            await self._client.post(
                self._completion_url,
                json=payload,
                headers=self._completion_headers,
            ),
        )

    def stream_completion(
        self, payload: dict[str, Any]
    ) -> AbstractAsyncContextManager[MeasurementResponse]:
        """Open a streaming completion, exposing the SSE lines as they arrive.

        Returns the context manager rather than awaiting it, so the caller
        controls when the stream opens and can timestamp each line itself.
        """
        return cast(
            AbstractAsyncContextManager[MeasurementResponse],
            self._client.stream(
                "POST", self._completion_url, json=payload, headers=self._completion_headers
            ),
        )


class HTTPMeasurementClient(_ConfiguredMeasurementClient):
    """The production measurement client, owning its HTTP connection pool.

    Used as an async context manager so the pool is closed deterministically.
    Runners take it as a ``client_factory`` argument, which is the seam a test
    replaces with a double.
    """

    #: Marks the class as satisfying the measurement port, for runners that
    #: accept either a factory or a pre-built client.
    _measurement_port = True

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        timeout: float,
        max_connections: int | None = None,
        max_keepalive_connections: int | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        completion_url: str | None = None,
        completion_headers: Mapping[str, str] | None = None,
    ) -> None:
        """Build a client with its own pool.

        Connection limits apply only when both are given; a partial pair would
        silently mix one explicit bound with httpx's default for the other.
        """
        if max_connections is not None and max_keepalive_connections is not None:
            limits = httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_keepalive_connections,
            )
            client = httpx.AsyncClient(
                timeout=httpx.Timeout(timeout), limits=limits, transport=transport
            )
        else:
            client = httpx.AsyncClient(timeout=httpx.Timeout(timeout), transport=transport)
        super().__init__(
            cast(_HTTPClient, client),
            base_url=base_url,
            api_key=api_key,
            completion_url=completion_url,
            completion_headers=completion_headers,
        )
        self._owned_client = client

    async def __aenter__(self) -> HTTPMeasurementClient:
        """Open the underlying connection pool."""
        await self._owned_client.__aenter__()
        return self

    async def __aexit__(self, *args: object) -> None:
        """Close the pool, whether or not the measurement run succeeded."""
        await self._owned_client.aclose()
