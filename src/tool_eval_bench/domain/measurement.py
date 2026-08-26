"""Ports for low-level benchmark measurement traffic.

The interface is intentionally semantic rather than HTTP-shaped. Runners ask
for a model list, exact token count, metrics, or a completion. The adapter owns
endpoint routing, authentication, and provider transport details. Streaming
still yields raw response lines so timing remains observable at the runner.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import AbstractAsyncContextManager
from typing import Any, Protocol


class MeasurementResponse(Protocol):
    """Raw response facts preserved for provider-specific measurements."""

    @property
    def status_code(self) -> int:
        """HTTP status of the response."""
        ...

    @property
    def text(self) -> str:
        """Full body as text. Reads the response, so not for streaming."""
        ...

    @property
    def headers(self) -> Mapping[str, str]:
        """Response headers. Rate-limit and timing headers are read from here."""
        ...

    def json(self) -> dict[str, Any]:
        """Parse the body as JSON."""
        ...

    def raise_for_status(self) -> Any:
        """Raise if the status is an error, so callers can classify failures."""
        ...

    async def aread(self) -> bytes:
        """Read the whole body. Used when a streamed response has to be replayed."""
        ...

    def aiter_lines(self) -> AsyncIterator[str]:
        """Yield the body line by line.

        Kept raw on purpose: throughput and speculative-decoding measurements
        time individual SSE lines as they arrive, so nothing may buffer them.
        """
        ...


class MeasurementClient(Protocol):
    """Measurement operations with raw completion streaming preserved."""

    async def tokenize(self, *, model: str, text: str) -> MeasurementResponse:
        """Count tokens with the server's own tokenizer.

        Context-pressure calibration needs the server's count, not a local
        estimate, or the fill overshoots or undershoots the window.
        """
        ...

    async def models(self) -> MeasurementResponse:
        """List the models the endpoint serves, for discovery and detection."""
        ...

    async def metrics(self, *, metrics_url: str | None = None) -> MeasurementResponse:
        """Scrape the backend's Prometheus metrics.

        ``metrics_url`` overrides the derived location for deployments that put
        metrics behind a different host or proxy from the API.
        """
        ...

    async def completion(self, payload: dict[str, Any]) -> MeasurementResponse:
        """Send one non-streaming completion and wait for the whole response."""
        ...

    def stream_completion(
        self, payload: dict[str, Any]
    ) -> AbstractAsyncContextManager[MeasurementResponse]:
        """Open a streaming completion.

        A context manager rather than a coroutine so the connection stays open
        while the caller times arriving lines, and closes deterministically.
        """
        ...


class MeasurementClientFactory(Protocol):
    """Composition seam that opens a configured measurement client."""

    def __call__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        timeout: float,
        max_connections: int | None = None,
        max_keepalive_connections: int | None = None,
        completion_url: str | None = None,
        completion_headers: Mapping[str, str] | None = None,
    ) -> AbstractAsyncContextManager[MeasurementClient]:
        """Open a configured measurement client.

        Runners receive a factory rather than a client so the domain never owns
        credentials or connection limits; the adapter layer supplies both.
        """
        ...
