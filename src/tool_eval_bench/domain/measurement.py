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
    def status_code(self) -> int: ...

    @property
    def text(self) -> str: ...

    @property
    def headers(self) -> Mapping[str, str]: ...

    def json(self) -> dict[str, Any]: ...

    def raise_for_status(self) -> Any: ...

    async def aread(self) -> bytes: ...

    def aiter_lines(self) -> AsyncIterator[str]: ...


class MeasurementClient(Protocol):
    """Measurement operations with raw completion streaming preserved."""

    async def tokenize(self, *, model: str, text: str) -> MeasurementResponse: ...

    async def models(self) -> MeasurementResponse: ...

    async def metrics(self, *, metrics_url: str | None = None) -> MeasurementResponse: ...

    async def completion(self, payload: dict[str, Any]) -> MeasurementResponse: ...

    def stream_completion(
        self, payload: dict[str, Any]
    ) -> AbstractAsyncContextManager[MeasurementResponse]: ...


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
    ) -> AbstractAsyncContextManager[MeasurementClient]: ...
