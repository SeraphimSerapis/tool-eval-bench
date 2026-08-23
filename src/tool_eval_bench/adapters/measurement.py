"""HTTP adapters for the low-level measurement port."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from typing import Any, Protocol, cast

import httpx

from tool_eval_bench.domain.measurement import MeasurementClient, MeasurementResponse
from tool_eval_bench.utils.urls import (
    chat_completions_url,
    metrics_request_target,
    models_url,
)


class _HTTPClient(Protocol):
    async def get(self, url: str, **kwargs: Any) -> httpx.Response: ...

    async def post(self, url: str, **kwargs: Any) -> httpx.Response: ...

    def stream(
        self, method: str, url: str, **kwargs: Any
    ) -> AbstractAsyncContextManager[httpx.Response]: ...


class _ConfiguredMeasurementClient:
    """Deep HTTP implementation shared by owned and test-injected clients."""

    def __init__(
        self,
        client: _HTTPClient,
        *,
        base_url: str,
        api_key: str | None,
        completion_url: str | None = None,
        completion_headers: Mapping[str, str] | None = None,
    ) -> None:
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
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    async def tokenize(self, *, model: str, text: str) -> MeasurementResponse:
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
        return cast(
            MeasurementResponse,
            await self._client.get(
                models_url(self._base_url), headers=self._json_headers(self._api_key)
            ),
        )

    async def metrics(self, *, metrics_url: str | None = None) -> MeasurementResponse:
        url, headers = metrics_request_target(self._base_url, metrics_url, self._api_key)
        return cast(MeasurementResponse, await self._client.get(url, headers=headers, timeout=5.0))

    async def completion(self, payload: dict[str, Any]) -> MeasurementResponse:
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
        return cast(
            AbstractAsyncContextManager[MeasurementResponse],
            self._client.stream(
                "POST", self._completion_url, json=payload, headers=self._completion_headers
            ),
        )


class HTTPMeasurementClient(_ConfiguredMeasurementClient):
    """Lifecycle-owning production adapter for benchmark measurement traffic."""

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
        await self._owned_client.__aenter__()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self._owned_client.aclose()


def bind_measurement_client(
    client: MeasurementClient | _HTTPClient,
    *,
    base_url: str,
    api_key: str | None,
    completion_url: str | None = None,
    completion_headers: Mapping[str, str] | None = None,
) -> MeasurementClient:
    """Bind legacy injected HTTP clients without exposing HTTP to runners.

    Production code always supplies ``HTTPMeasurementClient``. This adapter
    keeps existing test fakes and external callers on the semantic seam while
    ownership of URLs and credentials remains here.
    """
    if getattr(client, "_measurement_port", False) is True:
        return cast(MeasurementClient, client)
    return _ConfiguredMeasurementClient(
        cast(_HTTPClient, client),
        base_url=base_url,
        api_key=api_key,
        completion_url=completion_url,
        completion_headers=completion_headers,
    )
