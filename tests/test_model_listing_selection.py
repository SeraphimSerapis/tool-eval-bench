"""Model facts must describe the model under test, not the first ``/v1/models`` entry.

Multi-model servers (llama-swap, LiteLLM, Ollama, vLLM with LoRA modules) list
entries in an order that says nothing about ``--model``. Borrowing the first
entry recorded another model's root, context length and quantization, and moved
the comparison fingerprint whenever the listing was reordered.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from tool_eval_bench.runner.context_pressure import detect_context_size
from tool_eval_bench.utils import metadata
from tool_eval_bench.utils.fingerprint import comparison_fingerprint

QWEN = {"id": "qwen3-8b", "root": "Qwen/Qwen3-8B-AWQ", "max_model_len": 32768, "owned_by": "vllm"}
GEMMA = {
    "id": "gemma-3-27b",
    "root": "google/gemma-3-27b-it-FP8",
    "max_model_len": 8192,
    "owned_by": "vllm",
}
GGUF = {"id": "/models/x-Q4_K_M.gguf", "owned_by": "llamacpp"}

_REAL_CLIENT = httpx.AsyncClient


def _serve(monkeypatch: pytest.MonkeyPatch, listing: list[dict[str, Any]]) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"object": "list", "data": listing})
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "0.11.0"})
        return httpx.Response(404)

    monkeypatch.setattr(
        metadata.httpx,
        "AsyncClient",
        lambda **kwargs: _REAL_CLIENT(transport=httpx.MockTransport(respond), **kwargs),
    )


async def _context(monkeypatch: pytest.MonkeyPatch, listing: list[dict[str, Any]], model: str):
    _serve(monkeypatch, listing)
    return await metadata.collect_run_context(
        model=model, backend="vllm", base_url="http://h:8000/v1"
    )


def _fingerprint(context: Any) -> str:
    meta = context.to_dict()
    meta["git_sha"] = "pinned"
    return comparison_fingerprint({"model": context.model}, meta)


async def test_the_entry_matching_the_model_describes_the_run(monkeypatch) -> None:
    context = await _context(monkeypatch, [QWEN, GEMMA], "gemma-3-27b")

    assert context.server_model_id == "gemma-3-27b"
    assert context.server_model_root == "google/gemma-3-27b-it-FP8"
    assert context.max_model_len == 8192
    assert context.quantization == "FP8"


async def test_listing_order_does_not_move_the_fingerprint(monkeypatch) -> None:
    second = await _context(monkeypatch, [QWEN, GEMMA], "gemma-3-27b")
    first = await _context(monkeypatch, [GEMMA, QWEN], "gemma-3-27b")

    assert _fingerprint(second) == _fingerprint(first)


async def test_several_entries_and_no_match_leave_the_facts_unknown(monkeypatch) -> None:
    context = await _context(monkeypatch, [QWEN, GEMMA], "llama-3-70b")

    assert context.server_model_id is None
    assert context.server_model_root is None
    assert context.max_model_len is None
    assert context.quantization is None


async def test_a_single_entry_is_trusted_when_its_id_differs(monkeypatch) -> None:
    # llama.cpp lists its one model under a file path, not the requested name.
    _serve(monkeypatch, [GGUF])

    probe = await metadata._probe_models("http://h:8080", None, model="whatever")

    assert probe["server_model_id"] == "/models/x-Q4_K_M.gguf"


async def test_owned_by_still_comes_from_the_first_entry(monkeypatch) -> None:
    # owned_by declares the engine behind the endpoint, so it must not depend
    # on which entry matched.
    _serve(monkeypatch, [{**QWEN, "owned_by": "tensorfold"}, GEMMA])

    probe = await metadata._probe_models("http://h:8000", None, model="gemma-3-27b")

    assert probe["owned_by"] == "tensorfold"
    assert probe["server_model_id"] == "gemma-3-27b"


def _listing_client(listing: list[dict[str, Any]]) -> AsyncMock:
    client = AsyncMock()
    resp = MagicMock()
    resp.json.return_value = {"data": listing}
    client.models = AsyncMock(return_value=resp)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


@pytest.mark.parametrize(
    ("listing", "model", "expected"),
    [
        ([QWEN, GEMMA], "gemma-3-27b", 8192),
        ([GEMMA, QWEN], "gemma-3-27b", 8192),
        ([QWEN, GEMMA], "llama-3-70b", None),
        ([{"id": "/models/x.gguf", "max_model_len": 4096}], "anything", 4096),
    ],
    ids=["second", "first", "no-match", "single-entry"],
)
async def test_context_pressure_sizes_the_fill_from_the_matching_entry(
    listing: list[dict[str, Any]], model: str, expected: int | None
) -> None:
    client = _listing_client(listing)

    size = await detect_context_size("http://h:8000", model, client_factory=lambda **kwargs: client)

    assert size == expected
