"""Unit tests for utils/metadata.py backend probing.

Uses deterministic mocks so no live inference server is required.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_response(
    status_code: int = 200, json_data: Any | None = None, headers: dict | None = None
) -> MagicMock:
    """Build a mock httpx Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data if json_data is not None else {}
    resp.headers = headers or {}
    return resp


def _mock_async_client(responses: list[MagicMock]) -> MagicMock:
    """Build a mock AsyncClient whose get() returns responses in order."""
    client = MagicMock()
    client.get = AsyncMock(side_effect=responses)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.aclose = AsyncMock()
    return client


# ---------------------------------------------------------------------------
# _probe_models
# ---------------------------------------------------------------------------


class TestProbeModels:
    @pytest.mark.asyncio
    async def test_extracts_model_info(self) -> None:
        from tool_eval_bench.utils.metadata import _probe_models

        resp = _mock_response(
            200,
            {
                "data": [
                    {
                        "id": "qwen-7b",
                        "root": "Qwen/Qwen2.5-7B-Instruct",
                        "max_model_len": 32768,
                    }
                ]
            },
        )
        with patch(
            "tool_eval_bench.utils.metadata.httpx.AsyncClient",
            return_value=_mock_async_client([resp]),
        ):
            result = await _probe_models("http://localhost:8000", None)

        assert result["server_model_id"] == "qwen-7b"
        assert result["server_model_root"] == "Qwen/Qwen2.5-7B-Instruct"
        assert result["max_model_len"] == 32768

    @pytest.mark.asyncio
    async def test_returns_empty_on_404(self) -> None:
        from tool_eval_bench.utils.metadata import _probe_models

        resp = _mock_response(404)
        with patch(
            "tool_eval_bench.utils.metadata.httpx.AsyncClient",
            return_value=_mock_async_client([resp]),
        ):
            result = await _probe_models("http://localhost:8000", None)

        assert result == {}

    @pytest.mark.asyncio
    async def test_returns_empty_on_connection_error(self) -> None:
        from tool_eval_bench.utils.metadata import _probe_models

        client = _mock_async_client([])
        client.get = AsyncMock(side_effect=ConnectionError("refused"))
        with patch(
            "tool_eval_bench.utils.metadata.httpx.AsyncClient",
            return_value=client,
        ):
            result = await _probe_models("http://localhost:8000", None)

        assert result == {}

    @pytest.mark.asyncio
    async def test_uses_api_key(self) -> None:
        from tool_eval_bench.utils.metadata import _probe_models

        resp = _mock_response(200, {"data": [{"id": "m"}]})
        client = _mock_async_client([resp])
        with patch(
            "tool_eval_bench.utils.metadata.httpx.AsyncClient",
            return_value=client,
        ):
            await _probe_models("http://localhost:8000", "secret-key")

        call_kwargs = client.get.call_args.kwargs
        assert call_kwargs["headers"]["Authorization"] == "Bearer secret-key"


# ---------------------------------------------------------------------------
# _probe_vllm_version
# ---------------------------------------------------------------------------


class TestProbeVllmVersion:
    @pytest.mark.asyncio
    async def test_extracts_version(self) -> None:
        from tool_eval_bench.utils.metadata import _probe_vllm_version

        resp = _mock_response(200, {"version": "0.5.1"})
        with patch(
            "tool_eval_bench.utils.metadata.httpx.AsyncClient",
            return_value=_mock_async_client([resp]),
        ):
            result = await _probe_vllm_version("http://localhost:8000", None)

        assert result == {"engine_name": "vLLM", "engine_version": "0.5.1"}

    @pytest.mark.asyncio
    async def test_returns_empty_on_404(self) -> None:
        from tool_eval_bench.utils.metadata import _probe_vllm_version

        resp = _mock_response(404)
        with patch(
            "tool_eval_bench.utils.metadata.httpx.AsyncClient",
            return_value=_mock_async_client([resp]),
        ):
            result = await _probe_vllm_version("http://localhost:8000", None)

        assert result == {}


# ---------------------------------------------------------------------------
# _probe_llamacpp
# ---------------------------------------------------------------------------


class TestProbeLlamacpp:
    @pytest.mark.asyncio
    async def test_extracts_from_props(self) -> None:
        from tool_eval_bench.utils.metadata import _probe_llamacpp

        resp = _mock_response(200, {"build_info": "1234 (abc)", "total_slots": 1})
        with patch(
            "tool_eval_bench.utils.metadata.httpx.AsyncClient",
            return_value=_mock_async_client([resp]),
        ):
            result = await _probe_llamacpp("http://localhost:8080")

        assert result["engine_name"] == "llama.cpp"
        assert result["engine_version"] == "1234 (abc)"
        assert result["gpu_count"] == 1

    @pytest.mark.asyncio
    async def test_falls_back_to_health(self) -> None:
        from tool_eval_bench.utils.metadata import _probe_llamacpp

        props_resp = _mock_response(404)
        health_resp = _mock_response(200, {"build_number": 999})
        with patch(
            "tool_eval_bench.utils.metadata.httpx.AsyncClient",
            return_value=_mock_async_client([props_resp, health_resp]),
        ):
            result = await _probe_llamacpp("http://localhost:8080")

        assert result["engine_name"] == "llama.cpp"
        assert result["engine_version"] == "b999"

    @pytest.mark.asyncio
    async def test_returns_empty_when_both_fail(self) -> None:
        from tool_eval_bench.utils.metadata import _probe_llamacpp

        resp = _mock_response(500)
        with patch(
            "tool_eval_bench.utils.metadata.httpx.AsyncClient",
            return_value=_mock_async_client([resp, resp]),
        ):
            result = await _probe_llamacpp("http://localhost:8080")

        assert result == {}


# ---------------------------------------------------------------------------
# _probe_litellm
# ---------------------------------------------------------------------------


class TestProbeLitellm:
    @pytest.mark.asyncio
    async def test_extracts_from_header(self) -> None:
        from tool_eval_bench.utils.metadata import _probe_litellm

        resp = _mock_response(200, {"status": "healthy"}, {"x-litellm-version": "1.40.0"})
        with patch(
            "tool_eval_bench.utils.metadata.httpx.AsyncClient",
            return_value=_mock_async_client([resp]),
        ):
            result = await _probe_litellm("http://localhost:4000", None)

        assert result == {"engine_name": "LiteLLM", "engine_version": "1.40.0"}

    @pytest.mark.asyncio
    async def test_falls_back_to_body_version(self) -> None:
        from tool_eval_bench.utils.metadata import _probe_litellm

        resp = _mock_response(200, {"litellm_version": "1.39.0"})
        with patch(
            "tool_eval_bench.utils.metadata.httpx.AsyncClient",
            return_value=_mock_async_client([resp]),
        ):
            result = await _probe_litellm("http://localhost:4000", None)

        assert result == {"engine_name": "LiteLLM", "engine_version": "1.39.0"}

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_version(self) -> None:
        from tool_eval_bench.utils.metadata import _probe_litellm

        resp = _mock_response(200, {"status": "ok"})
        with patch(
            "tool_eval_bench.utils.metadata.httpx.AsyncClient",
            return_value=_mock_async_client([resp]),
        ):
            result = await _probe_litellm("http://localhost:4000", None)

        assert result == {}


# ---------------------------------------------------------------------------
# _guess_quantization
# ---------------------------------------------------------------------------


class TestGuessQuantization:
    def test_gguf_pattern(self) -> None:
        from tool_eval_bench.utils.metadata import _guess_quantization

        assert _guess_quantization("model-Q4_K_M.gguf") == "Q4_K_M"

    def test_autoround_int4(self) -> None:
        from tool_eval_bench.utils.metadata import _guess_quantization

        assert _guess_quantization("model-INT4-AutoRound") == "INT4-AutoRound"

    def test_autoround_without_bits(self) -> None:
        from tool_eval_bench.utils.metadata import _guess_quantization

        assert _guess_quantization("model-AutoRound") == "AutoRound"

    def test_common_quantizations(self) -> None:
        from tool_eval_bench.utils.metadata import _guess_quantization

        assert _guess_quantization("model-AWQ") == "AWQ"
        assert _guess_quantization("model-GPTQ") == "GPTQ"
        assert _guess_quantization("model-FP8") == "FP8"

    def test_none_for_unknown(self) -> None:
        from tool_eval_bench.utils.metadata import _guess_quantization

        assert _guess_quantization("unknown-model") is None

    def test_none_for_empty(self) -> None:
        from tool_eval_bench.utils.metadata import _guess_quantization

        assert _guess_quantization(None) is None
        assert _guess_quantization("") is None


# ---------------------------------------------------------------------------
# _probe_engine
# ---------------------------------------------------------------------------


class TestProbeEngine:
    @pytest.mark.asyncio
    async def test_vllm_probe(self) -> None:
        from tool_eval_bench.utils.metadata import _probe_engine

        models_resp = _mock_response(
            200,
            {
                "data": [
                    {
                        "id": "qwen-7b",
                        "root": "Qwen/Qwen2.5-7B-Instruct",
                        "max_model_len": 32768,
                    }
                ]
            },
        )
        version_resp = _mock_response(200, {"version": "0.5.1"})
        with patch(
            "tool_eval_bench.utils.metadata.httpx.AsyncClient",
            return_value=_mock_async_client([models_resp, version_resp]),
        ):
            result = await _probe_engine("http://localhost:8000", None, "vllm")

        assert result["engine_name"] == "vLLM"
        assert result["engine_version"] == "0.5.1"
        assert result["server_model_id"] == "qwen-7b"
        assert result["max_model_len"] == 32768
        assert "quantization" not in result

    @pytest.mark.asyncio
    async def test_llamacpp_probe(self) -> None:
        from tool_eval_bench.utils.metadata import _probe_engine

        models_resp = _mock_response(200, {"data": [{"id": "llama-model"}]})
        props_resp = _mock_response(200, {"build_info": "1234", "total_slots": 1})
        with patch(
            "tool_eval_bench.utils.metadata.httpx.AsyncClient",
            return_value=_mock_async_client([models_resp, props_resp]),
        ):
            result = await _probe_engine("http://localhost:8080", None, "llamacpp")

        assert result["engine_name"] == "llama.cpp"
        assert result["server_model_id"] == "llama-model"

    @pytest.mark.asyncio
    async def test_litellm_probe(self) -> None:
        from tool_eval_bench.utils.metadata import _probe_engine

        models_resp = _mock_response(200, {"data": [{"id": "gpt-4o"}]})
        health_resp = _mock_response(200, {"status": "healthy"}, {"x-litellm-version": "1.40.0"})
        with patch(
            "tool_eval_bench.utils.metadata.httpx.AsyncClient",
            return_value=_mock_async_client([models_resp, health_resp]),
        ):
            result = await _probe_engine("http://localhost:4000", None, "litellm")

        assert result["engine_name"] == "LiteLLM"
        assert result["server_model_id"] == "gpt-4o"

    @pytest.mark.asyncio
    async def test_unknown_backend_tries_all(self) -> None:
        from tool_eval_bench.utils.metadata import _probe_engine

        models_resp = _mock_response(200, {"data": [{"id": "fallback"}]})
        version_resp = _mock_response(404)
        health_resp = _mock_response(404)
        props_resp = _mock_response(200, {"build_number": 42})
        with patch(
            "tool_eval_bench.utils.metadata.httpx.AsyncClient",
            return_value=_mock_async_client([models_resp, version_resp, health_resp, props_resp]),
        ):
            result = await _probe_engine("http://localhost:9999", None, "unknown")

        assert result["engine_name"] == "llama.cpp"
        assert result["server_model_id"] == "fallback"

    @pytest.mark.asyncio
    async def test_ninfer_backend_uses_owned_by_fingerprint(self) -> None:
        from tool_eval_bench.utils.metadata import _probe_engine

        models_resp = _mock_response(200, {"data": [{"id": "qwen3.8-27b", "owned_by": "ninfer"}]})
        client = _mock_async_client([models_resp])
        with patch("tool_eval_bench.utils.metadata.httpx.AsyncClient", return_value=client):
            result = await _probe_engine("http://localhost:8000", None, "ninfer")

        assert result["engine_name"] == "NInfer"
        assert result["owned_by"] == "ninfer"
        assert client.get.await_count == 1

    @pytest.mark.asyncio
    async def test_ninfer_backend_requires_owned_by_fingerprint(self) -> None:
        from tool_eval_bench.utils.metadata import _probe_engine

        models_resp = _mock_response(200, {"data": [{"id": "model", "owned_by": "vllm"}]})
        client = _mock_async_client([models_resp])
        with patch("tool_eval_bench.utils.metadata.httpx.AsyncClient", return_value=client):
            result = await _probe_engine("http://localhost:8000", None, "ninfer")

        assert "engine_name" not in result
        assert client.get.await_count == 1

    @pytest.mark.asyncio
    async def test_quantization_inferred_from_model_name(self) -> None:
        from tool_eval_bench.utils.metadata import _probe_engine

        models_resp = _mock_response(
            200,
            {"data": [{"id": "qwen-7b-AWQ", "root": "Qwen/Qwen2.5-7B-AWQ"}]},
        )
        version_resp = _mock_response(200, {"version": "0.5.1"})
        with patch(
            "tool_eval_bench.utils.metadata.httpx.AsyncClient",
            return_value=_mock_async_client([models_resp, version_resp]),
        ):
            result = await _probe_engine("http://localhost:8000", None, "vllm")

        assert result["quantization"] == "AWQ"

    @pytest.mark.asyncio
    async def test_sglang_probe_sets_engine_name_without_extra_calls(self) -> None:
        from tool_eval_bench.utils.metadata import _probe_engine

        models_resp = _mock_response(200, {"data": [{"id": "sglang-model"}]})
        client = _mock_async_client([models_resp])
        with patch("tool_eval_bench.utils.metadata.httpx.AsyncClient", return_value=client):
            result = await _probe_engine("http://localhost:30000", None, "sglang")

        assert result["engine_name"] == "SGLang"
        assert result["server_model_id"] == "sglang-model"
        # Only the /v1/models probe should have fired — no speculative HTTP calls.
        assert client.get.await_count == 1

    @pytest.mark.asyncio
    async def test_gemini_probe_makes_no_requests(self) -> None:
        """A hosted API has no engine metadata and no /v1/models to ask."""
        from tool_eval_bench.utils.metadata import _probe_engine

        client = _mock_async_client([])
        with patch("tool_eval_bench.utils.metadata.httpx.AsyncClient", return_value=client):
            result = await _probe_engine(
                "https://generativelanguage.googleapis.com", "key", "gemini"
            )

        assert result == {"engine_name": "Google Gemini API"}
        assert client.get.await_count == 0


# ---------------------------------------------------------------------------
# detect_backend_from_metrics / probe_backend_hint
# ---------------------------------------------------------------------------


class TestDetectBackendFromMetrics:
    def test_identifies_vllm(self) -> None:
        from tool_eval_bench.utils.metadata import detect_backend_from_metrics

        text = (
            "# HELP vllm:num_requests_running Number of requests running.\n"
            "# TYPE vllm:num_requests_running gauge\n"
            "vllm:num_requests_running 3.0\n"
        )
        assert detect_backend_from_metrics(text) == ("vllm", "vLLM")

    def test_identifies_llamacpp(self) -> None:
        from tool_eval_bench.utils.metadata import detect_backend_from_metrics

        text = (
            "# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed.\n"
            "# TYPE llamacpp:prompt_tokens_total counter\n"
            "llamacpp:prompt_tokens_total 128\n"
        )
        assert detect_backend_from_metrics(text) == ("llamacpp", "llama.cpp")

    def test_identifies_sglang_legacy_prefix(self) -> None:
        from tool_eval_bench.utils.metadata import detect_backend_from_metrics

        assert detect_backend_from_metrics("sglang:num_running_reqs 1\n") == ("sglang", "SGLang")

    def test_identifies_sglang_underscore_prefix(self) -> None:
        from tool_eval_bench.utils.metadata import detect_backend_from_metrics

        assert detect_backend_from_metrics("sglang_num_running_reqs 1\n") == ("sglang", "SGLang")

    def test_returns_none_for_unrecognized_text(self) -> None:
        from tool_eval_bench.utils.metadata import detect_backend_from_metrics

        assert detect_backend_from_metrics("some_other_metric 1\n") is None
        assert detect_backend_from_metrics("") is None


class TestProbeBackendHint:
    @pytest.mark.asyncio
    async def test_detects_llamacpp_from_metrics_without_further_probes(self) -> None:
        from tool_eval_bench.utils.metadata import probe_backend_hint

        metrics_resp = _mock_response(200)
        metrics_resp.text = "llamacpp:prompt_tokens_total 5\n"
        client = _mock_async_client([metrics_resp])
        with patch("tool_eval_bench.utils.metadata.httpx.AsyncClient", return_value=client):
            result = await probe_backend_hint("http://192.168.10.239:8080")

        assert result == ("llamacpp", "llama.cpp")
        assert client.get.await_count == 1

    @pytest.mark.asyncio
    async def test_falls_back_to_llamacpp_props_probe_when_metrics_unavailable(self) -> None:
        """llama.cpp's --metrics flag is opt-in, so the /props path must work."""
        from tool_eval_bench.utils.metadata import probe_backend_hint

        metrics_resp = _mock_response(404)
        version_resp = _mock_response(404)  # llama.cpp 404s vLLM's /version
        ninfer_resp = _mock_response(404)  # /v1/models checked before generic /props
        props_resp = _mock_response(200, {"build_info": "1234", "total_slots": 1})
        with patch(
            "tool_eval_bench.utils.metadata.httpx.AsyncClient",
            return_value=_mock_async_client([metrics_resp, version_resp, ninfer_resp, props_resp]),
        ):
            result = await probe_backend_hint("http://localhost:8080")

        assert result == ("llamacpp", "llama.cpp")

    @pytest.mark.asyncio
    async def test_falls_back_to_vllm_version_probe(self) -> None:
        from tool_eval_bench.utils.metadata import probe_backend_hint

        metrics_resp = _mock_response(404)
        version_resp = _mock_response(200, {"version": "0.5.1"})
        with patch(
            "tool_eval_bench.utils.metadata.httpx.AsyncClient",
            return_value=_mock_async_client([metrics_resp, version_resp]),
        ):
            result = await probe_backend_hint("http://localhost:8000")

        assert result == ("vllm", "vLLM")

    @pytest.mark.asyncio
    async def test_vllm_wins_over_generic_llamacpp_health_fingerprint(self) -> None:
        """Regression: /version is specific, llama.cpp's /health is generic.

        A vLLM build whose /health returns a JSON body must not be reported as
        llama.cpp — the specific probe has to be consulted first.
        """
        from tool_eval_bench.utils.metadata import probe_backend_hint

        metrics_resp = _mock_response(404)
        version_resp = _mock_response(200, {"version": "0.11.0"})
        # Would match _probe_llamacpp if it were ever reached.
        health_resp = _mock_response(200, {"status": "ok"})
        with patch(
            "tool_eval_bench.utils.metadata.httpx.AsyncClient",
            return_value=_mock_async_client([metrics_resp, version_resp, health_resp]),
        ):
            result = await probe_backend_hint("http://localhost:8000")

        assert result == ("vllm", "vLLM")

    @pytest.mark.asyncio
    async def test_vllm_version_wins_before_ninfer_model_owner(self) -> None:
        from tool_eval_bench.utils.metadata import probe_backend_hint

        async def respond(url: str, **kwargs: Any) -> MagicMock:
            if url.endswith("/metrics"):
                return _mock_response(404)
            if url.endswith("/version"):
                return _mock_response(200, {"version": "0.11.0"})
            if url.endswith("/v1/models"):
                return _mock_response(200, {"data": [{"id": "m", "owned_by": "ninfer"}]})
            raise AssertionError(f"unexpected probe: {url}")

        client = _mock_async_client([])
        client.get = AsyncMock(side_effect=respond)
        with patch("tool_eval_bench.utils.metadata.httpx.AsyncClient", return_value=client):
            result = await probe_backend_hint("http://localhost:8000")

        assert result == ("vllm", "vLLM")
        assert [call.args[0] for call in client.get.await_args_list] == [
            "http://localhost:8000/metrics",
            "http://localhost:8000/version",
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("base", ["http://host:8080", "http://host:8080/v1"])
    async def test_root_level_probes_ignore_v1_suffix(self, base: str) -> None:
        """Regression: /props and /version live at the server root.

        Appending them to a ``.../v1`` base URL requests ``/v1/props``, which
        real llama.cpp and vLLM servers 404 — silently losing engine version
        metadata for anyone who passes the ``/v1`` form.
        """
        from tool_eval_bench.utils.metadata import _probe_llamacpp, _probe_vllm_version

        props_resp = _mock_response(200, {"build_info": "b10277", "total_slots": 2})
        client = _mock_async_client([props_resp])
        with patch("tool_eval_bench.utils.metadata.httpx.AsyncClient", return_value=client):
            result = await _probe_llamacpp(base)
        assert result["engine_version"] == "b10277"
        assert client.get.await_args.args[0] == "http://host:8080/props"

        version_resp = _mock_response(200, {"version": "0.11.0"})
        client = _mock_async_client([version_resp])
        with patch("tool_eval_bench.utils.metadata.httpx.AsyncClient", return_value=client):
            result = await _probe_vllm_version(base, None)
        assert result["engine_version"] == "0.11.0"
        assert client.get.await_args.args[0] == "http://host:8080/version"

    @pytest.mark.asyncio
    async def test_returns_none_when_nothing_matches(self) -> None:
        from tool_eval_bench.utils.metadata import probe_backend_hint

        responses = [_mock_response(404) for _ in range(5)]
        with patch(
            "tool_eval_bench.utils.metadata.httpx.AsyncClient",
            return_value=_mock_async_client(responses),
        ):
            result = await probe_backend_hint("http://localhost:9999")

        assert result is None

    @pytest.mark.asyncio
    async def test_detects_ninfer_from_owned_by(self) -> None:
        """NInfer answers /health with 200 {'status':'ok'} (the llama.cpp
        fingerprint); its /v1/models owned_by=='ninfer' must win first."""
        from tool_eval_bench.utils.metadata import probe_backend_hint

        metrics_resp = _mock_response(404)
        version_resp = _mock_response(404)
        models_resp = _mock_response(200, {"data": [{"id": "qwen3.8-27b", "owned_by": "ninfer"}]})
        # Would match _probe_llamacpp if reached.
        health_resp = _mock_response(200, {"status": "ok"})
        with patch(
            "tool_eval_bench.utils.metadata.httpx.AsyncClient",
            return_value=_mock_async_client([metrics_resp, version_resp, models_resp, health_resp]),
        ):
            result = await probe_backend_hint("http://localhost:8000")

        assert result == ("ninfer", "NInfer")


# ---------------------------------------------------------------------------
# collect_run_context
# ---------------------------------------------------------------------------


class TestCollectRunContext:
    @pytest.mark.asyncio
    async def test_collects_context(self) -> None:
        from tool_eval_bench.utils.metadata import collect_run_context

        models_resp = _mock_response(
            200,
            {
                "data": [
                    {
                        "id": "qwen-7b",
                        "root": "Qwen/Qwen2.5-7B-Instruct",
                        "max_model_len": 32768,
                    }
                ]
            },
        )
        version_resp = _mock_response(200, {"version": "0.5.1"})
        with patch(
            "tool_eval_bench.utils.metadata.httpx.AsyncClient",
            return_value=_mock_async_client([models_resp, version_resp]),
        ):
            ctx = await collect_run_context(
                model="qwen-7b",
                backend="vllm",
                base_url="http://localhost:8000",
                scenario_selector="all",
            )

        assert ctx.model == "qwen-7b"
        assert ctx.backend == "vllm"
        assert ctx.server_model_id == "qwen-7b"
        assert ctx.engine_name == "vLLM"
        assert ctx.tool_version is not None
        assert ctx.hostname is not None

    @pytest.mark.asyncio
    async def test_redacts_url(self) -> None:
        from tool_eval_bench.utils.metadata import collect_run_context

        resp = _mock_response(200, {"data": [{"id": "m"}]})
        with patch(
            "tool_eval_bench.utils.metadata.httpx.AsyncClient",
            return_value=_mock_async_client([resp]),
        ):
            ctx = await collect_run_context(
                model="m",
                backend="vllm",
                base_url="http://192.168.1.10:8000",
                redact_url=True,
                probe_engine=True,
            )

        assert ctx.base_url == "http://***:8000"

    @pytest.mark.asyncio
    async def test_probe_can_be_disabled(self) -> None:
        from tool_eval_bench.utils.metadata import collect_run_context

        client = _mock_async_client([])
        with patch(
            "tool_eval_bench.utils.metadata.httpx.AsyncClient",
            return_value=client,
        ):
            ctx = await collect_run_context(
                model="m",
                backend="vllm",
                base_url="http://localhost:8000",
                probe_engine=False,
            )

        assert client.get.call_count == 0
        assert ctx.server_model_id is None


# ---------------------------------------------------------------------------
# collect_run_metadata (legacy API)
# ---------------------------------------------------------------------------


class TestCollectRunMetadata:
    @pytest.mark.asyncio
    async def test_returns_metadata_dict(self) -> None:
        from tool_eval_bench.domain.models import BenchmarkConfig
        from tool_eval_bench.utils.metadata import collect_run_metadata

        resp = _mock_response(200, {"data": [{"id": "legacy"}]})
        config = BenchmarkConfig(
            model="legacy-model",
            backend="vllm",
            base_url="http://localhost:8000",
        )

        with patch(
            "tool_eval_bench.utils.metadata.httpx.AsyncClient",
            return_value=_mock_async_client([resp]),
        ):
            result = await collect_run_metadata(config)

        assert result["config"]["model"] == "legacy-model"
        assert result["backend_probe"]["server_model_id"] == "legacy"
