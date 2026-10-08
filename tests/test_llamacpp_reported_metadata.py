"""llama.cpp's server-reported context window and quantization, and nobody else's."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from tool_eval_bench.utils import metadata

# Trimmed from llama-server b11478-18b5f8b18 serving gemma-4-E4B-it-Q4_0.gguf
# with two slots. chat_template, most sampling params, and the Ollama-style
# "models" list are dropped; every field the probes read is as served.
LIVE_PROPS: dict[str, Any] = {
    "default_generation_settings": {
        "params": {"seed": 4294967295, "temperature": 1.0, "top_k": 64, "n_predict": -1},
        "n_ctx": 81920,
    },
    "total_slots": 2,
    "model_alias": "gemma4",
    "model_ftype": "Q4_0",
    "model_path": "/models/gemma-4-E4B-it-Q4_0.gguf",
    "modalities": {"vision": True, "video": True, "audio": True},
    "endpoint_slots": True,
    "endpoint_props": False,
    "endpoint_metrics": True,
    "build_info": "b11478-18b5f8b18",
    "is_sleeping": False,
}
LIVE_MODELS: dict[str, Any] = {
    "object": "list",
    "data": [
        {
            "id": "gemma4",
            "aliases": ["gemma4"],
            "object": "model",
            "owned_by": "llamacpp",
            "meta": {
                "vocab_type": 2,
                "n_vocab": 262144,
                "n_ctx": 81920,
                "n_ctx_train": 131072,
                "n_embd": 2560,
                "n_params": 7463013674,
                "size": 4574983336,
                "ftype": "Q4_0",
            },
        }
    ],
}
LIVE_HEADERS = {"server": "llama.cpp"}

# A llama-server-shaped /props and /v1/models meta that other servers might
# carry. None of it may reach their metadata.
LLAMA_LIKE_META = {"ftype": "Q4_0", "n_ctx": 81920, "n_ctx_train": 131072}
LLAMA_LIKE_PROPS = {
    "default_generation_settings": {"n_ctx": 4096},
    "total_slots": 3,
    "model_ftype": "Q8_0",
}


def _serve(monkeypatch, routes: dict[str, Any], headers: dict[str, str] | None = None) -> None:
    real_client = httpx.AsyncClient

    def respond(request: httpx.Request) -> httpx.Response:
        body = routes.get(request.url.path)
        if body is None:
            return httpx.Response(404)
        if isinstance(body, str):
            return httpx.Response(200, text=body)
        return httpx.Response(200, json=body, headers=headers or {})

    monkeypatch.setattr(
        metadata.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(respond), **kwargs),
    )


def _llamacpp(props: dict[str, Any], model_id: str = "gemma4") -> dict[str, Any]:
    """A llama-server that names itself nowhere, so only /props identifies it."""
    return {"/v1/models": {"data": [{"id": model_id}]}, "/props": props}


@pytest.mark.parametrize("backend", ["unknown", "llamacpp"])
async def test_live_llamacpp_records_context_window_and_quantization(monkeypatch, backend):
    _serve(
        monkeypatch,
        {"/v1/models": LIVE_MODELS, "/props": LIVE_PROPS, "/health": {"status": "ok"}},
        LIVE_HEADERS,
    )
    context = await metadata.collect_run_context(
        model="gemma4", backend=backend, base_url="http://test/v1"
    )
    assert context.engine_name == "llama.cpp"
    assert context.engine_version == "b11478-18b5f8b18"
    assert context.max_model_len == 81920
    assert context.quantization == "Q4_0"
    assert context.slot_count == 2


# llama_ftype_name() output and its LLAMA_FTYPE_* enum, minus the MOSTLY_ prefix.
UPSTREAM_FTYPE_ENUMS = {
    "all F32": "ALL_F32",
    "F16": "F16",
    "BF16": "BF16",
    "Q1_0": "Q1_0",
    "Q2_0": "Q2_0",
    "Q4_0": "Q4_0",
    "Q4_1": "Q4_1",
    "Q5_0": "Q5_0",
    "Q5_1": "Q5_1",
    "Q8_0": "Q8_0",
    "MXFP4 MoE": "MXFP4_MOE",
    "NVFP4": "NVFP4",
    "Q2_K - Medium": "Q2_K",
    "Q2_K - Small": "Q2_K_S",
    "Q3_K - Small": "Q3_K_S",
    "Q3_K - Medium": "Q3_K_M",
    "Q3_K - Large": "Q3_K_L",
    "Q4_K - Small": "Q4_K_S",
    "Q4_K - Medium": "Q4_K_M",
    "Q5_K - Small": "Q5_K_S",
    "Q5_K - Medium": "Q5_K_M",
    "Q6_K": "Q6_K",
    "TQ1_0 - 1.69 bpw ternary": "TQ1_0",
    "TQ2_0 - 2.06 bpw ternary": "TQ2_0",
    "IQ2_XXS - 2.0625 bpw": "IQ2_XXS",
    "IQ2_XS - 2.3125 bpw": "IQ2_XS",
    "IQ2_S - 2.5 bpw": "IQ2_S",
    "IQ2_M - 2.7 bpw": "IQ2_M",
    "IQ3_XS - 3.3 bpw": "IQ3_XS",
    "IQ3_XXS - 3.0625 bpw": "IQ3_XXS",
    "IQ1_S - 1.5625 bpw": "IQ1_S",
    "IQ1_M - 1.75 bpw": "IQ1_M",
    "IQ4_NL - 4.5 bpw": "IQ4_NL",
    "IQ4_XS - 4.25 bpw": "IQ4_XS",
    "IQ3_S - 3.4375 bpw": "IQ3_S",
    "IQ3_S mix - 3.66 bpw": "IQ3_M",
}
# GGUF filenames say F16 and F32, which the name heuristic does not label.
NAME_PATH_DIFFERS = {"F16": "FP16", "all F32": "FP32"}


def test_every_upstream_ftype_name_is_mapped():
    assert set(metadata._LLAMACPP_FTYPES) == set(UPSTREAM_FTYPE_ENUMS)


@pytest.mark.parametrize(("ftype", "enum"), UPSTREAM_FTYPE_ENUMS.items())
def test_ftype_label_matches_the_name_heuristic(ftype, enum):
    expected = NAME_PATH_DIFFERS.get(ftype) or metadata._guess_quantization(f"model-{enum}")
    assert expected
    assert metadata._llamacpp_quantization(ftype) == expected


@pytest.mark.parametrize(
    "ftype",
    [
        "?",
        "",
        "unknown, may not work",
        "(guessed) Q4_K - Medium",
        "Q4_K_M",
        "q4_0",
        " Q4_0",
        "Q4_0; drop table runs",
        None,
        4,
        True,
        ["Q4_0"],
        {"Q4_0": 1},
    ],
)
def test_unknown_llamacpp_ftypes_are_not_labels(ftype):
    assert metadata._llamacpp_quantization(ftype) is None


@pytest.mark.parametrize(
    ("ftype", "model_id", "expected"),
    [
        # A name that identifies a type wins: the file type cannot tell
        # UD-Q4_K_XL from Q4_K_M, and a conflict keeps main's label.
        ("Q4_K - Medium", "Qwen3-0.6B-UD-Q4_K_XL.gguf", "Q4_K_XL"),
        ("Q4_K - Medium", "model-Q8_0", "Q8_0"),
        # A name that identifies none takes the server's file type.
        ("Q4_0", "gemma4", "Q4_0"),
        ("F16", "Qwen3-0.6B-F16.gguf", "FP16"),
        ("Q4_K - Medium", "model-GGUF", "Q4_K_M"),
        # An unrecognized file type leaves the name's answer.
        ("garbage", "gemma4", None),
        ("(guessed) Q4_K - Medium", "model.gguf", "GGUF"),
        (["Q4_0"], "gemma4", None),
        (None, "model-GGUF", "GGUF"),
    ],
)
async def test_name_quantization_wins_and_reported_type_fills_in(
    monkeypatch, ftype, model_id, expected
):
    _serve(monkeypatch, _llamacpp({**LIVE_PROPS, "model_ftype": ftype}, model_id))
    for backend in ("unknown", "llamacpp"):
        info = await metadata._probe_engine("http://test", None, backend)
        assert info["engine_name"] == "llama.cpp"
        assert info.get("quantization") == expected


@pytest.mark.parametrize(
    "settings",
    [
        {"n_ctx": 0},
        {"n_ctx": -1},
        {"n_ctx": True},
        {"n_ctx": "81920"},
        {"n_ctx": 81920.0},
        {"n_ctx": None},
        {},
        "n_ctx=81920",
    ],
)
async def test_invalid_context_window_is_not_recorded(monkeypatch, settings):
    _serve(monkeypatch, _llamacpp({**LIVE_PROPS, "default_generation_settings": settings}))
    info = await metadata._probe_engine("http://test", None, "llamacpp")
    assert info["engine_name"] == "llama.cpp"
    assert info["slot_count"] == 2
    assert "max_model_len" not in info


async def test_smallest_context_window_is_recorded(monkeypatch):
    props = {**LIVE_PROPS, "default_generation_settings": {"n_ctx": 1}}
    _serve(monkeypatch, _llamacpp(props))
    info = await metadata._probe_engine("http://test", None, "llamacpp")
    assert info["max_model_len"] == 1


@pytest.mark.parametrize(("slots", "expected"), [(1, 1), (0, None), ("2", None), (True, None)])
async def test_slot_count_must_be_a_positive_int(monkeypatch, slots, expected):
    _serve(monkeypatch, _llamacpp({**LIVE_PROPS, "total_slots": slots}))
    info = await metadata._probe_engine("http://test", None, "llamacpp")
    assert info["engine_name"] == "llama.cpp"
    assert info.get("slot_count") == expected


async def test_router_mode_props_record_nothing_new(monkeypatch):
    # llama-server in router mode answers /props without ?model= with a dummy
    # (tools/server/server-models.cpp): n_ctx 0 and no slot count or file type.
    router_props = {
        "role": "router",
        "max_instances": 4,
        "models_autoload": True,
        "model_alias": "llama-server",
        "model_path": "none",
        "default_generation_settings": {"params": {}, "n_ctx": 0},
        "ui_settings": {},
        "build_info": "b11478-18b5f8b18",
        "cors_proxy_enabled": False,
    }
    _serve(
        monkeypatch,
        {
            "/v1/models": {"data": [{"id": "gemma4", "owned_by": "llamacpp"}]},
            "/props": router_props,
        },
    )
    info = await metadata._probe_engine("http://test", None, "unknown")
    assert info == {
        "engine_name": "llama.cpp",
        "engine_version": "b11478-18b5f8b18",
        "owned_by": "llamacpp",
        "server_model_id": "gemma4",
        "server_model_root": None,
    }


async def test_models_meta_is_not_a_fallback_for_props(monkeypatch):
    # /v1/models meta carries the same server values. Builds b9860 to b9926
    # have meta.ftype without /props model_ftype and get no reported type.
    props = {k: v for k, v in LIVE_PROPS.items() if k != "model_ftype"}
    props["default_generation_settings"] = {"params": {}}
    _serve(monkeypatch, {"/v1/models": LIVE_MODELS, "/props": props})
    info = await metadata._probe_engine("http://test", None, "llamacpp")
    assert info["engine_name"] == "llama.cpp"
    assert info["slot_count"] == 2
    assert "quantization" not in info
    assert "max_model_len" not in info


async def test_sleeping_server_still_reports_its_model(monkeypatch):
    # /props answers from cached metadata while the model is unloaded.
    _serve(monkeypatch, _llamacpp({**LIVE_PROPS, "is_sleeping": True}))
    info = await metadata._probe_engine("http://test", None, "unknown")
    assert info["max_model_len"] == 81920
    assert info["quantization"] == "Q4_0"


async def test_koboldcpp_shaped_props_gain_a_context_window(monkeypatch):
    # Current behaviour: a llama-server-shaped /props with no declared identity
    # is read as llama.cpp, so it now records n_ctx alongside the slot count.
    _serve(
        monkeypatch,
        {
            "/v1/models": {"data": [{"id": "koboldcpp/Qwen3-8B-Q4_K_M", "owned_by": "koboldcpp"}]},
            "/props": {"default_generation_settings": {"n_ctx": 8192}, "total_slots": 1},
        },
    )
    info = await metadata._probe_engine("http://test", None, "unknown")
    assert info == {
        "engine_name": "llama.cpp",
        "max_model_len": 8192,
        "owned_by": "koboldcpp",
        "quantization": "Q4_K_M",
        "server_model_id": "koboldcpp/Qwen3-8B-Q4_K_M",
        "server_model_root": None,
        "slot_count": 1,
    }


async def test_explicit_llamacpp_label_on_strata_records_none_of_its_props(monkeypatch):
    _serve(
        monkeypatch,
        {
            "/v1/models": {"data": [{"id": "m", "meta": LLAMA_LIKE_META}]},
            "/props": {**LLAMA_LIKE_PROPS, "build_info": "Strata 0.1.40.3"},
            "/health": {"status": "ok", "service": "strata"},
        },
    )
    info = await metadata._probe_engine("http://test", None, "llamacpp")
    assert info == {"server_model_id": "m", "server_model_root": None, "owned_by": None}


# Expected output is what origin/main a6362b2 produced for each server. A
# llama-like /props or /v1/models meta must not add or change a single key.
_VLLM_MODELS = {
    "data": [
        {
            "id": "Qwen3-8B-AWQ",
            "root": "Qwen/Qwen3-8B-AWQ",
            "max_model_len": 32768,
            "owned_by": "vllm",
            "meta": LLAMA_LIKE_META,
        }
    ]
}
_VLLM_INFO = {
    "engine_name": "vLLM",
    "engine_version": "0.11.0",
    "max_model_len": 32768,
    "owned_by": "vllm",
    "quantization": "AWQ",
    "server_model_id": "Qwen3-8B-AWQ",
    "server_model_root": "Qwen/Qwen3-8B-AWQ",
}
_STRATA_INFO = {
    "engine_name": "Strata",
    "engine_version": "0.1.40.3",
    "max_model_len": 4096,
    "owned_by": None,
    "quantization": "Q5_K_M",
    "server_model_id": "gemma-Q5_K_M",
    "server_model_root": None,
    "slot_count": 3,
}
_TABBY_INFO = {
    "engine_name": "TabbyAPI",
    "max_model_len": 4096,
    "owned_by": "tabbyAPI",
    "quantization": "EXL3",
    "server_model_id": "Qwen3-8B-exl3-4.0bpw",
    "server_model_root": None,
    "slot_count": 3,
}
_SGLANG_INFO = {
    "engine_name": "SGLang",
    "owned_by": "sglang",
    "quantization": "FP8",
    "server_model_id": "Qwen3-8B-FP8",
    "server_model_root": None,
}


@pytest.mark.parametrize(
    ("routes", "backends", "expected"),
    [
        pytest.param(
            {
                "/v1/models": _VLLM_MODELS,
                "/version": {"version": "0.11.0"},
                "/props": LLAMA_LIKE_PROPS,
            },
            ("vllm", "unknown"),
            _VLLM_INFO,
            id="vllm-with-llama-like-meta",
        ),
        pytest.param(
            {
                "/v1/models": {"data": [{"id": "m", "root": "org/m-FP8", "max_model_len": 8192}]},
                "/version": {"version": "0.11.0"},
            },
            ("vllm", "unknown"),
            {
                "engine_name": "vLLM",
                "engine_version": "0.11.0",
                "max_model_len": 8192,
                "owned_by": None,
                "quantization": "FP8",
                "server_model_id": "m",
                "server_model_root": "org/m-FP8",
            },
            id="vllm",
        ),
        pytest.param(
            {
                "/v1/models": {
                    "data": [{"id": "Qwen3-8B-FP8", "owned_by": "sglang", "meta": LLAMA_LIKE_META}]
                },
                "/props": LLAMA_LIKE_PROPS,
            },
            ("sglang", "unknown"),
            _SGLANG_INFO,
            id="sglang",
        ),
        pytest.param(
            {
                "/v1/models": {
                    "data": [
                        {
                            "id": "Qwen3-8B-exl3-4.0bpw",
                            "owned_by": "tabbyAPI",
                            "meta": LLAMA_LIKE_META,
                        }
                    ]
                },
                "/props": {**LLAMA_LIKE_PROPS, "build_info": "b1-abc"},
                "/.well-known/serviceinfo": {"software": {"name": "TabbyAPI"}},
            },
            ("tabbyapi", "unknown"),
            _TABBY_INFO,
            id="tabbyapi",
        ),
        pytest.param(
            {
                "/v1/models": {"data": [{"id": "gemma-Q5_K_M", "meta": LLAMA_LIKE_META}]},
                "/props": {**LLAMA_LIKE_PROPS, "build_info": "Strata 0.1.40.3"},
                "/health": {"status": "ok", "service": "strata", "max_context": 2048},
            },
            ("strata", "unknown"),
            _STRATA_INFO,
            id="strata",
        ),
        pytest.param(
            {
                "/v1/models": {"data": [{"id": "m-Q6_K", "meta": LLAMA_LIKE_META}]},
                "/props": LLAMA_LIKE_PROPS,
            },
            ("halogen",),
            {
                "engine_name": "Halogen Flash",
                "owned_by": None,
                "quantization": "Q6_K",
                "server_model_id": "m-Q6_K",
                "server_model_root": None,
            },
            id="halogen",
        ),
    ],
)
async def test_other_backends_ignore_llamacpp_metadata(monkeypatch, routes, backends, expected):
    _serve(monkeypatch, routes)
    for backend in backends:
        assert await metadata._probe_engine("http://test", None, backend) == expected
