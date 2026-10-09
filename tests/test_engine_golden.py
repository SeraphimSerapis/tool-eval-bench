"""Golden outcomes for everything the code decides from an engine's identity.

Each module used to carry its own chain of "what makes an engine an engine":
backend identification, spec detection, spec-live labelling, the reported
context window, and the probe sequence. These tables pin what each of them
answers for every recognised engine, written against the code before the
engine registry existed, so moving those facts into one place cannot change
an outcome without failing here. Expected values are literals on purpose: a
typo in the registry has to disagree with something.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from tool_eval_bench.domain.models import RunContext
from tool_eval_bench.runner.context_pressure import reported_context_window
from tool_eval_bench.runner.spec_detection import detect_spec_decoding
from tool_eval_bench.runner.spec_live import _parse_snapshot, compute_delta
from tool_eval_bench.utils import metadata

ACCEPT = "text/plain; version=0.0.4, */*;q=0.1"

# ---------------------------------------------------------------------------
# Metrics fixtures
# ---------------------------------------------------------------------------


def _counters(
    prefix: str, drafted: int, accepted: int = 0, steps: int = 0, labels: str = ""
) -> str:
    lab = f"{{{labels}}}" if labels else ""
    return (
        f"{prefix}spec_decode_num_draft_tokens_total{lab} {drafted}\n"
        f"{prefix}spec_decode_num_accepted_tokens_total{lab} {accepted}\n"
        f"{prefix}spec_decode_num_drafts_total{lab} {steps}\n"
    )


STRATA_NS = 'strata:live_state{state="generating"} 1\n'
LLAMACPP_LOAD = "llamacpp:prompt_tokens_total 212548\nllamacpp:tokens_predicted_total 1157\n"
HALOGEN_NS = "halogen:requests_total 3\n"
SGLANG_SPEC = (
    'sglang:spec_accept_length{tp_rank="0"} 2.5\nsglang:spec_accept_rate{tp_rank="0"} 0.6\n'
)
SGLANG_ENGINE = 'sglang:gen_throughput{tp_rank="0"} 12.0\n'


# ---------------------------------------------------------------------------
# Backend identification from /metrics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param(HALOGEN_NS, ("halogen", "Halogen Flash"), id="halogen"),
        pytest.param(
            "tensorfold:requests_total 1\n", ("tensorfold", "TensorFold"), id="tensorfold"
        ),
        pytest.param(STRATA_NS, ("strata", "Strata"), id="strata"),
        pytest.param("vllm:num_requests_running 1\n", ("vllm", "vLLM"), id="vllm"),
        pytest.param("sglang:num_running_reqs 1\n", ("sglang", "SGLang"), id="sglang-colon"),
        pytest.param("sglang_num_running_reqs 1\n", ("sglang", "SGLang"), id="sglang-underscore"),
        pytest.param(LLAMACPP_LOAD, ("llamacpp", "llama.cpp"), id="llamacpp"),
        # Compatibility aliases yield to the native namespace in either order.
        pytest.param(
            HALOGEN_NS + LLAMACPP_LOAD, ("halogen", "Halogen Flash"), id="halogen+llamacpp"
        ),
        pytest.param(
            LLAMACPP_LOAD + HALOGEN_NS, ("halogen", "Halogen Flash"), id="llamacpp+halogen"
        ),
        pytest.param(
            _counters("vllm:", 1) + STRATA_NS, ("strata", "Strata"), id="vllm-names+strata"
        ),
        pytest.param(
            "vllm:num_requests_running 1\ntensorfold:requests_total 1\n",
            ("tensorfold", "TensorFold"),
            id="vllm+tensorfold",
        ),
        pytest.param(
            LLAMACPP_LOAD + "sglang:num_running_reqs 1\n",
            ("sglang", "SGLang"),
            id="llamacpp+sglang",
        ),
        pytest.param(
            LLAMACPP_LOAD + "vllm:num_requests_running 1\n", ("vllm", "vLLM"), id="llamacpp+vllm"
        ),
        pytest.param(LLAMACPP_LOAD + STRATA_NS, ("strata", "Strata"), id="llamacpp+strata"),
        pytest.param(STRATA_NS + HALOGEN_NS, ("halogen", "Halogen Flash"), id="strata+halogen"),
        # A namespace inside a label value names no engine.
        pytest.param(
            'vllm:num_requests_running{model_name="acme/strata:7b"} 1\n',
            ("vllm", "vLLM"),
            id="strata-in-label",
        ),
        pytest.param(
            'vllm:num_requests_running{model_name="x/halogen:1"} 1\n',
            ("vllm", "vLLM"),
            id="halogen-in-label",
        ),
        pytest.param('other_metric{model_name="llamacpp:gemma"} 1\n', None, id="llamacpp-in-label"),
        pytest.param("# HELP strata:live_state x\n# TYPE vllm:x gauge\n", None, id="comments-only"),
        pytest.param("", None, id="empty"),
    ],
)
def test_metrics_namespace_identity(text: str, expected: tuple[str, str] | None) -> None:
    assert metadata.detect_backend_from_metrics(text) == expected


@pytest.mark.parametrize(
    ("source", "value", "expected"),
    [
        ("owned_by", "llamacpp", ("llamacpp", "llama.cpp")),
        ("owned_by", "ninfer", ("ninfer", "NInfer")),
        ("owned_by", "sglang", ("sglang", "SGLang")),
        ("owned_by", "tabbyapi", ("tabbyapi", "TabbyAPI")),
        ("owned_by", "TensorFold 1.2", ("tensorfold", "TensorFold")),
        ("owned_by", "vllm", None),
        ("owned_by", "strata", None),
        ("server", "llama.cpp", ("llamacpp", "llama.cpp")),
        ("server", "vLLM/0.9.1", ("vllm", "vLLM")),
        ("server", "sglang", ("sglang", "SGLang")),
        ("server", "LiteLLM", ("litellm", "LiteLLM")),
        ("server", "uvicorn", None),
        ("server", "strata", None),
        ("service", "Strata 0.1.40", ("strata", "Strata")),
        ("service", "llama.cpp", None),
        ("software", "tabbyAPI", ("tabbyapi", "TabbyAPI")),
        ("software", "tabby", None),
        ("build_info", "strata 0.1.40", ("strata", "Strata")),
        ("build_info", "b6123-abc1234", None),
        ("build_info", "", None),
        ("build_info", None, None),
    ],
)
def test_declared_identity_tokens(
    source: Any, value: Any, expected: tuple[str, str] | None
) -> None:
    assert metadata._declared(source, value) == expected


# ---------------------------------------------------------------------------
# Probe sequences and engine names
# ---------------------------------------------------------------------------


class _RecordingClient:
    """Answers GETs from a path table and records path plus the headers sent."""

    def __init__(self, routes: dict[str, httpx.Response] | None = None) -> None:
        self.routes = routes or {}
        self.calls: list[tuple[str, tuple[tuple[str, str], ...]]] = []

    async def get(self, url: str, headers: dict[str, str] | None = None) -> httpx.Response:
        path = url.removeprefix("http://engine.test:8000")
        self.calls.append((path, tuple(sorted((headers or {}).items()))))
        canned = self.routes.get(path)
        if canned is None:
            return httpx.Response(404, request=httpx.Request("GET", url))
        return httpx.Response(
            canned.status_code,
            headers=canned.headers,
            content=canned.content,
            request=httpx.Request("GET", url),
        )

    async def __aenter__(self) -> _RecordingClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


def _install(monkeypatch: pytest.MonkeyPatch, routes: dict[str, httpx.Response] | None = None):
    client = _RecordingClient(routes)
    monkeypatch.setattr(metadata.httpx, "AsyncClient", lambda **_: client)
    return client


AUTH = (("Authorization", "Bearer k"),)
METRICS_HEADERS = (("Accept", ACCEPT), ("Authorization", "Bearer k"))


@pytest.mark.asyncio
async def test_backend_hint_fetch_order_and_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _install(monkeypatch)

    assert await metadata.probe_backend_hint("http://engine.test:8000/v1", "k") is None
    assert client.calls == [
        ("/metrics", METRICS_HEADERS),
        ("/version", AUTH),
        ("/v1/models", AUTH),
        ("/health", AUTH),
        ("/.well-known/serviceinfo", AUTH),
        ("/props", AUTH),
    ]


@pytest.mark.asyncio
async def test_backend_hint_without_a_key_sends_only_accept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _install(monkeypatch)

    await metadata.probe_backend_hint("http://engine.test:8000")
    assert client.calls[0] == ("/metrics", (("Accept", ACCEPT),))
    assert all(headers == () for _, headers in client.calls[1:])


_LLAMA_PROPS = {
    "build_info": "b6123-abc1234",
    "default_generation_settings": {"n_ctx": 8192},
    "total_slots": 1,
}

# label -> (routes, expected engine_name, expected request paths)
_ENGINE_CASES: list[tuple[str, dict[str, httpx.Response], str | None, list[str]]] = [
    ("gemini", {}, "Google Gemini API", []),
    ("openai", {}, "OpenAI API", []),
    ("anthropic", {}, "Anthropic Messages API", []),
    ("GEMINI", {}, "Google Gemini API", []),
    (
        "vllm",
        {"/version": httpx.Response(200, json={"version": "0.9.1"})},
        "vLLM",
        ["/v1/models", "/version"],
    ),
    ("tensorfold", {}, "TensorFold", ["/v1/models", "/health"]),
    ("strata", {}, "Strata", ["/v1/models", "/props", "/health"]),
    ("tabbyapi", {}, "TabbyAPI", ["/v1/models", "/v1/model", "/props"]),
    ("halogen", {}, "Halogen Flash", ["/v1/models"]),
    ("sglang", {}, "SGLang", ["/v1/models"]),
    (
        "litellm",
        {"/health": httpx.Response(200, headers={"x-litellm-version": "1.2"})},
        "LiteLLM",
        ["/v1/models", "/health"],
    ),
    (
        "ninfer",
        {"/v1/models": httpx.Response(200, json={"data": [{"id": "m", "owned_by": "ninfer"}]})},
        "NInfer",
        ["/v1/models"],
    ),
    ("ninfer", {}, None, ["/v1/models"]),
    (
        "llamacpp",
        {"/props": httpx.Response(200, json=_LLAMA_PROPS)},
        "llama.cpp",
        ["/v1/models", "/props"],
    ),
    (
        "llama.cpp",
        {"/props": httpx.Response(200, json=_LLAMA_PROPS)},
        "llama.cpp",
        ["/v1/models", "/props"],
    ),
    (
        "llama_cpp",
        {"/props": httpx.Response(200, json=_LLAMA_PROPS)},
        "llama.cpp",
        ["/v1/models", "/props"],
    ),
    (
        "LlamaCpp",
        {"/props": httpx.Response(200, json=_LLAMA_PROPS)},
        "llama.cpp",
        ["/v1/models", "/props"],
    ),
    ("llamacpp", {}, None, ["/v1/models", "/props", "/health"]),
    (
        "unknown",
        {},
        None,
        ["/v1/models", "/version", "/health", "/.well-known/serviceinfo", "/props"],
    ),
    (
        "unknown",
        {"/health": httpx.Response(200, json={"service": "strata"})},
        "Strata",
        ["/v1/models", "/version", "/health", "/props"],
    ),
    (
        "something-else",
        {"/props": httpx.Response(200, json=_LLAMA_PROPS)},
        "llama.cpp",
        ["/v1/models", "/version", "/health", "/.well-known/serviceinfo", "/props"],
    ),
    (
        # owned_by tensorfold overrides whatever label the caller gave.
        "vllm",
        {"/v1/models": httpx.Response(200, json={"data": [{"id": "m", "owned_by": "TensorFold"}]})},
        "TensorFold",
        ["/v1/models", "/health"],
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("label", "routes", "engine_name", "paths"), _ENGINE_CASES)
async def test_engine_probe_name_and_fetch_order(
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    routes: dict[str, httpx.Response],
    engine_name: str | None,
    paths: list[str],
) -> None:
    client = _install(monkeypatch, routes)

    result = await metadata._probe_engine("http://engine.test:8000/v1", "k", label)

    assert result.get("engine_name") == engine_name
    assert [path for path, _ in client.calls] == paths
    assert all(headers == AUTH for _, headers in client.calls)


# ---------------------------------------------------------------------------
# Spec detection
# ---------------------------------------------------------------------------


class _MetricsOnly:
    def __init__(self, text: str | None) -> None:
        self.text = text

    async def metrics(self, *, metrics_url: str | None = None) -> httpx.Response:
        if self.text is None:
            return httpx.Response(404)
        return httpx.Response(200, text=self.text)


GENERIC = "Detected via Prometheus /metrics (spec_decode counters present)"
TF_DETAIL = "TensorFold draft counters available; proposer configuration unknown"
LC_DETAIL = (
    "llama.cpp draft counters available; per-request counts "
    "come from response timings (draft_n/draft_n_accepted)"
)
LC_NO_COUNTERS = (
    "llama.cpp detected — spec decode metrics available "
    "via per-request timings (draft_n/draft_n_accepted)"
)
STRATA_DETAIL = "Strata MTP draft counters available"

# (active, method, has_prometheus, has_per_request_timings, detail)
_SPEC_CASES: list[tuple[str, str | None, str, tuple[bool, str, bool, bool, str]]] = [
    ("unreachable", None, "auto", (False, "unknown", False, False, "")),
    ("vllm", _counters("vllm:", 10, 5, 4), "auto", (True, "unknown", True, False, GENERIC)),
    ("vllm-zero", _counters("vllm:", 0), "auto", (True, "unknown", True, False, GENERIC)),
    (
        "vllm-method-label",
        _counters("vllm:", 10, 5, 4, labels='spec_method="eagle3"'),
        "auto",
        (True, "eagle3", True, False, GENERIC),
    ),
    ("vllm-no-spec", "vllm:num_requests_running 1\n", "auto", (False, "unknown", False, False, "")),
    ("sglang-gauges", SGLANG_SPEC, "auto", (False, "unknown", False, False, "")),
    (
        "tensorfold-zero",
        _counters("tensorfold:", 0),
        "auto",
        (False, "unknown", True, False, TF_DETAIL),
    ),
    (
        "tensorfold",
        _counters("tensorfold:", 10, 5, 4),
        "auto",
        (True, "unknown", True, False, TF_DETAIL),
    ),
    (
        "llamacpp-zero",
        LLAMACPP_LOAD + _counters("llamacpp:", 0),
        "auto",
        (False, "unknown", True, True, LC_DETAIL),
    ),
    (
        "llamacpp",
        LLAMACPP_LOAD + _counters("llamacpp:", 10, 5, 4),
        "auto",
        (True, "unknown", True, True, LC_DETAIL),
    ),
    ("llamacpp-load-only", LLAMACPP_LOAD, "auto", (False, "unknown", False, True, LC_NO_COUNTERS)),
    (
        "llamacpp-load-only-hint",
        LLAMACPP_LOAD,
        "mtp",
        (True, "mtp", False, True, "Assumed active via --spec-method=mtp"),
    ),
    (
        "strata-zero",
        _counters("vllm:", 0) + STRATA_NS,
        "auto",
        (False, "mtp", True, False, STRATA_DETAIL),
    ),
    (
        "strata",
        _counters("vllm:", 10, 5, 4) + STRATA_NS,
        "auto",
        (True, "mtp", True, False, STRATA_DETAIL),
    ),
    (
        # Current behaviour (finding 3): detection forces mtp over an explicit
        # method label, while spec-live lets the label win.
        "strata-method-label",
        _counters("vllm:", 10, 5, 4, labels='spec_method="eagle"') + STRATA_NS,
        "auto",
        (True, "mtp", True, False, STRATA_DETAIL),
    ),
    (
        # Halogen exports llama.cpp metrics, and detection reads them as such.
        "halogen+llamacpp",
        HALOGEN_NS + _counters("llamacpp:", 10, 5, 4),
        "auto",
        (True, "unknown", True, True, LC_DETAIL),
    ),
    (
        # Detection checks llama.cpp before Strata, the reverse of identity.
        "strata+llamacpp",
        _counters("vllm:", 10, 5, 4) + STRATA_NS + _counters("llamacpp:", 10, 5, 4),
        "auto",
        (True, "unknown", True, True, LC_DETAIL),
    ),
    (
        "tensorfold+llamacpp",
        _counters("tensorfold:", 0) + _counters("llamacpp:", 10, 5, 4),
        "auto",
        (True, "unknown", True, False, TF_DETAIL),
    ),
    (
        "strata-in-label",
        _counters("vllm:", 0, labels='model_name="acme/strata:7b"'),
        "auto",
        (True, "unknown", True, False, GENERIC),
    ),
    (
        "llamacpp-in-label-with-counters",
        _counters("vllm:", 0, labels='model_name="llamacpp:gemma"'),
        "auto",
        (True, "unknown", True, False, GENERIC),
    ),
    (
        # Without spec counters the llama.cpp check is anchored too, so a
        # label value does not claim llama.cpp's per-request timings.
        "llamacpp-in-label-without-counters",
        'vllm:num_requests_running{model_name="acme/llamacpp:7b"} 1\n',
        "auto",
        (False, "unknown", False, False, ""),
    ),
    (
        "llamacpp-in-help-without-counters",
        "# HELP vllm:num_requests_running Like llamacpp:requests_processing.\n"
        "vllm:num_requests_running 1\n",
        "auto",
        (False, "unknown", False, False, ""),
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "hint", "expected"),
    [
        pytest.param(text, hint, expected, id=case_id)
        for case_id, text, hint, expected in _SPEC_CASES
    ],
)
async def test_spec_detection_outcome(
    text: str | None, hint: str, expected: tuple[bool, str, bool, bool, str]
) -> None:
    info = await detect_spec_decoding(_MetricsOnly(text), "http://test", backend_hint=hint)  # type: ignore[arg-type]

    assert (
        info.active,
        info.method,
        info.has_prometheus,
        info.has_per_request_timings,
        info.detail,
    ) == expected


# ---------------------------------------------------------------------------
# Spec-live labels
# ---------------------------------------------------------------------------

# (spec_backend, spec_metrics_source, spec_method)
_LIVE_CASES: list[tuple[str, str, tuple[str, str, str]]] = [
    ("empty", "", ("unknown", "unknown", "unknown")),
    ("vllm", _counters("vllm:", 10, 5, 4), ("vllm", "vllm", "unknown")),
    ("vllm-underscore", _counters("vllm_", 10, 5, 4), ("vllm", "vllm", "unknown")),
    (
        "vllm-method-label",
        _counters("vllm:", 10, 5, 4, labels='spec_method="eagle3"'),
        ("vllm", "vllm", "eagle3"),
    ),
    ("strata", _counters("vllm:", 10, 5, 4) + STRATA_NS, ("strata", "strata", "mtp")),
    ("strata-zero", _counters("vllm:", 0) + STRATA_NS, ("strata", "strata", "mtp")),
    (
        # Current behaviour (finding 3): spec-live lets a label win over mtp.
        "strata-method-label",
        _counters("vllm:", 10, 5, 4, labels='spec_method="eagle"') + STRATA_NS,
        ("strata", "strata", "eagle"),
    ),
    (
        "strata-in-label",
        _counters("vllm:", 10, 5, 4, labels='model_name="acme/strata:7b"'),
        ("vllm", "vllm", "unknown"),
    ),
    ("strata-without-counters", STRATA_NS, ("unknown", "unknown", "unknown")),
    ("sglang-gauges", SGLANG_SPEC, ("sglang", "sglang", "unknown")),
    ("sglang-engine-only", SGLANG_ENGINE, ("sglang", "unknown", "unknown")),
    ("sglang+vllm", SGLANG_SPEC + _counters("vllm:", 10, 5, 4), ("vllm", "sglang", "unknown")),
    (
        "llamacpp",
        LLAMACPP_LOAD + _counters("llamacpp:", 10, 5, 4),
        ("llamacpp", "llamacpp", "unknown"),
    ),
    ("llamacpp-zero", _counters("llamacpp:", 0), ("llamacpp", "llamacpp", "unknown")),
    ("llamacpp-load-only", LLAMACPP_LOAD, ("llamacpp", "unknown", "unknown")),
    (
        "halogen+llamacpp",
        HALOGEN_NS + _counters("llamacpp:", 10, 5, 4),
        ("llamacpp", "llamacpp", "unknown"),
    ),
    ("tensorfold", _counters("tensorfold:", 10, 5, 4), ("tensorfold", "tensorfold", "unknown")),
    (
        "tensorfold+vllm",
        _counters("tensorfold:", 10, 5, 4) + _counters("vllm:", 10, 5, 4),
        ("vllm", "vllm", "unknown"),
    ),
    (
        "vllm+llamacpp",
        _counters("vllm:", 10, 5, 4) + _counters("llamacpp:", 10, 5, 4),
        ("vllm", "vllm", "unknown"),
    ),
    (
        "tensorfold+llamacpp",
        _counters("tensorfold:", 10, 5, 4) + _counters("llamacpp:", 10, 5, 4),
        ("tensorfold", "tensorfold", "unknown"),
    ),
    (
        "strata+llamacpp",
        _counters("vllm:", 10, 5, 4) + STRATA_NS + _counters("llamacpp:", 10, 5, 4),
        ("strata", "strata", "mtp"),
    ),
]


@pytest.mark.parametrize(
    ("text", "expected"),
    [pytest.param(text, expected, id=case_id) for case_id, text, expected in _LIVE_CASES],
)
def test_spec_live_labels(text: str, expected: tuple[str, str, str]) -> None:
    snap = _parse_snapshot(text)
    delta = compute_delta(snap, snap)

    assert (snap.spec_backend, delta.spec_metrics_source, delta.spec_method) == expected
    assert snap.spec_method == delta.spec_method


# ---------------------------------------------------------------------------
# Reported context window
# ---------------------------------------------------------------------------


def _run_context(engine_name: str | None, window: int | None = 81920) -> RunContext:
    return RunContext(
        tool_version="test",
        git_sha=None,
        hostname="host",
        platform_info="linux",
        python_version="3.13",
        model="m",
        backend="unknown",
        base_url="http://test/v1",
        engine_name=engine_name,
        max_model_len=window,
    )


@pytest.mark.parametrize(
    ("engine_name", "expected"),
    [
        ("llama.cpp", 81920),
        ("Strata", 81920),
        ("vLLM", None),
        ("SGLang", None),
        ("TensorFold", None),
        ("TabbyAPI", None),
        ("Halogen Flash", None),
        ("LiteLLM", None),
        ("NInfer", None),
        ("Google Gemini API", None),
        ("OpenAI API", None),
        ("Anthropic Messages API", None),
        ("llamacpp", None),
        ("strata", None),
        ("LLAMA.CPP", None),
        (None, None),
    ],
)
def test_reported_context_window_per_engine(engine_name: str | None, expected: int | None) -> None:
    assert reported_context_window(_run_context(engine_name)) == expected


@pytest.mark.parametrize("window", [0, -1, None, True])
def test_reported_context_window_rejects_invalid_windows(window: Any) -> None:
    assert reported_context_window(_run_context("llama.cpp", window)) is None


# ---------------------------------------------------------------------------
# Backend labels
# ---------------------------------------------------------------------------

_BACKEND_CHOICES = [
    "vllm",
    "litellm",
    "llamacpp",
    "sglang",
    "gemini",
    "openai",
    "anthropic",
    "ninfer",
    "tensorfold",
    "halogen",
    "strata",
    "tabbyapi",
    "unknown",
]


def test_supported_backend_labels() -> None:
    from tool_eval_bench.application import service

    assert service._SUPPORTED_BACKENDS == {*_BACKEND_CHOICES, "llama.cpp", "llama_cpp"}


def test_schema_backend_choices() -> None:
    from tool_eval_bench.schema import ARGS_SCHEMA

    (backend,) = [arg for arg in ARGS_SCHEMA if arg["name"] == "backend"]
    assert backend["choices"] == _BACKEND_CHOICES


def test_unsupported_backend_message() -> None:
    from tool_eval_bench.application.service import BenchmarkService

    service = BenchmarkService(repo=None, reporter=None)
    with pytest.raises(ValueError) as excinfo:
        service._adapter_for("Bogus")
    assert str(excinfo.value) == (
        "Unsupported backend: Bogus. "
        "Supported: vllm, litellm, llamacpp, sglang, gemini, openai, anthropic, "
        "ninfer, tensorfold, halogen, strata, tabbyapi, unknown"
    )
