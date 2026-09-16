"""Provider-scoped endpoint settings: TOOL_EVAL_<PROVIDER>_* and --provider."""

from __future__ import annotations

import sys

import pytest

from tool_eval_bench.cli import dispatch
from tool_eval_bench.cli.provider_env import provider_env_prefix, resolve_provider


def test_no_provider_resolves_to_none() -> None:
    assert resolve_provider(None, {"TOOL_EVAL_GEMINI_BASE_URL": "x"}) is None
    assert resolve_provider("", {"TOOL_EVAL_GEMINI_BASE_URL": "x"}) is None


def test_hosted_provider_reads_triple_and_labels_backend() -> None:
    env = {
        "TOOL_EVAL_GEMINI_BASE_URL": "https://generativelanguage.googleapis.com ",
        "TOOL_EVAL_GEMINI_API_KEY": "k",
        "TOOL_EVAL_GEMINI_MODEL": "gemini-2.5-pro",
        # Generic vars must not leak into a provider run.
        "TOOL_EVAL_BASE_URL": "http://localhost:8000",
        "TOOL_EVAL_API_KEY": "other",
    }
    settings = resolve_provider("Gemini", env)
    assert settings is not None
    assert settings.base_url == "https://generativelanguage.googleapis.com"
    assert settings.api_key == "k"
    assert settings.model == "gemini-2.5-pro"
    assert settings.backend_label == "gemini"


def test_custom_provider_has_no_backend_label_and_optional_fields() -> None:
    settings = resolve_provider("vllm-a", {"TOOL_EVAL_VLLM_A_BASE_URL": "http://a:8000"})
    assert settings is not None
    assert settings.base_url == "http://a:8000"
    assert settings.api_key is None
    assert settings.model is None
    # Detection decides the label for a self-hosted box.
    assert settings.backend_label is None


def test_empty_optional_values_are_none() -> None:
    env = {
        "TOOL_EVAL_OPENAI_BASE_URL": "https://api.openai.com",
        "TOOL_EVAL_OPENAI_API_KEY": "   ",
        "TOOL_EVAL_OPENAI_MODEL": "",
    }
    settings = resolve_provider("openai", env)
    assert settings is not None
    assert settings.api_key is None
    assert settings.model is None


def test_missing_base_url_names_the_variable() -> None:
    with pytest.raises(ValueError, match="TOOL_EVAL_ANTHROPIC_BASE_URL is not set"):
        resolve_provider("anthropic", {"TOOL_EVAL_ANTHROPIC_API_KEY": "k"})


@pytest.mark.parametrize(
    ("name", "prefix"),
    [
        ("gemini", "TOOL_EVAL_GEMINI_"),
        ("llama.cpp", "TOOL_EVAL_LLAMA_CPP_"),
        ("vllm-b2", "TOOL_EVAL_VLLM_B2_"),
    ],
)
def test_prefix_normalizes_separators(name: str, prefix: str) -> None:
    assert provider_env_prefix(name) == prefix


@pytest.mark.parametrize("name", ["1abc", "a b", "", "x/y", "-lead"])
def test_prefix_rejects_names_that_cannot_form_env_vars(name: str) -> None:
    with pytest.raises(ValueError, match="Invalid provider name"):
        provider_env_prefix(name)


def _probe_via_main(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> dict[str, object]:
    """Run ``main()`` in probe mode and capture what the cascade resolved."""
    seen: dict[str, object] = {}

    def fake_probe(console, base_url, api_key, headless=False, wire_format="openai", headers=None):
        seen["base_url"] = base_url
        seen["api_key"] = api_key
        seen["headers"] = headers

    monkeypatch.setattr(dispatch, "_load_dotenv", lambda: None)
    monkeypatch.setattr(dispatch, "_probe_server", fake_probe)
    monkeypatch.setattr(sys, "argv", ["tool-eval-bench", "--probe", *argv])
    dispatch.main()
    return seen


def test_main_prefers_provider_over_generic_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOOL_EVAL_BASE_URL", "http://generic:8000")
    monkeypatch.setenv("TOOL_EVAL_API_KEY", "generic-key")
    monkeypatch.setenv("TOOL_EVAL_OPENAI_BASE_URL", "https://api.openai.com")
    monkeypatch.setenv("TOOL_EVAL_OPENAI_API_KEY", "openai-key")

    seen = _probe_via_main(monkeypatch, ["--provider", "openai"])
    assert (seen["base_url"], seen["api_key"]) == ("https://api.openai.com", "openai-key")


def test_main_reads_provider_from_env_and_lets_flags_win(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TOOL_EVAL_PROVIDER", "anthropic")
    monkeypatch.setenv("TOOL_EVAL_ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1")
    monkeypatch.setenv("TOOL_EVAL_ANTHROPIC_API_KEY", "anthropic-key")

    seen = _probe_via_main(monkeypatch, ["--api-key", "flag-key"])
    assert (seen["base_url"], seen["api_key"]) == ("https://api.anthropic.com/v1", "flag-key")


def test_main_provider_does_not_inherit_generic_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TOOL_EVAL_API_KEY", "generic-key")
    monkeypatch.setenv("TOOL_EVAL_LOCAL_BASE_URL", "http://gpu-box:8080")

    seen = _probe_via_main(monkeypatch, ["--provider", "local"])
    assert (seen["base_url"], seen["api_key"]) == ("http://gpu-box:8080", None)


def test_main_rejects_provider_without_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TOOL_EVAL_NOPE_BASE_URL", raising=False)
    monkeypatch.setattr(dispatch, "_load_dotenv", lambda: None)
    monkeypatch.setattr(sys, "argv", ["tool-eval-bench", "--probe", "--provider", "nope"])
    with pytest.raises(SystemExit) as exc:
        dispatch.main()
    assert exc.value.code == 2


def test_provider_reads_headers_and_session_header() -> None:
    settings = resolve_provider(
        "zen",
        {
            "TOOL_EVAL_ZEN_BASE_URL": "https://opencode.ai/zen/go/v1/messages",
            "TOOL_EVAL_ZEN_HEADERS": "User-Agent=my-agent/1.0; X-Trace: abc",
            "TOOL_EVAL_ZEN_SESSION_HEADER": "x-opencode-session",
        },
    )
    assert settings is not None
    assert settings.headers == {"User-Agent": "my-agent/1.0", "X-Trace": "abc"}
    assert settings.session_header == "x-opencode-session"


def test_main_merges_provider_headers_with_flags_and_mints_a_session_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TOOL_EVAL_ZEN_BASE_URL", "https://opencode.ai/zen/go/v1/messages")
    monkeypatch.setenv("TOOL_EVAL_ZEN_HEADERS", "User-Agent=env-agent;X-Env=1")
    monkeypatch.setenv("TOOL_EVAL_ZEN_SESSION_HEADER", "x-opencode-session")

    seen = _probe_via_main(monkeypatch, ["--provider", "zen", "--header", "User-Agent=flag-agent"])
    headers = seen["headers"]
    assert isinstance(headers, dict)
    # The flag overrides the provider's value; the provider's other header stays.
    assert headers["User-Agent"] == "flag-agent"
    assert headers["X-Env"] == "1"
    # A pre-flight request is a conversation of its own.
    assert len(headers["x-opencode-session"]) == 32


def test_main_reads_generic_headers_without_a_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOOL_EVAL_BASE_URL", "http://gpu-box:8080")
    monkeypatch.setenv("TOOL_EVAL_HEADERS", "X-Generic=yes")
    monkeypatch.delenv("TOOL_EVAL_SESSION_HEADER", raising=False)

    seen = _probe_via_main(monkeypatch, ["--session-header", "x-session"])
    headers = seen["headers"]
    assert isinstance(headers, dict)
    assert headers["X-Generic"] == "yes"
    assert "x-session" in headers


def test_main_rejects_a_malformed_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOOL_EVAL_BASE_URL", "http://gpu-box:8080")
    monkeypatch.setattr(dispatch, "_load_dotenv", lambda: None)
    monkeypatch.setattr(sys, "argv", ["tool-eval-bench", "--probe", "--header", "no-separator"])
    with pytest.raises(SystemExit) as exc:
        dispatch.main()
    assert exc.value.code == 2
