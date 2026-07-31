"""Tests for utils.tokenizers — offline tokenizer auto-detection."""

from __future__ import annotations

from pathlib import Path

import pytest

from tool_eval_bench.utils.tokenizers import (
    format_candidates,
    hf_cache_roots,
    iter_cached_repos,
    resolve_tokenizer,
    tokenizer_in_repo,
    tokenizer_near_path,
)

_USE_SHA = "<use-sha>"


def _make_repo(
    cache: Path,
    repo_id: str,
    *,
    sha: str = "abc123",
    ref: str | None = _USE_SHA,
    tokenizer: bool = True,
) -> Path:
    """Create a fake HF cache entry for ``repo_id``.

    ``ref`` defaults to ``sha``; pass None for no refs/main, or another value
    to simulate a dangling ref.
    """
    if ref is _USE_SHA:
        ref = sha
    repo_dir = cache / f"models--{repo_id.replace('/', '--')}"
    snapshot = repo_dir / "snapshots" / sha
    snapshot.mkdir(parents=True)
    if tokenizer:
        (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
    if ref is not None:
        (repo_dir / "refs").mkdir(parents=True, exist_ok=True)
        (repo_dir / "refs" / "main").write_text(ref, encoding="utf-8")
    return repo_dir


@pytest.fixture
def hf_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An isolated HF hub cache, with all other cache locations neutralised.

    HOME is redirected too, otherwise the developer's real
    ``~/.cache/huggingface/hub`` leaks into the scan.
    """
    cache = tmp_path / "hub"
    cache.mkdir()
    for var in ("HF_HOME", "TRANSFORMERS_CACHE", "HF_HUB_CACHE", "XDG_CACHE_HOME"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(cache))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    return cache


# -- cache discovery --------------------------------------------------------


def test_hf_cache_roots_prefers_env(hf_cache: Path) -> None:
    assert hf_cache_roots()[0] == hf_cache


def test_hf_cache_roots_skips_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "nope"))
    assert all(root.is_dir() for root in hf_cache_roots())


def test_hf_cache_roots_deduplicates(hf_cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_CACHE", str(hf_cache))
    assert hf_cache_roots().count(hf_cache) == 1


def test_iter_cached_repos_maps_repo_ids(hf_cache: Path) -> None:
    _make_repo(hf_cache, "Qwen/Qwen3.6-35B-A3B-FP8")
    _make_repo(hf_cache, "gpt2")
    (hf_cache / "not-a-repo").mkdir()

    repos = iter_cached_repos()

    assert set(repos) == {"Qwen/Qwen3.6-35B-A3B-FP8", "gpt2"}


# -- snapshot selection -----------------------------------------------------


def test_tokenizer_in_repo_follows_refs_main(hf_cache: Path) -> None:
    repo = _make_repo(hf_cache, "org/model", sha="pinned")
    other = repo / "snapshots" / "stale"
    other.mkdir()
    (other / "tokenizer.json").write_text("{}", encoding="utf-8")

    found = tokenizer_in_repo(repo)

    assert found is not None
    assert found.parent.name == "pinned"


def test_tokenizer_in_repo_falls_back_when_ref_missing(hf_cache: Path) -> None:
    repo = _make_repo(hf_cache, "org/model", sha="only", ref=None)
    assert tokenizer_in_repo(repo) == repo / "snapshots" / "only" / "tokenizer.json"


def test_tokenizer_in_repo_ignores_snapshot_without_tokenizer(hf_cache: Path) -> None:
    repo = _make_repo(hf_cache, "org/model", tokenizer=False)
    assert tokenizer_in_repo(repo) is None


def test_tokenizer_in_repo_handles_dangling_ref(hf_cache: Path) -> None:
    """A refs/main pointing at a snapshot without tokenizer.json still resolves."""
    repo = _make_repo(hf_cache, "org/model", sha="good", ref="gone")
    assert tokenizer_in_repo(repo) == repo / "snapshots" / "good" / "tokenizer.json"


# -- local paths ------------------------------------------------------------


def test_tokenizer_near_path_directory(tmp_path: Path) -> None:
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    assert tokenizer_near_path(str(tmp_path)) == tmp_path / "tokenizer.json"


def test_tokenizer_near_path_file_itself(tmp_path: Path) -> None:
    target = tmp_path / "tokenizer.json"
    target.write_text("{}", encoding="utf-8")
    assert tokenizer_near_path(str(target)) == target


def test_tokenizer_near_path_gguf_sibling(tmp_path: Path) -> None:
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    gguf = tmp_path / "model-Q4_K_M.gguf"
    gguf.write_bytes(b"GGUF")
    assert tokenizer_near_path(str(gguf)) == tmp_path / "tokenizer.json"


def test_tokenizer_near_path_missing(tmp_path: Path) -> None:
    assert tokenizer_near_path(str(tmp_path)) is None
    assert tokenizer_near_path("") is None


# -- resolution order -------------------------------------------------------


def test_explicit_wins(hf_cache: Path) -> None:
    _make_repo(hf_cache, "org/model")
    resolution = resolve_tokenizer("org/model", explicit="/custom/tokenizer.json")

    assert resolution.path == "/custom/tokenizer.json"
    assert resolution.source == "explicit"
    assert bool(resolution) is True


def test_exact_repo_id_from_hf_cache(hf_cache: Path) -> None:
    _make_repo(hf_cache, "Qwen/Qwen3.6-35B-A3B-FP8")

    resolution = resolve_tokenizer("Qwen/Qwen3.6-35B-A3B-FP8")

    assert resolution.source == "hf-cache"
    assert resolution.path is not None
    assert resolution.path.endswith("tokenizer.json")


def test_vllm_root_resolves_alias(hf_cache: Path) -> None:
    """The served alias is meaningless; /v1/models root carries the real id."""
    _make_repo(hf_cache, "Qwen/Qwen3.6-35B-A3B-FP8")

    resolution = resolve_tokenizer("my-model", model_root="Qwen/Qwen3.6-35B-A3B-FP8")

    assert resolution.source == "hf-cache"
    assert resolution.detail == "Qwen/Qwen3.6-35B-A3B-FP8"


def test_local_model_directory(tmp_path: Path, hf_cache: Path) -> None:
    model_dir = tmp_path / "models" / "Qwen3.6"
    model_dir.mkdir(parents=True)
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    resolution = resolve_tokenizer("qwen", model_root=str(model_dir))

    assert resolution.source == "model-path"
    assert resolution.path == str(model_dir / "tokenizer.json")


def test_alias_matches_single_cached_repo(hf_cache: Path) -> None:
    _make_repo(hf_cache, "Qwen/Qwen3.6-35B-A3B-FP8")
    _make_repo(hf_cache, "gpt2")

    resolution = resolve_tokenizer("qwen3.6-35b-a3b-fp8")

    assert resolution.source == "hf-cache-alias"
    assert resolution.detail == "Qwen/Qwen3.6-35B-A3B-FP8"


def test_ambiguous_alias_is_refused(hf_cache: Path) -> None:
    """A wrong-family tokenizer silently skews token counts — never guess."""
    _make_repo(hf_cache, "Intel/Qwen3.6-27B-int4-AutoRound")
    _make_repo(hf_cache, "unsloth/Qwen3.6-27B-int4-AutoRound")

    resolution = resolve_tokenizer("Qwen3.6-27B-int4-AutoRound")

    assert resolution.path is None


def test_llamacpp_model_path_fallback(tmp_path: Path, hf_cache: Path) -> None:
    gguf_dir = tmp_path / "gguf"
    gguf_dir.mkdir()
    (gguf_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    gguf = gguf_dir / "model.gguf"
    gguf.write_bytes(b"GGUF")

    resolution = resolve_tokenizer("unknown-alias", model_path=str(gguf))

    assert resolution.source == "model-path"
    assert resolution.path == str(gguf_dir / "tokenizer.json")


def test_unresolved_reports_candidates(hf_cache: Path) -> None:
    _make_repo(hf_cache, "gpt2")
    _make_repo(hf_cache, "deepseek-ai/DeepSeek-V4-Flash")

    resolution = resolve_tokenizer("totally-unrelated-name")

    assert resolution.path is None
    assert resolution.source == "none"
    assert bool(resolution) is False
    assert resolution.candidates == ["deepseek-ai/DeepSeek-V4-Flash", "gpt2"]


def test_empty_cache_resolves_to_nothing(hf_cache: Path) -> None:
    resolution = resolve_tokenizer("any-model")
    assert resolution.path is None
    assert resolution.candidates == []


# -- error-message rendering ------------------------------------------------


def test_format_candidates_lists_entries() -> None:
    assert format_candidates(["a/b", "c/d"]) == "  - a/b\n  - c/d"


def test_format_candidates_truncates() -> None:
    rendered = format_candidates([f"org/m{i}" for i in range(12)], limit=3)
    assert rendered.endswith("… and 9 more")


def test_format_candidates_empty() -> None:
    assert format_candidates([]) == ""


# -- CLI wiring -------------------------------------------------------------


def test_perf_helper_prefers_explicit_and_skips_probe(
    hf_cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit --tokenizer short-circuits both the cache scan and /props."""
    from rich.console import Console

    from tool_eval_bench.cli import perf

    def _boom(_: str) -> str | None:
        raise AssertionError("/props must not be probed when --tokenizer is given")

    monkeypatch.setattr(perf, "_probe_llamacpp_model_path", _boom)

    result = perf._resolve_benchy_tokenizer(
        Console(), "m", "m", "http://localhost:8000/v1", "/explicit/tokenizer.json"
    )

    assert result == "/explicit/tokenizer.json"


def test_perf_helper_finds_cached_tokenizer(
    hf_cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rich.console import Console

    from tool_eval_bench.cli import perf

    _make_repo(hf_cache, "Qwen/Qwen3.6-35B-A3B-FP8")
    monkeypatch.setattr(perf, "_probe_llamacpp_model_path", lambda _: None)
    console = Console(record=True, width=100)

    result = perf._resolve_benchy_tokenizer(
        console, "alias", "Qwen/Qwen3.6-35B-A3B-FP8", "http://localhost:8000/v1", None
    )

    assert result is not None
    assert result.endswith("tokenizer.json")
    assert "Qwen/Qwen3.6-35B-A3B-FP8" in console.export_text()


def test_perf_helper_falls_back_to_llamacpp_props(
    tmp_path: Path, hf_cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With nothing in the cache, /props.model_path is the last resort."""
    from rich.console import Console

    from tool_eval_bench.cli import perf

    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF")
    monkeypatch.setattr(perf, "_probe_llamacpp_model_path", lambda _: str(gguf))

    result = perf._resolve_benchy_tokenizer(
        Console(), "alias", "alias", "http://localhost:8080/v1", None
    )

    assert result == str(tmp_path / "tokenizer.json")


def test_perf_helper_returns_none_when_unresolved(
    hf_cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rich.console import Console

    from tool_eval_bench.cli import perf

    monkeypatch.setattr(perf, "_probe_llamacpp_model_path", lambda _: None)

    assert (
        perf._resolve_benchy_tokenizer(Console(), "m", "m", "http://localhost:8000/v1", None)
        is None
    )


class _FakeResponse:
    def __init__(self, status_code: int, payload: object) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> object:
        return self._payload


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"model_path": "/models/m.gguf"}, "/models/m.gguf"),
        ({"default_generation_settings": {"model": "/models/m.gguf"}}, "/models/m.gguf"),
        ({}, None),
        ([], None),
    ],
)
def test_probe_llamacpp_model_path(
    payload: object, expected: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    import httpx

    from tool_eval_bench.cli import perf

    seen: list[str] = []

    def fake_get(url: str, **kwargs: object) -> _FakeResponse:
        seen.append(url)
        return _FakeResponse(200, payload)

    monkeypatch.setattr(httpx, "get", fake_get)

    assert perf._probe_llamacpp_model_path("http://localhost:8080/v1") == expected
    assert seen == ["http://localhost:8080/props"]


def test_probe_llamacpp_model_path_swallows_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    from tool_eval_bench.cli import perf

    def fake_get(url: str, **kwargs: object) -> _FakeResponse:
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", fake_get)

    assert perf._probe_llamacpp_model_path("http://localhost:8080/v1") is None
