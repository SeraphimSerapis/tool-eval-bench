"""Locate a local HuggingFace tokenizer without touching the network.

llama-benchy always needs a tokenizer to construct benchmark prompts.  We run
it with ``HF_HUB_OFFLINE=1`` (see :mod:`tool_eval_bench.runner.llama_benchy`),
so on hosts with an empty HuggingFace cache it fails unless the user passes
``--tokenizer /path/to/tokenizer.json`` — which means hunting through
``~/.cache/huggingface/hub`` by hand.

This module does that hunt automatically.  It is deliberately dependency-free
(stdlib ``pathlib`` only): ``huggingface_hub`` is not a runtime dependency, and
shelling out to the ``hf`` CLI would mean parsing a human-readable table.

The HF cache layout we rely on::

    <cache_root>/models--Qwen--Qwen3.6-35B-A3B-FP8/
        refs/main                      -> <sha>
        snapshots/<sha>/tokenizer.json

Resolution order (see :func:`resolve_tokenizer`):

1. An explicit ``--tokenizer`` value always wins.
2. The served model id / vLLM ``root`` is a local path → look there.
3. The id looks like ``org/name`` → exact HF cache lookup.
4. The id is an alias (``qwen3-coder``) → normalised match against cached
   repos, accepted only when exactly one repo matches.
5. llama.cpp ``/props.model_path`` → the GGUF's sibling directory.
6. Nothing found → return the cached repo ids so the caller can show the user
   what *is* available instead of making them go looking.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tool_eval_bench.domain.wire_hygiene import is_control_special_token

logger = logging.getLogger(__name__)

_TOKENIZER_FILE = "tokenizer.json"
_SPECIAL_TOKENS_MAP_FILE = "special_tokens_map.json"
_REPO_PREFIX = "models--"


@dataclass(frozen=True)
class TokenizerResolution:
    """Outcome of a tokenizer lookup.

    ``path`` is None when nothing was found; ``candidates`` then holds the
    repo ids present in the local HF cache, for a helpful error message.
    """

    path: str | None = None
    source: str = "none"
    detail: str = ""
    candidates: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.path is not None


# ---------------------------------------------------------------------------
# HF cache discovery
# ---------------------------------------------------------------------------


def hf_cache_roots() -> list[Path]:
    """Return existing HuggingFace hub cache directories, most specific first.

    Honours ``HUGGINGFACE_HUB_CACHE``, ``HF_HUB_CACHE``, ``TRANSFORMERS_CACHE``
    and ``HF_HOME`` before falling back to ``~/.cache/huggingface/hub``.
    """
    candidates: list[Path] = []

    for var in ("HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE", "TRANSFORMERS_CACHE"):
        value = os.environ.get(var)
        if value:
            candidates.append(Path(value).expanduser())

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        candidates.append(Path(hf_home).expanduser() / "hub")

    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        candidates.append(Path(xdg).expanduser() / "huggingface" / "hub")

    candidates.append(Path.home() / ".cache" / "huggingface" / "hub")

    roots: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            if not candidate.is_dir():
                continue
            resolved = candidate.resolve()
        except OSError:  # pragma: no cover - unreadable path
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        roots.append(candidate)
    return roots


def _repo_id_from_dir(name: str) -> str:
    """``models--Qwen--Qwen3.6-35B`` → ``Qwen/Qwen3.6-35B``."""
    return name[len(_REPO_PREFIX) :].replace("--", "/")


def iter_cached_repos(roots: list[Path] | None = None) -> dict[str, Path]:
    """Map ``repo_id`` → repo directory for every repo in the HF cache."""
    repos: dict[str, Path] = {}
    for root in roots if roots is not None else hf_cache_roots():
        try:
            entries = sorted(root.iterdir())
        except OSError:  # pragma: no cover - unreadable cache root
            continue
        for entry in entries:
            if not entry.is_dir() or not entry.name.startswith(_REPO_PREFIX):
                continue
            repos.setdefault(_repo_id_from_dir(entry.name), entry)
    return repos


def tokenizer_in_repo(repo_dir: Path) -> Path | None:
    """Find ``tokenizer.json`` inside a cached repo directory.

    Prefers the snapshot pinned by ``refs/main``; otherwise falls back to the
    most recently modified snapshot that actually has the file.
    """
    ref = repo_dir / "refs" / "main"
    try:
        if ref.is_file():
            sha = ref.read_text(encoding="utf-8").strip()
            pinned = repo_dir / "snapshots" / sha / _TOKENIZER_FILE
            if sha and pinned.is_file():
                return pinned
    except OSError:  # pragma: no cover - unreadable ref
        pass

    snapshots = repo_dir / "snapshots"
    try:
        found = [
            snapshot / _TOKENIZER_FILE
            for snapshot in snapshots.iterdir()
            if (snapshot / _TOKENIZER_FILE).is_file()
        ]
    except OSError:
        return None
    if not found:
        return None
    return max(found, key=lambda p: p.stat().st_mtime)


# ---------------------------------------------------------------------------
# Local path handling
# ---------------------------------------------------------------------------


def tokenizer_near_path(raw: str) -> Path | None:
    """Find a tokenizer for a local model path (a directory, or a weights file).

    Accepts a ``tokenizer.json`` directly, a directory containing one, or a
    model file (``.gguf``, ``.safetensors``) whose sibling directory has one.
    """
    if not raw:
        return None
    path = Path(raw).expanduser()
    try:
        if path.is_file():
            if path.name == _TOKENIZER_FILE:
                return path
            sibling = path.parent / _TOKENIZER_FILE
            return sibling if sibling.is_file() else None
        if path.is_dir():
            direct = path / _TOKENIZER_FILE
            return direct if direct.is_file() else None
    except OSError:  # pragma: no cover - permission errors
        return None
    return None


def _looks_like_path(value: str) -> bool:
    return value.startswith(("/", "~", "./", "../")) or (len(value) > 2 and value[1] == ":")


# ---------------------------------------------------------------------------
# Alias matching
# ---------------------------------------------------------------------------


def _normalise(value: str) -> str:
    """Lowercase and strip separators so ``Qwen3.6-35B`` == ``qwen3_6_35b``."""
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _fuzzy_match(alias: str, repos: dict[str, Path]) -> tuple[str, Path] | None:
    """Match a served alias against cached repo ids.

    Only an unambiguous match is accepted: a tokenizer from the wrong model
    family silently changes real prompt token counts, which is worse than
    asking the user for ``--tokenizer``.
    """
    target = _normalise(alias)
    if not target:
        return None

    exact: list[tuple[str, Path]] = []
    partial: list[tuple[str, Path]] = []
    for repo_id, repo_dir in repos.items():
        full = _normalise(repo_id)
        name = _normalise(repo_id.split("/")[-1])
        if target in (full, name):
            exact.append((repo_id, repo_dir))
        elif target in name or name in target:
            partial.append((repo_id, repo_dir))

    for bucket in (exact, partial):
        if len(bucket) == 1:
            return bucket[0]
        if len(bucket) > 1:
            logger.debug(
                "tokenizer alias %r matched %d cached repos, refusing to guess",
                alias,
                len(bucket),
            )
            return None
    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def resolve_tokenizer(
    model: str,
    *,
    explicit: str | None = None,
    model_root: str | None = None,
    model_path: str | None = None,
) -> TokenizerResolution:
    """Resolve a local tokenizer for ``model``.

    Parameters
    ----------
    model:
        The model id used against the API (the served alias).
    explicit:
        A user-supplied ``--tokenizer`` value; returned unchanged if truthy.
    model_root:
        The ``root`` field from ``/v1/models`` — on vLLM this is the real HF
        repo id or local path behind the alias.
    model_path:
        A backend-reported weights path, e.g. llama.cpp ``/props.model_path``.
    """
    if explicit:
        return TokenizerResolution(path=explicit, source="explicit", detail="--tokenizer")

    repos = iter_cached_repos()
    candidates = sorted(repos)
    ids = [value for value in (model_root, model) if value]

    # 2. Local model directory / file (vLLM root is often an absolute path).
    for value in ids:
        if not _looks_like_path(value):
            continue
        found = tokenizer_near_path(value)
        if found:
            return TokenizerResolution(
                path=str(found), source="model-path", detail=value, candidates=candidates
            )

    # 3. Exact HF repo id.
    for value in ids:
        repo_dir = repos.get(value)
        if repo_dir is None:
            continue
        found = tokenizer_in_repo(repo_dir)
        if found:
            return TokenizerResolution(
                path=str(found), source="hf-cache", detail=value, candidates=candidates
            )

    # 4. Alias → unique cached repo.
    for value in ids:
        match = _fuzzy_match(value, repos)
        if match is None:
            continue
        repo_id, repo_dir = match
        found = tokenizer_in_repo(repo_dir)
        if found:
            return TokenizerResolution(
                path=str(found),
                source="hf-cache-alias",
                detail=repo_id,
                candidates=candidates,
            )

    # 5. Backend-reported weights path (llama.cpp GGUF sibling).
    if model_path:
        found = tokenizer_near_path(model_path)
        if found:
            return TokenizerResolution(
                path=str(found), source="model-path", detail=model_path, candidates=candidates
            )

    # 6. Nothing usable — report what the cache does hold.
    return TokenizerResolution(candidates=candidates)


def format_candidates(candidates: list[str], limit: int = 10) -> str:
    """Render cached repo ids as an indented bullet list for error output."""
    if not candidates:
        return ""
    shown = candidates[:limit]
    lines = [f"  - {repo_id}" for repo_id in shown]
    if len(candidates) > limit:
        lines.append(f"  … and {len(candidates) - limit} more")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Special-token extraction (stdlib-only)
# ---------------------------------------------------------------------------


def _load_json_dict(path: Path) -> dict[str, Any] | None:
    """Read a JSON object from *path*, returning None if missing or corrupt."""
    try:
        if not path.is_file():
            return None
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _iter_special_map_strings(value: Any) -> list[str]:
    """Extract string token contents from one ``special_tokens_map.json`` entry."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        content = value.get("content")
        return [content] if isinstance(content, str) else []
    if isinstance(value, list):
        found: list[str] = []
        for item in value:
            found.extend(_iter_special_map_strings(item))
        return found
    return []


def extract_special_tokens(tokenizer_path: str | Path) -> frozenset[str]:
    """Extract chat-template/control special tokens from a local HuggingFace tokenizer.

    Accepts a path to ``tokenizer.json`` or a directory containing it, and also
    inspects a sibling ``special_tokens_map.json`` when present. Only entries
    marked ``special: true`` in ``added_tokens`` (or listed in
    ``special_tokens_map.json``) that match bracketed/angle-delimited control
    syntax (see :func:`tool_eval_bench.domain.wire_hygiene.is_control_special_token`)
    are returned, filtering out plain words, short fragments, and standard HTML
    tags. Returns ``frozenset()`` on missing, unreadable, or corrupt files.
    """
    if not tokenizer_path:
        return frozenset()
    raw_path = Path(tokenizer_path).expanduser()
    try:
        is_dir = raw_path.is_dir()
    except OSError:
        return frozenset()

    tok_file = raw_path / _TOKENIZER_FILE if is_dir else raw_path
    special_map_file = (raw_path if is_dir else raw_path.parent) / _SPECIAL_TOKENS_MAP_FILE

    tokens: set[str] = set()

    tok_data = _load_json_dict(tok_file)
    if tok_data is not None:
        added_tokens = tok_data.get("added_tokens")
        if isinstance(added_tokens, list):
            for entry in added_tokens:
                if not isinstance(entry, dict) or entry.get("special") is not True:
                    continue
                content = entry.get("content")
                if isinstance(content, str) and is_control_special_token(content):
                    tokens.add(content)

    map_data = _load_json_dict(special_map_file)
    if map_data is not None:
        for value in map_data.values():
            for candidate in _iter_special_map_strings(value):
                if is_control_special_token(candidate):
                    tokens.add(candidate)

    return frozenset(tokens)
