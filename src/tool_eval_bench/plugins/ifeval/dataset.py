"""IFEval dataset loader — download, cache, and parse from HuggingFace API.

Downloads the ``train`` split (only split) of ``google/IFEval`` via the
HuggingFace Datasets Server REST API.  Cached to ``data/ifeval/prompts.jsonl``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tool_eval_bench.plugins.hf_utils import read_cache_revision, write_cache_manifest

logger = logging.getLogger(__name__)

_HF_API_BASE = "https://datasets-server.huggingface.co/rows"
_DATASET = "google/IFEval"
_CONFIG = "default"
_SPLIT = "train"
_PAGE_SIZE = 100
# Commit of the HuggingFace dataset repository that ``datasets`` downloads are
# pinned to.  The REST fallback cannot be pinned (see ``hf_utils``).
_REVISION = "966cd89545d6b6acfd7638bc708b98261ca58e84"

_CACHE_DIR = Path("data") / "ifeval"
_CACHE_FILE = _CACHE_DIR / "prompts.jsonl"

OnDownloadProgress = Callable[[int, int], None]


@dataclass(slots=True)
class IFEvalItem:
    """A single IFEval prompt with its constraints."""

    key: int
    prompt: str
    instruction_id_list: list[str]
    kwargs: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "prompt": self.prompt,
            "instruction_id_list": self.instruction_id_list,
        }


def _find_cache_file() -> Path:
    return _CACHE_FILE


def _save_to_cache(path: Path, items: list[IFEvalItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            row = {
                "key": item.key,
                "prompt": item.prompt,
                "instruction_id_list": item.instruction_id_list,
                "kwargs": item.kwargs,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("Cached %d IFEval items to %s", len(items), path)


def _load_from_cache(path: Path) -> list[IFEvalItem]:
    items: list[IFEvalItem] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            items.append(
                IFEvalItem(
                    key=row["key"],
                    prompt=row["prompt"],
                    instruction_id_list=row["instruction_id_list"],
                    kwargs=row["kwargs"],
                )
            )
    return items


def _rows_to_items(rows: list[dict]) -> list[IFEvalItem]:
    """Convert raw row dicts to IFEvalItem objects."""
    items: list[IFEvalItem] = []
    for row in rows:
        items.append(
            IFEvalItem(
                key=int(row["key"]),
                prompt=row["prompt"],
                instruction_id_list=list(row["instruction_id_list"]),
                kwargs=list(row["kwargs"]),
            )
        )
    return items


def _download_dataset(
    on_progress: OnDownloadProgress | None = None,
) -> tuple[list[IFEvalItem], str]:
    """Download from HuggingFace — tries ``datasets`` lib first, REST API fallback.

    Returns ``(items, method)`` where method is ``"datasets"`` or ``"rest_api"``.
    """
    from tool_eval_bench.plugins.hf_utils import (
        download_rows_paginated,
        load_via_datasets_lib,
    )

    # Fast path: datasets library (no rate limits)
    rows = load_via_datasets_lib(
        _DATASET,
        _CONFIG,
        _SPLIT,
        revision=_REVISION,
        on_progress=on_progress,
    )
    if rows is not None:
        return _rows_to_items(rows), "datasets"

    # Fallback: REST API with resume support
    partial_path = _CACHE_DIR / "prompts.partial.jsonl"
    rows = download_rows_paginated(
        _DATASET,
        _CONFIG,
        _SPLIT,
        page_size=_PAGE_SIZE,
        partial_path=partial_path,
        on_progress=on_progress,
    )
    return _rows_to_items(rows), "rest_api"


def load_dataset(
    *,
    force_download: bool = False,
    on_progress: OnDownloadProgress | None = None,
) -> list[IFEvalItem]:
    """Load IFEval items, downloading and caching if needed."""
    cache_path = _find_cache_file()

    if cache_path.exists() and not force_download:
        logger.info("Loading IFEval from cache: %s", cache_path)
        return _load_from_cache(cache_path)

    logger.info("Downloading IFEval from HuggingFace…")
    items, method = _download_dataset(on_progress=on_progress)
    logger.info("Downloaded %d items via %s", len(items), method)

    # Save to final cache and clean up any partial file
    _save_to_cache(cache_path, items)
    write_cache_manifest(
        cache_path,
        dataset=_DATASET,
        config=_CONFIG,
        split=_SPLIT,
        method=method,
        revision=_REVISION,
    )
    partial_path = _CACHE_DIR / "prompts.partial.jsonl"
    partial_path.unlink(missing_ok=True)

    return items


def dataset_revision() -> str:
    """Revision behind the cached prompts, or ``"unknown"`` when it was not pinned."""
    return read_cache_revision(_find_cache_file())
