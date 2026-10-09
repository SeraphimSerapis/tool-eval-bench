"""GSM8K dataset loader — download, cache, and parse from HuggingFace API.

Downloads the test split of ``openai/gsm8k`` via the HuggingFace
Datasets Server REST API (no ``datasets`` library required).  Results
are cached locally under ``data/gsm8k/test.jsonl``.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tool_eval_bench.plugins.hf_utils import read_cache_revision, write_cache_manifest

logger = logging.getLogger(__name__)

# HuggingFace Datasets Server API (public, no auth required for gsm8k)
_HF_API_BASE = "https://datasets-server.huggingface.co/rows"
_DATASET = "openai/gsm8k"
_CONFIG = "main"
_SPLIT = "test"
_PAGE_SIZE = 100
# Commit of the HuggingFace dataset repository that ``datasets`` downloads are
# pinned to.  The REST fallback cannot be pinned (see ``hf_utils``).
_REVISION = "740312add88f781978c0658806c59bc2815b9866"

# Cache location relative to project root
_CACHE_DIR = Path("data") / "gsm8k"
_CACHE_FILE = _CACHE_DIR / "test.jsonl"

# Progress callback: (downloaded_rows, total_rows) -> None
OnDownloadProgress = Callable[[int, int], None]


@dataclass(slots=True)
class GSM8KItem:
    """A single GSM8K test question with ground-truth answer."""

    index: int
    question: str
    raw_answer: str  # Full solution with chain-of-thought
    ground_truth: float  # Numeric answer extracted from ``#### N``

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "question": self.question,
            "ground_truth": self.ground_truth,
        }


def _extract_ground_truth(raw_answer: str) -> float:
    """Extract the numeric answer from the ``#### N`` pattern.

    GSM8K answers always end with ``#### <number>``.  The number may
    contain commas (e.g. ``70,000``) which we strip before parsing.
    """
    match = re.search(r"####\s*([^\n]+)", raw_answer)
    if not match:
        raise ValueError(f"No #### pattern found in answer: {raw_answer[:100]!r}")
    num_str = match.group(1).strip().replace(",", "")
    return float(num_str)


def _find_cache_file() -> Path:
    """Resolve cache file path, searching from cwd upward for the project root."""
    # Try relative to cwd first (normal usage)
    cwd_path = Path.cwd() / _CACHE_FILE
    if cwd_path.exists():
        return cwd_path

    # Try relative to the package location (installed in .venv)
    pkg_path = Path(__file__).resolve().parents[3] / _CACHE_FILE
    if pkg_path.exists():
        return pkg_path

    # Default to cwd (will be created on download)
    return cwd_path


def load_dataset(
    *,
    force_download: bool = False,
    on_progress: OnDownloadProgress | None = None,
) -> list[GSM8KItem]:
    """Load GSM8K test split, downloading from HuggingFace if not cached.

    Parameters
    ----------
    force_download : bool
        Re-download even if cached.
    on_progress : callable, optional
        ``(downloaded_rows, total_rows)`` callback during download.

    Returns a list of ``GSM8KItem`` instances.
    """
    cache_path = _find_cache_file()

    if cache_path.exists() and not force_download:
        logger.info("Loading GSM8K from cache: %s", cache_path)
        return _load_from_cache(cache_path)

    logger.info("Downloading GSM8K test split from HuggingFace...")
    items, method = _download_dataset(on_progress=on_progress)
    logger.info("Downloaded %d items via %s", len(items), method)

    # Cache for next time
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    _save_to_cache(cache_path, items)
    write_cache_manifest(
        cache_path,
        dataset=_DATASET,
        config=_CONFIG,
        split=_SPLIT,
        method=method,
        revision=_REVISION,
    )
    logger.info("Cached %d items to %s", len(items), cache_path)

    return items


def dataset_revision() -> str:
    """Revision behind the cached test split, or ``"unknown"`` when it was not pinned."""
    return read_cache_revision(_find_cache_file())


def _rows_to_items(rows: list[dict]) -> list[GSM8KItem]:
    """Convert raw row dicts to GSM8KItem objects."""
    items: list[GSM8KItem] = []
    for i, row in enumerate(rows):
        try:
            gt = _extract_ground_truth(row["answer"])
            items.append(
                GSM8KItem(
                    index=i,
                    question=row["question"],
                    raw_answer=row["answer"],
                    ground_truth=gt,
                )
            )
        except (ValueError, KeyError) as exc:
            logger.warning("Skipping row %d: %s", i, exc)
    return items


def _download_dataset(
    *,
    on_progress: OnDownloadProgress | None = None,
) -> tuple[list[GSM8KItem], str]:
    """Download from HuggingFace — tries ``datasets`` lib first, REST API fallback.

    Returns ``(items, method)`` where method is ``"datasets"`` or ``"rest_api"``.
    """
    from tool_eval_bench.plugins.hf_utils import load_via_datasets_lib

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

    # Fallback: REST API (original httpx-based downloader)
    import httpx

    items: list[GSM8KItem] = []
    offset = 0
    total = 0

    with httpx.Client(timeout=30.0) as client:
        while True:
            url = (
                f"{_HF_API_BASE}?dataset={_DATASET}"
                f"&config={_CONFIG}&split={_SPLIT}"
                f"&offset={offset}&length={_PAGE_SIZE}"
            )
            resp = client.get(url)
            resp.raise_for_status()
            try:
                data = resp.json()
            except Exception as exc:
                raise RuntimeError(
                    f"HuggingFace API returned non-JSON at offset {offset}: {exc}"
                ) from exc

            rows_data = data.get("rows", [])
            if not rows_data:
                break

            total = data.get("num_rows_total", total)

            for row_data in rows_data:
                row = row_data["row"]
                idx = row_data["row_idx"]
                try:
                    gt = _extract_ground_truth(row["answer"])
                    items.append(
                        GSM8KItem(
                            index=idx,
                            question=row["question"],
                            raw_answer=row["answer"],
                            ground_truth=gt,
                        )
                    )
                except (ValueError, KeyError) as exc:
                    logger.warning("Skipping row %d: %s", idx, exc)

            offset += len(rows_data)
            logger.debug("Downloaded %d / %d rows", offset, total)

            if on_progress:
                on_progress(offset, total)

            if offset >= total:
                break

    return items, "rest_api"


def _load_from_cache(path: Path) -> list[GSM8KItem]:
    """Read items from a JSONL cache file."""
    items: list[GSM8KItem] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            items.append(
                GSM8KItem(
                    index=d["index"],
                    question=d["question"],
                    raw_answer=d["raw_answer"],
                    ground_truth=d["ground_truth"],
                )
            )
    return items


def _save_to_cache(path: Path, items: list[GSM8KItem]) -> None:
    """Write items to a JSONL cache file."""
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            d = {
                "index": item.index,
                "question": item.question,
                "raw_answer": item.raw_answer,
                "ground_truth": item.ground_truth,
            }
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
