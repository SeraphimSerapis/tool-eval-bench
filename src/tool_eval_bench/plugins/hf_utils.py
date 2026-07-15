"""Shared HuggingFace Datasets Server API utilities.

Provides a resilient paginated downloader with retry/backoff for the
HuggingFace Datasets Server REST API.  Used by GSM8K, MMLU, and IFEval
dataset loaders.

Features:
- Exponential backoff on 429 (rate limit) and 5xx errors
- Retry-After header support
- Inter-page throttling to avoid triggering rate limits
- Resume support via partial cache files
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_HF_API_BASE = "https://datasets-server.huggingface.co/rows"

# Retry configuration
_MAX_RETRIES = 5
_INITIAL_BACKOFF_S = 2.0
_MAX_BACKOFF_S = 60.0
_INTER_PAGE_DELAY_S = 0.15  # Throttle between pages to avoid 429

OnDownloadProgress = Callable[[int, int], None]
OnRetry = Callable[[int, int, float], None]  # (attempt, max_retries, wait_seconds)


def _fetch_with_retry(
    url: str,
    *,
    max_retries: int = _MAX_RETRIES,
    on_retry: OnRetry | None = None,
) -> dict[str, Any]:
    """Fetch a URL with exponential backoff on 429 and transient errors.

    Returns the parsed JSON response.
    """
    backoff = _INITIAL_BACKOFF_S
    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                # Rate limited — check Retry-After header
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                wait = float(retry_after) if retry_after else backoff

                if attempt < max_retries:
                    if on_retry:
                        on_retry(attempt, max_retries, wait)
                    logger.warning(
                        "HTTP 429 (rate limited) on attempt %d/%d — waiting %.1fs before retry",
                        attempt,
                        max_retries,
                        wait,
                    )
                    time.sleep(wait)
                    backoff = min(backoff * 2, _MAX_BACKOFF_S)
                    continue
                raise
            if exc.code >= 500 and attempt < max_retries:
                # Transient server error — retry
                if on_retry:
                    on_retry(attempt, max_retries, backoff)
                logger.warning(
                    "HTTP %d on attempt %d/%d — waiting %.1fs",
                    exc.code,
                    attempt,
                    max_retries,
                    backoff,
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, _MAX_BACKOFF_S)
                continue
            raise
        except (TimeoutError, OSError) as exc:
            if attempt < max_retries:
                if on_retry:
                    on_retry(attempt, max_retries, backoff)
                logger.warning(
                    "Network error on attempt %d/%d (%s) — waiting %.1fs",
                    attempt,
                    max_retries,
                    exc,
                    backoff,
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, _MAX_BACKOFF_S)
                continue
            raise
    raise RuntimeError(f"Failed to fetch {url} after {max_retries} attempts")


def get_dataset_info(
    dataset: str,
    config: str,
) -> dict[str, Any]:
    """Fetch dataset info (splits, sizes) from HuggingFace."""
    url = f"https://datasets-server.huggingface.co/info?dataset={dataset}&config={config}"
    return _fetch_with_retry(url)


def _count_lines(path: Path) -> int:
    """Count non-empty lines in a JSONL file (for resume offset)."""
    count = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _append_rows_to_file(path: Path, rows: list[dict[str, Any]]) -> None:
    """Append rows as JSONL to a partial cache file."""
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_rows_from_file(path: Path) -> list[dict[str, Any]]:
    """Read all rows from a JSONL file."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def download_rows_paginated(
    dataset: str,
    config: str,
    split: str,
    *,
    page_size: int = 100,
    partial_path: Path | None = None,
    on_progress: OnDownloadProgress | None = None,
    on_retry: OnRetry | None = None,
) -> list[dict[str, Any]]:
    """Download all rows from a HuggingFace dataset split with pagination.

    Returns a list of row dicts (each is ``data["rows"][i]["row"]``).
    Handles rate limiting with exponential backoff and adds inter-page
    throttling to avoid triggering 429s.

    If ``partial_path`` is provided, rows are appended incrementally
    as each page is fetched.  On failure, the partial file persists and
    subsequent calls resume from where they left off — no re-downloading
    already-fetched pages.
    """
    info = get_dataset_info(dataset, config)
    total = info["dataset_info"]["splits"][split]["num_examples"]
    logger.info("%s/%s %s split has %d items", dataset, config, split, total)

    # Resume support: check for existing partial file
    offset = 0
    if partial_path is not None:
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        if partial_path.exists():
            offset = _count_lines(partial_path)
            if offset >= total:
                # Already complete — read and return
                logger.info("Partial cache is complete (%d/%d), reading", offset, total)
                return _read_rows_from_file(partial_path)
            if offset > 0:
                logger.info(
                    "Resuming download from row %d/%d (%.0f%%)",
                    offset,
                    total,
                    offset / total * 100,
                )
                if on_progress:
                    on_progress(offset, total)

    all_rows: list[dict[str, Any]] = []

    while offset < total:
        url = (
            f"{_HF_API_BASE}?dataset={dataset}&config={config}"
            f"&split={split}&offset={offset}&length={page_size}"
        )
        data = _fetch_with_retry(url, on_retry=on_retry)

        rows = data.get("rows", [])
        if not rows:
            break

        page_rows = [row_obj["row"] for row_obj in rows]
        all_rows.extend(page_rows)

        # Incremental save if partial_path is set
        if partial_path is not None:
            _append_rows_to_file(partial_path, page_rows)

        offset += len(rows)
        if on_progress:
            on_progress(offset, total)

        # Throttle between pages to stay under rate limits
        if offset < total:
            time.sleep(_INTER_PAGE_DELAY_S)

    # If we resumed, prepend the already-cached rows
    if partial_path is not None and partial_path.exists():
        return _read_rows_from_file(partial_path)

    return all_rows


def load_via_datasets_lib(
    dataset: str,
    config: str,
    split: str,
    *,
    on_progress: OnDownloadProgress | None = None,
) -> list[dict[str, Any]] | None:
    """Try loading a dataset via the ``datasets`` library (fast path).

    Downloads directly from the HuggingFace git repo — no datasets-server
    API, no 429 rate limits.  Returns ``None`` if the ``datasets`` library
    is not installed, letting the caller fall back to the REST API.

    Install with: ``pip install tool-eval-bench[hf]``
    """
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError:
        return None

    logger.info(
        "Using 'datasets' library for %s/%s %s (fast path, no rate limits)",
        dataset,
        config,
        split,
    )

    try:
        ds = load_dataset(dataset, config, split=split, trust_remote_code=False)
    except Exception as exc:
        logger.warning(
            "datasets library failed for %s/%s %s: %s — falling back to REST API",
            dataset,
            config,
            split,
            exc,
        )
        return None

    total = len(ds)
    rows: list[dict[str, Any]] = []
    for i, item in enumerate(ds):
        # datasets returns dict-like objects; convert to plain dict
        rows.append(dict(item))
        if on_progress and (i + 1) % 100 == 0:
            on_progress(i + 1, total)

    if on_progress:
        on_progress(total, total)

    return rows
