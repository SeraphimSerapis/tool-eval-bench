"""MMLU dataset loader — download, cache, and parse from HuggingFace API.

Downloads the test split of ``cais/mmlu`` (all config) via the HuggingFace
Datasets Server REST API (no ``datasets`` library required).  Results
are cached locally under ``data/mmlu/test.jsonl``.

Also downloads the ``dev`` split (5 examples per subject) for few-shot
prompting, cached to ``data/mmlu/dev.jsonl``.
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
_DATASET = "cais/mmlu"
_CONFIG = "all"
_PAGE_SIZE = 100
# Commit of the HuggingFace dataset repository that ``datasets`` downloads are
# pinned to.  The REST fallback cannot be pinned (see ``hf_utils``).
_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"

_CACHE_DIR = Path("data") / "mmlu"
_CACHE_TEST = _CACHE_DIR / "test.jsonl"
_CACHE_DEV = _CACHE_DIR / "dev.jsonl"

OnDownloadProgress = Callable[[int, int], None]

# 57 MMLU subjects → 4 categories
SUBJECT_CATEGORIES: dict[str, str] = {
    # STEM
    "abstract_algebra": "STEM",
    "anatomy": "STEM",
    "astronomy": "STEM",
    "college_biology": "STEM",
    "college_chemistry": "STEM",
    "college_computer_science": "STEM",
    "college_mathematics": "STEM",
    "college_physics": "STEM",
    "computer_security": "STEM",
    "conceptual_physics": "STEM",
    "electrical_engineering": "STEM",
    "elementary_mathematics": "STEM",
    "high_school_biology": "STEM",
    "high_school_chemistry": "STEM",
    "high_school_computer_science": "STEM",
    "high_school_mathematics": "STEM",
    "high_school_physics": "STEM",
    "high_school_statistics": "STEM",
    "machine_learning": "STEM",
    # Humanities
    "formal_logic": "Humanities",
    "high_school_european_history": "Humanities",
    "high_school_us_history": "Humanities",
    "high_school_world_history": "Humanities",
    "international_law": "Humanities",
    "jurisprudence": "Humanities",
    "logical_fallacies": "Humanities",
    "moral_disputes": "Humanities",
    "moral_scenarios": "Humanities",
    "philosophy": "Humanities",
    "prehistory": "Humanities",
    "professional_law": "Humanities",
    "world_religions": "Humanities",
    # Social Sciences
    "econometrics": "Social Sciences",
    "high_school_geography": "Social Sciences",
    "high_school_government_and_politics": "Social Sciences",
    "high_school_macroeconomics": "Social Sciences",
    "high_school_microeconomics": "Social Sciences",
    "high_school_psychology": "Social Sciences",
    "human_sexuality": "Social Sciences",
    "professional_psychology": "Social Sciences",
    "public_relations": "Social Sciences",
    "security_studies": "Social Sciences",
    "sociology": "Social Sciences",
    "us_foreign_policy": "Social Sciences",
    # Other
    "business_ethics": "Other",
    "clinical_knowledge": "Other",
    "college_medicine": "Other",
    "global_facts": "Other",
    "human_aging": "Other",
    "management": "Other",
    "marketing": "Other",
    "medical_genetics": "Other",
    "miscellaneous": "Other",
    "nutrition": "Other",
    "professional_accounting": "Other",
    "professional_medicine": "Other",
    "virology": "Other",
}

CATEGORIES = sorted(set(SUBJECT_CATEGORIES.values()))


@dataclass(slots=True)
class MMLUItem:
    """A single MMLU question."""

    index: int
    question: str
    subject: str
    choices: list[str]  # Always 4 items
    answer: int  # 0-3 (index into choices)

    @property
    def answer_letter(self) -> str:
        return "ABCD"[self.answer]

    @property
    def category(self) -> str:
        return SUBJECT_CATEGORIES.get(self.subject, "Other")

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "question": self.question,
            "subject": self.subject,
            "choices": self.choices,
            "answer": self.answer,
            "answer_letter": self.answer_letter,
        }


def _find_cache_file(split: str = "test") -> Path:
    return _CACHE_TEST if split == "test" else _CACHE_DEV


def _save_to_cache(path: Path, items: list[MMLUItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            row = {
                "index": item.index,
                "question": item.question,
                "subject": item.subject,
                "choices": item.choices,
                "answer": item.answer,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("Cached %d MMLU items to %s", len(items), path)


def _load_from_cache(path: Path) -> list[MMLUItem]:
    items: list[MMLUItem] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            items.append(
                MMLUItem(
                    index=row["index"],
                    question=row["question"],
                    subject=row["subject"],
                    choices=row["choices"],
                    answer=row["answer"],
                )
            )
    return items


def _rows_to_items(rows: list[dict]) -> list[MMLUItem]:
    """Convert raw row dicts to MMLUItem objects."""
    items: list[MMLUItem] = []
    for row in rows:
        items.append(
            MMLUItem(
                index=len(items),
                question=row["question"],
                subject=row["subject"],
                choices=list(row["choices"]),
                answer=int(row["answer"]),
            )
        )
    return items


def _download_dataset(
    split: str = "test",
    on_progress: OnDownloadProgress | None = None,
) -> tuple[list[MMLUItem], str]:
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
        split,
        revision=_REVISION,
        on_progress=on_progress,
    )
    if rows is not None:
        return _rows_to_items(rows), "datasets"

    # Fallback: REST API with resume support
    partial_path = _CACHE_DIR / f"{split}.partial.jsonl"
    rows = download_rows_paginated(
        _DATASET,
        _CONFIG,
        split,
        page_size=_PAGE_SIZE,
        partial_path=partial_path,
        on_progress=on_progress,
    )
    return _rows_to_items(rows), "rest_api"


def load_dataset(
    split: str = "test",
    *,
    force_download: bool = False,
    on_progress: OnDownloadProgress | None = None,
) -> list[MMLUItem]:
    """Load MMLU items, downloading and caching if needed."""
    cache_path = _find_cache_file(split)

    if cache_path.exists() and not force_download:
        logger.info("Loading MMLU %s from cache: %s", split, cache_path)
        return _load_from_cache(cache_path)

    logger.info("Downloading MMLU %s split from HuggingFace…", split)
    items, method = _download_dataset(split=split, on_progress=on_progress)
    logger.info("Downloaded %d items via %s", len(items), method)

    # Save to final cache and clean up any partial file
    _save_to_cache(cache_path, items)
    write_cache_manifest(
        cache_path,
        dataset=_DATASET,
        config=_CONFIG,
        split=split,
        method=method,
        revision=_REVISION,
    )
    partial_path = _CACHE_DIR / f"{split}.partial.jsonl"
    partial_path.unlink(missing_ok=True)

    return items


def dataset_revision() -> str:
    """Revision behind the cached test split, or ``"unknown"`` when it was not pinned."""
    return read_cache_revision(_find_cache_file("test"))
