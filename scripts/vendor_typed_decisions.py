#!/usr/bin/env python3
"""Vendor the typed-decisions test split into the decision plugin.

    uv run --no-project --with pyarrow python scripts/vendor_typed_decisions.py

Downloads one pinned parquet file from Hugging Face, checks its sha256, keeps
the columns the benchmark sends or scores (``id``, ``workflow``, ``state``,
``questions``, ``gold``), decodes their JSON strings into objects, and writes
gzipped JSONL plus a manifest the loader verifies. Row content is otherwise
unchanged. Only the ``test`` split is vendored: several published models were
fine-tuned on ``train``, so it is never benchmark material.

pyarrow is needed only here, so it stays out of the project's dependencies.
The downloaded file is untrusted input: every field is type-checked before it
is written, and nothing in it is executed or used as a path.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

DATASET = "LocalLLaMA/typed-decisions"
REVISION = "e039ebffcc280174dd354227424fb2b249f191de"
SOURCE_PATH = "all/test-00000-of-00001.parquet"
# The Hub's LFS object id for SOURCE_PATH at REVISION, which is its sha256.
SOURCE_SHA256 = "4f294f218ea1da27f3efef936359389c62ea4d3973a41457732990f1d31b647c"
SOURCE_URL = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{SOURCE_PATH}"
USER_AGENT = "OpenAI File Downloader, XaiImageApiFetch/1.0"

OUT_DIR = (
    Path(__file__).resolve().parent.parent
    / "src/tool_eval_bench/plugins/decision/vendor/typed_decisions"
)
DATA_NAME = "test.jsonl.gz"
MANIFEST_NAME = "manifest.json"

KEPT_COLUMNS = ("id", "workflow", "state", "questions", "gold")
JSON_COLUMNS = ("state", "questions", "gold")


def convert_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep ``KEPT_COLUMNS`` and decode the JSON-string columns.

    Raises ``ValueError`` on a row from another split or with an unexpected type,
    so a changed upstream cannot slip into the vendored file.
    """
    converted = []
    for row in rows:
        if row.get("split") != "test":
            raise ValueError(f"row {row.get('id')!r} is not from the test split")
        out: dict[str, Any] = {}
        for column in KEPT_COLUMNS:
            value = row.get(column)
            if not isinstance(value, str):
                raise ValueError(f"row {row.get('id')!r}: {column} is not a string")
            out[column] = json.loads(value) if column in JSON_COLUMNS else value
        if not isinstance(out["questions"], dict) or not isinstance(out["gold"], dict):
            raise ValueError(f"row {out['id']!r}: questions and gold must be JSON objects")
        converted.append(out)
    return converted


def encode_jsonl(rows: list[dict[str, Any]]) -> bytes:
    # Key order is preserved, not sorted: a choice question's option order is
    # part of the request.
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")


def gzip_deterministic(data: bytes) -> bytes:
    buffer = io.BytesIO()
    # A fixed mtime and no file name keep regeneration byte-stable on one zlib.
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, compresslevel=9, mtime=0) as gz:
        gz.write(data)
    return buffer.getvalue()


def download() -> bytes:
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": USER_AGENT})  # noqa: S310
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        body: bytes = response.read()
    digest = hashlib.sha256(body).hexdigest()
    if digest != SOURCE_SHA256:
        raise SystemExit(
            f"{SOURCE_PATH}: sha256 {digest} does not match the pinned {SOURCE_SHA256}"
        )
    return body


def main() -> int:
    import pyarrow.parquet as pq  # only available under `uv run --with pyarrow`

    source = download()
    rows = convert_rows(pq.read_table(io.BytesIO(source)).to_pylist())
    jsonl = encode_jsonl(rows)
    compressed = gzip_deterministic(jsonl)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / DATA_NAME).write_bytes(compressed)
    manifest = {
        "generated_by": "scripts/vendor_typed_decisions.py (do not edit by hand)",
        "dataset": DATASET,
        "url": f"https://huggingface.co/datasets/{DATASET}",
        "revision": REVISION,
        "license": "Apache-2.0",
        "split": "test",
        "source": {"path": SOURCE_PATH, "sha256": SOURCE_SHA256},
        "data": {
            "path": DATA_NAME,
            "rows": len(rows),
            "columns": list(KEPT_COLUMNS),
            # Hash of the decompressed JSONL, so a different zlib build cannot
            # change it.
            "jsonl_sha256": hashlib.sha256(jsonl).hexdigest(),
        },
    }
    (OUT_DIR / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"{len(rows)} rows: {len(jsonl):,} bytes JSONL, {len(compressed):,} bytes gzipped "
        f"-> {OUT_DIR / DATA_NAME}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
