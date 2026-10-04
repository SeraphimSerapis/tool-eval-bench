"""Validation for a user-supplied system prompt override.

One rule for every entry point (CLI, ``BenchmarkService``, the Python API), so the
text that is persisted, fingerprinted, and sent to the model is the same text no
matter how it arrived.
"""

from __future__ import annotations

#: Upper bound on the override, in UTF-8 bytes.  The text is stored verbatim in the
#: run config and the run metadata, so an unbounded file (a pasted log, a secrets
#: file, a binary) would become durable run data.  The bundled examples are ~5 KiB.
MAX_SYSTEM_PROMPT_BYTES = 32 * 1024


def normalize_system_prompt(text: str) -> str:
    """Return ``text`` in the canonical form that is persisted and sent.

    A leading byte-order mark and trailing whitespace carry no meaning for the
    model but would split cohorts: the same words read from a file (which ends in
    a newline) and passed inline (which does not) must fingerprint identically.

    Raises:
        ValueError: if the prompt is blank, is not encodable as UTF-8, or exceeds
            ``MAX_SYSTEM_PROMPT_BYTES``.
    """
    cleaned = text.lstrip("\ufeff").rstrip()
    if not cleaned.strip():
        raise ValueError("system prompt must not be empty")
    try:
        size = len(cleaned.encode("utf-8"))
    except UnicodeEncodeError:
        raise ValueError("system prompt is not valid UTF-8") from None
    if size > MAX_SYSTEM_PROMPT_BYTES:
        raise ValueError(f"system prompt exceeds the {MAX_SYSTEM_PROMPT_BYTES}-byte limit")
    return cleaned
