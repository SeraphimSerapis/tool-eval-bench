"""Headless error codes for machine-readable JSONL error events.

These constants are the canonical error codes emitted in ``--json`` mode
when the benchmark fails before scenario execution begins.  External
integrators (e.g. sparkrun) can exhaustively match on these values.

Each constant maps to a specific exit code:
    - Exit 1: runtime error, generic failure
    - Exit 2: connection/HTTP error (server unreachable or bad response)
    - Exit 3: server responded but has no models loaded

Usage in CLI:
    ``_headless_error(NO_SERVER, "...", exit_code=2)``
"""

from __future__ import annotations

# -- Connection errors (exit code 2) ----------------------------------------
CONNECTION_FAILED = "connection_failed"
"""Server is unreachable (TCP connection refused, DNS failure, timeout)."""

HTTP_ERROR = "http_error"
"""Server responded with an HTTP error status (4xx, 5xx)."""

DETECTION_FAILED = "detection_failed"
"""Server probing failed during auto-discovery (unexpected exception)."""

INVALID_RESPONSE = "invalid_response"
"""Server returned a response that could not be parsed as expected JSON."""

# -- Model errors (exit code 3) ---------------------------------------------
NO_MODELS = "no_models"
"""Server responded but the model list is empty — no models loaded."""

MODEL_NOT_AVAILABLE = "model_not_available"
"""Model is listed in /v1/models but returns an error (4xx/5xx) when actually
used for inference.  This catches the case where a model appears loaded but
fails on real requests — without this check the benchmark silently produces
misleading pass/partial/fail scores (issue #19)."""

# -- Discovery errors (exit code 1) -----------------------------------------
NO_SERVER = "no_server"
"""Auto-discovery found no responsive inference server on localhost."""
