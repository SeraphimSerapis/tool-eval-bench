"""Persist-safe run config with its comparison fingerprint.

Kept out of the application layer so the CLI helpers can import it without
loading the benchmark service, httpx, and SQLite.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tool_eval_bench.utils.ids import build_config_fingerprint
from tool_eval_bench.utils.urls import endpoint_identity, redact_url


def with_config_fingerprint(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return persist-safe config with a stable comparison fingerprint.

    Plugin runs receive the real endpoint so requests can authenticate, but the
    returned mapping is written to SQLite and reports.  Do not retain endpoint
    hosts, URL userinfo, or query-string credentials there.  The opaque endpoint
    identity keeps distinct deployments from being accidentally compared while
    deliberately ignoring credentials and ephemeral query parameters.
    """
    persisted = dict(config)
    fingerprint_config = dict(config)
    base_url = config.get("base_url")
    if isinstance(base_url, str):
        persisted["base_url"] = redact_url(base_url)
        fingerprint_config["base_url"] = endpoint_identity(base_url)
    return {
        **persisted,
        "config_fingerprint": build_config_fingerprint(fingerprint_config),
    }
