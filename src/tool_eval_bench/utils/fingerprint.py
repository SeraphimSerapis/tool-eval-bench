"""Persist-safe run config with its comparison fingerprint.

Kept out of the application layer so the CLI helpers can import it without
loading the benchmark service, httpx, and SQLite.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tool_eval_bench.utils.ids import build_config_fingerprint
from tool_eval_bench.utils.urls import endpoint_identity, redact_url

#: Deployment facts that make two runs comparable.  A change in any of them puts
#: a run in a different cohort, so they are folded into the fingerprint.
COMPARISON_METADATA_KEYS = (
    "server_model_id",
    "server_model_root",
    "engine_name",
    "engine_version",
    "max_model_len",
    "quantization",
    "gpu_count",
    "slot_count",
    "spec_decoding",
)


def comparison_fingerprint(config: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    """Fingerprint a run's comparable config together with its code and deployment.

    The fingerprint answers "are these two runs comparable?".  Scenarios,
    evaluators and measurement loops are code, so two runs from different
    commits are not comparable even when every flag matches, and neither are
    runs against different engines or quantizations.  Every run type uses this
    one payload shape; changing it re-cohorts every historical run.
    """
    from tool_eval_bench import __version__

    deployment = {
        key: metadata.get(key) for key in COMPARISON_METADATA_KEYS if metadata.get(key) is not None
    }
    return build_config_fingerprint(
        {
            "config": dict(config),
            "deployment": deployment,
            "tool_version": __version__,
            "git_sha": metadata.get("git_sha"),
        }
    )


def with_config_fingerprint(
    config: Mapping[str, Any], metadata: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Return persist-safe config with a stable comparison fingerprint.

    Plugin runs receive the real endpoint so requests can authenticate, but the
    returned mapping is written to SQLite and reports.  Do not retain endpoint
    hosts, URL userinfo, or query-string credentials there.  The opaque endpoint
    identity keeps distinct deployments from being accidentally compared while
    deliberately ignoring credentials and ephemeral query parameters.

    ``metadata`` is the run's ``RunContext.to_dict()``; its code identity and
    deployment facts join the fingerprint exactly as they do for scored runs.
    """
    persisted = dict(config)
    fingerprint_config = dict(config)
    base_url = config.get("base_url")
    if isinstance(base_url, str):
        persisted["base_url"] = redact_url(base_url)
        fingerprint_config["base_url"] = endpoint_identity(base_url)
    return {
        **persisted,
        "config_fingerprint": comparison_fingerprint(fingerprint_config, metadata or {}),
    }
