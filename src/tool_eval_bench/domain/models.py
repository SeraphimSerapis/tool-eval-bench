"""Core configuration model for benchmark runs."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(slots=True)
class BenchmarkConfig:
    """Server connection and run configuration."""
    model: str
    backend: str
    base_url: str
    api_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("api_key", None)  # never persist credentials
        return d


@dataclass(slots=True)
class RunContext:
    """Full execution context for a benchmark run (issue #6).

    Three tiers of metadata:
      Tier 1 — Always available (local environment)
      Tier 2 — CLI parameters (explicit user choices)
      Tier 3 — Inference engine probe (best-effort, may be None)
    """

    # -- Tier 1: local environment (always available) --
    tool_version: str
    git_sha: str | None
    hostname: str
    platform_info: str
    python_version: str

    # -- Tier 2: CLI parameters --
    model: str
    backend: str
    base_url: str                       # redacted for reports
    temperature: float = 0.0
    max_turns: int = 8
    timeout_seconds: float = 60.0
    seed: int | None = None
    scenario_selector: str = "all"      # "all (69)" / "short (15)" / "TC-01,TC-07"
    trials: int = 1
    parallel: int = 1
    error_rate: float = 0.0
    thinking_enabled: bool = True
    extra_params: dict[str, Any] | None = None
    context_pressure: float | None = None

    # -- Tier 3: inference engine (best-effort) --
    server_model_id: str | None = None
    server_model_root: str | None = None
    engine_name: str | None = None      # "vLLM" / "llama.cpp" / "LiteLLM"
    engine_version: str | None = None   # e.g. "0.8.5"
    max_model_len: int | None = None
    quantization: str | None = None
    gpu_count: int | None = None
    spec_decoding: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict for metadata_json storage."""
        d = asdict(self)
        # Strip None values to keep the blob compact
        return {k: v for k, v in d.items() if v is not None}
