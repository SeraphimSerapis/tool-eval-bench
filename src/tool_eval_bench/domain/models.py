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
