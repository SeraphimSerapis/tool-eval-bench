"""Application services that compose domain logic with concrete adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tool_eval_bench.application.service import BenchmarkService

__all__ = ["BenchmarkService"]


def __getattr__(name: str) -> Any:
    if name == "BenchmarkService":
        from tool_eval_bench.application.service import BenchmarkService

        return BenchmarkService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
