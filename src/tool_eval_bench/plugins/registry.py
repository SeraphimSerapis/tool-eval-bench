"""Plugin registry — maps short names to BenchmarkPlugin instances."""

from __future__ import annotations

from tool_eval_bench.domain.plugin import BenchmarkPlugin


def _load_builtin_plugins() -> dict[str, type[BenchmarkPlugin]]:
    """Lazily import built-in plugins to avoid circular imports."""
    from tool_eval_bench.plugins.gsm8k.plugin import GSM8KPlugin
    from tool_eval_bench.plugins.ifeval.plugin import IFEvalPlugin
    from tool_eval_bench.plugins.mmlu.plugin import MMLUPlugin
    from tool_eval_bench.plugins.needle.plugin import NeedlePlugin

    return {
        "gsm8k": GSM8KPlugin,
        "ifeval": IFEvalPlugin,
        "mmlu": MMLUPlugin,
        "needle": NeedlePlugin,
    }


def get_plugin(name: str, **kwargs) -> BenchmarkPlugin:
    """Look up a benchmark plugin by short name and return an instance.

    Raises ``KeyError`` if the name is unknown.
    """
    registry = _load_builtin_plugins()
    cls = registry.get(name)
    if cls is None:
        available = ", ".join(sorted(registry))
        raise KeyError(f"Unknown benchmark plugin: {name!r}. Available: {available}")
    return cls(**kwargs)


def available_plugins() -> list[str]:
    """Return the list of registered plugin names."""
    return sorted(_load_builtin_plugins())
