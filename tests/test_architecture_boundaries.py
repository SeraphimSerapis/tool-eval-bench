"""Regression tests for the repository's layered import boundaries."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tool_eval_bench.adapters.base import (
    BackendAdapter as LegacyBackendAdapter,
)
from tool_eval_bench.adapters.base import (
    ChatCompletionResult as LegacyChatCompletionResult,
)
from tool_eval_bench.adapters.base import (
    ProviderToolCall as LegacyProviderToolCall,
)
from tool_eval_bench.application.service import BenchmarkService as ApplicationBenchmarkService
from tool_eval_bench.domain.adapters import (
    BackendAdapter,
    ChatCompletionResult,
    ProviderToolCall,
)
from tool_eval_bench.runner.service import BenchmarkService as LegacyBenchmarkService

PACKAGE_ROOT = Path(__file__).parents[1] / "src" / "tool_eval_bench"

# These rules mirror docs/architecture.md. Entries include the current layer
# because imports between modules in the same package are always allowed.
LAYER_IMPORT_RULES: dict[str, frozenset[str]] = {
    "domain": frozenset({"domain"}),
    "evals": frozenset({"domain", "evals"}),
    "runner": frozenset({"domain", "evals", "runner", "utils"}),
    "plugins": frozenset({"domain", "plugins"}),
    "storage": frozenset({"domain", "storage"}),
    "application": frozenset(
        {"adapters", "application", "domain", "evals", "runner", "storage", "utils"}
    ),
    "adapters": frozenset({"adapters", "domain", "utils"}),
    "utils": frozenset({"domain", "utils"}),
    "cli": frozenset(
        {
            "adapters",
            "api",
            "application",
            "cli",
            "compare_reports",
            "domain",
            "evals",
            "plugins",
            "runner",
            "storage",
            "utils",
        }
    ),
}

# Compatibility modules may deliberately point against the normal dependency
# direction. Keep each exception exact so new imports cannot hide behind a
# broad package-level exemption.
COMPATIBILITY_IMPORT_ALLOWLIST: frozenset[tuple[str, str]] = frozenset(
    {("runner/service.py", "application")}
)


def _internal_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("tool_eval_bench"):
                imports.add(node.module)
        elif isinstance(node, ast.Import):
            imports.update(
                alias.name for alias in node.names if alias.name.startswith("tool_eval_bench")
            )
    return imports


@pytest.mark.parametrize("layer", sorted(LAYER_IMPORT_RULES))
def test_layer_import_boundaries_are_recursive_and_table_driven(layer: str) -> None:
    allowed_layers = LAYER_IMPORT_RULES[layer]
    violations: list[str] = []
    for path in sorted((PACKAGE_ROOT / layer).rglob("*.py")):
        relative_path = path.relative_to(PACKAGE_ROOT).as_posix()
        for imported in _internal_imports(path):
            parts = imported.split(".")
            imported_layer = parts[1] if len(parts) > 1 else ""
            if not imported_layer:
                continue
            if imported_layer in allowed_layers:
                continue
            if (relative_path, imported_layer) in COMPATIBILITY_IMPORT_ALLOWLIST:
                continue
            violations.append(f"{relative_path}: {imported}")

    assert violations == []


def test_legacy_adapter_contract_exports_preserve_identity() -> None:
    assert LegacyBackendAdapter is BackendAdapter
    assert LegacyChatCompletionResult is ChatCompletionResult
    assert LegacyProviderToolCall is ProviderToolCall


def test_legacy_runner_service_export_preserves_identity() -> None:
    assert LegacyBenchmarkService is ApplicationBenchmarkService


def test_first_party_dispatch_uses_application_service_directly() -> None:
    imports = _internal_imports(PACKAGE_ROOT / "cli" / "dispatch.py")

    assert "tool_eval_bench.application.service" in imports
    assert "tool_eval_bench.runner.service" not in imports
