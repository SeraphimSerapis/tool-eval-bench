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

# Finer than the layer table: a layer may reach a package while still being
# barred from one module inside it. Keyed by the importing layer.
FORBIDDEN_MODULES: dict[str, frozenset[str]] = {
    # The delivery layer formats reports (storage.reports is a renderer), but
    # must not open the database. Connection lifetime, query shape, and the
    # write-then-persist ordering are the application layer's to own; a CLI
    # module that opens its own repository leaks a WAL connection on every
    # early return, which is exactly how it went wrong before.
    "cli": frozenset({"tool_eval_bench.storage.db"}),
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


@pytest.mark.parametrize("layer", sorted(FORBIDDEN_MODULES))
def test_a_layer_cannot_reach_a_module_barred_to_it(layer: str) -> None:
    barred = FORBIDDEN_MODULES[layer]
    violations = [
        f"{path.relative_to(PACKAGE_ROOT).as_posix()}: {imported}"
        for path in sorted((PACKAGE_ROOT / layer).rglob("*.py"))
        for imported in _internal_imports(path)
        if imported in barred
    ]

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


def test_no_test_module_patches_methods_onto_the_httpx_client_class() -> None:
    """Attaching to `httpx.AsyncClient` itself would silently shadow a future release.

    `tests/conftest.py` used to do this at import time for the whole session,
    so an httpx version adding a same-named method would have been overridden
    everywhere with nothing pointing at the cause.
    """
    tests_root = Path(__file__).parent
    violations = [
        f"{path.relative_to(tests_root)}:{node.lineno}"
        for path in sorted(tests_root.rglob("*.py"))
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Attribute)
        and target.value.attr == "AsyncClient"
    ]

    assert violations == []


def test_no_module_reaches_for_a_private_argparse_action_class() -> None:
    """`argparse._StoreTrueAction` and friends are absent from `argparse.__all__`.

    `cli/parser.py` used to branch on them to decide how to recreate a flag in
    a focused help parser. `Action.const` is documented and says the same
    thing, so nothing needs the private classes any more.
    """
    violations = [
        f"{path.relative_to(PACKAGE_ROOT).as_posix()}:{node.lineno}"
        for path in sorted(PACKAGE_ROOT.rglob("*.py"))
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "argparse"
        and node.attr.startswith("_")
    ]

    assert violations == []
