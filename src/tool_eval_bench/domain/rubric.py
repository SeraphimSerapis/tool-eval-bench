"""Explicit rubric versions and AST fingerprint gate for scoring stability.

Leaderboard cohorts group runs by ``EVAL_RUBRIC_VERSION``, ``LOCALE_VERSION``,
and ``NOISE_VERSION`` instead of unrelated commits or package versions.

Updating ``RUBRIC_AST_FINGERPRINT``:
- When a change under ``evals/`` or ``runner/orchestrator.py`` alters scoring
  verdicts, localized prompts, or noise injection, bump the relevant version
  constant (``EVAL_RUBRIC_VERSION``, ``LOCALE_VERSION``, or ``NOISE_VERSION``)
  and refresh ``RUBRIC_AST_FINGERPRINT``.
- For strictly verdict-preserving refactors (renames, helper extractions,
  formatting, typing), update ``RUBRIC_AST_FINGERPRINT`` only and explain why
  the refactor preserves verdicts in the PR description.
- Docstring, comment, and whitespace edits are ignored by the AST normalizer
  and do not change ``RUBRIC_AST_FINGERPRINT``.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

EVAL_RUBRIC_VERSION: str = "1"
LOCALE_VERSION: str = "1"
NOISE_VERSION: str = "1"

# Pinned 16-hex SHA-256 digest of normalized ASTs across ``evals/**/*.py`` and
# ``runner/orchestrator.py``. Verified in CI by
# ``tests/test_rubric_cohorts.py::test_rubric_ast_fingerprint_matches_source``.
RUBRIC_AST_FINGERPRINT: str = "e17954df6945eaa7"

_DOCSTRING_OWNERS = (
    ast.Module,
    ast.ClassDef,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
)


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    """Remove module, class, and function docstrings in place."""
    for node in ast.walk(tree):
        if isinstance(node, _DOCSTRING_OWNERS):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body = node.body[1:]
    return tree


def _canonical_ast_dump(node: ast.AST) -> str:
    """Serialize *node* deterministically across Python 3.11+ versions.

    ``ast.dump(tree, include_attributes=False)`` changed its default formatting
    for empty fields between Python 3.11 and 3.13, and Python 3.12 added
    ``type_params=[]`` to definitions. Omitting attributes, empty lists, and
    unset optional fields matches Python 3.13's compact dump while producing
    identical output on Python 3.11, 3.12, and 3.13.
    """

    def _fmt(obj: object) -> str:
        if isinstance(obj, ast.AST):
            parts: list[str] = []
            for field in obj._fields:
                value = getattr(obj, field)
                if value is None and not (isinstance(obj, ast.Constant) and field == "value"):
                    continue
                if isinstance(value, list) and not value:
                    continue
                parts.append(f"{field}={_fmt(value)}")
            return f"{type(obj).__name__}({', '.join(parts)})"
        if isinstance(obj, list):
            return "[" + ", ".join(_fmt(item) for item in obj) + "]"
        return repr(obj)

    return _fmt(node)


def _rubric_source_paths(package_root: Path) -> list[Path]:
    """Return every rubric-governing Python file under *package_root* in stable order."""
    evals_dir = package_root / "evals"
    if not evals_dir.is_dir():
        raise FileNotFoundError(f"Expected evals directory at {evals_dir}")

    orchestrator_path = package_root / "runner" / "orchestrator.py"
    if not orchestrator_path.is_file():
        raise FileNotFoundError(f"Expected orchestrator module at {orchestrator_path}")

    paths = [*evals_dir.rglob("*.py"), orchestrator_path]
    return sorted(paths, key=lambda p: p.relative_to(package_root).as_posix())


def compute_rubric_ast_fingerprint(package_root: Path | None = None) -> str:
    """Compute a deterministic 16-hex SHA-256 digest of scoring and orchestration ASTs.

    Walks every ``.py`` file under ``evals/`` plus ``runner/orchestrator.py``,
    parses each file into an AST, strips module/class/function docstrings so
    comment, whitespace, and docstring edits never alter the digest, and hashes
    the normalized AST dump keyed by POSIX relative path.
    """
    root = (
        Path(package_root).resolve()
        if package_root is not None
        else Path(__file__).resolve().parent.parent
    )
    digest = hashlib.sha256()
    for path in _rubric_source_paths(root):
        relative = path.relative_to(root).as_posix()
        source = path.read_text(encoding="utf-8")
        tree = _strip_docstrings(ast.parse(source, filename=relative))
        dumped = _canonical_ast_dump(tree)
        digest.update(f"{relative}\n{dumped}\n".encode())
    return digest.hexdigest()[:16]
