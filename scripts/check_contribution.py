"""Enforce mechanical pull-request contribution requirements."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

CHANGELOG_TYPES = ("added", "changed", "fixed", "removed", "security")
FRAGMENT_NAME = re.compile(
    rf"^(?:[0-9]+|\+[a-z0-9][a-z0-9-]*)\.(?:{'|'.join(CHANGELOG_TYPES)})\.md$"
)
RUNTIME_FILES = {
    ".env.example",
    "Dockerfile",
    "compose.yaml",
    "docker-compose.yml",
    "pyproject.toml",
    "uv.lock",
}


def _pull_request(event: dict[str, Any]) -> dict[str, Any]:
    pull_request = event.get("pull_request")
    if not isinstance(pull_request, dict):
        raise ValueError("event does not contain a pull_request object")
    return pull_request


def _labels(pull_request: dict[str, Any]) -> set[str]:
    labels = pull_request.get("labels", [])
    return {
        label["name"]
        for label in labels
        if isinstance(label, dict) and isinstance(label.get("name"), str)
    }


def _repo_name(ref: object) -> str | None:
    if not isinstance(ref, dict):
        return None
    repo = ref.get("repo")
    if not isinstance(repo, dict):
        return None
    name = repo.get("full_name")
    return name if isinstance(name, str) else None


def _is_runtime_file(path: str) -> bool:
    return path.startswith("src/") or path in RUNTIME_FILES


def _is_production_python(path: str) -> bool:
    return path.startswith("src/") and path.endswith(".py")


def _is_test_file(path: str) -> bool:
    return path.startswith("tests/")


def _added_fragments(added_files: set[str]) -> list[str]:
    return sorted(
        path
        for path in added_files
        if path.startswith("changelog.d/") and path != "changelog.d/README.md"
    )


def evaluate_policy(
    *,
    event: dict[str, Any],
    changed_files: set[str],
    added_files: set[str],
    fragment_contents: dict[str, str | None],
) -> list[str]:
    """Return actionable policy failures for one pull request."""
    pull_request = _pull_request(event)
    labels = _labels(pull_request)
    errors: list[str] = []

    head_repo = _repo_name(pull_request.get("head"))
    base_repo = _repo_name(pull_request.get("base"))
    if head_repo and base_repo and head_repo != base_repo:
        if not pull_request.get("maintainer_can_modify", False):
            errors.append(
                "Fork pull requests must enable 'Allow edits from maintainers'. Enable it in "
                "the pull request sidebar, then rerun this check."
            )

    if "tests-not-needed" not in labels:
        production_changed = any(_is_production_python(path) for path in changed_files)
        tests_changed = any(_is_test_file(path) for path in changed_files)
        if production_changed and not tests_changed:
            errors.append(
                "Production Python changed without a tests/** change. Add regression coverage "
                "or ask a maintainer to apply the tests-not-needed label."
            )

    if "skip-changelog" not in labels:
        if "CHANGELOG.md" in changed_files:
            errors.append(
                "CHANGELOG.md is generated. Revert that edit and add a changelog.d/ fragment "
                "instead."
            )

        fragments = _added_fragments(added_files)
        for path in fragments:
            name = Path(path).name
            if not FRAGMENT_NAME.fullmatch(name):
                errors.append(
                    f"Invalid changelog fragment name: {path}. Use "
                    "<issue-or-PR-number>.<type>.md or +<slug>.<type>.md."
                )
                continue
            content = fragment_contents.get(path)
            if content is None or not content.strip():
                errors.append(f"Changelog fragment is empty: {path}.")

        runtime_changed = any(_is_runtime_file(path) for path in changed_files)
        valid_fragments = [path for path in fragments if FRAGMENT_NAME.fullmatch(Path(path).name)]
        if runtime_changed and not valid_fragments:
            errors.append(
                "Runtime or packaging files changed without a changelog fragment. Add "
                "changelog.d/<issue-or-+slug>.<type>.md or ask a maintainer to apply the "
                "skip-changelog label."
            )

    return errors


def _git_files(root: Path, base_sha: str, head_sha: str, *, diff_filter: str) -> set[str]:
    # Git receives separate arguments and never invokes a shell. The commit IDs
    # and paths may come from the PR event, but they cannot become commands.
    try:
        result = subprocess.run(  # noqa: S603
            [
                "git",
                "diff",
                "--name-only",
                "-z",
                f"--diff-filter={diff_filter}",
                f"{base_sha}...{head_sha}",
                "--",
            ],
            cwd=root,
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or b"").decode("utf-8", "replace").strip()
        raise SystemExit(
            f"Cannot diff {base_sha}...{head_sha}: {detail or 'git diff failed'}. "
            "Fetch the pull-request commits first "
            "(git fetch origin '+refs/pull/<number>/head') and rerun."
        ) from exc
    return {path.decode("utf-8") for path in result.stdout.split(b"\0") if path}


def _git_file_text(root: Path, head_sha: str, path: str) -> str | None:
    result = subprocess.run(  # noqa: S603
        ["git", "show", f"{head_sha}:{path}"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    try:
        return result.stdout.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _sha(pull_request: dict[str, Any], side: str) -> str:
    ref = pull_request.get(side)
    if not isinstance(ref, dict) or not isinstance(ref.get("sha"), str):
        raise ValueError(f"pull_request.{side}.sha is missing")
    return ref["sha"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--event-path",
        type=Path,
        default=Path(os.environ.get("GITHUB_EVENT_PATH", "")),
        help="GitHub pull_request event JSON",
    )
    parser.add_argument("--base-sha", help="Override the event base commit")
    parser.add_argument("--head-sha", help="Override the event head commit")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.event_path.is_file():
        raise SystemExit("--event-path must name a pull_request event JSON file")

    event = json.loads(args.event_path.read_text(encoding="utf-8"))
    pull_request = _pull_request(event)
    if pull_request.get("state", "open") != "open":
        print("Pull request is closed; contributor policy check skipped.")
        return 0
    base_sha = args.base_sha or _sha(pull_request, "base")
    head_sha = args.head_sha or _sha(pull_request, "head")
    root = args.root.resolve()
    changed_files = _git_files(root, base_sha, head_sha, diff_filter="ACDMR")
    added_files = _git_files(root, base_sha, head_sha, diff_filter="A")
    fragments = _added_fragments(added_files)
    errors = evaluate_policy(
        event=event,
        changed_files=changed_files,
        added_files=added_files,
        fragment_contents={path: _git_file_text(root, head_sha, path) for path in fragments},
    )

    if errors:
        print("Contributor policy failed:")
        for error in errors:
            print(f"  - {error}")
        print("See CONTRIBUTING.md#pull-request-policy for details.")
        return 1

    print("Contributor policy passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
