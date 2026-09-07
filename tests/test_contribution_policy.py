"""Tests for the pull-request contribution policy."""

from __future__ import annotations

import json
import subprocess

import pytest

from scripts.check_contribution import _git_files, evaluate_policy, main


def _event(
    *,
    fork: bool = False,
    maintainer_can_modify: bool = True,
    labels: tuple[str, ...] = (),
) -> dict:
    head_repo = "contributor/tool-eval-bench" if fork else "owner/tool-eval-bench"
    return {
        "pull_request": {
            "maintainer_can_modify": maintainer_can_modify,
            "head": {"repo": {"full_name": head_repo}},
            "base": {"repo": {"full_name": "owner/tool-eval-bench"}},
            "labels": [{"name": label} for label in labels],
        }
    }


def _fragment(name: str = "123.fixed.md") -> str:
    return f"changelog.d/{name}"


def test_source_change_requires_test_and_changelog() -> None:
    errors = evaluate_policy(
        event=_event(),
        changed_files={"src/tool_eval_bench/runner/service.py"},
        added_files=set(),
        fragment_contents={},
    )

    assert errors == [
        "Production Python changed without a tests/** change. Add regression coverage or ask "
        "a maintainer to apply the tests-not-needed label.",
        "Runtime or packaging files changed without a changelog fragment. Add "
        "changelog.d/<issue-or-+slug>.<type>.md or ask a maintainer to apply the "
        "skip-changelog label.",
    ]


def test_source_change_with_required_artifacts_passes() -> None:
    fragment = _fragment()

    errors = evaluate_policy(
        event=_event(),
        changed_files={
            "src/tool_eval_bench/runner/service.py",
            "tests/test_service.py",
            fragment,
        },
        added_files={fragment},
        fragment_contents={fragment: "Fixed it.\n"},
    )

    assert errors == []


def test_changelog_fragment_must_be_new_valid_and_nonempty() -> None:
    invalid = _fragment("notes.md")
    empty = _fragment("124.fixed.md")

    errors = evaluate_policy(
        event=_event(),
        changed_files={invalid, empty},
        added_files={invalid, empty},
        fragment_contents={invalid: "Notes.\n", empty: " \n"},
    )

    assert errors == [
        "Changelog fragment is empty: changelog.d/124.fixed.md.",
        "Invalid changelog fragment name: changelog.d/notes.md. Use "
        "<issue-or-PR-number>.<type>.md or +<slug>.<type>.md.",
    ]


def test_generated_changelog_edit_is_rejected() -> None:
    errors = evaluate_policy(
        event=_event(),
        changed_files={"CHANGELOG.md"},
        added_files=set(),
        fragment_contents={},
    )

    assert errors == [
        "CHANGELOG.md is generated. Revert that edit and add a changelog.d/ fragment instead."
    ]


def test_maintainer_labels_allow_deliberate_exceptions() -> None:
    errors = evaluate_policy(
        event=_event(labels=("tests-not-needed", "skip-changelog")),
        changed_files={"src/tool_eval_bench/domain/models.py", "CHANGELOG.md"},
        added_files=set(),
        fragment_contents={},
    )

    assert errors == []


def test_fork_pull_request_requires_maintainer_edits() -> None:
    errors = evaluate_policy(
        event=_event(fork=True, maintainer_can_modify=False),
        changed_files={"docs/example.md"},
        added_files=set(),
        fragment_contents={},
    )

    assert errors == [
        "Fork pull requests must enable 'Allow edits from maintainers'. Enable it in the pull "
        "request sidebar, then rerun this check."
    ]


def test_fork_with_maintainer_edits_and_same_repo_branch_pass() -> None:
    fork_errors = evaluate_policy(
        event=_event(fork=True, maintainer_can_modify=True),
        changed_files={"docs/example.md"},
        added_files=set(),
        fragment_contents={},
    )
    same_repo_errors = evaluate_policy(
        event=_event(fork=False, maintainer_can_modify=False),
        changed_files={"docs/example.md"},
        added_files=set(),
        fragment_contents={},
    )

    assert fork_errors == []
    assert same_repo_errors == []


def test_packaging_change_requires_changelog_but_not_python_test() -> None:
    errors = evaluate_policy(
        event=_event(),
        changed_files={"pyproject.toml"},
        added_files=set(),
        fragment_contents={},
    )

    assert errors == [
        "Runtime or packaging files changed without a changelog fragment. Add "
        "changelog.d/<issue-or-+slug>.<type>.md or ask a maintainer to apply the "
        "skip-changelog label."
    ]


def test_closed_pull_request_skips_diff(tmp_path, capsys) -> None:
    event_path = tmp_path / "event.json"
    event_path.write_text(
        json.dumps({"pull_request": {"state": "closed", "labels": []}}),
        encoding="utf-8",
    )

    assert main(["--event-path", str(event_path), "--root", str(tmp_path)]) == 0
    assert "skipped" in capsys.readouterr().out


def test_git_files_missing_sha_reports_git_error(tmp_path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)

    with pytest.raises(SystemExit, match="Cannot diff"):
        _git_files(tmp_path, "0" * 40, "1" * 40, diff_filter="ACDMR")
