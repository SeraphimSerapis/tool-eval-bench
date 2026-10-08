"""Tests for the pull-request contribution policy."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from scripts.check_contribution import _git_files, evaluate_policy, main
from tool_eval_bench.utils.metadata import _GIT_REPOSITORY_ENV_VARS

MISSING_FRAGMENT = (
    "Runtime or packaging files changed without a changelog fragment. Add "
    "changelog.d/<issue-or-+slug>.<type>.md, update the unreleased fragment this change "
    "amends, or ask a maintainer to apply the skip-changelog label."
)


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
        written_files=set(),
        fragment_contents={},
    )

    assert errors == [
        "Production Python changed without a tests/** change. Add regression coverage or ask "
        "a maintainer to apply the tests-not-needed label.",
        MISSING_FRAGMENT,
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
        written_files={fragment},
        fragment_contents={fragment: "Fixed it.\n"},
    )

    assert errors == []


def test_changelog_fragment_must_be_valid_and_nonempty() -> None:
    invalid = _fragment("notes.md")
    empty = _fragment("124.fixed.md")

    errors = evaluate_policy(
        event=_event(),
        changed_files={invalid, empty},
        written_files={invalid, empty},
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
        written_files=set(),
        fragment_contents={},
    )

    assert errors == [
        "CHANGELOG.md is generated. Revert that edit and add a changelog.d/ fragment instead."
    ]


def test_maintainer_labels_allow_deliberate_exceptions() -> None:
    errors = evaluate_policy(
        event=_event(labels=("tests-not-needed", "skip-changelog")),
        changed_files={"src/tool_eval_bench/domain/models.py", "CHANGELOG.md"},
        written_files=set(),
        fragment_contents={},
    )

    assert errors == []


def test_fork_pull_request_requires_maintainer_edits() -> None:
    errors = evaluate_policy(
        event=_event(fork=True, maintainer_can_modify=False),
        changed_files={"docs/example.md"},
        written_files=set(),
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
        written_files=set(),
        fragment_contents={},
    )
    same_repo_errors = evaluate_policy(
        event=_event(fork=False, maintainer_can_modify=False),
        changed_files={"docs/example.md"},
        written_files=set(),
        fragment_contents={},
    )

    assert fork_errors == []
    assert same_repo_errors == []


def test_packaging_change_requires_changelog_but_not_python_test() -> None:
    errors = evaluate_policy(
        event=_event(),
        changed_files={"pyproject.toml"},
        written_files=set(),
        fragment_contents={},
    )

    assert errors == [MISSING_FRAGMENT]


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


@pytest.fixture(autouse=True)
def _isolated_from_outer_repository(monkeypatch: pytest.MonkeyPatch) -> None:
    # A pre-push hook in a linked worktree exports GIT_DIR. Inherited, it would
    # point these temporary repositories, and main()'s git calls, at the real one.
    for name in _GIT_REPOSITORY_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _git(repo: Path, *args: str) -> str:
    # Arguments are literals from this file; no shell and no outside input.
    result = subprocess.run(  # noqa: S603
        [
            "git",
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.com",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "--no-verify", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _policy_args_after(
    tmp_path: Path,
    change: Callable[[Path], object],
    *,
    release_on_base: bool = False,
) -> list[str]:
    """Build a base with an unreleased fragment, commit ``change`` on a branch, return argv.

    The branch always changes src/ and tests/, so only the changelog rule decides
    the result. With ``release_on_base`` the base branch moves on and deletes the
    fragment, as ``towncrier build`` does at release.
    """
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "changelog.d").mkdir()
    (repo / "src" / "feature.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "tests" / "test_feature.py").write_text("def test_it(): pass\n", encoding="utf-8")
    (repo / "changelog.d" / "README.md").write_text("How fragments work.\n", encoding="utf-8")
    (repo / "changelog.d" / "+feature.added.md").write_text(
        "Added the feature.\n", encoding="utf-8"
    )
    _git(repo, "init", "-q")
    fork_point = _commit(repo, "base")
    base = fork_point
    if release_on_base:
        (repo / "changelog.d" / "+feature.added.md").unlink()
        base = _commit(repo, "release")
        _git(repo, "checkout", "-q", fork_point)

    (repo / "src" / "feature.py").write_text("VALUE = 2\n", encoding="utf-8")
    (repo / "tests" / "test_feature.py").write_text("def test_it(): assert 1\n", encoding="utf-8")
    change(repo / "changelog.d")
    head = _commit(repo, "head")

    event = _event()
    event["pull_request"]["base"]["sha"] = base
    event["pull_request"]["head"]["sha"] = head
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps(event), encoding="utf-8")
    return ["--event-path", str(event_path), "--root", str(repo)]


def _edit_fragment(text: str) -> Callable[[Path], object]:
    return lambda changelog: (changelog / "+feature.added.md").write_text(text, encoding="utf-8")


def test_editing_an_unreleased_fragment_satisfies_the_changelog_rule(tmp_path, capsys) -> None:
    argv = _policy_args_after(tmp_path, _edit_fragment("Added the renamed feature.\n"))

    assert main(argv) == 0
    assert "Contributor policy passed." in capsys.readouterr().out


def test_renaming_a_fragment_counts_as_adding_it(tmp_path, capsys) -> None:
    def retype(changelog: Path) -> None:
        (changelog / "+feature.added.md").rename(changelog / "+feature.changed.md")

    assert main(_policy_args_after(tmp_path, retype)) == 0
    assert "Contributor policy passed." in capsys.readouterr().out


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(lambda changelog: None, id="untouched"),
        pytest.param(lambda changelog: (changelog / "+feature.added.md").unlink(), id="deleted"),
        pytest.param(_edit_fragment("Added   the feature.\n\n"), id="whitespace-only"),
        pytest.param(_edit_fragment("Added the feature.\r\n"), id="line-endings-only"),
        pytest.param(
            lambda changelog: (changelog / "README.md").write_text("Edited.\n", encoding="utf-8"),
            id="readme",
        ),
    ],
)
def test_fragment_changes_that_do_not_satisfy_the_changelog_rule(
    tmp_path, capsys, change: Callable[[Path], object]
) -> None:
    assert main(_policy_args_after(tmp_path, change)) == 1
    assert MISSING_FRAGMENT in capsys.readouterr().out


def test_emptying_an_unreleased_fragment_is_rejected(tmp_path, capsys) -> None:
    assert main(_policy_args_after(tmp_path, _edit_fragment(" \n"))) == 1
    assert "Changelog fragment is empty: changelog.d/+feature.added.md." in (
        capsys.readouterr().out
    )


def test_editing_a_fragment_released_since_the_branch_point_is_rejected(tmp_path, capsys) -> None:
    argv = _policy_args_after(
        tmp_path, _edit_fragment("Added the renamed feature.\n"), release_on_base=True
    )

    assert main(argv) == 1
    out = capsys.readouterr().out
    assert "Changelog fragment changelog.d/+feature.added.md was already released." in out
    assert MISSING_FRAGMENT in out


def test_moving_a_module_out_of_src_still_counts_as_a_source_change(tmp_path, capsys) -> None:
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "feature.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "init", "-q")
    base = _commit(repo, "base")
    (repo / "lib").mkdir()
    (repo / "src" / "feature.py").rename(repo / "lib" / "feature.py")
    head = _commit(repo, "move")
    event = _event()
    event["pull_request"]["base"]["sha"] = base
    event["pull_request"]["head"]["sha"] = head
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps(event), encoding="utf-8")

    assert main(["--event-path", str(event_path), "--root", str(repo)]) == 1
    out = capsys.readouterr().out
    assert "Production Python changed without a tests/** change." in out
    assert MISSING_FRAGMENT in out
