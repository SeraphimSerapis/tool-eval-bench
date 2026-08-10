"""A published score is only useful if you can tell which code produced it.

Two failure modes are covered here:
  * the reported git SHA must belong to *this* package, not to whatever
    repository the user happened to run the CLI from;
  * two runs built from different commits must not share a config_fingerprint,
    since the scenarios and evaluators are themselves code.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tool_eval_bench.application.service import _build_run_config
from tool_eval_bench.domain.scenarios import Category, ScenarioDefinition
from tool_eval_bench.utils.metadata import _git_env_without_repository, _git_sha


def _scenario(sid: str) -> ScenarioDefinition:
    return ScenarioDefinition(
        id=sid,
        title=sid,
        category=Category.A,
        user_message="",
        description="",
        handle_tool_call=lambda state, call: None,
        evaluate=lambda state: None,  # type: ignore[arg-type,return-value]
    )


def _config(metadata: dict) -> dict:
    return _build_run_config(
        model="m",
        backend="vllm",
        base_url="http://localhost:8000",
        scenarios=[_scenario("TC-01")],
        temperature=0.0,
        timeout_seconds=120.0,
        max_turns=8,
        seed=None,
        reference_date=None,
        concurrency=1,
        error_rate=0.0,
        alpha=0.7,
        extra_params=None,
        context_pressure_config=None,
        weight_by_difficulty=False,
        metadata=metadata,
    )


class TestGitShaProvenance:
    def test_sha_is_resolved_against_the_package_not_the_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Running from an unrelated repo must not attribute its SHA to the run."""
        unrelated = tmp_path / "unrelated"
        unrelated.mkdir()
        clean_env = _git_env_without_repository()
        subprocess.run(["git", "init", "-q"], cwd=unrelated, check=True, env=clean_env)
        subprocess.run(
            [
                "git",
                "-c",
                "user.email=a@b",
                "-c",
                "user.name=t",
                "commit",
                "-q",
                "--allow-empty",
                "-m",
                "unrelated",
            ],
            cwd=unrelated,
            check=True,
            env=clean_env,
        )
        foreign = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], cwd=unrelated, env=clean_env
            )
            .decode()
            .strip()
        )

        monkeypatch.chdir(unrelated)
        monkeypatch.setenv("GIT_DIR", str(unrelated / ".git"))
        monkeypatch.setenv("GIT_WORK_TREE", str(unrelated))
        sha = _git_sha()

        assert sha != foreign
        if sha is not None:
            assert sha.startswith(_package_head())
        else:
            # Installed without git metadata — still better than a foreign SHA.
            assert _package_head() is None

    def test_dirty_tree_is_flagged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A run from an edited tree is not reproducible from the SHA alone."""
        import tool_eval_bench.utils.metadata as metadata_module

        calls: list[tuple[str, ...]] = []

        def fake_check_output(cmd, **kwargs):  # type: ignore[no-untyped-def]
            args = tuple(cmd[3:])
            calls.append(args)
            if args == ("rev-parse", "--is-inside-work-tree"):
                return b"true\n"
            if args == ("rev-parse", "--short", "HEAD"):
                return b"abc1234\n"
            if args == ("status", "--porcelain"):
                return b" M src/tool_eval_bench/foo.py\n"
            raise AssertionError(f"unexpected git call: {args}")

        monkeypatch.setattr(metadata_module.subprocess, "check_output", fake_check_output)

        assert _git_sha() == "abc1234-dirty"

    def test_clean_tree_has_no_suffix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import tool_eval_bench.utils.metadata as metadata_module

        def fake_check_output(cmd, **kwargs):  # type: ignore[no-untyped-def]
            args = tuple(cmd[3:])
            if args == ("rev-parse", "--is-inside-work-tree"):
                return b"true\n"
            if args == ("rev-parse", "--short", "HEAD"):
                return b"abc1234\n"
            return b""

        monkeypatch.setattr(metadata_module.subprocess, "check_output", fake_check_output)

        assert _git_sha() == "abc1234"

    def test_non_checkout_reports_nothing_rather_than_guessing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Installed wheels have no git metadata — None is the honest answer."""
        import tool_eval_bench.utils.metadata as metadata_module

        def fake_check_output(cmd, **kwargs):  # type: ignore[no-untyped-def]
            raise subprocess.CalledProcessError(128, cmd)

        monkeypatch.setattr(metadata_module.subprocess, "check_output", fake_check_output)

        assert _git_sha() is None

    def test_git_missing_from_path_is_not_fatal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import tool_eval_bench.utils.metadata as metadata_module

        def fake_check_output(cmd, **kwargs):  # type: ignore[no-untyped-def]
            raise FileNotFoundError("git")

        monkeypatch.setattr(metadata_module.subprocess, "check_output", fake_check_output)

        assert _git_sha() is None


class TestFingerprintIncludesCodeIdentity:
    def test_different_commits_are_not_comparable(self) -> None:
        a = _config({"git_sha": "aaaaaaa"})
        b = _config({"git_sha": "bbbbbbb"})

        assert a["config_fingerprint"] != b["config_fingerprint"]

    def test_same_commit_and_flags_are_comparable(self) -> None:
        a = _config({"git_sha": "aaaaaaa"})
        b = _config({"git_sha": "aaaaaaa"})

        assert a["config_fingerprint"] == b["config_fingerprint"]

    def test_dirty_tree_differs_from_its_base_commit(self) -> None:
        clean = _config({"git_sha": "aaaaaaa"})
        dirty = _config({"git_sha": "aaaaaaa-dirty"})

        assert clean["config_fingerprint"] != dirty["config_fingerprint"]

    def test_missing_sha_is_tolerated(self) -> None:
        assert _config({})["config_fingerprint"]


def _package_head() -> str | None:
    """The HEAD of the checkout the installed package lives in, if any."""
    import tool_eval_bench

    package_root = Path(tool_eval_bench.__file__).resolve().parent
    try:
        out = subprocess.check_output(  # noqa: S603 — fixed argv, test-only
            ["git", "-C", str(package_root), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            env=_git_env_without_repository(),
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.decode().strip()
