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

from tool_eval_bench.application.run_config import RunSettings, build_run_config
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


def _settings() -> RunSettings:
    return RunSettings(
        model="m",
        backend="vllm",
        base_url="http://localhost:8000",
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
    )


def _config(metadata: dict) -> dict:
    return build_run_config(_settings(), scenarios=[_scenario("TC-01")], metadata=metadata)


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
            if args == ("rev-parse", "--show-toplevel"):
                return f"{_source_root()}\n".encode()
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
            if args == ("rev-parse", "--show-toplevel"):
                return f"{_source_root()}\n".encode()
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


class TestGitShaEnclosingRepository:
    """Only this project's own checkout may lend the run its SHA.

    Each case lays out a package copy inside a temporary repository and points
    the module's ``__file__`` at it, which is all ``_git_sha`` anchors on.
    """

    @staticmethod
    def _repo(root: Path, *, ignore: str = "") -> None:
        env = _git_env_without_repository()
        root.mkdir(parents=True, exist_ok=True)
        if ignore:
            (root / ".gitignore").write_text(ignore, encoding="utf-8")
        subprocess.run(["git", "init", "-q"], cwd=root, check=True, env=env)
        subprocess.run(["git", "add", "-A"], cwd=root, check=True, env=env)
        subprocess.run(
            ["git", "-c", "user.email=a@b", "-c", "user.name=t", "commit", "-q", "-m", "init"],
            cwd=root,
            check=True,
            env=env,
        )

    @staticmethod
    def _package_at(package_dir: Path) -> Path:
        """Create ``<package_dir>/utils/metadata.py`` and return that path."""
        module = package_dir / "utils" / "metadata.py"
        module.parent.mkdir(parents=True)
        module.write_text("", encoding="utf-8")
        return module

    @staticmethod
    def _pyproject(root: Path, name: str) -> None:
        (root / "pyproject.toml").write_text(f'[project]\nname = "{name}"\n', encoding="utf-8")

    def _sha_for(self, module: Path, monkeypatch: pytest.MonkeyPatch) -> str | None:
        import tool_eval_bench.utils.metadata as metadata_module

        monkeypatch.setattr(metadata_module, "__file__", str(module))
        return _git_sha()

    def test_wheel_in_an_ignored_venv_of_another_repo_has_no_sha(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        userproj = tmp_path / "userproj"
        site = userproj / ".venv" / "lib" / "python3" / "site-packages"
        module = self._package_at(site / "tool_eval_bench")
        self._pyproject(userproj, "userproj")
        self._repo(userproj, ignore=".venv/\n")

        assert self._sha_for(module, monkeypatch) is None

    def test_own_checkout_reports_its_sha(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checkout = tmp_path / "checkout"
        module = self._package_at(checkout / "src" / "tool_eval_bench")
        self._pyproject(checkout, "tool-eval-bench")
        self._repo(checkout)
        head = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=checkout,
                env=_git_env_without_repository(),
            )
            .decode()
            .strip()
        )

        assert self._sha_for(module, monkeypatch) == head

    def test_src_layout_of_another_project_has_no_sha(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checkout = tmp_path / "other"
        module = self._package_at(checkout / "src" / "tool_eval_bench")
        self._pyproject(checkout, "someone-elses-fork")
        self._repo(checkout)

        assert self._sha_for(module, monkeypatch) is None

    def test_copy_vendored_below_the_top_level_has_no_sha(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The copy is tracked and even carries this project's pyproject, but
        # the work tree it belongs to is the enclosing repository.
        userproj = tmp_path / "userproj"
        vendored = userproj / "vendor" / "tool-eval-bench"
        module = self._package_at(vendored / "src" / "tool_eval_bench")
        self._pyproject(vendored, "tool-eval-bench")
        self._repo(userproj)

        assert self._sha_for(module, monkeypatch) is None


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

    def test_context_window_changes_comparison_cohort(self) -> None:
        small = _config({"engine_name": "TensorFold", "max_model_len": 8192})
        large = _config({"engine_name": "TensorFold", "max_model_len": 32768})
        assert small["config_fingerprint"] != large["config_fingerprint"]
        assert (
            _config({})["config_fingerprint"]
            == _config({"max_model_len": None})["config_fingerprint"]
        )

    def test_server_slot_count_changes_comparison_cohort(self) -> None:
        one_slot = _config({"git_sha": "aaaaaaa", "slot_count": 1})
        three_slots = _config({"git_sha": "aaaaaaa", "slot_count": 3})

        assert one_slot["config_fingerprint"] != three_slots["config_fingerprint"]

    def test_recorded_thinking_flag_does_not_change_comparison_cohort(self) -> None:
        # The flag restates extra_params, which the config already fingerprints,
        # so correcting how it is derived must not re-cohort history.
        on = _config({"git_sha": "aaaaaaa", "thinking_enabled": True})
        off = _config({"git_sha": "aaaaaaa", "thinking_enabled": False})

        assert on["config_fingerprint"] == off["config_fingerprint"]


class TestSystemPromptProvenance:
    """A custom system prompt is a scoring condition: it must change the cohort."""

    def test_default_run_persists_no_system_prompt_key(self) -> None:
        """The key is absent, not null.

        The fingerprint is a hash of this mapping, so persisting ``None`` on every
        run would have re-cohorted every historical run the first time this option
        shipped. Absence also means "built-in prompt" to the resume check.
        """
        config = _config({})

        assert "system_prompt" not in config

    def test_default_fingerprint_ignores_the_feature(self) -> None:
        """A default run fingerprints exactly as it did before the override existed."""
        import dataclasses

        from tool_eval_bench import __version__
        from tool_eval_bench.application.run_config import build_run_config
        from tool_eval_bench.utils.ids import build_config_fingerprint

        config = build_run_config(
            dataclasses.replace(_settings(), system_prompt=None),
            scenarios=[_scenario("TC-01")],
            metadata={},
        )
        persisted = dict(config)
        fingerprint = persisted.pop("config_fingerprint")
        # Persisted for resume, but endpoint_id is what tells endpoints apart.
        persisted.pop("base_url")
        # The payload shape predates this PR; rebuilt here so a future change to
        # the default config has to be deliberate.
        expected = build_config_fingerprint(
            {
                "config": {**persisted, "scenario_ids": sorted(persisted["scenario_ids"])},
                "deployment": {},
                "tool_version": __version__,
                "git_sha": None,
            }
        )

        assert fingerprint == expected

    def test_override_is_persisted_and_changes_cohort(self) -> None:
        import dataclasses

        custom = dataclasses.replace(_settings(), system_prompt="You are a strict evaluator.")
        default = build_run_config(_settings(), scenarios=[_scenario("TC-01")], metadata={})
        overridden = build_run_config(custom, scenarios=[_scenario("TC-01")], metadata={})

        assert overridden["system_prompt"] == "You are a strict evaluator."
        assert default["config_fingerprint"] != overridden["config_fingerprint"]

    def test_two_runs_with_the_same_override_are_comparable(self) -> None:
        import dataclasses

        custom = dataclasses.replace(_settings(), system_prompt="You are a strict evaluator.")
        a = build_run_config(
            custom, scenarios=[_scenario("TC-01")], metadata={"git_sha": "aaaaaaa"}
        )
        b = build_run_config(
            custom, scenarios=[_scenario("TC-01")], metadata={"git_sha": "aaaaaaa"}
        )

        assert a["config_fingerprint"] == b["config_fingerprint"]


class TestSystemPromptResumeCompatibility:
    """Resuming a run with a different system prompt must be flagged."""

    @staticmethod
    def _mismatches(
        previous: dict, *, system_prompt: str | None, drop: tuple[str, ...] = ()
    ) -> list[str]:
        import dataclasses

        from tool_eval_bench.cli.dispatch import _resume_config_mismatches
        from tool_eval_bench.cli.legacy_parser import _make_parser

        settings = dataclasses.replace(_settings(), system_prompt=system_prompt)
        scenarios = [_scenario("TC-01")]
        current_config = build_run_config(settings, scenarios=scenarios, metadata={})
        args = _make_parser().parse_args(
            [] if system_prompt is None else ["--system-prompt", system_prompt]
        )
        # ``drop`` is applied after the merge: the prior run is modelled on the
        # current config, so a key the current run sets has to be removed here to
        # model a persisted run that never recorded it.
        prior = {**current_config, **previous}
        for key in drop:
            prior.pop(key, None)
        return _resume_config_mismatches(
            prior,
            model=settings.model,
            backend=settings.backend,
            base_url=settings.base_url,
            scenarios=scenarios,
            args=args,
            extra_params=None,
            scenario_packs=None,
            context_pressure=None,
        )

    def test_same_prompt_resume_is_clean(self) -> None:
        assert "system_prompt" not in self._mismatches({}, system_prompt=None)

    def test_resuming_a_keyless_run_with_an_override_is_flagged(self) -> None:
        """A run that predates this option persists no key; it used the built-in prompt.

        Without this, completed scenarios graded under the built-in persona would be
        merged with the remainder graded under an override, and the finished run
        would fingerprint as the override-only cohort.
        """
        assert "system_prompt" in self._mismatches(
            {}, system_prompt="You are a pirate.", drop=("system_prompt",)
        )

    def test_resuming_a_keyless_run_without_an_override_is_clean(self) -> None:
        assert "system_prompt" not in self._mismatches(
            {}, system_prompt=None, drop=("system_prompt",)
        )

    def test_changed_prompt_is_flagged_in_both_directions(self) -> None:
        # Default run resumed with an override...
        assert "system_prompt" in self._mismatches(
            {"system_prompt": None}, system_prompt="Be terse."
        )
        # ...and an overridden run resumed without the flag.
        assert "system_prompt" in self._mismatches(
            {"system_prompt": "Be terse."}, system_prompt=None
        )


def _source_root() -> Path:
    """The checkout that holds the imported package under ``src/``."""
    import tool_eval_bench

    return Path(tool_eval_bench.__file__).resolve().parent.parent.parent


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
