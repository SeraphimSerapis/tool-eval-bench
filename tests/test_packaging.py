"""Container packaging contracts that do not require Docker at test time."""

from __future__ import annotations

import tomllib
from pathlib import Path
from subprocess import run

from tool_eval_bench.utils.metadata import _git_env_without_repository

ROOT = Path(__file__).parents[1]


def test_bundled_yaml_scenarios_are_declared_as_package_data() -> None:
    with (ROOT / "pyproject.toml").open("rb") as file:
        config = tomllib.load(file)

    package_data = config["tool"]["setuptools"]["package-data"]["tool_eval_bench"]

    assert "evals/yaml_scenarios/*.yaml" in package_data
    assert (ROOT / "src/tool_eval_bench/evals/yaml_scenarios/weather.yaml").is_file()


def test_build_backend_supports_declared_spdx_license_expression() -> None:
    with (ROOT / "pyproject.toml").open("rb") as file:
        config = tomllib.load(file)

    assert "setuptools>=77" in config["build-system"]["requires"]
    assert config["project"]["license"] == "MIT"


def test_docker_build_uses_git_only_in_the_builder_stage() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "FROM python:3.12-slim@sha256:" in dockerfile
    assert dockerfile.count("python:3.12-slim@sha256:") == 2
    assert "FROM ghcr.io/astral-sh/uv:0.10.8@sha256:" in dockerfile
    assert "COPY .git ./.git" in dockerfile
    assert "COPY pyproject.toml uv.lock README.md ./" in dockerfile
    assert "uv sync --locked --no-dev --no-editable" in dockerfile
    assert "ARG BUILD_VERSION" in dockerfile
    assert "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_TOOL_EVAL_BENCH" in dockerfile
    assert "COPY --from=build /opt/venv /opt/venv" in dockerfile
    assert "COPY --from=build /build/.git" not in dockerfile


def test_docker_runtime_is_unprivileged_and_owns_its_unmounted_outputs() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "USER tool-eval" in dockerfile
    assert "install -d --owner=tool-eval --group=tool-eval /app/data /app/runs" in dockerfile
    assert 'VOLUME ["/app/data", "/app/runs"]' in dockerfile


def test_docker_context_keeps_git_metadata_available_to_the_builder() -> None:
    ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()

    assert ".git" not in ignored


def test_lockfile_is_tracked_and_not_ignored(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert "uv.lock" not in ignored
    assert (ROOT / "uv.lock").is_file()
    monkeypatch.setenv("GIT_DIR", str(ROOT / "not-the-repository"))
    tracked = run(  # noqa: S603 -- fixed git executable and repository-local path
        ["git", "-C", str(ROOT), "ls-files", "--error-unmatch", "uv.lock"],
        check=False,
        capture_output=True,
        env=_git_env_without_repository(),
        text=True,
        encoding="utf-8",
    )
    assert tracked.returncode == 0, tracked.stderr


def test_compose_persists_reports_and_sqlite_history() -> None:
    compose = (ROOT / "docker-compose.yaml").read_text(encoding="utf-8")

    assert "./runs:/app/runs" in compose
    assert "./data:/app/data" in compose
    assert (
        'user: "${LOCAL_UID:?set LOCAL_UID to id -u}:${LOCAL_GID:?set LOCAL_GID to id -g}"'
        in compose
    )
