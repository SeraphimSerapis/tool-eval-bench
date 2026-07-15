"""Packaging regressions for non-Python benchmark assets."""

from __future__ import annotations

import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]


def test_bundled_yaml_scenarios_are_declared_as_package_data() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as file:
        config = tomllib.load(file)

    package_data = config["tool"]["setuptools"]["package-data"]["tool_eval_bench"]

    assert "evals/yaml_scenarios/*.yaml" in package_data
    assert (PROJECT_ROOT / "src/tool_eval_bench/evals/yaml_scenarios/weather.yaml").is_file()


def test_build_backend_supports_declared_spdx_license_expression() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as file:
        config = tomllib.load(file)

    assert "setuptools>=77" in config["build-system"]["requires"]
    assert config["project"]["license"] == "MIT"
