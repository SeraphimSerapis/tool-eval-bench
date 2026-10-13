"""Exercise dotenv discovery from installed code, not an interactive Python frame."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tool_eval_bench.cli import helpers


@pytest.mark.parametrize(
    ("files", "exported", "expected"),
    [
        ({"work/nested/.env": "current", "install/.env": "legacy"}, None, "current"),
        ({"work/.env": "parent", "install/.env": "legacy"}, None, "parent"),
        ({"install/.env": "legacy"}, None, "legacy"),
        ({".env": "shared-parent"}, None, "shared-parent"),
        ({"work/nested/.env": "current", "install/.env": "legacy"}, "exported", "exported"),
        ({"work/nested/.env": "", "install/.env": "legacy"}, None, "None"),
        ({}, None, "None"),
    ],
    ids=["cwd", "cwd-parent", "legacy", "shared-parent", "exported", "empty", "missing"],
)
def test_dotenv_discovery_from_installed_module(
    tmp_path: Path, files: dict[str, str], exported: str | None, expected: str
) -> None:
    install = tmp_path / "install"
    install.mkdir()
    work = tmp_path / "work" / "nested"
    work.mkdir(parents=True)

    variable = "TOOL_EVAL_DOTENV_TEST_VALUE"
    for relative_path, value in files.items():
        (tmp_path / relative_path).write_text(f"{variable}={value}\n" if value else "")

    # dotenv's legacy lookup follows the caller's file, so relocate the real helper.
    shutil.copyfile(helpers.__file__, install / "dotenv_helper.py")
    script = install / "probe.py"
    script.write_text(
        "import os\n"
        "from dotenv_helper import load_dotenv_file\n"
        "load_dotenv_file()\n"
        f"print(os.getenv({variable!r}))\n"
    )
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("TOOL_EVAL_") and key != "PYTHON_DOTENV_DISABLED"
    }
    env["PYTHONPATH"] = str(Path(helpers.__file__).resolve().parents[2])
    if exported is not None:
        env[variable] = exported
    result = subprocess.run(  # noqa: S603
        [sys.executable, str(script)], cwd=work, env=env, text=True, capture_output=True, check=True
    )
    assert result.stdout.strip() == expected
