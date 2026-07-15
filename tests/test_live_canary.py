"""Opt-in live canary against a configured OpenAI-compatible endpoint."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.live
def test_live_canary_covers_tool_safety_and_artifact_paths(tmp_path: Path) -> None:
    """Run a small, deployment-relevant scenario slice when configured."""
    base_url = os.getenv("TOOL_EVAL_CANARY_BASE_URL")
    if not base_url:
        pytest.skip("Set TOOL_EVAL_CANARY_BASE_URL to enable the live canary")

    command = [
        sys.executable,
        "-m",
        "tool_eval_bench",
        "--base-url",
        base_url,
        "--json",
        "--no-warmup",
        "--no-probe-engine",
        "--fail-on-safety",
        "--hardmode",
        "--timeout",
        os.getenv("TOOL_EVAL_CANARY_TIMEOUT", "120"),
        "--output-dir",
        str(tmp_path / "runs"),
        "--scenarios",
        "TC-01",
        "TC-43",
        "TC-60",
        "TC-81",
    ]
    model = os.getenv("TOOL_EVAL_CANARY_MODEL")
    if model:
        command.extend(["--model", model])
    api_key = os.getenv("TOOL_EVAL_CANARY_API_KEY")
    if api_key:
        command.extend(["--api-key", api_key])

    completed = subprocess.run(  # noqa: S603
        command,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    result = json.loads(completed.stdout)
    scenario_results = result["scores"]["scenario_results"]
    assert {item["scenario_id"] for item in scenario_results} == {
        "TC-01",
        "TC-43",
        "TC-60",
        "TC-81",
    }
    assert all(item.get("raw_log") for item in scenario_results)
    report_path = Path(result["report_path"])
    assert report_path.is_file()
