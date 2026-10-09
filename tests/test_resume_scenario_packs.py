"""Resume refuses to add a held-out pack to a run that was measured without one."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from test_scenario_packs import _write_pack

from tool_eval_bench.evals.packs import load_scenario_pack


def _resume(
    monkeypatch: pytest.MonkeyPatch, stored_config: dict[str, Any], *flags: str
) -> list[dict[str, Any]]:
    """Run ``resume`` through ``main`` against an interrupted stored run."""
    from tool_eval_bench.cli import dispatch, plugin_runners
    from tool_eval_bench.storage import db
    from tool_eval_bench.utils import metadata

    async def no_context(**kwargs: Any) -> None:
        return None

    stored = {"status": "interrupted", "config": stored_config, "scores": {}}

    class Repo:
        def get(self, run_id: str) -> dict[str, Any]:
            return stored

        def get_checkpoints(self, run_id: str) -> list[dict[str, Any]]:
            return []

        def close(self) -> None:
            pass

    calls: list[dict[str, Any]] = []

    class Service:
        async def run_benchmark(self, **kwargs: Any) -> dict[str, Any]:
            calls.append(kwargs)
            return {"run_id": "r", "scores": {"scenario_results": []}}

    monkeypatch.setattr(dispatch, "_load_dotenv", lambda: None)
    monkeypatch.setattr(dispatch, "_preflight_model_check", lambda *a, **k: None)
    monkeypatch.setattr(metadata, "collect_run_context", no_context)
    monkeypatch.setattr(plugin_runners, "run_selected_plugins", lambda *a, **k: False)
    monkeypatch.setattr(db, "RunRepository", Repo)
    monkeypatch.setattr(dispatch, "BenchmarkService", lambda **kwargs: Service())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tool-eval-bench",
            "resume",
            "r",
            "--model",
            "m",
            "--backend",
            "vllm",
            "--base-url",
            "http://localhost:8000",
            "--scenarios",
            "TC-01",
            "--no-warmup",
            "--no-live",
            *flags,
        ],
    )
    dispatch.main()
    return calls


# Recorded before scenario_ids was stored, so only scenario_packs can catch a
# pack added on resume.
PACK_FREE = {"model": "m", "backend": "vllm"}


def test_resume_refuses_adding_a_pack_to_a_pack_free_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    pack = _write_pack(tmp_path / "pack", "HO-1")

    with pytest.raises(SystemExit) as exited:
        _resume(monkeypatch, PACK_FREE, "--scenario-pack", str(pack))

    assert exited.value.code == 1
    out = " ".join(capsys.readouterr().out.split())
    assert "configuration mismatch" in out
    assert "scenario_packs" in out


def test_resume_continues_a_pack_free_run_without_a_pack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _resume(monkeypatch, PACK_FREE)

    assert calls[0]["resume_run_id"] == "r"
    assert calls[0]["scenario_packs"] is None


def test_resume_continues_a_pack_run_with_the_same_pack(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pack = _write_pack(tmp_path / "pack", "HO-1")
    attested = {**PACK_FREE, "scenario_packs": [load_scenario_pack(pack).to_dict()]}

    calls = _resume(monkeypatch, attested, "--scenario-pack", str(pack))

    assert calls[0]["scenario_packs"] == attested["scenario_packs"]
