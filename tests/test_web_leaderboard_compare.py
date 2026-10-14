"""Tests for the web UI leaderboard, CSV/JSON export, and head-to-head comparison routes."""

from __future__ import annotations

import csv
import io
from typing import Any

import pytest

from tests.web_support import TEST_TOKEN, web_client
from tool_eval_bench.storage.db import RunRepository
from tool_eval_bench.web import create_app


def _make_completed_run(
    run_id: str,
    *,
    model: str,
    final_score: int,
    backend: str = "vllm",
    fingerprint: str = "fp-standard-15",
    results: list[dict[str, Any]] | None = None,
    include_pack: bool = False,
) -> dict[str, Any]:
    scenario_results = results or [
        {
            "scenario_id": "TC-01",
            "status": "pass",
            "points": 2,
            "summary": "Passed TC-01",
            "raw_log": f"trace for {model} TC-01",
            "duration_seconds": 1.0,
        },
        {
            "scenario_id": "TC-02",
            "status": "pass" if final_score >= 80 else "fail",
            "points": 2 if final_score >= 80 else 0,
            "summary": "Result TC-02",
            "raw_log": f"trace for {model} TC-02",
            "duration_seconds": 1.5,
        },
    ]
    packs: list[dict[str, Any]] = []
    if include_pack:
        scenario_results = [
            *scenario_results,
            {
                "scenario_id": "PACK-01",
                "status": "pass",
                "points": 2,
                "summary": "Private held-out summary",
                "raw_log": "PRIVATE HELD OUT TRACE",
                "duration_seconds": 2.0,
            },
        ]
        packs.append(
            {
                "name": "secret-pack",
                "scenario_count": 1,
                "scenario_ids": ["PACK-01"],
                "content_hash": "1234567890abcdef",
            }
        )

    return {
        "run_id": run_id,
        "status": "completed",
        "run_type": "tool_eval",
        "config": {
            "model": model,
            "backend": backend,
            "scenario_count": len(scenario_results),
            "config_fingerprint": fingerprint,
            "scenario_packs": packs or None,
        },
        "metadata": {
            "model": model,
            "backend": backend,
            "tool_version": "2.8.0",
            "git_sha": "abc1234",
            "engine_name": "vLLM",
            "quantization": "fp8",
        },
        "scores": {
            "final_score": final_score,
            "rating": "★★★★ Good" if final_score >= 75 else "★★ Weak",
            "total_points": sum(r["points"] for r in scenario_results),
            "max_points": 2 * len(scenario_results),
            "completion_rate": 100,
            "total_tokens": 4000,
            "category_scores": [
                {"category": "A", "label": "Tool Selection", "percent": final_score},
            ],
            "scenario_results": scenario_results,
        },
    }


@pytest.fixture()
def leaderboard_db(tmp_path, monkeypatch: pytest.MonkeyPatch) -> str:
    db_path = str(tmp_path / "benchmarks.sqlite")
    with RunRepository(db_path=db_path) as repo:
        repo.upsert_scenario_run(
            _make_completed_run(
                "20261014T100000Z_qwen",
                model="/models/models--Qwen--Qwen3-32B/snapshots/123",
                final_score=92,
                fingerprint="fp-qwen",
            )
        )
        repo.upsert_scenario_run(
            _make_completed_run(
                "20261014T110000Z_llama",
                model="meta-llama/Llama-3.3-70B",
                final_score=50,
                fingerprint="fp-llama",
            )
        )
        repo.upsert_scenario_run(
            _make_completed_run(
                "20261014T120000Z_pack",
                model="google/gemini-2.5-pro",
                final_score=95,
                fingerprint="fp-gemini-pack",
                include_pack=True,
            )
        )
    monkeypatch.setattr(
        "tool_eval_bench.application.run_queries._repository",
        lambda: RunRepository(db_path=db_path),
    )
    return db_path


@pytest.mark.asyncio
async def test_leaderboard_groups_cohorts_and_filters(leaderboard_db: str) -> None:
    app = create_app(token=TEST_TOKEN)
    async with web_client(app) as client:
        resp = await client.get("/api/leaderboard")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["rows"]) == 3
        assert len(data["cohorts"]) == 2  # standard 2-scenario cohort + held-out pack cohort
        assert data["categories"] == ["A"]

        models = {r["display_model"]: r for r in data["rows"]}
        assert "Qwen/Qwen3-32B" in models
        assert models["Qwen/Qwen3-32B"]["cohort_rank"] == 1
        assert models["meta-llama/Llama-3.3-70B"]["cohort_rank"] == 2

        # Filter by search query
        q_resp = await client.get("/api/leaderboard", params={"q": "Qwen3-32B"})
        assert q_resp.status_code == 200
        assert len(q_resp.json()["rows"]) == 1

        # Filter by cohort fingerprint
        cohort_fp = models["Qwen/Qwen3-32B"]["cohort_fingerprint"]
        c_resp = await client.get("/api/leaderboard", params={"cohort": cohort_fp})
        assert c_resp.status_code == 200
        assert len(c_resp.json()["rows"]) == 2


@pytest.mark.asyncio
async def test_export_endpoint_serves_csv_and_json(leaderboard_db: str) -> None:
    app = create_app(token=TEST_TOKEN)
    async with web_client(app) as client:
        json_resp = await client.get("/api/export", params={"format": "json"})
        assert json_resp.status_code == 200
        assert "application/json" in json_resp.headers["content-type"]
        exported = json_resp.json()
        assert len(exported) == 3

        csv_resp = await client.get("/api/export", params={"format": "csv"})
        assert csv_resp.status_code == 200
        assert "text/csv" in csv_resp.headers["content-type"]
        rows = list(csv.DictReader(io.StringIO(csv_resp.text)))
        assert len(rows) == 3
        assert "cat_A" in rows[0]

        bad_resp = await client.get("/api/export", params={"format": "xml"})
        assert bad_resp.status_code == 400


@pytest.mark.asyncio
async def test_compare_endpoint_computes_deltas_mcnemar_and_redacts_held_out(
    leaderboard_db: str,
) -> None:
    app = create_app(token=TEST_TOKEN)
    async with web_client(app) as client:
        resp = await client.get(
            "/api/compare",
            params={
                "left": "20261014T110000Z_llama",
                "right": "20261014T120000Z_pack",
            },
        )
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["config_mismatch"] is True
        assert any("scenario_count" in d for d in payload["config_diffs"])
        assert payload["summary"]["improved"] == 1
        assert payload["summary"]["unchanged"] == 1
        assert payload["mcnemar"]["discordant_pairs"] == 1
        assert payload["mcnemar"]["b_only_pass"] == 1

        deltas = {d["scenario_id"]: d for d in payload["deltas"]}
        assert deltas["TC-02"]["outcome"] == "improved"
        assert deltas["TC-02"]["points_delta"] == 2
        assert deltas["PACK-01"]["outcome"] == "added"
        assert deltas["PACK-01"]["title"] == "held out"
        assert deltas["PACK-01"]["b"]["raw_log"] == "held out"
        assert "PRIVATE HELD OUT TRACE" not in resp.text

        # Missing params -> 400, unknown run -> 404
        assert (await client.get("/api/compare", params={"left": "only-one"})).status_code == 400
        assert (
            await client.get(
                "/api/compare", params={"left": "unknown", "right": "20261014T100000Z_qwen"}
            )
        ).status_code == 404
