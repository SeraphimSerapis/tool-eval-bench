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
        assert (
            await client.get(
                "/api/compare", params={"left": "20261014T100000Z_qwen", "right": "unknown"}
            )
        ).status_code == 404


@pytest.mark.asyncio
async def test_leaderboard_and_export_filter_before_window_so_non_tool_runs_do_not_evict_scores(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = str(tmp_path / "benchmarks.sqlite")
    with RunRepository(db_path=db_path) as repo:
        repo.upsert_scenario_run(
            _make_completed_run(
                "20261014T090000Z_target",
                model="Qwen/Qwen3-32B",
                final_score=91,
                fingerprint="fp-target",
            )
        )
        # Insert more newer non-eligible rows (perf, plugin, interrupted) than the window.
        for idx in range(4):
            repo.upsert_scenario_run(
                {
                    "run_id": f"20261014T10000{idx}Z_perf{idx}",
                    "status": "completed",
                    "run_type": "perf",
                    "model": "Qwen/Qwen3-32B",
                    "config": {"model": "Qwen/Qwen3-32B", "backend": "vllm"},
                    "metadata": {},
                    "scores": {"samples": 2, "results": []},
                }
            )
        for idx in range(3):
            repo.upsert_scenario_run(
                {
                    "run_id": f"20261014T11000{idx}Z_gsm{idx}",
                    "status": "completed",
                    "run_type": "gsm8k",
                    "model": "Qwen/Qwen3-32B",
                    "config": {"model": "Qwen/Qwen3-32B", "backend": "vllm"},
                    "metadata": {},
                    "scores": {"accuracy": 85.0},
                }
            )
        for idx in range(3):
            interrupted = _make_completed_run(
                f"20261014T12000{idx}Z_int{idx}",
                model="Qwen/Qwen3-32B",
                final_score=0,
                fingerprint="fp-target",
            )
            interrupted["status"] = "interrupted"
            repo.upsert_scenario_run(interrupted)

    monkeypatch.setattr(
        "tool_eval_bench.application.run_queries._repository",
        lambda: RunRepository(db_path=db_path),
    )
    monkeypatch.setattr("tool_eval_bench.web.app._LEADERBOARD_WINDOW", 5)

    app = create_app(token=TEST_TOKEN)
    async with web_client(app) as client:
        lb = await client.get("/api/leaderboard")
        assert lb.status_code == 200
        assert [r["run_id"] for r in lb.json()["rows"]] == ["20261014T090000Z_target"]

        exp_json = await client.get("/api/export", params={"format": "json"})
        assert exp_json.status_code == 200
        assert [r["run_id"] for r in exp_json.json()] == ["20261014T090000Z_target"]

        exp_csv = await client.get("/api/export", params={"format": "csv"})
        assert exp_csv.status_code == 200
        csv_rows = list(csv.DictReader(io.StringIO(exp_csv.text)))
        assert [r["run_id"] for r in csv_rows] == ["20261014T090000Z_target"]


@pytest.mark.asyncio
async def test_compare_surfaces_comparability_warnings_when_rubric_locale_dirty_or_incomplete(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pathlib import Path

    db_path = str(tmp_path / "benchmarks.sqlite")
    run_a = _make_completed_run(
        "20261014T100000Z_a",
        model="Qwen/Qwen3-8B",
        final_score=100,
        fingerprint="fp-shared",
    )
    run_a["metadata"].update(
        {
            "eval_rubric_version": "rubric-v1",
            "locale_version": "en-v1",
            "git_sha": "abc1234-dirty",
        }
    )

    run_b = _make_completed_run(
        "20261014T110000Z_b",
        model="Qwen/Qwen3-8B",
        final_score=50,
        fingerprint="fp-shared",
    )
    run_b["status"] = "interrupted"
    run_b["scores"]["completion_rate"] = 50.0
    run_b["metadata"].update(
        {
            "eval_rubric_version": "rubric-v2",
            "locale_version": "de-v2",
            "git_sha": "def5678",
        }
    )

    with RunRepository(db_path=db_path) as repo:
        repo.upsert_scenario_run(run_a)
        repo.upsert_scenario_run(run_b)

    monkeypatch.setattr(
        "tool_eval_bench.application.run_queries._repository",
        lambda: RunRepository(db_path=db_path),
    )

    app = create_app(token=TEST_TOKEN)
    async with web_client(app) as client:
        resp = await client.get(
            "/api/compare",
            params={"left": "20261014T100000Z_a", "right": "20261014T110000Z_b"},
        )
        assert resp.status_code == 200
        warnings = resp.json()["paired_stats"]["comparability_warnings"]

    assert any("dirty git working tree" in w for w in warnings), warnings
    assert any("Incomplete run" in w for w in warnings), warnings
    assert any("Eval rubric version mismatch" in w for w in warnings), warnings
    assert any("Locale version mismatch" in w for w in warnings), warnings

    compare_js = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "tool_eval_bench"
        / "web"
        / "static"
        / "compare.js"
    ).read_text(encoding="utf-8")
    assert "comparability_warnings" in compare_js


@pytest.mark.asyncio
async def test_leaderboard_invalid_limit_and_backend_filter(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = str(tmp_path / "benchmarks.sqlite")
    with RunRepository(db_path=db_path) as repo:
        repo.upsert_scenario_run(
            _make_completed_run(
                "20261014T100000Z_vllm",
                model="Qwen/Qwen3-8B",
                final_score=90,
                backend="vllm",
                fingerprint="fp-vllm",
            )
        )
        repo.upsert_scenario_run(
            _make_completed_run(
                "20261014T110000Z_sglang",
                model="Qwen/Qwen3-8B",
                final_score=85,
                backend="sglang",
                fingerprint="fp-sglang",
            )
        )
    monkeypatch.setattr(
        "tool_eval_bench.application.run_queries._repository",
        lambda: RunRepository(db_path=db_path),
    )
    app = create_app(token=TEST_TOKEN)
    async with web_client(app) as client:
        assert (await client.get("/api/leaderboard", params={"limit": "nope"})).status_code == 400
        sg = await client.get("/api/leaderboard", params={"backend": "SGLANG"})
        assert sg.status_code == 200
        assert [r["run_id"] for r in sg.json()["rows"]] == ["20261014T110000Z_sglang"]


@pytest.mark.asyncio
async def test_compare_rejects_non_scenario_runs_and_handles_removed_and_regressed_deltas(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = str(tmp_path / "benchmarks.sqlite")
    with RunRepository(db_path=db_path) as repo:
        repo.upsert_scenario_run(
            _make_completed_run(
                "20261014T100000Z_left",
                model="Qwen/Qwen3-8B",
                final_score=100,
                fingerprint="fp-left",
                results=[
                    {
                        "scenario_id": "TC-01",
                        "status": "pass",
                        "points": 2,
                        "duration_seconds": 1.0,
                    },
                    {
                        "scenario_id": "TC-02",
                        "status": "pass",
                        "points": 2,
                        "duration_seconds": 1.0,
                    },
                ],
            )
        )
        repo.upsert_scenario_run(
            _make_completed_run(
                "20261014T110000Z_right",
                model="Qwen/Qwen3-8B",
                final_score=0,
                fingerprint="fp-right",
                results=[
                    {
                        "scenario_id": "TC-01",
                        "status": "fail",
                        "points": 0,
                        "duration_seconds": 1.5,
                    },
                ],
            )
        )
        repo.upsert_scenario_run(
            {
                "run_id": "20261014T120000Z_perf",
                "status": "completed",
                "run_type": "perf",
                "model": "Qwen/Qwen3-8B",
                "config": {"model": "Qwen/Qwen3-8B", "backend": "vllm"},
                "metadata": {},
                "scores": {"samples": 1, "results": []},
            }
        )
        # Same standard config keys but different fingerprints -> generic diff fallback.
        repo.upsert_scenario_run(
            _make_completed_run(
                "20261014T130000Z_fp_only",
                model="Qwen/Qwen3-8B",
                final_score=100,
                fingerprint="fp-other",
                results=[
                    {
                        "scenario_id": "TC-01",
                        "status": "pass",
                        "points": 2,
                        "duration_seconds": 1.0,
                    },
                    {
                        "scenario_id": "TC-02",
                        "status": "pass",
                        "points": 2,
                        "duration_seconds": 1.0,
                    },
                ],
            )
        )
    monkeypatch.setattr(
        "tool_eval_bench.application.run_queries._repository",
        lambda: RunRepository(db_path=db_path),
    )

    app = create_app(token=TEST_TOKEN)
    async with web_client(app) as client:
        non_sc = await client.get(
            "/api/compare",
            params={"left": "20261014T100000Z_left", "right": "20261014T120000Z_perf"},
        )
        assert non_sc.status_code == 400
        assert "scenario results" in non_sc.json()["error"]

        one_sided = await client.get(
            "/api/compare",
            params={"left": "20261014T100000Z_left", "right": "20261014T110000Z_right"},
        )
        assert one_sided.status_code == 200
        payload = one_sided.json()
        assert payload["summary"]["regressed"] == 1
        by_id = {d["scenario_id"]: d for d in payload["deltas"]}
        assert by_id["TC-01"]["outcome"] == "regressed"
        assert by_id["TC-01"]["points_delta"] == -2
        assert by_id["TC-02"]["outcome"] == "removed"
        assert by_id["TC-02"]["b"] is None

        fp_only = await client.get(
            "/api/compare",
            params={"left": "20261014T100000Z_left", "right": "20261014T130000Z_fp_only"},
        )
        assert fp_only.status_code == 200
        assert fp_only.json()["config_diffs"] == ["config fingerprints differ"]


@pytest.mark.asyncio
async def test_web_compare_handles_null_points_without_500(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = str(tmp_path / "benchmarks.sqlite")
    out_dir = str(tmp_path / "runs")
    monkeypatch.setattr("tool_eval_bench.storage.db._default_db_path", lambda: db_path)

    with RunRepository(db_path=db_path) as repo:
        for idx, pts in (("run-a", None), ("run-b", 2)):
            repo.upsert_scenario_run(
                {
                    "run_id": idx,
                    "run_type": "tool_eval",
                    "status": "completed",
                    "config": {"model": idx, "backend": "vllm", "base_url": "http://***:8000"},
                    "metadata": {"model": idx},
                    "scores": {
                        "final_score": 0 if pts is None else 100,
                        "scenario_results": [
                            {
                                "scenario_id": "TC-01",
                                "status": "fail" if pts is None else "pass",
                                "points": pts,
                            }
                        ],
                    },
                }
            )

    app = create_app(output_dir=out_dir, token=TEST_TOKEN)
    async with web_client(app) as client:
        resp = await client.get("/api/compare", params={"left": "run-a", "right": "run-b"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["summary"]["total_points_a"] == 0
        assert body["summary"]["total_points_b"] == 2
        assert body["summary"]["total_delta"] == 2


@pytest.mark.asyncio
async def test_csv_export_neutralizes_formula_injection_in_web_and_cli_without_touching_numbers(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rich.console import Console

    from tests.web_support import AUTH_HEADERS
    from tool_eval_bench.cli import leaderboard
    from tool_eval_bench.web.app import _sanitize_csv_cell as _web_sanitize_csv_cell

    db_path = str(tmp_path / "benchmarks.sqlite")
    out_dir_runs = str(tmp_path / "runs")
    monkeypatch.setattr("tool_eval_bench.storage.db._default_db_path", lambda: db_path)

    # Unit-level boundary check on helper behavior for numbers, booleans, empty, and prefixes.
    for fn in (_web_sanitize_csv_cell, leaderboard._sanitize_csv_cell):
        assert fn("=1+1") == "'=1+1"
        assert fn("+SUM(A1:A2)") == "'+SUM(A1:A2)"
        assert fn("-2+3") == "'-2+3"
        assert fn("@IMP") == "'@IMP"
        assert fn("\ttabbed") == "'\ttabbed"
        assert fn("\rcr") == "'\rcr"
        assert fn("safe-model") == "safe-model"
        assert fn("") == ""
        assert fn(-42) == -42
        assert fn(-0.25) == -0.25

    with RunRepository(db_path=db_path) as repo:
        repo.upsert_scenario_run(
            {
                "run_id": "csv-formula-run",
                "run_type": "tool_eval",
                "status": "completed",
                "config": {
                    "model": "=CMD|' /C calc'!A0",
                    "backend": "@gateway",
                    "base_url": "http://***:8000",
                    "scenario_count": 1,
                    "temperature": 0.0,
                },
                "metadata": {
                    "model": "=CMD|' /C calc'!A0",
                    "engine_name": "+vLLM",
                    "engine_version": "-0.8.5",
                    "quantization": "\tAWQ",
                    "server_model_root": "\r/models/qwen",
                    "tool_version": "0.1.0",
                    "max_model_len": 32768,
                    "temperature": 0.0,
                },
                "scores": {
                    "final_score": 85,
                    "rating": "★★★★ Good",
                    "total_points": 17,
                    "max_points": 20,
                    "total_tokens": 1024,
                    "scenario_results": [{"scenario_id": "TC-01", "status": "pass", "points": 2}],
                    "category_scores": [{"category": "A", "percent": 85.0}],
                },
            }
        )
        repo.upsert_scenario_run(
            {
                "run_id": "csv-clean-run",
                "run_type": "tool_eval",
                "status": "completed",
                "config": {
                    "model": "qwen2.5-7b",
                    "backend": "vllm",
                    "base_url": "http://***:8000",
                    "scenario_count": 1,
                    "temperature": 0.2,
                },
                "metadata": {
                    "model": "qwen2.5-7b",
                    "engine_name": "vLLM",
                    "engine_version": "0.8.5",
                    "quantization": "FP8",
                    "server_model_root": "/models/qwen2.5-7b",
                    "temperature": 0.2,
                },
                "scores": {
                    "final_score": 90,
                    "rating": "★★★★★ Excellent",
                    "total_points": 18,
                    "max_points": 20,
                    "total_tokens": 900,
                    "scenario_results": [{"scenario_id": "TC-01", "status": "pass", "points": 2}],
                    "category_scores": [{"category": "A", "percent": 90.0}],
                },
            }
        )

    # 1. Web CSV and JSON export.
    app = create_app(token=TEST_TOKEN, output_dir=out_dir_runs)
    async with web_client(app) as client:
        csv_resp = await client.get("/api/export?format=csv", headers=AUTH_HEADERS)
        assert csv_resp.status_code == 200
        web_rows = {r["run_id"]: r for r in csv.DictReader(io.StringIO(csv_resp.text))}

        json_resp = await client.get("/api/export?format=json", headers=AUTH_HEADERS)
        assert json_resp.status_code == 200
        json_rows = {r["run_id"]: r for r in json_resp.json()}

    formula_row = web_rows["csv-formula-run"]
    assert formula_row["model"].startswith("'=CMD")
    assert formula_row["backend"] == "'@gateway"
    assert formula_row["cohort"].startswith("'@gateway/1")
    assert formula_row["engine_name"] == "'+vLLM"
    assert formula_row["engine_version"] == "'-0.8.5"
    assert formula_row["quantization"] == "'\tAWQ"
    assert formula_row["server_model_root"] == "'\r/models/qwen"
    assert formula_row["final_score"] == "85"
    assert formula_row["temperature"] == "0.0"

    clean_row = web_rows["csv-clean-run"]
    assert clean_row["model"] == "qwen2.5-7b"
    assert clean_row["engine_name"] == "vLLM"
    assert clean_row["final_score"] == "90"
    assert clean_row["temperature"] == "0.2"

    # JSON export preserves raw strings without spreadsheet quoting.
    assert json_rows["csv-formula-run"]["model"] == "=CMD|' /C calc'!A0"

    # 2. CLI CSV export + bracketed output path escaping.
    out_dir = tmp_path / "dir[bold]"
    out_dir.mkdir(parents=True)
    out_csv = out_dir / "export[1].csv"
    cli_buf = io.StringIO()
    cli_console = Console(file=cli_buf, width=140, no_color=True)
    leaderboard.export_runs(cli_console, fmt="csv", output=str(out_csv))
    cli_out = cli_buf.getvalue().replace("\n", "")
    assert "dir[bold]" in cli_out
    assert "export[1].csv" in cli_out

    with out_csv.open("r", encoding="utf-8", newline="") as fh:
        cli_rows = {r["run_id"]: r for r in csv.DictReader(fh)}
    assert cli_rows["csv-formula-run"]["model"].startswith("'=CMD")
    assert cli_rows["csv-formula-run"]["engine_name"] == "'+vLLM"
    assert cli_rows["csv-formula-run"]["engine_version"] == "'-0.8.5"
    assert cli_rows["csv-formula-run"]["quantization"] == "'\tAWQ"
    assert cli_rows["csv-clean-run"]["model"] == "qwen2.5-7b"
    assert cli_rows["csv-clean-run"]["final_score"] == "90"
