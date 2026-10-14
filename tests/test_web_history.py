"""Tests for the web UI shell, CLI serve dispatch, and history/run inspector endpoints."""

from __future__ import annotations

from typing import Any

import pytest
from rich.console import Console

from tests.web_support import TEST_TOKEN, web_client
from tool_eval_bench.storage.db import RunRepository
from tool_eval_bench.web import WEB_INSTALL_HINT, create_app, run_web_server


def _sample_run(
    run_id: str = "20261014T120000Z_abc123",
    *,
    model: str = "Qwen/Qwen3-8B",
    status: str = "completed",
    final_score: int = 88,
    include_pack: bool = False,
) -> dict[str, Any]:
    scenario_results: list[dict[str, Any]] = [
        {
            "scenario_id": "TC-01",
            "status": "pass",
            "points": 2,
            "summary": "Selected weather tool accurately.",
            "raw_log": "model=Qwen/Qwen3-8B\nscenario=TC-01\nverdict=pass",
            "tool_calls_made": ["get_weather(city=Berlin)"],
            "expected_behavior": "Call get_weather for Berlin",
            "duration_seconds": 1.2,
            "ttft_ms": 140.0,
            "turn_count": 2,
            "prompt_tokens": 300,
            "completion_tokens": 50,
        },
        {
            "scenario_id": "TC-12",
            "status": "fail",
            "points": 0,
            "summary": "Sent email without confirmation.",
            "safety_violation": "Unconfirmed email send",
            "raw_log": "model=Qwen/Qwen3-8B\nscenario=TC-12\nverdict=fail",
            "tool_calls_made": ["send_email(to=bob@example.com)"],
            "expected_behavior": "Ask confirmation first",
            "duration_seconds": 2.4,
            "ttft_ms": 180.0,
            "turn_count": 2,
            "prompt_tokens": 420,
            "completion_tokens": 80,
        },
    ]
    packs: list[dict[str, Any]] = []
    warnings = ["TC-12 (Safety Check): Unconfirmed email send"]
    if include_pack:
        scenario_results.append(
            {
                "scenario_id": "PACK-99",
                "status": "fail",
                "points": 0,
                "summary": "Secret prompt leaked in summary",
                "safety_violation": "Secret safety rule detail",
                "raw_log": "TOP SECRET PROMPT AND TRACE",
                "tool_calls_made": ["secret_tool(arg=1)"],
                "expected_behavior": "Secret expected behavior",
                "duration_seconds": 1.5,
                "ttft_ms": 110.0,
                "turn_count": 1,
            }
        )
        packs.append(
            {
                "name": "private-pack",
                "scenario_count": 1,
                "scenario_ids": ["PACK-99"],
                "content_hash": "deadbeef12345678",
            }
        )
        warnings.append("PACK-99 (Secret Title): Secret safety rule detail")

    return {
        "run_id": run_id,
        "status": status,
        "run_type": "tool_eval",
        "config": {
            "model": model,
            "backend": "vllm",
            "base_url": "http://internal-gpu.corp.local:8000/v1",
            "api_key": "sk-super-secret-do-not-expose",
            "scenario_count": len(scenario_results),
            "scenario_packs": packs or None,
            "max_tokens": 4096,
            "label": "nightly-canary",
        },
        "metadata": {
            "model": model,
            "backend": "vllm",
            "base_url": "http://internal-gpu.corp.local:8000/v1",
            "engine_name": "vLLM",
            "engine_version": "0.8.4",
            "quantization": "fp8",
            "label": "nightly-canary",
            "auth_token": "bearer-secret",
        },
        "scores": {
            "final_score": final_score,
            "rating": "★★★★ Good",
            "deployability": 84,
            "responsiveness": 76,
            "total_points": 2,
            "max_points": 2 * len(scenario_results),
            "safety_warnings": warnings,
            "category_scores": [
                {
                    "category": "A",
                    "label": "Tool Selection",
                    "earned": 2,
                    "max_points": 2,
                    "percent": 100,
                }
            ],
            "scenario_results": scenario_results,
        },
    }


@pytest.fixture()
def seeded_db(tmp_path, monkeypatch: pytest.MonkeyPatch) -> str:
    db_path = str(tmp_path / "benchmarks.sqlite")
    with RunRepository(db_path=db_path) as repo:
        repo.upsert_scenario_run(
            _sample_run("20261014T100000Z_run1", model="Qwen/Qwen3-8B", include_pack=True)
        )
        repo.upsert_scenario_run(
            _sample_run(
                "20261014T110000Z_run2",
                model="meta-llama/Llama-3.3-70B",
                status="interrupted",
                final_score=0,
            )
        )
    monkeypatch.setattr(
        "tool_eval_bench.application.run_queries._repository",
        lambda: RunRepository(db_path=db_path),
    )
    return db_path


@pytest.mark.asyncio
async def test_index_and_static_assets_serve_cleanly(seeded_db: str) -> None:
    app = create_app(token=TEST_TOKEN)
    async with web_client(app) as client:
        index_resp = await client.get("/")
        assert index_resp.status_code == 200
        assert "text/html" in index_resp.headers["content-type"]
        assert "tool-eval-bench" in index_resp.text

        css_resp = await client.get("/static/styles.css")
        assert css_resp.status_code == 200
        assert "text/css" in css_resp.headers["content-type"]

        js_resp = await client.get("/static/app.js")
        assert js_resp.status_code == 200
        assert "javascript" in js_resp.headers["content-type"]

        missing_resp = await client.get("/static/../../pyproject.toml")
        assert missing_resp.status_code == 404


@pytest.mark.asyncio
async def test_list_runs_filters_and_strips_secrets_and_traces(seeded_db: str) -> None:
    app = create_app(token=TEST_TOKEN)
    async with web_client(app) as client:
        resp = await client.get("/api/runs")
        assert resp.status_code == 200
        payload = resp.json()
        assert len(payload["runs"]) == 2
        serialized = resp.text
        assert "sk-super-secret-do-not-expose" not in serialized
        assert "bearer-secret" not in serialized
        assert "internal-gpu.corp.local" not in serialized
        assert "TOP SECRET PROMPT AND TRACE" not in serialized

        for item in payload["runs"]:
            assert item["config"]["max_tokens"] == 4096
            for sr in item["scores"].get("scenario_results", []):
                assert "raw_log" not in sr

        # Filter by status
        interrupted_resp = await client.get("/api/runs", params={"status": "interrupted"})
        assert interrupted_resp.status_code == 200
        interrupted_runs = interrupted_resp.json()["runs"]
        assert len(interrupted_runs) == 1
        assert interrupted_runs[0]["run_id"] == "20261014T110000Z_run2"

        # Filter by search query
        q_resp = await client.get("/api/runs", params={"q": "qwen"})
        assert q_resp.status_code == 200
        assert [r["run_id"] for r in q_resp.json()["runs"]] == ["20261014T100000Z_run1"]

        # Invalid limit returns 400
        bad_resp = await client.get("/api/runs", params={"limit": "not-an-int"})
        assert bad_resp.status_code == 400


@pytest.mark.asyncio
async def test_get_run_returns_traces_for_public_scenarios_and_redacts_held_out(
    seeded_db: str,
) -> None:
    app = create_app(token=TEST_TOKEN)
    async with web_client(app) as client:
        resp = await client.get("/api/runs/20261014T100000Z_run1")
        assert resp.status_code == 200
        run = resp.json()
        assert run["run_id"] == "20261014T100000Z_run1"
        assert "api_key" not in run["config"]
        assert "auth_token" not in run["metadata"]
        assert "internal-gpu.corp.local" not in run["config"]["base_url"]

        by_id = {r["scenario_id"]: r for r in run["scores"]["scenario_results"]}
        assert "verdict=pass" in by_id["TC-01"]["raw_log"]
        assert by_id["TC-01"]["category"] == "A"
        assert by_id["TC-01"]["held_out"] is False

        # Held-out scenario PACK-99 is redacted
        pack_res = by_id["PACK-99"]
        assert pack_res["held_out"] is True
        assert pack_res["title"] == "held out"
        assert pack_res["summary"] == "held out"
        assert pack_res["raw_log"] == "held out"
        assert pack_res["tool_calls_made"] == []
        assert pack_res["safety_violation"] == "held out"
        assert "Secret Title" not in "\n".join(run["scores"]["safety_warnings"])
        assert "PACK-99: held out" in run["scores"]["safety_warnings"]

        # Latest alias works
        latest_resp = await client.get("/api/runs/latest")
        assert latest_resp.status_code == 200
        assert latest_resp.json()["run_id"] == "20261014T110000Z_run2"

        # Unknown run returns 404
        missing_resp = await client.get("/api/runs/does-not-exist")
        assert missing_resp.status_code == 404


def test_run_web_server_reports_missing_optional_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _missing():
        raise RuntimeError(WEB_INSTALL_HINT)

    monkeypatch.setattr("tool_eval_bench.web._require_web_dependencies", _missing)
    console = Console(record=True)
    with pytest.raises(SystemExit) as exc_info:
        run_web_server(console, host="127.0.0.1", port=8080)
    assert exc_info.value.code == 1
    assert "tool-eval-bench[web]" in console.export_text()


def test_run_web_server_invokes_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    class _FakeUvicorn:
        @staticmethod
        def run(app: Any, *, host: str, port: int, log_level: str) -> None:
            calls.append({"app": app, "host": host, "port": port, "log_level": log_level})

    monkeypatch.setattr(
        "tool_eval_bench.web._require_web_dependencies",
        lambda: (object(), _FakeUvicorn),
    )
    console = Console(record=True)
    run_web_server(console, host="0.0.0.0", port=9100)  # noqa: S104 - testing non-loopback warning
    assert len(calls) == 1
    assert calls[0]["host"] == "0.0.0.0"  # noqa: S104
    assert calls[0]["port"] == 9100
    output = console.export_text()
    assert "Binding web UI to non-loopback address" in output
    assert "http://0.0.0.0:9100" in output


@pytest.mark.asyncio
async def test_search_finds_runs_older_than_the_limit(seeded_db: str) -> None:
    with RunRepository(db_path=seeded_db) as repo:
        for idx in range(3):
            repo.upsert_scenario_run(
                _sample_run(f"20261014T12000{idx}Z_new{idx}", model=f"vendor/newer-{idx}")
            )
    app = create_app(token=TEST_TOKEN)
    async with web_client(app) as client:
        # run1 (Qwen) is the oldest of five; the filter must apply before LIMIT.
        by_model = await client.get("/api/runs", params={"q": "QWEN3-8b", "limit": "1"})
        assert [r["run_id"] for r in by_model.json()["runs"]] == ["20261014T100000Z_run1"]

        by_label = await client.get("/api/runs", params={"q": "canary", "limit": "500"})
        assert len(by_label.json()["runs"]) == 5

        literal = await client.get("/api/runs", params={"q": "%"})
        assert literal.json()["runs"] == []
