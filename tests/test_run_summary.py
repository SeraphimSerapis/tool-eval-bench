"""What the web UI shows for each stored run type: headlines, details, and same-session links."""

from __future__ import annotations

import re
from importlib.resources import files
from pathlib import Path
from typing import Any

import pytest

from tests.web_support import TEST_TOKEN, web_client
from tool_eval_bench.application.run_summary import (
    RUN_TYPE_LABELS,
    detail,
    headline,
    session_entry,
    type_label,
)
from tool_eval_bench.storage.db import RunRepository
from tool_eval_bench.web import create_app


def _run(run_type: str, scores: dict[str, Any], **config: Any) -> dict[str, Any]:
    return {"run_id": f"r-{run_type}", "run_type": run_type, "status": "completed",
            "config": config, "scores": scores}  # fmt: skip


SPEC = _run(
    "spec-bench",
    {
        "samples": 2,
        "failed": 0,
        "results": [
            {
                "prompt_type": "code",
                "depth": 0,
                "tg_tps": 80.0,
                "effective_tg_tps": 120.0,
                "acceptance_rate": 0.8,
                "acceptance_length": 3.2,
                "speedup_ratio": None,
                "ttft_ms": 41.6,
            },
            {
                "prompt_type": "filler",
                "depth": 4096,
                "tg_tps": 70.0,
                "effective_tg_tps": 100.0,
                "acceptance_rate": 0.6,
                "acceptance_length": 2.4,
                "speedup_ratio": None,
                "ttft_ms": 90.2,
            },
        ],  # fmt: skip
    },
    method="mtp",
)
PERF = _run(
    "perf",
    {
        "samples": 3,
        "results": [
            {"depth": 0, "concurrency": 1, "pp_tps": 3000.0, "tg_tps": 55.0, "ttft_ms": 120.0},
            {"depth": 0, "concurrency": 4, "pp_tps": 5000.0, "tg_tps": 160.0, "ttft_ms": 300.0},
            {"depth": 32768, "concurrency": 1, "pp_tps": 2500.0, "tg_tps": 40.0, "ttft_ms": 900.0},
        ],
    },
    perf_preset="long",
    pp=2048,
    tg=128,
)
PRESSURE = _run(
    "context-pressure",
    {
        "levels": 3,
        "planned_levels": 3,
        "breaking_point": 0.75,
        "first_degradation": 0.9,
        "level_results": [
            {"ratio": 0.5, "fill_tokens": 60000, "score_pct": 100.0, "pass_count": 5},
            {"ratio": 0.75, "fill_tokens": 90000, "score_pct": 100.0, "pass_count": 5},
            {
                "ratio": 0.9,
                "fill_tokens": 108000,
                "score_pct": 40.0,
                "pass_count": 2,
                "error": "Sweep level failed: http://10.0.0.5:8000 refused",
            },
        ],  # fmt: skip
    },
)
NEEDLE = _run(
    "needle",
    {
        "final_score": 75,
        "accuracy": 75.0,
        "effective_context": 65536,
        "cells": [
            {"context_tokens": 32768, "depth_percent": 50.0, "found": True},
            {"context_tokens": 65536, "depth_percent": 50.0, "found": True},
            {"context_tokens": 131072, "depth_percent": 50.0, "found": False},
            {"context_tokens": 131072, "depth_percent": 90.0, "found": True},
            {"context_tokens": 262144, "depth_percent": 50.0, "found": False, "error": True},
        ],
    },
)
GSM8K = _run("gsm8k", {"final_score": 90, "accuracy": 90.0, "correct": 45, "total": 50,
                       "errors": 0, "dataset_size": 1319, "n_shots": 8})  # fmt: skip


class TestHeadline:
    def test_spec_decode_shows_effective_decode_and_acceptance(self) -> None:
        assert headline(SPEC) == {
            "label": "effective decode",
            "value": 110.0,
            "unit": "tok/s",
            "detail": "70% accepted",
        }

    def test_throughput_uses_single_request_decode_at_the_shallowest_depth(self) -> None:
        figure = headline(PERF)
        assert figure is not None
        assert (figure["label"], figure["value"], figure["unit"]) == ("decode", 55.0, "tok/s")

    def test_pressure_reports_where_it_holds_or_degrades(self) -> None:
        assert headline(PRESSURE) == {"label": "holds to", "value": 75.0, "unit": "% fill",
                                      "detail": None}  # fmt: skip
        degraded = _run("context-pressure", {"breaking_point": None, "first_degradation": 0.5})
        figure = headline(degraded)
        assert figure is not None and figure["label"] == "degrades at"
        assert headline(_run("context-pressure", {"levels": 0})) is None

    def test_needle_counts_errored_cells_out_of_accuracy(self) -> None:
        figure = headline(NEEDLE)
        assert figure is not None
        assert (figure["value"], figure["detail"]) == (75.0, "effective to 65,536 tokens")

    def test_accuracy_plugins_use_the_card_reader(self) -> None:
        figure = headline(GSM8K)
        assert figure is not None
        assert (figure["value"], figure["detail"]) == (90.0, "50 of 1,319 q, 8-shot")

    def test_every_request_errored_has_no_figure(self) -> None:
        assert headline(_run("gsm8k", {"correct": 0, "total": 5, "errors": 5})) is None

    def test_tool_eval_and_empty_runs_have_no_headline(self) -> None:
        assert headline(_run("tool_eval", {"final_score": 90})) is None
        assert headline({"run_id": "old", "scores": {"final_score": 90}}) is None
        assert headline(_run("spec-bench", {"results": []})) is None
        assert headline(_run("perf", {"samples": 3})) is None

    def test_unknown_modes_fall_back_to_a_stored_accuracy(self) -> None:
        figure = headline(_run("toolscale", {"accuracy": 62.5}))
        assert figure is not None and (figure["label"], figure["value"]) == ("accuracy", 62.5)


class TestDetail:
    def test_spec_decode_table_drops_columns_nothing_measured(self) -> None:
        result = detail(SPEC)
        assert result is not None and result["table"] is not None
        keys = [column["key"] for column in result["table"]["columns"]]
        assert "speedup_ratio" not in keys
        assert result["table"]["rows"][0]["acceptance_rate"] == 80.0
        assert {"label": "Method", "value": "mtp", "unit": ""} in result["stats"]
        assert {"label": "Mean acceptance", "value": 70.0, "unit": "%"} in result["stats"]

    def test_throughput_table_has_one_row_per_cell(self) -> None:
        result = detail(PERF)
        assert result is not None and result["table"] is not None
        rows = result["table"]["rows"]
        assert [(row["depth"], row["concurrency"]) for row in rows] == [(0, 1), (0, 4), (32768, 1)]
        # Above concurrency 1 the stored rate is the batch total.
        assert (rows[1]["tg_tps"], rows[1]["tg_tps_total"]) == (40.0, 160.0)

    def test_pressure_table_flags_errors_without_their_text(self) -> None:
        result = detail(PRESSURE)
        assert result is not None and result["table"] is not None
        assert [row["outcome"] for row in result["table"]["rows"]] == ["", "", "error"]
        assert "10.0.0.5" not in repr(result)

    def test_needle_table_marks_each_cell(self) -> None:
        result = detail(NEEDLE)
        assert result is not None and result["table"] is not None
        outcomes = [row["outcome"] for row in result["table"]["rows"]]
        assert outcomes == ["found", "found", "missed", "found", "error"]

    def test_tool_eval_has_no_detail(self) -> None:
        assert detail(_run("tool_eval", {})) is None

    def test_generic_stats_skip_non_numbers(self) -> None:
        result = detail(_run("decision", {"accuracy": 80.0, "rating": "Good", "flag": True}))
        assert result == {"stats": [{"label": "Accuracy", "value": 80.0, "unit": ""}],
                          "table": None}  # fmt: skip


def test_session_entry_keeps_the_score_only_for_tool_calls() -> None:
    assert session_entry(NEEDLE)["final_score"] is None
    tool = session_entry(_run("tool_eval", {"final_score": 93}))
    assert (tool["final_score"], tool["type_label"], tool["headline"]) == (93, "Tool calls", None)


def test_unknown_types_get_a_readable_label() -> None:
    assert type_label("spec-bench") == "Spec decode"
    assert type_label("new_mode") == "New mode"


def test_the_run_type_filter_offers_only_stored_types() -> None:
    html = files("tool_eval_bench.web.static").joinpath("index.html").read_text(encoding="utf-8")
    select = html[html.index('id="history-type-filter"') :]
    select = select[: select.index("</select>")]
    values = {value for value in re.findall(r'<option value="([^"]*)"', select) if value}
    assert values <= set(RUN_TYPE_LABELS)
    assert {"spec-bench", "context-pressure"} <= values


@pytest.fixture()
def session_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    with RunRepository() as repo:
        for run_id, run_type, scores in (
            ("tool", "tool_eval", {"final_score": 93, "scenario_results": []}),
            ("spec", "spec-bench", SPEC["scores"]),
            ("needle", "needle", NEEDLE["scores"]),
        ):
            repo.upsert_scenario_run(
                {
                    "run_id": run_id,
                    "run_type": run_type,
                    "config": {"model": "qwen3.8-flash-next"},
                    "metadata": {"session_id": "sess-1"},
                    "scores": scores,
                }
            )
        repo.upsert_scenario_run(
            {
                "run_id": "other",
                "config": {"model": "qwen3.8-flash-next"},
                "metadata": {"session_id": "sess-2"},
                "scores": {"final_score": 50},
            }
        )


@pytest.mark.asyncio
async def test_a_run_lists_the_other_runs_of_its_session(session_db: None) -> None:
    app = create_app(token=TEST_TOKEN)
    async with web_client(app) as client:
        tool = (await client.get("/api/runs/tool")).json()
        spec = (await client.get("/api/runs/spec")).json()
        listed = (await client.get("/api/runs")).json()["runs"]

    assert [entry["run_id"] for entry in tool["session"]] == ["spec", "needle"]
    assert tool["session"][0]["headline"]["label"] == "effective decode"
    assert tool["detail"] is None
    assert [entry["run_id"] for entry in spec["session"]] == ["tool", "needle"]
    assert spec["session"][0]["final_score"] == 93
    assert spec["detail"]["table"]["rows"]
    by_id = {run["run_id"]: run for run in listed}
    assert by_id["spec"]["type_label"] == "Spec decode"
    assert by_id["spec"]["headline"]["unit"] == "tok/s"
    assert by_id["tool"]["headline"] is None


@pytest.mark.asyncio
async def test_a_run_without_a_session_lists_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with RunRepository() as repo:
        repo.upsert_scenario_run({"run_id": "lone", "config": {"model": "m"}, "scores": {}})
    app = create_app(token=TEST_TOKEN)
    async with web_client(app) as client:
        payload = (await client.get("/api/runs/lone")).json()
    assert payload["session"] == []
