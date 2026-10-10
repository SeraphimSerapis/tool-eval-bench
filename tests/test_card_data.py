"""Mapping stored runs onto a results card, and loading them by session or run id."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Collection
from typing import Any

import pytest

from tool_eval_bench.application import card_data
from tool_eval_bench.application.card_data import (
    CardDataUnavailable,
    build_card_data,
    load_card_data,
)
from tool_eval_bench.domain.card import NeedleCell, PerfPoint
from tool_eval_bench.evals.scenarios import ALL_SCENARIOS, ALL_SCENARIOS_WITH_HARDMODE

# Values that must never reach a card. Each one is planted somewhere a careless
# mapping could pick it up.
PRIVATE = {
    "hostname": "PRIVATE-HOST-prometheus",
    "base_url": "http://10.9.8.7:8000/PRIVATE-URL",
    "session_id": "PRIVATE-SESSION-1234",
    "run_id": "2026-10-10T12-00-00Z_PRIVATE-RUN",
    "label": "PRIVATE-LABEL tim's secret tuning",
    "tokenizer": "/home/alice/PRIVATE-TOKENIZER/tokenizer.json",
    "system_prompt": "PRIVATE-SYSTEM-PROMPT you are a pirate",
    "model_dir": "PRIVATE-MODEL-DIR",
}


def _metadata(**overrides: Any) -> dict[str, Any]:
    metadata = {
        "tool_version": "3.1.0",
        "git_sha": "abc1234",
        "hostname": PRIVATE["hostname"],
        "platform_info": "Linux-PRIVATE",
        "base_url": PRIVATE["base_url"],
        "session_id": PRIVATE["session_id"],
        "label": PRIVATE["label"],
        "system_prompt": PRIVATE["system_prompt"],
        "server_model_id": "qwen",
        "server_model_root": f"/home/alice/{PRIVATE['model_dir']}/Qwen3-32B-AWQ",
        "engine_name": "vLLM",
        "engine_version": "0.11.0",
        "quantization": "awq",
        "max_model_len": 131072,
    }
    metadata.update(overrides)
    return metadata


def _perf_row(
    depth: int,
    concurrency: int,
    *,
    pp_tps: float = 1000.0,
    tg_tps: float = 50.0,
    ttft_ms: float = 200.0,
    context_prefill: bool = False,
    pp: int = 2048,
    tg: int = 128,
    pp_estimated: bool = False,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "requested_pp": depth if context_prefill else pp,
        "requested_depth": depth,
        "pp_tokens": pp,
        "tg_tokens": tg,
        "depth": depth,
        "concurrency": concurrency,
        "ttft_ms": ttft_ms,
        "total_ms": ttft_ms + 1000,
        "pp_tps": pp_tps,
        "tg_tps": tg_tps,
        "pp_estimated": pp_estimated,
        "calibration_confidence": "llama-benchy",
    }
    if context_prefill:
        row["is_context_prefill"] = True
    return row


def _tool_eval_run(created_at: str = "2026-10-10T12:05:00+00:00", **scores: Any) -> dict:
    base_scores: dict[str, Any] = {
        "final_score": 82,
        "total_points": 82,
        "max_points": 100,
        "rating": "★★★★ Good",
        "category_scores": [
            {
                "category": "A",
                "label": "Tool Selection",
                "earned": 6,
                "max": 6,
                "percent": 100.0,
                "pass_count": 3,
                "partial_count": 0,
                "fail_count": 0,
            },
            {
                "category": "K",
                "label": "Safety & Boundaries",
                "earned": 3,
                "max": 6,
                "percent": 50.0,
                "pass_count": 1,
                "partial_count": 1,
                "fail_count": 1,
            },
        ],
        "scenario_results": [],
    }
    base_scores.update(scores)
    return {
        "run_id": PRIVATE["run_id"],
        "created_at": created_at,
        "status": "completed",
        "model": "qwen",
        "run_type": "tool_eval",
        "session_id": PRIVATE["session_id"],
        "report_path": f"/home/alice/runs/{PRIVATE['label']}.md",
        "config": {
            "model": "qwen",
            "backend": "vllm",
            "base_url": PRIVATE["base_url"],
            "seed": 7,
            "system_prompt": PRIVATE["system_prompt"],
            "config_fingerprint": "0123456789abcdef",
        },
        "scores": base_scores,
        "metadata": _metadata(),
    }


def _perf_run(created_at: str = "2026-10-10T12:00:00+00:00", **config: Any) -> dict:
    return {
        "run_id": PRIVATE["run_id"] + "-perf",
        "created_at": created_at,
        "status": "completed",
        "model": "qwen",
        "run_type": "perf",
        "config": {
            "model": "qwen",
            "base_url": PRIVATE["base_url"],
            "mode": "perf-only",
            "pp": 2048,
            "tg": 128,
            "tokenizer": PRIVATE["tokenizer"],
            "benchy_args": ["--api-key", "***"],
            "perf_preset": "long",
            "config_fingerprint": "fedcba9876543210",
            **config,
        },
        "scores": {
            "samples": 4,
            "results": [
                _perf_row(0, 1, tg_tps=60.0, ttft_ms=100.0),
                _perf_row(0, 1, tg_tps=40.0, ttft_ms=300.0),
                _perf_row(0, 4, tg_tps=160.0),
                _perf_row(32768, 1, pp_tps=3000.0, tg_tps=30.0, ttft_ms=900.0),
            ],
        },
        "metadata": _metadata(),
    }


def _needle_run(created_at: str = "2026-10-10T12:10:00+00:00", *, cells: bool = True) -> dict:
    scores: dict[str, Any] = {
        "final_score": 75,
        "accuracy": 75.0,
        "rating": "★★★ Adequate",
        "retrieved": 3,
        "total": 4,
        "context_lengths": [1024, 4096],
        "depths": [0.0, 1.0],
        "by_length": {"1024": 100.0, "4096": 50.0},
        "effective_context": 1024,
    }
    if cells:
        scores["cells"] = [
            {"context_tokens": 1024, "depth_percent": 0.0, "found": True},
            {"context_tokens": 1024, "depth_percent": 1.0, "found": True},
            {"context_tokens": 4096, "depth_percent": 0.0, "found": True},
            {"context_tokens": 4096, "depth_percent": 1.0, "found": False},
        ]
    return {
        "run_id": PRIVATE["run_id"] + "-needle",
        "created_at": created_at,
        "status": "completed",
        "model": "qwen",
        "run_type": "needle",
        "config": {"model": "qwen", "base_url": PRIVATE["base_url"], "seed": None},
        "scores": scores,
        "metadata": _metadata(),
    }


def _plugin_run(kind: str, accuracy: float, created_at: str, **scores: Any) -> dict:
    return {
        "run_id": f"{PRIVATE['run_id']}-{kind}",
        "created_at": created_at,
        "status": "completed",
        "model": "qwen",
        "run_type": kind,
        "config": {"model": "qwen", "base_url": PRIVATE["base_url"], "mode": kind},
        "scores": {"final_score": round(accuracy), "accuracy": accuracy, **scores},
        "metadata": _metadata(),
    }


def _full_session() -> list[dict]:
    return [
        _perf_run(),
        _tool_eval_run(),
        _needle_run(),
        _plugin_run("gsm8k", 91.5, "2026-10-10T12:11:00+00:00", total=200, n_shots=8),
        _plugin_run("mmlu", 70.0, "2026-10-10T12:12:00+00:00", total=500, n_shots=5),
        _plugin_run("ifeval", 80.0, "2026-10-10T12:13:00+00:00", total=541),
    ]


class TestFullSession:
    def test_metadata_comes_from_named_fields(self) -> None:
        card = build_card_data(_full_session(), hardware="  RTX 5090 32GB ")
        assert card.model == "Qwen3-32B-AWQ"
        assert card.engine == "vLLM"
        assert card.engine_version == "0.11.0"
        assert card.quantization == "awq"
        assert card.max_context == 131072
        assert card.tool_version == "3.1.0"
        assert card.hardware == "RTX 5090 32GB"
        assert card.seed == 7
        # The newest run on the card dates it.
        assert card.created_at == "2026-10-10T12:13:00+00:00"
        # The scored run's fingerprint, cut to ten characters.
        assert card.fingerprint == "0123456789"

    def test_tool_eval_summary(self) -> None:
        summary = build_card_data(_full_session()).tool_eval
        assert summary is not None
        assert summary.score == 82.0
        assert summary.rating == "★★★★ Good"
        assert (summary.passed, summary.partial, summary.failed, summary.total) == (4, 1, 1, 6)
        assert [(c.code, c.name, c.percent) for c in summary.categories] == [
            ("A", "Tool Selection", 100.0),
            ("K", "Safety & Boundaries", 50.0),
        ]
        assert summary.safety_capped is False
        assert summary.hardmode is False

    def test_perf_sweep_and_workload(self) -> None:
        card = build_card_data(_full_session())
        assert card.perf_preset == "long"
        assert (card.perf_pp, card.perf_tg) == (2048, 128)
        assert [(p.depth, p.concurrency) for p in card.perf] == [(0, 1), (0, 4), (32768, 1)]

    def test_needle_grid(self) -> None:
        needle = build_card_data(_full_session()).needle
        assert needle is not None
        assert needle.accuracy == 0.75
        assert needle.effective_context == 1024
        assert needle.cells[-1] == NeedleCell(context_tokens=4096, depth_percent=1.0, found=False)
        assert len(needle.cells) == 4

    def test_accuracy_results_in_fixed_order_with_details(self) -> None:
        accuracy = build_card_data(_full_session()).accuracy
        assert [(a.name, a.accuracy, a.detail) for a in accuracy] == [
            ("GSM8K", 0.915, "200 q, 8-shot"),
            ("MMLU", 0.70, "500 q, 5-shot"),
            ("IFEval", 0.80, "541 prompts"),
        ]

    def test_newest_run_of_each_kind_wins(self) -> None:
        old = _plugin_run("gsm8k", 10.0, "2026-10-09T00:00:00+00:00", total=20, n_shots=8)
        card = build_card_data([*_full_session(), old])
        assert card.accuracy[0].accuracy == 0.915

    def test_input_order_does_not_matter(self) -> None:
        runs = _full_session()
        assert build_card_data(list(reversed(runs))) == build_card_data(runs)


class TestPerfCells:
    def test_repeated_rows_in_a_cell_are_averaged(self) -> None:
        point = build_card_data([_perf_run()]).perf[0]
        assert point == PerfPoint(
            depth=0,
            concurrency=1,
            pp_tps=1000.0,
            tg_tps=50.0,
            tg_tps_total=50.0,
            ttft_ms=200.0,
        )

    def test_concurrent_tg_is_stored_as_the_batch_total(self) -> None:
        point = build_card_data([_perf_run()]).perf[1]
        assert point.concurrency == 4
        assert point.tg_tps_total == 160.0
        assert point.tg_tps == 40.0

    def test_measured_per_request_rate_beats_the_estimate(self) -> None:
        run = _perf_run()
        run["scores"]["results"] = [
            {**_perf_row(0, 4, tg_tps=160.0), "tg_req_tps": 46.0},
            {**_perf_row(0, 4, tg_tps=140.0), "tg_req_tps": 42.0},
        ]
        point = build_card_data([run]).perf[0]
        assert point.tg_tps == 44.0
        assert point.tg_tps_total == 150.0

    def test_a_row_without_the_measured_rate_falls_back_for_the_cell(self) -> None:
        run = _perf_run()
        run["scores"]["results"] = [
            {**_perf_row(0, 4, tg_tps=160.0), "tg_req_tps": 46.0},
            _perf_row(0, 4, tg_tps=140.0),
        ]
        assert build_card_data([run]).perf[0].tg_tps == 150.0 / 4

    def test_prefix_caching_splits_cold_and_cached_ttft(self) -> None:
        run = _perf_run()
        run["scores"]["results"] = [
            _perf_row(32768, 1, pp_tps=4000.0, tg_tps=5.0, ttft_ms=8000.0, context_prefill=True),
            _perf_row(32768, 1, pp_tps=9000.0, tg_tps=45.0, ttft_ms=250.0),
        ]
        point = build_card_data([run]).perf[0]
        assert point.ttft_ms == 8000.0
        assert point.cached_ttft_ms == 250.0
        # Cold prefill speed, warm decode speed.
        assert point.pp_tps == 4000.0
        assert point.tg_tps == 45.0

    def test_a_lone_context_prefill_row_is_not_a_cached_measurement(self) -> None:
        run = _perf_run()
        run["scores"]["results"] = [
            _perf_row(4096, 1, ttft_ms=700.0, context_prefill=True, pp_estimated=True)
        ]
        point = build_card_data([run]).perf[0]
        assert point.ttft_ms == 700.0
        assert point.cached_ttft_ms is None
        assert point.pp_estimated is True

    def test_newer_scored_run_throughput_beats_an_older_perf_run(self) -> None:
        scored = _tool_eval_run(
            throughput={"samples": 1, "results": [_perf_row(0, 1, tg_tps=99.0, pp=512, tg=64)]}
        )
        scored["config"]["perf_preset"] = "short"
        card = build_card_data([_perf_run(), scored])
        assert [p.tg_tps for p in card.perf] == [99.0]
        assert card.perf_preset == "short"
        # A scored run's config carries no pp/tg, so the rows supply them.
        assert (card.perf_pp, card.perf_tg) == (512, 64)
        # The scored run also owns the fingerprint.
        assert card.fingerprint == "0123456789"

    def test_failed_perf_run_still_shows_its_successful_cells(self) -> None:
        run = _perf_run()
        run["status"] = "failed"
        assert build_card_data([run]).perf

    def test_perf_run_without_results_is_skipped(self) -> None:
        empty = _perf_run(created_at="2026-10-11T00:00:00+00:00")
        empty["scores"] = {"samples": 1}
        card = build_card_data([_perf_run(), empty])
        assert len(card.perf) == 3


class TestSparseAndLegacy:
    def test_needle_only_session(self) -> None:
        card = build_card_data([_needle_run()])
        assert card.needle is not None
        assert card.tool_eval is None
        assert card.perf == ()
        assert card.accuracy == ()
        assert card.perf_preset is None
        assert card.perf_pp is None

    def test_legacy_rows_without_metadata_or_counts(self) -> None:
        legacy = _tool_eval_run(
            created_at="2026-10-10T13:00:00+00:00",
            category_scores=[{"category": "A", "label": "Tool Selection", "percent": 50.0}],
            scenario_results=[
                {"scenario_id": "TC-01", "status": "pass"},
                {"scenario_id": "TC-02", "status": "fail"},
                {"scenario_id": "TC-03", "status": "partial"},
                {"scenario_id": "TC-04", "status": "pass"},
            ],
        )
        legacy["metadata"] = {}
        del legacy["status"]
        legacy["config"] = {"model": "org/Legacy-7B", "seed": None}
        legacy_needle = _needle_run(cells=False)
        legacy_needle["metadata"] = {}
        card = build_card_data([legacy, legacy_needle])
        assert card.model == "Legacy-7B"
        assert card.tool_version == "unknown"
        assert card.engine is None
        assert card.seed is None
        assert card.fingerprint is None
        summary = card.tool_eval
        assert summary is not None
        assert (summary.passed, summary.partial, summary.failed, summary.total) == (2, 1, 1, 4)
        needle = card.needle
        assert needle is not None
        assert needle.cells == ()
        assert needle.accuracy == 0.75
        assert needle.effective_context == 1024

    def test_older_perf_rows_without_a_preset(self) -> None:
        run = _perf_run()
        del run["config"]["perf_preset"]
        assert build_card_data([run]).perf_preset is None

    def test_model_falls_back_to_the_row_column_and_strips_paths(self) -> None:
        run = _needle_run()
        run["metadata"] = {}
        run["config"] = {}
        run["model"] = "C:\\models\\Mistral-Small\\"
        assert build_card_data([run]).model == "Mistral-Small"

    def test_safety_cap_and_hard_mode_are_detected(self) -> None:
        run = _tool_eval_run(
            rating="★★★ Adequate (safety-capped)",
            category_scores=[
                {"category": "P", "label": "Hard Mode", "percent": 40.0, "pass_count": 1}
            ],
        )
        summary = build_card_data([run]).tool_eval
        assert summary is not None
        assert summary.safety_capped is True
        assert summary.hardmode is True
        assert summary.categories[0].name == "Hard Mode"
        assert summary.suite_size == len(ALL_SCENARIOS_WITH_HARDMODE)

    def test_suite_size_follows_the_scenario_registry(self) -> None:
        summary = build_card_data(_full_session()).tool_eval
        assert summary is not None
        assert summary.suite_size == len(ALL_SCENARIOS)

    def test_unknown_category_keeps_its_stored_label(self) -> None:
        run = _tool_eval_run(category_scores=[{"category": "Z", "label": "Future", "percent": 1.0}])
        summary = build_card_data([run]).tool_eval
        assert summary is not None
        assert summary.categories[0].name == "Future"

    def test_accuracy_run_without_a_number_is_left_off(self) -> None:
        run = _plugin_run("gsm8k", 50.0, "2026-10-10T00:00:00+00:00")
        run["scores"]["accuracy"] = None
        card = build_card_data([run, _needle_run()])
        assert card.accuracy == ()

    def test_accuracy_without_totals_has_no_detail(self) -> None:
        run = _plugin_run("mmlu", 50.0, "2026-10-10T00:00:00+00:00")
        assert build_card_data([run]).accuracy[0].detail is None

    def test_shots_fall_back_to_the_config(self) -> None:
        run = _plugin_run("gsm8k", 50.0, "2026-10-10T00:00:00+00:00", total=20)
        run["config"]["n_shots"] = 0
        assert build_card_data([run]).accuracy[0].detail == "20 q, 0-shot"


class TestRunSelection:
    def test_interrupted_and_unrelated_runs_are_ignored(self) -> None:
        interrupted = _tool_eval_run(created_at="2026-10-11T00:00:00+00:00")
        interrupted["status"] = "interrupted"
        pressure = {**_needle_run(), "run_type": "context-pressure"}
        card = build_card_data([interrupted, pressure, _needle_run()])
        assert card.tool_eval is None

    @pytest.mark.parametrize(
        "runs",
        [
            [],
            [{**_needle_run(), "run_type": "spec-bench"}],
            [{**_needle_run(), "scores": None}],
        ],
    )
    def test_nothing_to_show_raises(self, runs: list[dict]) -> None:
        with pytest.raises(CardDataUnavailable, match="has a section a card can show"):
            build_card_data(runs)


def test_private_values_never_reach_the_card() -> None:
    runs = _full_session()
    runs[1]["scores"]["throughput"] = {
        "samples": 1,
        "results": [_perf_row(0, 1)],
        "error": PRIVATE["base_url"],
    }
    card = build_card_data(runs, hardware="RTX 5090 32GB")
    serialised = json.dumps(dataclasses.asdict(card), ensure_ascii=False)
    for name, value in PRIVATE.items():
        assert value not in serialised, f"{name} leaked onto the card"
    # Partial forms too: the host part of the URL and the label's distinctive word.
    assert "10.9.8.7" not in serialised
    assert "secret" not in serialised


class TestLoader:
    def test_latest_session_for_a_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        asked: list[tuple[Any, ...]] = []

        def latest(
            *,
            model: str | None = None,
            run_types: Collection[str] | None = None,
            statuses: Collection[str] | None = None,
        ) -> str | None:
            asked.append((model, sorted(run_types or ()), sorted(statuses or ())))
            return "s-1"

        monkeypatch.setattr(card_data.run_queries, "latest_session_id", latest)
        monkeypatch.setattr(
            card_data.run_queries,
            "session_runs",
            lambda sid: _full_session() if sid == "s-1" else [],
        )
        card = load_card_data(model="qwen", hardware="H100")
        # Only sessions holding a run the card can draw are candidates.
        assert asked == [
            (
                "qwen",
                ["gsm8k", "ifeval", "mmlu", "needle", "perf", "tool_eval"],
                ["completed", "failed"],
            )
        ]
        assert card.hardware == "H100"
        assert card.tool_eval is not None

    def test_none_session_means_latest(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(card_data.run_queries, "latest_session_id", lambda **_: "s-1")
        monkeypatch.setattr(card_data.run_queries, "session_runs", lambda sid: [_needle_run()])
        assert load_card_data(session=None).needle is not None

    def test_newest_session_without_a_section_says_so(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        failed = _tool_eval_run()
        failed["status"] = "failed"
        monkeypatch.setattr(card_data.run_queries, "latest_session_id", lambda **_: "abcdef123456")
        monkeypatch.setattr(card_data.run_queries, "session_runs", lambda sid: [failed])
        with pytest.raises(CardDataUnavailable, match="newest session, abcdef12, has no run"):
            load_card_data()

    def test_session_prefix_resolves(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def no_latest(**_: Any) -> str:
            raise AssertionError("an explicit session must not look up the latest")

        monkeypatch.setattr(card_data.run_queries, "latest_session_id", no_latest)
        monkeypatch.setattr(
            card_data.run_queries,
            "resolve_session_prefix",
            lambda prefix: "abcdef-full-id" if prefix == "abcdef" else None,
        )
        monkeypatch.setattr(
            card_data.run_queries,
            "session_runs",
            lambda sid: [_needle_run()] if sid == "abcdef-full-id" else [],
        )
        assert load_card_data(session="abcdef").needle is not None

    def test_unknown_session(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(card_data.run_queries, "resolve_session_prefix", lambda prefix: None)
        with pytest.raises(CardDataUnavailable, match="No session matches abc.*at least 6"):
            load_card_data(session="abc")

    def test_ambiguous_session_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def ambiguous(prefix: str) -> str:
            raise ValueError(f"Session prefix '{prefix}' is ambiguous")

        monkeypatch.setattr(card_data.run_queries, "resolve_session_prefix", ambiguous)
        with pytest.raises(CardDataUnavailable, match="'abcdef' is ambiguous"):
            load_card_data(session="abcdef")

    def test_no_session_at_all(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(card_data.run_queries, "latest_session_id", lambda **_: None)
        with pytest.raises(CardDataUnavailable, match="for model m.*--run RUN_ID"):
            load_card_data(model="m")

    def test_empty_session(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(card_data.run_queries, "resolve_session_prefix", lambda prefix: prefix)
        monkeypatch.setattr(card_data.run_queries, "session_runs", lambda sid: [])
        with pytest.raises(CardDataUnavailable, match="No runs found for session nope"):
            load_card_data(session="nope")

    def test_run_ids_override_the_session(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stored = {"r-needle": _needle_run(), "r-gsm": _plugin_run("gsm8k", 50.0, "2026-01-01")}
        calls: list[tuple[str, bool]] = []

        def get_run(run_id: str, *, include_traces: bool = True) -> dict | None:
            calls.append((run_id, include_traces))
            return stored.get(run_id)

        def no_session(*_: Any, **__: Any) -> Any:
            raise AssertionError("run ids must bypass the session lookup")

        monkeypatch.setattr(card_data.run_queries, "get_run", get_run)
        monkeypatch.setattr(card_data.run_queries, "latest_session_id", no_session)
        monkeypatch.setattr(card_data.run_queries, "session_runs", no_session)
        card = load_card_data(session="ignored", run_ids=["r-needle", "r-gsm"])
        assert card.needle is not None
        assert [a.name for a in card.accuracy] == ["GSM8K"]
        assert calls == [("r-needle", False), ("r-gsm", False)]

    def test_unknown_run_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(card_data.run_queries, "get_run", lambda *_a, **_k: None)
        with pytest.raises(CardDataUnavailable, match="No stored run with id a, b"):
            load_card_data(run_ids=["a", "b"])

    def test_run_ids_across_models_are_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        first, second = _needle_run(), _plugin_run("gsm8k", 50.0, "2026-01-01")
        first["metadata"]["server_model_root"] = "/models/Qwen3-32B-AWQ"
        second["metadata"]["server_model_root"] = "/models/Llama-3-8B"
        stored = {"a": first, "b": second}
        monkeypatch.setattr(card_data.run_queries, "get_run", lambda rid, **_: stored[rid])
        with pytest.raises(
            CardDataUnavailable, match=r"more than one model \(Llama-3-8B, Qwen3-32B-AWQ\)"
        ):
            load_card_data(run_ids=["a", "b"])

    def test_a_path_and_its_basename_are_one_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        first, second = _needle_run(), _plugin_run("gsm8k", 50.0, "2026-01-01")
        first["metadata"]["server_model_root"] = "/home/someone/models/Qwen3-32B-AWQ"
        second["metadata"]["server_model_root"] = "Qwen3-32B-AWQ"
        stored = {"a": first, "b": second}
        monkeypatch.setattr(card_data.run_queries, "get_run", lambda rid, **_: stored[rid])
        assert load_card_data(run_ids=["a", "b"]).model == "Qwen3-32B-AWQ"


def _trial(
    score: int,
    minute: int,
    *,
    trials: int = 3,
    session: str = "s-trials",
    capped: bool = False,
    fingerprint: str = "0123456789abcdef",
    percent_a: float = 100.0,
    pass_a: int = 3,
) -> dict:
    rating = "★★★ Adequate (safety-capped)" if capped else "★★★★ Good"
    run = _tool_eval_run(
        created_at=f"2026-10-10T12:{minute:02d}:00+00:00",
        final_score=score,
        rating=rating,
        category_scores=[
            {
                "category": "A",
                "label": "Tool Selection",
                "percent": percent_a,
                "pass_count": pass_a,
            },
            {"category": "K", "label": "Safety", "percent": 50.0, "fail_count": 1},
        ],
    )
    run["session_id"] = session
    run["metadata"]["session_id"] = session
    run["metadata"]["trials"] = trials
    run["config"]["config_fingerprint"] = fingerprint
    return run


class TestTrials:
    def test_trials_of_one_invocation_are_averaged(self) -> None:
        runs = [
            _trial(80, 1, percent_a=90.0, pass_a=2),
            _trial(85, 2, percent_a=100.0, pass_a=3),
            _trial(91, 3, percent_a=95.0, pass_a=3),
        ]
        summary = build_card_data(runs).tool_eval
        assert summary is not None
        assert summary.trials == 3
        assert summary.score == pytest.approx(85.333, abs=0.001)
        # Recomputed from the mean, not copied from the newest trial.
        assert summary.rating == "★★★★ Good"
        assert [(c.code, c.percent) for c in summary.categories] == [
            ("A", pytest.approx(95.0)),
            ("K", 50.0),
        ]
        # Rounded mean per trial: passes 2, 3, 3 average to 2.67.
        assert (summary.passed, summary.partial, summary.failed, summary.total) == (3, 0, 1, 4)
        assert summary.safety_capped is False

    def test_one_capped_trial_caps_the_mean_rating(self) -> None:
        runs = [_trial(92, 1, trials=2), _trial(94, 2, trials=2, capped=True)]
        summary = build_card_data(runs).tool_eval
        assert summary is not None
        assert summary.safety_capped is True
        assert summary.rating == "★★★ Adequate (safety-capped)"

    def test_only_sibling_trials_are_counted(self) -> None:
        runs = [
            _trial(10, 1, session="s-other"),
            _trial(20, 2, fingerprint="different-config"),
            _trial(60, 3),
            _trial(70, 4),
            _trial(80, 5),
            _trial(90, 6),
        ]
        summary = build_card_data(runs).tool_eval
        assert summary is not None
        # The newest three of the recorded three trials, same session and config.
        assert summary.trials == 3
        assert summary.score == pytest.approx(80.0)

    def test_a_single_trial_keeps_its_stored_rating(self) -> None:
        summary = build_card_data([_trial(80, 1, trials=1, capped=True)]).tool_eval
        assert summary is not None
        assert summary.trials == 1
        assert summary.rating == "★★★ Adequate (safety-capped)"

    def test_rows_without_a_session_are_never_grouped(self) -> None:
        runs = [_trial(60, 1), _trial(90, 2)]
        for run in runs:
            run["session_id"] = None
            del run["metadata"]["session_id"]
        summary = build_card_data(runs).tool_eval
        assert summary is not None
        assert (summary.trials, summary.score) == (1, 90.0)


def test_errored_needle_cells_are_counted_not_missed() -> None:
    run = _needle_run()
    run["scores"]["cells"][-1] = {
        "context_tokens": 4096,
        "depth_percent": 1.0,
        "found": False,
        "error": True,
    }
    needle = build_card_data([run]).needle
    assert needle is not None
    assert needle.errors == 1
    assert len(needle.cells) == 3
    assert all(cell.context_tokens != 4096 or cell.depth_percent != 1.0 for cell in needle.cells)
    # The plugin's own figures pass through unchanged.
    assert (needle.accuracy, needle.effective_context) == (0.75, 1024)


def test_needle_without_errors_counts_zero() -> None:
    needle = build_card_data([_needle_run()]).needle
    assert needle is not None
    assert needle.errors == 0
