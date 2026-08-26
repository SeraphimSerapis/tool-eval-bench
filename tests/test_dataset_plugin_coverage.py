"""High-value deterministic coverage for reporting, performance CLI, and datasets."""

from __future__ import annotations

from pathlib import Path

import pytest
from rich.console import Console


def test_dataset_cache_roundtrips_and_download_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tool_eval_bench.plugins import hf_utils
    from tool_eval_bench.plugins.gsm8k import dataset as gsm
    from tool_eval_bench.plugins.ifeval import dataset as ifeval
    from tool_eval_bench.plugins.mmlu import dataset as mmlu

    gsm_path = tmp_path / "gsm.jsonl"
    gsm_items = [gsm.GSM8KItem(0, "q", "work #### 1,200", 1200.0)]
    gsm._save_to_cache(gsm_path, gsm_items)
    assert gsm._load_from_cache(gsm_path)[0].to_dict()["ground_truth"] == 1200.0
    assert (
        gsm._rows_to_items(
            [{"question": "q", "answer": "x #### 3"}, {"question": "bad", "answer": "none"}]
        )[0].ground_truth
        == 3
    )

    if_path = tmp_path / "if.jsonl"
    if_items = [ifeval.IFEvalItem(1, "p", ["id"], [{"x": 1}])]
    ifeval._save_to_cache(if_path, if_items)
    assert ifeval._load_from_cache(if_path)[0].to_dict()["key"] == 1
    assert (
        ifeval._rows_to_items(
            [{"key": "2", "prompt": "p2", "instruction_id_list": ["x"], "kwargs": [{}]}]
        )[0].key
        == 2
    )

    mm_path = tmp_path / "mm.jsonl"
    mm_items = [mmlu.MMLUItem(0, "q", "anatomy", ["a", "b", "c", "d"], 1)]
    mmlu._save_to_cache(mm_path, mm_items)
    loaded = mmlu._load_from_cache(mm_path)[0]
    assert loaded.answer_letter == "B"
    assert loaded.category == "STEM"
    assert loaded.to_dict()["answer_letter"] == "B"
    assert mmlu.MMLUItem(1, "q", "unknown", ["a", "b", "c", "d"], 0).category == "Other"

    rows = [
        {
            "question": "q",
            "subject": "anatomy",
            "choices": ["a", "b", "c", "d"],
            "answer": "2",
        }
    ]

    def library_rows(dataset: str, *args: object, **kwargs: object):
        if dataset == "openai/gsm8k":
            return [{"question": "q", "answer": "work #### 3"}]
        if dataset == "google/IFEval":
            return [{"key": "2", "prompt": "p", "instruction_id_list": ["x"], "kwargs": [{}]}]
        return rows

    monkeypatch.setattr(hf_utils, "load_via_datasets_lib", library_rows)
    assert mmlu._download_dataset()[0][0].answer == 2
    assert gsm._download_dataset()[0][0].ground_truth == 3
    assert ifeval._download_dataset()[0][0].key == 2

    monkeypatch.setattr(hf_utils, "load_via_datasets_lib", lambda *args, **kwargs: None)
    monkeypatch.setattr(hf_utils, "download_rows_paginated", lambda *args, **kwargs: rows)
    assert mmlu._download_dataset()[1] == "rest_api"


def test_dataset_load_uses_cache_or_download_and_cleans_partial(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tool_eval_bench.plugins.ifeval import dataset as ifeval
    from tool_eval_bench.plugins.mmlu import dataset as mmlu

    if_cache = tmp_path / "ifeval" / "prompts.jsonl"
    monkeypatch.setattr(ifeval, "_CACHE_DIR", if_cache.parent)
    monkeypatch.setattr(ifeval, "_CACHE_FILE", if_cache)
    monkeypatch.setattr(ifeval, "_find_cache_file", lambda: if_cache)
    item = ifeval.IFEvalItem(1, "p", [], [])
    monkeypatch.setattr(ifeval, "_download_dataset", lambda **kwargs: ([item], "test"))
    assert ifeval.load_dataset(force_download=True) == [item]
    assert ifeval.load_dataset() == [item]

    mm_cache = tmp_path / "mmlu" / "test.jsonl"
    monkeypatch.setattr(mmlu, "_CACHE_DIR", mm_cache.parent)
    monkeypatch.setattr(mmlu, "_find_cache_file", lambda split="test": mm_cache)
    mm_item = mmlu.MMLUItem(0, "q", "anatomy", ["a", "b", "c", "d"], 0)
    monkeypatch.setattr(mmlu, "_download_dataset", lambda **kwargs: ([mm_item], "test"))
    assert mmlu.load_dataset(force_download=True) == [mm_item]
    assert mmlu.load_dataset() == [mm_item]


def test_plugin_runners_complete_cached_lifecycles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import argparse

    from tool_eval_bench.cli import plugin_runners
    from tool_eval_bench.domain.plugin import BenchmarkResult
    from tool_eval_bench.plugins.gsm8k import dataset as gd
    from tool_eval_bench.plugins.gsm8k import plugin as gp
    from tool_eval_bench.plugins.ifeval import dataset as idata
    from tool_eval_bench.plugins.ifeval import plugin as ip
    from tool_eval_bench.plugins.mmlu import dataset as md
    from tool_eval_bench.plugins.mmlu import plugin as mp

    cache = tmp_path / "cache"
    cache.write_text("cached", encoding="utf-8")
    monkeypatch.setattr(gd, "_find_cache_file", lambda: cache)
    monkeypatch.setattr(gd, "load_dataset", lambda **kw: [gd.GSM8KItem(0, "q", "#### 1", 1)])
    monkeypatch.setattr(md, "_find_cache_file", lambda split="test": cache)
    monkeypatch.setattr(
        md,
        "load_dataset",
        lambda split="test", **kw: [md.MMLUItem(0, "q", "anatomy", ["a"] * 4, 0)],
    )
    monkeypatch.setattr(idata, "_find_cache_file", lambda: cache)
    monkeypatch.setattr(idata, "load_dataset", lambda **kw: [idata.IFEvalItem(0, "p", ["x"], [{}])])

    async def gsm_run(self, adapter, *, on_progress=None, **kw):
        await on_progress(
            1, 3, {"correct": True, "question": "q", "extracted_answer": 1, "ground_truth": 1}
        )
        await on_progress(2, 3, {"correct": False, "question": "x" * 100})
        await on_progress(3, 3, {"is_error": True, "question": "bad"})
        return BenchmarkResult(
            "gsm8k",
            50,
            "50%",
            "Weak",
            details={
                "total": 3,
                "correct": 1,
                "errors": 1,
                "answered": 2,
                "completion_rate": 66.67,
            },
            duration_seconds=1,
            total_tokens=10,
        )

    async def mmlu_run(self, adapter, *, on_progress=None, **kw):
        await on_progress(
            1,
            3,
            {
                "correct": True,
                "subject": "anatomy",
                "question": "q",
                "extracted_answer": "A",
                "ground_truth": "A",
            },
        )
        await on_progress(2, 3, {"correct": False, "subject": "anatomy", "question": "x" * 100})
        await on_progress(3, 3, {"is_error": True, "subject": "anatomy", "question": "bad"})
        return BenchmarkResult(
            "mmlu",
            50,
            "50%",
            "Weak",
            details={
                "total": 3,
                "correct": 1,
                "errors": 1,
                "answered": 2,
                "completion_rate": 66.67,
                "categories": {"STEM": {"accuracy": 50.0}},
            },
            duration_seconds=1,
            total_tokens=10,
        )

    async def ifeval_run(self, adapter, *, on_progress=None, **kw):
        await on_progress(
            1,
            3,
            {"prompt_pass": True, "instructions_passed": 2, "instructions_total": 2, "prompt": "p"},
        )
        await on_progress(
            2,
            3,
            {
                "prompt_pass": False,
                "instructions_passed": 1,
                "instructions_total": 2,
                "prompt": "x" * 100,
            },
        )
        await on_progress(3, 3, {"is_error": True, "prompt": "bad"})
        return BenchmarkResult(
            "ifeval",
            50,
            "50%",
            "Weak",
            details={
                "total": 3,
                "prompts_passed": 1,
                "errors": 1,
                "answered": 2,
                "completion_rate": 66.67,
                "prompt_accuracy": 50.0,
                "instructions_passed": 3,
                "instructions_total": 4,
                "instruction_accuracy": 75.0,
            },
            duration_seconds=1,
            total_tokens=10,
        )

    for cls, run in (
        (gp.GSM8KPlugin, gsm_run),
        (mp.MMLUPlugin, mmlu_run),
        (ip.IFEvalPlugin, ifeval_run),
    ):
        monkeypatch.setattr(cls, "run", run)
        monkeypatch.setattr(cls, "render_report_section", lambda self, result: ["report"])
    monkeypatch.setattr(plugin_runners, "_with_config_fingerprint", lambda value: value)
    monkeypatch.setattr(plugin_runners, "_metadata_for_storage", lambda value: {})
    persisted: list[dict] = []
    monkeypatch.setattr(plugin_runners, "_persist_plugin_run", persisted.append)
    args = argparse.Namespace(
        gsm8k_shots=1,
        gsm8k_limit=3,
        gsm8k_shuffle=True,
        mmlu_shots=1,
        mmlu_limit=3,
        mmlu_subjects="STEM",
        ifeval_limit=3,
        seed=1,
        parallel=2,
        temperature=0.0,
        timeout=1.0,
    )
    console = Console(record=True, width=180)

    plugin_runners._run_gsm8k_benchmark(
        console, "m", "Display", "url", None, args, output_dir=str(tmp_path)
    )
    plugin_runners._run_mmlu_benchmark(
        console, "m", "Display", "url", None, args, output_dir=str(tmp_path)
    )
    plugin_runners._run_ifeval_benchmark(
        console, "m", "Display", "url", None, args, output_dir=str(tmp_path)
    )

    assert [item["run_type"] for item in persisted] == ["gsm8k", "mmlu", "ifeval"]
    output = console.export_text()
    assert "IFEval Prompt Accuracy" in output
    assert output.count("counted in accuracy; 2/3 answered") == 3
    assert "excluded from accuracy" not in output
    assert all(item["scores"]["completion_rate"] == 66.67 for item in persisted)


def test_gsm8k_rest_download_and_load_dataset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tool_eval_bench.plugins import hf_utils
    from tool_eval_bench.plugins.gsm8k import dataset as gsm

    monkeypatch.setattr(hf_utils, "load_via_datasets_lib", lambda *a, **k: None)

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "num_rows_total": 2,
                "rows": [
                    {"row_idx": 0, "row": {"question": "q", "answer": "work #### 2"}},
                    {"row_idx": 1, "row": {"question": "bad", "answer": "none"}},
                ],
            }

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def get(self, url):
            return Response()

    import httpx

    monkeypatch.setattr(httpx, "Client", Client)
    progress: list[tuple[int, int]] = []
    items, method = gsm._download_dataset(on_progress=lambda a, b: progress.append((a, b)))
    assert method == "rest_api" and len(items) == 1 and progress == [(2, 2)]

    cache = tmp_path / "test.jsonl"
    monkeypatch.setattr(gsm, "_find_cache_file", lambda: cache)
    monkeypatch.setattr(gsm, "_download_dataset", lambda **kw: (items, "test"))
    assert gsm.load_dataset(force_download=True) == items
    assert gsm.load_dataset() == items
