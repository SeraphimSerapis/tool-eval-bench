"""Shared dataset loading: cache hit, download, resume wording, and failure."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from tool_eval_bench.cli.plugin_datasets import load_dataset_with_progress


def _console() -> tuple[Console, StringIO]:
    buffer = StringIO()
    return Console(file=buffer, width=200, no_color=True), buffer


def test_cache_hit_loads_without_downloading(tmp_path: Path) -> None:
    cache = tmp_path / "test.jsonl"
    cache.write_text("{}", encoding="utf-8")
    console, buffer = _console()
    calls: list[dict] = []

    def load(**kwargs):
        calls.append(kwargs)
        return ["a", "b", "c"]

    items = load_dataset_with_progress(
        console,
        name="GSM8K",
        noun="questions",
        cache_path=cache,
        load=load,
        cache_note="data/gsm8k/test.jsonl",
    )

    assert items == ["a", "b", "c"]
    assert calls == [{}], "a cache hit must not pass a progress callback"
    output = buffer.getvalue()
    assert "Loading GSM8K from cache" in output
    assert "3 questions" in output
    assert "Downloading" not in output


def test_download_reports_progress_and_cache_location(tmp_path: Path) -> None:
    console, buffer = _console()

    def load(on_progress=None):
        if on_progress:
            on_progress(50, 100)
        return ["x"] * 7

    items = load_dataset_with_progress(
        console,
        name="IFEval",
        noun="prompts",
        cache_path=tmp_path / "missing.jsonl",
        load=load,
        cache_note="data/ifeval/prompts.jsonl",
    )

    assert items is not None and len(items) == 7
    output = buffer.getvalue()
    assert "Downloaded" in output
    assert "7 prompts" in output
    assert "data/ifeval/prompts.jsonl" in output


def _capture_status(console: Console, sink: list[str]) -> None:
    """Record spinner text, which a non-terminal console never writes out."""
    original = console.status

    def status(message, **kwargs):
        sink.append(str(message))
        return original(message, **kwargs)

    console.status = status  # type: ignore[method-assign]


def test_a_partial_download_switches_the_wording_to_resuming(tmp_path: Path) -> None:
    partial = tmp_path / "test.partial.jsonl"
    partial.write_text("{}", encoding="utf-8")
    console, _ = _console()
    messages: list[str] = []
    _capture_status(console, messages)

    load_dataset_with_progress(
        console,
        name="MMLU",
        noun="questions",
        cache_path=tmp_path / "missing.jsonl",
        load=lambda **kw: ["y"],
        cache_note="data/mmlu/test.jsonl",
        partial_path=partial,
    )

    assert any("Resuming MMLU download" in m for m in messages)


def test_without_a_partial_file_the_wording_stays_downloading(tmp_path: Path) -> None:
    console, _ = _console()
    messages: list[str] = []
    _capture_status(console, messages)

    load_dataset_with_progress(
        console,
        name="MMLU",
        noun="questions",
        cache_path=tmp_path / "missing.jsonl",
        load=lambda **kw: ["y"],
        cache_note="data/mmlu/test.jsonl",
        partial_path=tmp_path / "absent.partial.jsonl",
    )

    assert any("Downloading MMLU dataset" in m for m in messages)
    assert not any("Resuming" in m for m in messages)


def test_failure_returns_none_and_explains_why(tmp_path: Path) -> None:
    console, buffer = _console()

    def load(on_progress=None):
        raise RuntimeError("429 Too Many Requests")

    items = load_dataset_with_progress(
        console,
        name="MMLU",
        noun="questions",
        cache_path=tmp_path / "missing.jsonl",
        load=load,
        cache_note="data/mmlu/test.jsonl",
        partial_path=tmp_path / "test.partial.jsonl",
    )

    assert items is None, "the caller relies on None to stop cleanly"
    output = buffer.getvalue()
    assert "Failed to download MMLU dataset" in output
    assert "429 Too Many Requests" in output
    assert "rate limiting" in output
    assert "Progress is saved" in output, "a resumable loader must say so"


def test_a_non_resumable_failure_omits_the_resume_hint(tmp_path: Path) -> None:
    console, buffer = _console()

    def load(on_progress=None):
        raise RuntimeError("boom")

    items = load_dataset_with_progress(
        console,
        name="GSM8K",
        noun="questions",
        cache_path=tmp_path / "missing.jsonl",
        load=load,
        cache_note="data/gsm8k/test.jsonl",
    )

    assert items is None
    output = buffer.getvalue()
    assert "Failed to download GSM8K dataset" in output
    assert "Progress is saved" not in output, "GSM8K downloads cannot resume"


def test_ifeval_and_mmlu_cache_roundtrip_and_download_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tool_eval_bench.plugins import hf_utils
    from tool_eval_bench.plugins.ifeval import dataset as ifeval
    from tool_eval_bench.plugins.mmlu import dataset as mmlu

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

    def library_rows(dataset: str, *args: object, **kwargs: object) -> list[dict[str, object]]:
        if dataset == "google/IFEval":
            return [{"key": "2", "prompt": "p", "instruction_id_list": ["x"], "kwargs": [{}]}]
        return rows

    monkeypatch.setattr(hf_utils, "load_via_datasets_lib", library_rows)
    assert mmlu._download_dataset()[0][0].answer == 2
    assert ifeval._download_dataset()[0][0].key == 2

    monkeypatch.setattr(hf_utils, "load_via_datasets_lib", lambda *args, **kwargs: None)
    monkeypatch.setattr(hf_utils, "download_rows_paginated", lambda *args, **kwargs: rows)
    assert mmlu._download_dataset()[1] == "rest_api"


def test_ifeval_and_mmlu_load_dataset_uses_cache_or_download(
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
