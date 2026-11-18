"""Tests for pluggable benchmarks with mock adapters and preloaded items."""

from __future__ import annotations

from typing import Any

import pytest

from tool_eval_bench.adapters.base import (
    BackendAdapter,
    ChatCompletionResult,
)
from tool_eval_bench.plugins.gsm8k.dataset import GSM8KItem
from tool_eval_bench.plugins.gsm8k.plugin import GSM8KPlugin
from tool_eval_bench.plugins.ifeval.dataset import IFEvalItem
from tool_eval_bench.plugins.ifeval.plugin import IFEvalPlugin
from tool_eval_bench.plugins.mmlu.dataset import MMLUItem
from tool_eval_bench.plugins.mmlu.plugin import MMLUPlugin


class MockAdapter(BackendAdapter):
    """Simple mock adapter that returns pre-packaged model responses."""

    def __init__(self, response_content: str, *, fail_after: int | None = None) -> None:
        self.response_content = response_content
        self.fail_after = fail_after
        self.calls: list[dict] = []

    async def chat_completion(
        self,
        *,
        model: str,
        messages: list[dict] | list[Any],
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout_seconds: float = 60.0,
        api_key: str | None = None,
        base_url: str = "",
        extra_params: dict | None = None,
        stream: bool = False,
        response_format: dict | None = None,
        parallel_tool_calls: bool | None = True,
    ) -> ChatCompletionResult:
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "tools": tools,
                "tool_choice": tool_choice,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout_seconds": timeout_seconds,
                "api_key": api_key,
                "base_url": base_url,
                "extra_params": extra_params,
                "stream": stream,
                "response_format": response_format,
                "parallel_tool_calls": parallel_tool_calls,
            }
        )
        if self.fail_after is not None and len(self.calls) > self.fail_after:
            raise TimeoutError("synthetic timeout")
        return ChatCompletionResult(
            content=self.response_content,
            tool_calls=[],
            raw_response={"choices": [{"message": {"content": self.response_content}}]},
            elapsed_ms=15.0,
        )


@pytest.mark.asyncio
async def test_gsm8k_plugin_run() -> None:
    """GSM8K evaluation: sequential and parallel loops."""
    items = [
        GSM8KItem(
            index=0,
            question="What is 2+2?",
            raw_answer="#### 4",
            ground_truth=4.0,
        ),
        GSM8KItem(
            index=1,
            question="What is 3+3?",
            raw_answer="#### 6",
            ground_truth=6.0,
        ),
    ]

    # Test sequential run (concurrency=1)
    adapter_seq = MockAdapter("The answer is 4. #### 4")
    plugin = GSM8KPlugin()
    result_seq = await plugin.run(
        adapter_seq,
        model="test-model",
        base_url="http://localhost:8000",
        _preloaded_items=items,
        concurrency=1,
    )
    assert result_seq.plugin_name == "gsm8k"
    # 1 out of 2 correct (both adapter calls return 4)
    assert result_seq.score == 50.0
    assert len(adapter_seq.calls) == 2

    # Test parallel run (concurrency=2)
    adapter_par = MockAdapter("The answer is 6. #### 6")
    result_par = await plugin.run(
        adapter_par,
        model="test-model",
        base_url="http://localhost:8000",
        _preloaded_items=items,
        concurrency=2,
    )
    assert result_par.plugin_name == "gsm8k"
    # 1 out of 2 correct (both adapter calls return 6)
    assert result_par.score == 50.0
    assert len(adapter_par.calls) == 2


@pytest.mark.asyncio
async def test_plugin_errors_use_total_denominator_and_mark_incomplete() -> None:
    """A partial request must not become a perfect answered-only score."""
    gsm_items = [
        GSM8KItem(0, "Q1", "#### 1", 1.0),
        GSM8KItem(1, "Q2", "#### 1", 1.0),
    ]
    gsm_result = await GSM8KPlugin().run(
        MockAdapter("#### 1", fail_after=1),
        model="m",
        base_url="http://localhost:8000",
        _preloaded_items=gsm_items,
        concurrency=1,
    )
    assert gsm_result.score == 50.0
    assert gsm_result.details["status"] == "incomplete"
    assert gsm_result.details["answered"] == 1

    mmlu_items = [
        MMLUItem(0, "Q1", "abstract_algebra", ["a", "b", "c", "d"], 0),
        MMLUItem(1, "Q2", "abstract_algebra", ["a", "b", "c", "d"], 0),
    ]
    mmlu_result = await MMLUPlugin().run(
        MockAdapter("A", fail_after=1),
        model="m",
        base_url="http://localhost:8000",
        _preloaded_items={"test": mmlu_items, "dev": []},
        n_shots=0,
        concurrency=1,
    )
    assert mmlu_result.score == 50.0
    assert mmlu_result.details["status"] == "incomplete"

    ifeval_items = [
        IFEvalItem(0, "P1", ["punctuation:no_comma"], [{}]),
        IFEvalItem(1, "P2", ["punctuation:no_comma"], [{}]),
    ]
    ifeval_result = await IFEvalPlugin().run(
        MockAdapter("No commas here.", fail_after=1),
        model="m",
        base_url="http://localhost:8000",
        _preloaded_items=ifeval_items,
        concurrency=1,
    )
    assert ifeval_result.score == 50.0
    assert ifeval_result.details["status"] == "incomplete"


@pytest.mark.asyncio
async def test_plugins_forward_seed_and_reject_invalid_concurrency() -> None:
    items = [GSM8KItem(0, "Q", "#### 1", 1.0)]
    gsm_adapter = MockAdapter("#### 1")
    await GSM8KPlugin().run(
        gsm_adapter,
        model="m",
        base_url="http://localhost:8000",
        seed=17,
        _preloaded_items=items,
    )
    assert gsm_adapter.calls[0]["extra_params"] == {"seed": 17}

    mmlu_adapter = MockAdapter("A")
    await MMLUPlugin().run(
        mmlu_adapter,
        model="m",
        base_url="http://localhost:8000",
        seed=17,
        n_shots=0,
        _preloaded_items={"test": [MMLUItem(0, "Q", "anatomy", ["a"] * 4, 0)], "dev": []},
    )
    assert mmlu_adapter.calls[0]["extra_params"] == {"seed": 17}

    ifeval_adapter = MockAdapter("No commas")
    await IFEvalPlugin().run(
        ifeval_adapter,
        model="m",
        base_url="http://localhost:8000",
        seed=17,
        _preloaded_items=[IFEvalItem(0, "P", ["punctuation:no_comma"], [{}])],
    )
    assert ifeval_adapter.calls[0]["extra_params"] == {"seed": 17}

    with pytest.raises(ValueError, match="concurrency"):
        await GSM8KPlugin().run(
            MockAdapter("#### 1"),
            model="m",
            base_url="http://localhost:8000",
            concurrency=0,
            _preloaded_items=items,
        )
    with pytest.raises(ValueError, match="concurrency"):
        await MMLUPlugin().run(
            MockAdapter("A"),
            model="m",
            base_url="http://localhost:8000",
            concurrency=0,
            _preloaded_items={"test": [], "dev": []},
        )
    with pytest.raises(ValueError, match="concurrency"):
        await IFEvalPlugin().run(
            MockAdapter("No commas"),
            model="m",
            base_url="http://localhost:8000",
            concurrency=0,
            _preloaded_items=[],
        )


@pytest.mark.asyncio
async def test_mmlu_empty_preloaded_data_does_not_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tool_eval_bench.plugins.mmlu import plugin as mmlu_plugin

    def fail_download(*args: Any, **kwargs: Any) -> list[MMLUItem]:
        raise AssertionError("empty preloaded MMLU should not download")

    monkeypatch.setattr(mmlu_plugin, "load_dataset", fail_download)
    # An empty selection fails rather than saving a 0/0 result.
    with pytest.raises(ValueError, match="no questions"):
        await MMLUPlugin().run(
            MockAdapter("A"),
            model="m",
            base_url="http://localhost:8000",
            _preloaded_items={"test": [], "dev": []},
        )


@pytest.mark.asyncio
async def test_mmlu_plugin_run() -> None:
    """MMLU evaluation: sequential and parallel loops."""
    choices = ["A", "B", "C", "D"]
    items = [
        MMLUItem(
            index=0,
            question="Q1",
            subject="abstract_algebra",
            choices=choices,
            answer=0,
        ),
        MMLUItem(
            index=1,
            question="Q2",
            subject="abstract_algebra",
            choices=choices,
            answer=1,
        ),
    ]

    # Test sequential run
    adapter_seq = MockAdapter("A")
    plugin = MMLUPlugin()
    result_seq = await plugin.run(
        adapter_seq,
        model="test-model",
        base_url="http://localhost:8000",
        _preloaded_items={"test": items, "dev": []},
        concurrency=1,
        n_shots=0,
    )
    assert result_seq.plugin_name == "mmlu"
    assert result_seq.score == 50.0  # index 0 matches A
    assert len(adapter_seq.calls) == 2

    # Test parallel run
    adapter_par = MockAdapter("B")
    result_par = await plugin.run(
        adapter_par,
        model="test-model",
        base_url="http://localhost:8000",
        _preloaded_items={"test": items, "dev": []},
        concurrency=2,
        n_shots=0,
    )
    assert result_par.plugin_name == "mmlu"
    assert result_par.score == 50.0
    assert len(adapter_par.calls) == 2


@pytest.mark.asyncio
async def test_ifeval_plugin_run() -> None:
    """IFEval evaluation: sequential and parallel loops."""
    items = [
        IFEvalItem(
            key=0,
            prompt="P1",
            instruction_id_list=["punctuation:no_comma"],
            kwargs=[{"comma_number": 0}],
        ),
        IFEvalItem(
            key=1,
            prompt="P2",
            instruction_id_list=["punctuation:no_comma"],
            kwargs=[{"comma_number": 0}],
        ),
    ]

    # Test sequential run
    adapter_seq = MockAdapter("Response without any commas.")
    plugin = IFEvalPlugin()
    result_seq = await plugin.run(
        adapter_seq,
        model="test-model",
        base_url="http://localhost:8000",
        _preloaded_items=items,
        concurrency=1,
    )
    assert result_seq.plugin_name == "ifeval"
    assert result_seq.score == 100.0  # both follow instructions
    assert len(adapter_seq.calls) == 2

    # Test parallel run
    adapter_par = MockAdapter("Response, with, commas.")
    result_par = await plugin.run(
        adapter_par,
        model="test-model",
        base_url="http://localhost:8000",
        _preloaded_items=items,
        concurrency=2,
    )
    assert result_par.plugin_name == "ifeval"
    assert result_par.score == 0.0  # both fail instructions
    assert len(adapter_par.calls) == 2


def test_plugin_runners_complete_cached_lifecycles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    import argparse

    from rich.console import Console

    from tool_eval_bench.cli import plugin_runners
    from tool_eval_bench.domain.plugin import BenchmarkResult
    from tool_eval_bench.plugins.gsm8k import dataset as gd
    from tool_eval_bench.plugins.ifeval import dataset as idata
    from tool_eval_bench.plugins.mmlu import dataset as md

    cache = tmp_path / "cache"
    cache.write_text("cached", encoding="utf-8")
    monkeypatch.setattr(gd, "_find_cache_file", lambda: cache)
    monkeypatch.setattr(gd, "load_dataset", lambda **kw: [GSM8KItem(0, "q", "#### 1", 1)])
    monkeypatch.setattr(md, "_find_cache_file", lambda split="test": cache)
    monkeypatch.setattr(
        md,
        "load_dataset",
        lambda split="test", **kw: [MMLUItem(0, "q", "anatomy", ["a"] * 4, 0)],
    )
    monkeypatch.setattr(idata, "_find_cache_file", lambda: cache)
    monkeypatch.setattr(idata, "load_dataset", lambda **kw: [IFEvalItem(0, "p", ["x"], [{}])])

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
        (GSM8KPlugin, gsm_run),
        (MMLUPlugin, mmlu_run),
        (IFEvalPlugin, ifeval_run),
    ):
        monkeypatch.setattr(cls, "run", run)
        monkeypatch.setattr(cls, "render_report_section", lambda self, result: ["report"])
    monkeypatch.setattr(
        "tool_eval_bench.application.mode_runs.with_config_fingerprint", lambda value, *_: value
    )
    persisted: list[dict] = []
    monkeypatch.setattr("tool_eval_bench.application.run_queries.persist_run", persisted.append)
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
