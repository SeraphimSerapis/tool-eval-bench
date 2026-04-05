"""End-to-end smoke test for the full benchmark pipeline.

TEST-01 from the critical review: Exercises the complete path:
  CLI args → service → orchestrator → adapter → evaluator → storage → report

Uses a mock adapter (no real server) and a temp directory for storage,
verifying that all layers integrate correctly.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

import pytest

from tool_eval_bench.adapters.base import BackendAdapter, ChatCompletionResult, ProviderToolCall
from tool_eval_bench.runner.service import BenchmarkService
from tool_eval_bench.storage.db import RunRepository
from tool_eval_bench.storage.reports import MarkdownReporter


# ---------------------------------------------------------------------------
# Mock adapter that produces correct tool calls for TC-01
# ---------------------------------------------------------------------------


class SmokeMockAdapter(BackendAdapter):
    """Returns a get_weather call then a text answer — enough to PASS TC-01."""

    def __init__(self) -> None:
        self._call = 0

    async def chat_completion(self, **kwargs: Any) -> ChatCompletionResult:
        self._call += 1
        if self._call == 1:
            return ChatCompletionResult(
                content="",
                tool_calls=[
                    ProviderToolCall(
                        id="call_smoke_1",
                        name="get_weather",
                        arguments_str=json.dumps({"location": "Berlin"}),
                    )
                ],
                elapsed_ms=15.0,
                ttft_ms=5.0,
            )
        return ChatCompletionResult(
            content="Berlin is 8°C and overcast.",
            elapsed_ms=20.0,
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_e2e_single_scenario_pipeline() -> None:
    """Full pipeline: run TC-01 through service, check DB + report."""
    from tool_eval_bench.evals.scenarios import SCENARIOS

    tc01 = next(s for s in SCENARIOS if s.id == "TC-01")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test.sqlite")
        reports_root = str(Path(tmpdir) / "runs")

        repo = RunRepository(db_path=db_path)
        reporter = MarkdownReporter(root=reports_root)
        service = BenchmarkService(repo=repo, reporter=reporter)

        # Monkey-patch adapter creation to use our mock
        service._adapter_for = lambda backend: SmokeMockAdapter()

        result = await service.run_benchmark(
            model="smoke-test-model",
            backend="vllm",
            base_url="http://localhost:9999",
            scenarios=[tc01],
            temperature=0.0,
            timeout_seconds=10.0,
        )

        # --- Verify service return ---
        assert result["status"] == "completed"
        assert "run_id" in result
        assert "report_path" in result

        scores = result["scores"]
        assert scores["final_score"] == 100  # 1 scenario, perfect
        assert scores["total_points"] == 2
        assert scores["max_points"] == 2

        # --- Verify SQLite persistence ---
        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT run_id, status FROM scenario_runs").fetchall()
        assert len(rows) == 1
        assert rows[0][1] == "completed"
        conn.close()

        # --- Verify Markdown report ---
        report_path = result["report_path"]
        assert Path(report_path).exists()
        report_content = Path(report_path).read_text()
        assert "TC-01" in report_content
        assert "smoke-test-model" in report_content

        repo.close()


@pytest.mark.asyncio
async def test_e2e_fail_scenario_persists() -> None:
    """A failing scenario should still be persisted correctly."""
    from tool_eval_bench.evals.scenarios import SCENARIOS

    tc01 = next(s for s in SCENARIOS if s.id == "TC-01")

    class FailMockAdapter(BackendAdapter):
        async def chat_completion(self, **kwargs: Any) -> ChatCompletionResult:
            return ChatCompletionResult(content="I think Berlin is cold.", elapsed_ms=10.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test.sqlite")
        repo = RunRepository(db_path=db_path)
        reporter = MarkdownReporter(root=str(Path(tmpdir) / "runs"))
        service = BenchmarkService(repo=repo, reporter=reporter)
        service._adapter_for = lambda backend: FailMockAdapter()

        result = await service.run_benchmark(
            model="fail-model",
            backend="vllm",
            base_url="http://localhost:9999",
            scenarios=[tc01],
        )

        assert result["scores"]["final_score"] == 0
        assert result["scores"]["total_points"] == 0

        # Still persisted
        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT status FROM scenario_runs").fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "completed"
        conn.close()
        repo.close()


@pytest.mark.asyncio
async def test_e2e_multiple_scenarios() -> None:
    """Run 3 base scenarios (TC-01, TC-02, TC-03) and verify scoring."""
    from tool_eval_bench.evals.scenarios import SCENARIOS

    cat_a = [s for s in SCENARIOS if s.id in ("TC-01", "TC-02", "TC-03")]
    assert len(cat_a) == 3

    class MultiMockAdapter(BackendAdapter):
        """Responds to every scenario with a direct answer (no tools)."""
        async def chat_completion(self, **kwargs: Any) -> ChatCompletionResult:
            return ChatCompletionResult(
                content="Here's the information you requested.",
                elapsed_ms=10.0,
            )

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test.sqlite")
        repo = RunRepository(db_path=db_path)
        reporter = MarkdownReporter(root=str(Path(tmpdir) / "runs"))
        service = BenchmarkService(repo=repo, reporter=reporter)
        service._adapter_for = lambda backend: MultiMockAdapter()

        result = await service.run_benchmark(
            model="multi-test",
            backend="vllm",
            base_url="http://localhost:9999",
            scenarios=cat_a,
        )

        scores = result["scores"]
        # All 3 scenarios should FAIL (no tools called for Cat A)
        sr_list = scores["scenario_results"]
        assert len(sr_list) == 3
        assert all(sr["status"] == "fail" for sr in sr_list)
        assert scores["total_points"] == 0

        # Report should list all 3 scenarios
        report = Path(result["report_path"]).read_text()
        for sc_id in ("TC-01", "TC-02", "TC-03"):
            assert sc_id in report

        repo.close()


@pytest.mark.asyncio
async def test_e2e_callbacks_invoked() -> None:
    """Verify that on_scenario_start and on_scenario_result callbacks fire."""
    from tool_eval_bench.evals.scenarios import SCENARIOS

    tc01 = next(s for s in SCENARIOS if s.id == "TC-01")
    started: list[str] = []
    finished: list[str] = []

    async def on_start(scenario, idx, total):
        started.append(scenario.id)

    async def on_result(scenario, result, idx, total):
        finished.append(scenario.id)

    with tempfile.TemporaryDirectory() as tmpdir:
        repo = RunRepository(db_path=str(Path(tmpdir) / "test.sqlite"))
        reporter = MarkdownReporter(root=str(Path(tmpdir) / "runs"))
        service = BenchmarkService(repo=repo, reporter=reporter)
        service._adapter_for = lambda backend: SmokeMockAdapter()

        await service.run_benchmark(
            model="cb-test",
            backend="vllm",
            base_url="http://localhost:9999",
            scenarios=[tc01],
            on_scenario_start=on_start,
            on_scenario_result=on_result,
        )

        assert started == ["TC-01"]
        assert finished == ["TC-01"]
        repo.close()


@pytest.mark.asyncio
async def test_e2e_random_tool_model_scores_low() -> None:
    """METH-04: A model that picks random tools should score very poorly.

    This synthetic bad-model baseline validates that evaluators are
    discriminative — not just checking that *a* tool was called.
    """
    from tool_eval_bench.evals.scenarios import SCENARIOS

    # Use 5 base scenarios across different categories
    test_scenarios = [s for s in SCENARIOS if s.id in ("TC-01", "TC-04", "TC-07", "TC-10", "TC-13")]
    assert len(test_scenarios) == 5

    class RandomToolAdapter(BackendAdapter):
        """Always calls 'calculator' regardless of the scenario, then answers."""
        def __init__(self) -> None:
            self._calls: dict[int, int] = {}  # per-instance call counter

        async def chat_completion(self, **kwargs: Any) -> ChatCompletionResult:
            call_id = id(self)
            count = self._calls.get(call_id, 0)
            self._calls[call_id] = count + 1

            if count == 0:
                return ChatCompletionResult(
                    content="",
                    tool_calls=[
                        ProviderToolCall(
                            id=f"rnd_{count}",
                            name="calculator",
                            arguments_str=json.dumps({"expression": "1+1"}),
                        )
                    ],
                    elapsed_ms=10.0,
                )
            return ChatCompletionResult(content="The result is 2.", elapsed_ms=10.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        repo = RunRepository(db_path=str(Path(tmpdir) / "test.sqlite"))
        reporter = MarkdownReporter(root=str(Path(tmpdir) / "runs"))
        service = BenchmarkService(repo=repo, reporter=reporter)
        service._adapter_for = lambda backend: RandomToolAdapter()

        result = await service.run_benchmark(
            model="random-tool-model",
            backend="vllm",
            base_url="http://localhost:9999",
            scenarios=test_scenarios,
        )

        scores = result["scores"]
        # A model calling calculator for every scenario should score poorly
        assert scores["final_score"] < 40, (
            f"Random tool model scored {scores['final_score']}/100 — "
            f"evaluators may not be discriminative enough"
        )
        repo.close()


@pytest.mark.asyncio
async def test_e2e_text_only_model_fails_all_tool_scenarios() -> None:
    """METH-04: Model that never uses tools should fail all Categories A-C."""
    from tool_eval_bench.evals.scenarios import SCENARIOS

    # Categories A, B, C all require tool usage
    tool_scenarios = [s for s in SCENARIOS if s.id in ("TC-01", "TC-02", "TC-03",
                                                       "TC-04", "TC-05", "TC-06",
                                                       "TC-07", "TC-08", "TC-09")]
    assert len(tool_scenarios) == 9

    class TextOnlyAdapter(BackendAdapter):
        async def chat_completion(self, **kwargs: Any) -> ChatCompletionResult:
            return ChatCompletionResult(
                content="Based on my knowledge, the answer is: something useful.",
                elapsed_ms=10.0,
            )

    with tempfile.TemporaryDirectory() as tmpdir:
        repo = RunRepository(db_path=str(Path(tmpdir) / "test.sqlite"))
        reporter = MarkdownReporter(root=str(Path(tmpdir) / "runs"))
        service = BenchmarkService(repo=repo, reporter=reporter)
        service._adapter_for = lambda backend: TextOnlyAdapter()

        result = await service.run_benchmark(
            model="text-only-model",
            backend="vllm",
            base_url="http://localhost:9999",
            scenarios=tool_scenarios,
        )

        scores = result["scores"]
        # 0 tools called = 0 points for all tool-requiring scenarios
        assert scores["total_points"] == 0
        assert scores["final_score"] == 0
        repo.close()


@pytest.mark.asyncio
async def test_e2e_report_includes_tool_overhead() -> None:
    """PERF-03: Report should include tool definition token overhead."""
    from tool_eval_bench.evals.scenarios import SCENARIOS

    tc01 = next(s for s in SCENARIOS if s.id == "TC-01")

    with tempfile.TemporaryDirectory() as tmpdir:
        repo = RunRepository(db_path=str(Path(tmpdir) / "test.sqlite"))
        reporter = MarkdownReporter(root=str(Path(tmpdir) / "runs"))
        service = BenchmarkService(repo=repo, reporter=reporter)
        service._adapter_for = lambda backend: SmokeMockAdapter()

        result = await service.run_benchmark(
            model="overhead-test",
            backend="vllm",
            base_url="http://localhost:9999",
            scenarios=[tc01],
        )

        report = Path(result["report_path"]).read_text()
        assert "Tool Definition Overhead" in report
        assert "tokens" in report
        repo.close()


@pytest.mark.asyncio
async def test_e2e_no_think_threaded_to_adapter() -> None:
    """Extra params (--no-think) should reach the adapter's chat_completion call."""
    from tool_eval_bench.evals.scenarios import SCENARIOS

    tc01 = next(s for s in SCENARIOS if s.id == "TC-01")
    captured_kwargs: list[dict] = []

    class CapturingAdapter(BackendAdapter):
        def __init__(self) -> None:
            self._call = 0

        async def chat_completion(self, **kwargs: Any) -> ChatCompletionResult:
            captured_kwargs.append(kwargs)
            self._call += 1
            if self._call == 1:
                return ChatCompletionResult(
                    content="",
                    tool_calls=[
                        ProviderToolCall(
                            id="cap_1", name="get_weather",
                            arguments_str=json.dumps({"location": "Berlin"}),
                        )
                    ],
                    elapsed_ms=10.0,
                )
            return ChatCompletionResult(content="Berlin is 8°C.", elapsed_ms=10.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        repo = RunRepository(db_path=str(Path(tmpdir) / "test.sqlite"))
        reporter = MarkdownReporter(root=str(Path(tmpdir) / "runs"))
        service = BenchmarkService(repo=repo, reporter=reporter)
        service._adapter_for = lambda backend: CapturingAdapter()

        await service.run_benchmark(
            model="think-test",
            backend="vllm",
            base_url="http://localhost:9999",
            scenarios=[tc01],
            extra_params={"chat_template_kwargs": {"enable_thinking": False}},
        )

        # Verify the extra_params reached the adapter
        assert len(captured_kwargs) >= 1
        first_call = captured_kwargs[0]
        extra = first_call.get("extra_params") or {}
        assert extra.get("chat_template_kwargs") == {"enable_thinking": False}
        repo.close()


@pytest.mark.asyncio
async def test_e2e_sampling_params_threaded_to_adapter() -> None:
    """Sampling params (top_p, top_k, etc.) should reach the adapter."""
    from tool_eval_bench.evals.scenarios import SCENARIOS

    tc01 = next(s for s in SCENARIOS if s.id == "TC-01")
    captured_kwargs: list[dict] = []

    class CapturingAdapter(BackendAdapter):
        def __init__(self) -> None:
            self._call = 0

        async def chat_completion(self, **kwargs: Any) -> ChatCompletionResult:
            captured_kwargs.append(kwargs)
            self._call += 1
            if self._call == 1:
                return ChatCompletionResult(
                    content="",
                    tool_calls=[
                        ProviderToolCall(
                            id="cap_1", name="get_weather",
                            arguments_str=json.dumps({"location": "Berlin"}),
                        )
                    ],
                    elapsed_ms=10.0,
                )
            return ChatCompletionResult(content="Berlin is 8°C.", elapsed_ms=10.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        repo = RunRepository(db_path=str(Path(tmpdir) / "test.sqlite"))
        reporter = MarkdownReporter(root=str(Path(tmpdir) / "runs"))
        service = BenchmarkService(repo=repo, reporter=reporter)
        service._adapter_for = lambda backend: CapturingAdapter()

        await service.run_benchmark(
            model="sampling-test",
            backend="vllm",
            base_url="http://localhost:9999",
            scenarios=[tc01],
            extra_params={"top_p": 0.9, "top_k": 40, "min_p": 0.05},
        )

        assert len(captured_kwargs) >= 1
        extra = captured_kwargs[0].get("extra_params") or {}
        assert extra.get("top_p") == 0.9
        assert extra.get("top_k") == 40
        assert extra.get("min_p") == 0.05
        repo.close()

