"""Tests to close coverage gaps across speculative, storage, noise, async_tools, and reports modules."""

from __future__ import annotations

import json
import time

import httpx
import pytest

from tool_eval_bench.runner.speculative import (
    SpecDecodeSample,
    _metrics_url,
    _get_prompt_for_type,
    scrape_spec_metrics,
    detect_spec_decoding,
)
from tool_eval_bench.runner.async_tools import (
    AsyncToolExecutor,
    AsyncToolResult,
    AsyncToolSpec,
    AsyncToolStatus,
    create_example_async_specs,
    format_async_status,
)
from tool_eval_bench.evals.noise import (
    enrich_calendar,
    enrich_code_execution,
    enrich_contacts,
    enrich_email,
    enrich_file_read,
    enrich_file_search,
    enrich_payload,
    enrich_reminder,
    enrich_search,
    enrich_stock,
    enrich_translation,
    enrich_weather,
)
from tool_eval_bench.storage.db import RunRepository
from tool_eval_bench.storage.reports import MarkdownReporter, _render_run_context
from tool_eval_bench.domain.models import RunContext
from tool_eval_bench.domain.scenarios import (
    Category,
    CategoryScore,
    ModelScoreSummary,
    ScenarioResult,
    ScenarioStatus,
)


# ---------------------------------------------------------------------------
# speculative: _metrics_url
# ---------------------------------------------------------------------------

class TestMetricsUrl:
    def test_strips_v1(self):
        assert _metrics_url("http://host:8000/v1") == "http://host:8000/metrics"

    def test_plain_url(self):
        assert _metrics_url("http://host:8000") == "http://host:8000/metrics"

    def test_trailing_slash(self):
        assert _metrics_url("http://host:8000/") == "http://host:8000/metrics"


# ---------------------------------------------------------------------------
# speculative: _get_prompt_for_type
# ---------------------------------------------------------------------------

class TestGetPromptForType:
    def test_code(self):
        p = _get_prompt_for_type("code")
        assert p is not None and "Python" in p

    def test_structured(self):
        p = _get_prompt_for_type("structured")
        assert p is not None and "JSON" in p

    def test_filler_returns_none(self):
        assert _get_prompt_for_type("filler") is None

    def test_unknown_returns_none(self):
        assert _get_prompt_for_type("xyz") is None


# ---------------------------------------------------------------------------
# speculative: SpecDecodeSample edge cases
# ---------------------------------------------------------------------------

class TestSpecDecodeSampleEdges:
    def test_effective_zero_total_ms(self):
        s = SpecDecodeSample(tg_tokens=50, total_ms=0)
        assert s.effective_tg_tps == 0.0

    def test_goodput_zero_total_ms(self):
        s = SpecDecodeSample(tg_tokens=50, total_ms=0, accepted_tokens_delta=40)
        assert s.goodput == 0.0

    def test_speedup_zero_baseline(self):
        s = SpecDecodeSample(tg_tokens=50, total_ms=1000, baseline_tg_tps=0.0)
        assert s.speedup_ratio is None


# ---------------------------------------------------------------------------
# speculative: scrape_spec_metrics (async, mock transport)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_scrape_spec_metrics_active():
    body = "spec_decode_num_accepted_tokens 100\nspec_decode_num_draft_tokens 200\nspec_decode_num_drafts 50\n"
    transport = httpx.MockTransport(lambda r: httpx.Response(200, text=body))
    async with httpx.AsyncClient(transport=transport) as client:
        c = await scrape_spec_metrics(client, "http://host:8000/v1")
    assert c is not None
    assert c.accepted_tokens == pytest.approx(100)


@pytest.mark.asyncio
async def test_scrape_spec_metrics_no_counters():
    body = "some_other_metric 42\n"
    transport = httpx.MockTransport(lambda r: httpx.Response(200, text=body))
    async with httpx.AsyncClient(transport=transport) as client:
        c = await scrape_spec_metrics(client, "http://host:8000/v1")
    assert c is None


@pytest.mark.asyncio
async def test_scrape_spec_metrics_non_200():
    transport = httpx.MockTransport(lambda r: httpx.Response(404))
    async with httpx.AsyncClient(transport=transport) as client:
        c = await scrape_spec_metrics(client, "http://host:8000/v1")
    assert c is None


@pytest.mark.asyncio
async def test_scrape_spec_metrics_exception():
    def explode(r):
        raise ConnectionError("down")
    transport = httpx.MockTransport(explode)
    async with httpx.AsyncClient(transport=transport) as client:
        c = await scrape_spec_metrics(client, "http://host:8000/v1")
    assert c is None


@pytest.mark.asyncio
async def test_scrape_spec_metrics_zero_but_present():
    body = "spec_decode_num_accepted_tokens 0\nspec_decode_num_draft_tokens 0\n"
    transport = httpx.MockTransport(lambda r: httpx.Response(200, text=body))
    async with httpx.AsyncClient(transport=transport) as client:
        c = await scrape_spec_metrics(client, "http://host:8000/v1")
    assert c is not None  # metric names present


# ---------------------------------------------------------------------------
# speculative: detect_spec_decoding
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_detect_spec_decoding_via_prometheus():
    body = "spec_decode_num_accepted_tokens 100\nspec_decode_num_draft_tokens 200\n"
    transport = httpx.MockTransport(lambda r: httpx.Response(200, text=body))
    async with httpx.AsyncClient(transport=transport) as client:
        info = await detect_spec_decoding(client, "http://host:8000/v1")
    assert info.active is True
    assert info.has_prometheus is True
    assert info.method == "draft_model"


@pytest.mark.asyncio
async def test_detect_spec_decoding_eagle():
    body = "spec_decode_eagle_accepted 100\nspec_decode_num_draft_tokens 200\n"
    transport = httpx.MockTransport(lambda r: httpx.Response(200, text=body))
    async with httpx.AsyncClient(transport=transport) as client:
        info = await detect_spec_decoding(client, "http://host:8000/v1")
    assert info.method == "eagle"


@pytest.mark.asyncio
async def test_detect_spec_decoding_ngram():
    body = "spec_decode_ngram_accepted 100\nspec_decode_num_draft_tokens 200\n"
    transport = httpx.MockTransport(lambda r: httpx.Response(200, text=body))
    async with httpx.AsyncClient(transport=transport) as client:
        info = await detect_spec_decoding(client, "http://host:8000/v1")
    assert info.method == "ngram"


@pytest.mark.asyncio
async def test_detect_spec_decoding_mtp():
    body = "spec_decode_mtp_accepted 100\nspec_decode_num_draft_tokens 200\n"
    transport = httpx.MockTransport(lambda r: httpx.Response(200, text=body))
    async with httpx.AsyncClient(transport=transport) as client:
        info = await detect_spec_decoding(client, "http://host:8000/v1")
    assert info.method == "mtp"


@pytest.mark.asyncio
async def test_detect_spec_decoding_user_hint():
    transport = httpx.MockTransport(lambda r: httpx.Response(404))
    async with httpx.AsyncClient(transport=transport) as client:
        info = await detect_spec_decoding(client, "http://host:8000/v1", backend_hint="mtp")
    assert info.active is True
    assert info.method == "mtp"


@pytest.mark.asyncio
async def test_detect_spec_decoding_no_metrics():
    transport = httpx.MockTransport(lambda r: httpx.Response(404))
    async with httpx.AsyncClient(transport=transport) as client:
        info = await detect_spec_decoding(client, "http://host:8000/v1")
    assert info.active is False


@pytest.mark.asyncio
async def test_detect_spec_decoding_connection_error():
    def explode(r):
        raise ConnectionError("down")
    transport = httpx.MockTransport(explode)
    async with httpx.AsyncClient(transport=transport) as client:
        info = await detect_spec_decoding(client, "http://host:8000/v1")
    assert info.active is False


# ---------------------------------------------------------------------------
# async_tools: AsyncToolExecutor
# ---------------------------------------------------------------------------

class TestAsyncToolExecutor:
    def test_start_unregistered_tool(self):
        ex = AsyncToolExecutor()
        r = ex.start_tool("unknown")
        assert r.status == AsyncToolStatus.COMPLETED
        assert "not registered" in str(r.result)

    def test_start_registered_tool(self):
        ex = AsyncToolExecutor()
        ex.register_tool(AsyncToolSpec(tool_name="t", duration_ms=5000))
        r = ex.start_tool("t")
        assert r.status == AsyncToolStatus.PENDING
        assert r.handle.startswith("async_t_")

    def test_poll_unknown_handle(self):
        ex = AsyncToolExecutor()
        r = ex.poll_tool("no_such_handle")
        assert r.status == AsyncToolStatus.FAILED

    def test_poll_completed(self):
        ex = AsyncToolExecutor()
        ex.register_tool(AsyncToolSpec(tool_name="fast", duration_ms=0.001, final_result={"ok": True}))
        r = ex.start_tool("fast")
        time.sleep(0.01)
        r2 = ex.poll_tool(r.handle)
        assert r2.status == AsyncToolStatus.COMPLETED
        assert r2.result == {"ok": True}

    def test_poll_running_with_intermediate(self):
        ex = AsyncToolExecutor()
        ex.register_tool(AsyncToolSpec(
            tool_name="slow", duration_ms=100_000,
            supports_streaming=True,
            intermediate_results=["partial_a", "partial_b"],
        ))
        r = ex.start_tool("slow")
        r2 = ex.poll_tool(r.handle)
        assert r2.status == AsyncToolStatus.RUNNING
        assert r2.progress_percent is not None

    def test_cancel_tool(self):
        ex = AsyncToolExecutor()
        ex.register_tool(AsyncToolSpec(tool_name="c", duration_ms=100_000))
        r = ex.start_tool("c")
        cr = ex.cancel_tool(r.handle)
        assert cr.status == AsyncToolStatus.CANCELLED

    def test_cancel_unknown(self):
        ex = AsyncToolExecutor()
        cr = ex.cancel_tool("nope")
        assert cr.status == AsyncToolStatus.CANCELLED

    def test_simulated_failure(self):
        ex = AsyncToolExecutor()
        ex.register_tool(AsyncToolSpec(
            tool_name="fail", duration_ms=0.001,
            simulate_failure=True, failure_at_percent=0.0,
        ))
        r = ex.start_tool("fail")
        time.sleep(0.01)
        r2 = ex.poll_tool(r.handle)
        assert r2.status == AsyncToolStatus.FAILED

    def test_poll_unregistered_spec(self):
        ex = AsyncToolExecutor()
        ex._started_at["fake_unknown_1"] = time.monotonic()
        r = ex.poll_tool("fake_unknown_1")
        assert r.status == AsyncToolStatus.COMPLETED
        assert "No spec" in str(r.result)


# ---------------------------------------------------------------------------
# async_tools: format_async_status
# ---------------------------------------------------------------------------

class TestFormatAsyncStatus:
    def test_pending(self):
        r = AsyncToolResult(status=AsyncToolStatus.PENDING, handle="h1")
        s = format_async_status(r)
        assert "pending" in s and "h1" in s

    def test_running(self):
        r = AsyncToolResult(status=AsyncToolStatus.RUNNING, handle="h2", progress_percent=0.5)
        s = format_async_status(r)
        parsed = json.loads(s)
        assert parsed["status"] == "running"

    def test_running_with_intermediate(self):
        r = AsyncToolResult(status=AsyncToolStatus.RUNNING, handle="h3", progress_percent=0.5, intermediate_data={"x": 1})
        s = format_async_status(r)
        parsed = json.loads(s)
        assert parsed["intermediate_data"] == {"x": 1}

    def test_completed(self):
        r = AsyncToolResult(status=AsyncToolStatus.COMPLETED, handle="h4", result=42)
        s = format_async_status(r)
        parsed = json.loads(s)
        assert parsed["result"] == 42

    def test_failed(self):
        r = AsyncToolResult(status=AsyncToolStatus.FAILED, handle="h5", error="boom")
        s = format_async_status(r)
        assert "failed" in s and "boom" in s

    def test_cancelled(self):
        r = AsyncToolResult(status=AsyncToolStatus.CANCELLED, handle="h6")
        s = format_async_status(r)
        assert "cancelled" in s


def test_create_example_async_specs():
    specs = create_example_async_specs()
    assert len(specs) == 3
    names = {s.tool_name for s in specs}
    assert "search_files" in names
    assert "web_search" in names


# ---------------------------------------------------------------------------
# noise: enrichment functions
# ---------------------------------------------------------------------------

class TestNoiseEnrichment:
    def test_enrich_weather(self):
        r = enrich_weather({"temperature": 20})
        assert r["temperature"] == 20
        assert "wind_speed_kmh" in r
        assert "station_id" in r

    def test_enrich_search(self):
        r = enrich_search({"results": [{"title": "a"}, {"title": "b"}]})
        assert len(r["results"]) == 2
        assert r["results"][0]["rank"] == 1
        assert "total_results" in r

    def test_enrich_file_search(self):
        r = enrich_file_search({"results": [{"name": "f.txt"}]})
        assert r["results"][0]["size_bytes"] == 28_416
        assert "search_time_ms" in r

    def test_enrich_file_read(self):
        r = enrich_file_read({"content": "hello\nworld"})
        assert r["line_count"] == 2
        assert r["encoding"] == "utf-8"

    def test_enrich_email(self):
        r = enrich_email({"status": "sent"})
        assert r["delivery_status"] == "accepted"

    def test_enrich_calendar(self):
        r = enrich_calendar({"event_id": "e1"})
        assert r["calendar_id"] == "cal_primary"

    def test_enrich_contacts(self):
        r = enrich_contacts({"results": [{"name": "A"}, {"name": "B"}]})
        assert len(r["results"]) == 2
        assert r["results"][0]["department"] == "Engineering"

    def test_enrich_stock(self):
        r = enrich_stock({"price": 100.0})
        assert r["exchange"] == "NASDAQ"
        assert r["day_high"] > r["price"]

    def test_enrich_translation(self):
        r = enrich_translation({"translated": "hola mundo"})
        assert r["confidence"] == 0.98
        assert r["word_count"] == 2

    def test_enrich_code_execution(self):
        r = enrich_code_execution({"output": "42"})
        assert "sandbox_id" in r

    def test_enrich_reminder(self):
        r = enrich_reminder({"reminder_id": "r1"})
        assert "notification_channels" in r


class TestEnrichPayload:
    def test_known_tool(self):
        r = enrich_payload("get_weather", {"temperature": 10})
        assert "wind_speed_kmh" in r

    def test_unknown_tool_passthrough(self):
        orig = {"foo": "bar"}
        assert enrich_payload("unknown_tool", orig) is orig

    def test_error_payload(self):
        r = enrich_payload("get_weather", {"error": "timeout"})
        assert r["error_code"] == "ERR_TOOL_UNAVAILABLE"

    def test_non_dict_passthrough(self):
        assert enrich_payload("get_weather", "string") == "string"

    def test_calculator_passthrough(self):
        orig = {"result": 42}
        assert enrich_payload("calculator", orig) is orig


# ---------------------------------------------------------------------------
# storage/db: additional coverage
# ---------------------------------------------------------------------------

class TestRunRepositoryExtended:
    def test_get_nonexistent(self, tmp_path):
        repo = RunRepository(db_path=str(tmp_path / "test.sqlite"))
        assert repo.get("no_such_id") is None
        repo.close()

    def test_get_latest_empty(self, tmp_path):
        repo = RunRepository(db_path=str(tmp_path / "test.sqlite"))
        assert repo.get_latest() is None
        repo.close()

    def test_get_latest_with_data(self, tmp_path):
        repo = RunRepository(db_path=str(tmp_path / "test.sqlite"))
        repo.upsert_scenario_run({"run_id": "r1", "config": {"model": "m"}, "scores": {}, "metadata": {}})
        latest = repo.get_latest()
        assert latest is not None and latest["run_id"] == "r1"
        repo.close()

    def test_get_latest_filtered_by_model(self, tmp_path):
        repo = RunRepository(db_path=str(tmp_path / "test.sqlite"))
        repo.upsert_scenario_run({"run_id": "r1", "config": {"model": "a"}, "scores": {}, "metadata": {}})
        repo.upsert_scenario_run({"run_id": "r2", "config": {"model": "b"}, "scores": {}, "metadata": {}})
        assert repo.get_latest(model="a")["run_id"] == "r1"
        assert repo.get_latest(model="c") is None
        repo.close()

    def test_get_scenario_results(self, tmp_path):
        repo = RunRepository(db_path=str(tmp_path / "test.sqlite"))
        repo.upsert_scenario_run({
            "run_id": "r1", "config": {"model": "m"},
            "scores": {"scenario_results": [{"id": "TC-01", "status": "pass"}]},
            "metadata": {},
        })
        results = repo.get_scenario_results("r1")
        assert results is not None and len(results) == 1
        repo.close()

    def test_get_scenario_results_missing(self, tmp_path):
        repo = RunRepository(db_path=str(tmp_path / "test.sqlite"))
        assert repo.get_scenario_results("missing") is None
        repo.close()

    def test_get_scenario_results_no_scores(self, tmp_path):
        repo = RunRepository(db_path=str(tmp_path / "test.sqlite"))
        repo.upsert_scenario_run({"run_id": "r1", "config": {"model": "m"}, "metadata": {}})
        assert repo.get_scenario_results("r1") is None
        repo.close()

    def test_upsert_updates_existing(self, tmp_path):
        repo = RunRepository(db_path=str(tmp_path / "test.sqlite"))
        repo.upsert_scenario_run({"run_id": "r1", "config": {"model": "m"}, "scores": {"v": 1}, "metadata": {}})
        repo.upsert_scenario_run({"run_id": "r1", "config": {"model": "m"}, "scores": {"v": 2}, "metadata": {}})
        assert repo.get("r1")["scores"]["v"] == 2
        repo.close()

    def test_list_filter_by_model(self, tmp_path):
        repo = RunRepository(db_path=str(tmp_path / "test.sqlite"))
        repo.upsert_scenario_run({"run_id": "r1", "config": {"model": "a"}, "scores": {}, "metadata": {}})
        repo.upsert_scenario_run({"run_id": "r2", "config": {"model": "b"}, "scores": {}, "metadata": {}})
        assert len(repo.list(model="a")) == 1
        repo.close()

    def test_del_safety_net(self, tmp_path):
        repo = RunRepository(db_path=str(tmp_path / "test.sqlite"))
        del repo  # should not raise


# ---------------------------------------------------------------------------
# storage/reports: _render_run_context
# ---------------------------------------------------------------------------

def _make_run_context(**overrides) -> RunContext:
    defaults = dict(
        tool_version="1.3.0", git_sha="abc1234", hostname="testhost",
        platform_info="Linux-6.1", python_version="3.11.5",
        model="test-model", backend="vllm", base_url="http://localhost:8000",
    )
    defaults.update(overrides)
    return RunContext(**defaults)


class TestRenderRunContext:
    def test_basic_context(self):
        ctx = _make_run_context()
        md = _render_run_context(ctx)
        text = "\n".join(md)
        assert "## Run Context" in text
        assert "test-model" in text
        assert "## Environment" in text  # no engine info

    def test_with_engine_info(self):
        ctx = _make_run_context(engine_name="vLLM", engine_version="0.8.5", max_model_len=32768, gpu_count=2)
        md = _render_run_context(ctx)
        text = "\n".join(md)
        assert "## Inference Engine" in text
        assert "vLLM 0.8.5" in text
        assert "32,768" in text

    def test_with_quantization_and_spec(self):
        ctx = _make_run_context(engine_name="vLLM", quantization="AWQ", spec_decoding="mtp")
        md = _render_run_context(ctx)
        text = "\n".join(md)
        assert "AWQ" in text
        assert "mtp" in text

    def test_with_context_pressure(self):
        ctx = _make_run_context(context_pressure=0.75)
        md = _render_run_context(ctx)
        text = "\n".join(md)
        assert "75%" in text

    def test_with_extra_params(self):
        ctx = _make_run_context(extra_params={"top_k": 40})
        md = _render_run_context(ctx)
        text = "\n".join(md)
        assert "top_k" in text

    def test_server_model_root_shown(self):
        ctx = _make_run_context(server_model_root="Qwen/Qwen-27B")
        md = _render_run_context(ctx)
        text = "\n".join(md)
        assert "Qwen/Qwen-27B" in text


# ---------------------------------------------------------------------------
# storage/reports: spec decode report + run_context in reports
# ---------------------------------------------------------------------------

class TestSpecDecodeReport:
    def test_basic_spec_report(self, tmp_path):
        reporter = MarkdownReporter(root=str(tmp_path))
        samples = [
            SpecDecodeSample(tg_tokens=128, total_ms=2000, ttft_ms=200, tg_tps=50.0, spec_method="mtp", prompt_type="code"),
            SpecDecodeSample(tg_tokens=128, total_ms=3000, ttft_ms=300, tg_tps=40.0, spec_method="mtp", prompt_type="structured"),
        ]
        path = reporter.write_spec_decode_report("spec_001", "test-model", samples)
        content = path.read_text()
        assert "Speculative Decoding Benchmark" in content
        assert "mtp" in content
        assert "Interpretation Guide" in content

    def test_spec_report_with_acceptance_rate(self, tmp_path):
        reporter = MarkdownReporter(root=str(tmp_path))
        samples = [
            SpecDecodeSample(
                tg_tokens=128, total_ms=2000, ttft_ms=200, tg_tps=50.0,
                spec_method="mtp", prompt_type="code",
                acceptance_rate=0.85, acceptance_length=3.2,
            ),
        ]
        path = reporter.write_spec_decode_report("spec_002", "test-model", samples)
        content = path.read_text()
        assert "α (accept)" in content
        assert "85.0%" in content
        assert "Acceptance Rate by Prompt Type" in content

    def test_spec_report_no_acceptance_rate(self, tmp_path):
        reporter = MarkdownReporter(root=str(tmp_path))
        samples = [SpecDecodeSample(tg_tokens=64, total_ms=1000, tg_tps=30.0)]
        path = reporter.write_spec_decode_report("spec_003", "model", samples)
        content = path.read_text()
        assert "[!NOTE]" in content
        assert "not available" in content


class TestScenarioReportWithContext:
    def _make_summary(self):
        results = [ScenarioResult(
            scenario_id="TC-01", status=ScenarioStatus.PASS, points=2,
            summary="ok", raw_log="trace",
        )]
        return ModelScoreSummary(
            scenario_results=results,
            category_scores=[CategoryScore(Category.A, "A", 2, 2, 100)],
            final_score=100, total_points=2, max_points=2, rating="★★★★★",
        )

    def test_report_with_run_context(self, tmp_path):
        reporter = MarkdownReporter(root=str(tmp_path))
        ctx = _make_run_context()
        path = reporter.write_scenario_report("r1", "m", self._make_summary(), run_context=ctx)
        content = path.read_text()
        assert "## Run Context" in content
        assert "v1.3.0" in content

    def test_report_with_deployability(self, tmp_path):
        reporter = MarkdownReporter(root=str(tmp_path))
        summary = self._make_summary()
        summary.deployability = 85
        summary.responsiveness = 90
        summary.median_turn_ms = 1500.0
        summary.alpha = 0.7
        path = reporter.write_scenario_report("r2", "m", summary)
        content = path.read_text()
        assert "Deployability" in content
        assert "85" in content

    def test_report_with_context_pressure(self, tmp_path):
        reporter = MarkdownReporter(root=str(tmp_path))
        cp = {"ratio": 0.5, "fill_tokens": 4000, "context_size": 8192}
        path = reporter.write_scenario_report("r3", "m", self._make_summary(), context_pressure_config=cp)
        content = path.read_text()
        assert "Context Pressure" in content
        assert "50%" in content

    def test_throughput_report_with_context(self, tmp_path):
        from dataclasses import dataclass

        @dataclass
        class FakeSample:
            pp_tps: float = 1000.0
            tg_tps: float = 50.0
            ttft_ms: float = 100.0
            total_ms: float = 3000.0
            pp_tokens: int = 2048
            tg_tokens: int = 128
            label_pp: int = 2048
            label_depth: int = 0
            concurrency: int = 1
            error: str | None = None

        reporter = MarkdownReporter(root=str(tmp_path))
        ctx = _make_run_context()
        path = reporter.write_throughput_report("tp1", "model", [FakeSample()], run_context=ctx)
        content = path.read_text()
        assert "v1.3.0" in content
