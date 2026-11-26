"""Tests for the public headless API (tool_eval_bench.api) and schema module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tool_eval_bench.api import (
    ARGS_SCHEMA,
    OUTPUT_SCHEMA_VERSION,
    format_result,
    run_benchmark,
)
from tool_eval_bench.schema import ARGS_SCHEMA as SCHEMA_DIRECT
from tool_eval_bench.schema import get_schema

# ---------------------------------------------------------------------------
# format_result tests
# ---------------------------------------------------------------------------


class TestFormatResult:
    """Test the versioned envelope wrapper."""

    def test_adds_schema_version(self):
        data = {"run_id": "test-123", "scores": {}}
        result = format_result(data)
        assert result["schema_version"] == OUTPUT_SCHEMA_VERSION

    def test_adds_tool_version(self):
        from tool_eval_bench import __version__

        data = {"run_id": "test-123", "scores": {}}
        result = format_result(data)
        assert result["tool_eval_bench_version"] == __version__

    def test_preserves_run_data(self):
        data = {
            "run_id": "test-123",
            "status": "completed",
            "config": {"model": "qwen"},
            "scores": {"final_score": 87, "max_points": 138},
        }
        result = format_result(data)
        assert result["run_id"] == "test-123"
        assert result["status"] == "completed"
        assert result["config"]["model"] == "qwen"

    def test_promotes_spark_arena_fields(self):
        data = {
            "run_id": "test-123",
            "scores": {
                "final_score": 87,
                "rating": "★★★★ Good",
                "safety_warnings": ["TC-K1 failed"],
                "deployability": 82,
                "responsiveness": 72,
                "weighted_score": 84,
                # TC-03 timed out: excluded from max_points, still a scenario.
                "max_points": 4,
                "excluded_scenarios": ["TC-03"],
                "scenario_results": [
                    {"scenario_id": "TC-01", "status": "pass", "points": 2},
                    {"scenario_id": "TC-02", "status": "fail", "points": 0},
                    {
                        "scenario_id": "TC-03",
                        "status": "fail",
                        "points": 0,
                        "failure_kind": "timeout",
                    },
                ],
            },
        }
        result = format_result(data)
        assert result["final_score"] == 87
        assert result["rating"] == "★★★★ Good"
        assert result["safety_warnings"] == ["TC-K1 failed"]
        assert result["deployability"] == 82
        assert result["responsiveness"] == 72
        assert result["weighted_score"] == 84
        assert result["total_scenarios"] == 3

    def test_total_scenarios_counts_a_run_whose_every_scenario_was_excluded(self):
        timed_out = {"scenario_id": "TC-01", "status": "fail", "points": 0}
        result = format_result({"scores": {"max_points": 0, "scenario_results": [timed_out]}})
        assert result["total_scenarios"] == 1

    def test_weighted_score_is_none_without_difficulty_weighting(self):
        assert format_result({"scores": {"final_score": 50}})["weighted_score"] is None

    def test_empty_scores_returns_none_fields(self):
        result = format_result({"scores": {}})
        assert result["final_score"] is None
        assert result["rating"] is None
        assert result["safety_warnings"] == []
        assert result["total_scenarios"] is None

    def test_null_scores_falls_back_to_none_fields(self):
        result = format_result({"run_id": "null-scores", "scores": None})
        assert result["run_id"] == "null-scores"
        assert result["final_score"] is None
        assert result["safety_warnings"] == []
        assert result["total_scenarios"] is None

    def test_json_serializable(self):
        data = {
            "run_id": "test-123",
            "scores": {"final_score": 87, "max_points": 138},
        }
        result = format_result(data)
        text = json.dumps(result)
        assert "schema_version" in text


# ---------------------------------------------------------------------------
# ARGS_SCHEMA tests
# ---------------------------------------------------------------------------


class TestArgsSchema:
    """Test the machine-readable argument schema."""

    def test_schema_is_list(self):
        assert isinstance(ARGS_SCHEMA, list)

    def test_schema_not_empty(self):
        assert len(ARGS_SCHEMA) > 10  # we have ~20+ args

    def test_all_entries_have_required_fields(self):
        for entry in ARGS_SCHEMA:
            assert "name" in entry, f"Missing 'name' in {entry}"
            assert "type" in entry, f"Missing 'type' in {entry}"
            assert "description" in entry, f"Missing 'description' in {entry}"
            # default can be None, so just check key exists
            assert "default" in entry, f"Missing 'default' in {entry}"

    def test_known_args_present(self):
        names = {e["name"] for e in ARGS_SCHEMA}
        assert "backend" in names
        assert "temperature" in names
        assert "short" in names
        assert "hardmode" in names
        assert "timeout" in names
        assert "parallel" in names
        assert "alpha" in names

    def test_re_export_matches(self):
        """ARGS_SCHEMA from api.py should be the same object as from schema.py."""
        assert ARGS_SCHEMA is SCHEMA_DIRECT

    def test_get_schema_includes_version(self):
        schema = get_schema()
        assert "schema_version" in schema
        assert "args" in schema
        assert schema["args"] is ARGS_SCHEMA

    def test_schema_size(self):
        """Schema should cover all public args — update this when adding flags."""
        assert len(ARGS_SCHEMA) >= 40, (
            f"ARGS_SCHEMA has only {len(ARGS_SCHEMA)} entries — did you forget to "
            "add a new flag to schema.py?"
        )

    def test_all_parser_args_in_schema_or_hidden(self):
        """Every public parser argument must appear in ARGS_SCHEMA.

        This is the canonical drift-detection test.  It builds the real
        argparse parser (via ``_make_parser()``) and checks that every
        non-suppressed dest is either:
          - listed in ARGS_SCHEMA, OR
          - listed in _HIDDEN_ARGS (intentionally suppressed).

        Fail here means a CLI flag was added to bench.py without a
        corresponding entry in schema.py.
        """
        from tool_eval_bench.cli.bench import _HIDDEN_ARGS, _make_parser

        parser = _make_parser()
        schema_names = {entry["name"] for entry in ARGS_SCHEMA}

        missing: list[str] = []
        for action in parser._actions:
            dest = action.dest
            # argparse.SUPPRESS actions have dest == SUPPRESS; skip
            if dest == "==SUPPRESS==":
                continue
            if dest in _HIDDEN_ARGS:
                continue
            if dest not in schema_names:
                missing.append(dest)

        assert not missing, (
            "The following CLI args are public (not in _HIDDEN_ARGS) but missing "
            "from ARGS_SCHEMA in schema.py — add them:\n  " + "\n  ".join(sorted(missing))
        )


# ---------------------------------------------------------------------------
# run_benchmark tests (mocked service)
# ---------------------------------------------------------------------------


class TestRunBenchmark:
    """Test the programmatic run_benchmark entry point."""

    @pytest.fixture(autouse=True)
    def no_detection(self, monkeypatch):
        """These tests cover orchestration and envelope; detection has its own module."""

        async def keep(backend, **kwargs):
            return backend

        monkeypatch.setattr("tool_eval_bench.api.detect_backend", keep)
        monkeypatch.setattr("tool_eval_bench.api.build_run_context", AsyncMock(return_value=None))

    @pytest.mark.asyncio
    async def test_runs_selected_scenarios_without_persistence_and_forwards_options(
        self, monkeypatch, tmp_path
    ):
        from tool_eval_bench.adapters.base import BackendAdapter, ChatCompletionResult
        from tool_eval_bench.application.service import BenchmarkService
        from tool_eval_bench.evals.scenarios import SCENARIOS

        monkeypatch.chdir(tmp_path)

        class StaticAdapter(BackendAdapter):
            async def chat_completion(self, **kwargs):
                return ChatCompletionResult(content="ok", finish_reason="stop")

        monkeypatch.setattr(BenchmarkService, "_adapter_for", lambda *a, **k: StaticAdapter())
        started: list[str] = []
        finished: list[str] = []

        async def on_start(sc, idx, total):
            started.append(sc.id)

        async def on_result(sc, res, idx, total):
            finished.append(sc.id)

        result = await run_benchmark(
            model="test-model",
            base_url="http://localhost:8000",
            scenarios=[SCENARIOS[0]],
            weight_by_difficulty=True,
            system_prompt="You are a strict evaluator.",
            on_scenario_start=on_start,
            on_scenario_result=on_result,
            persist=False,
        )

        assert result["schema_version"] == OUTPUT_SCHEMA_VERSION
        assert result["total_scenarios"] == 1
        assert result["config"]["weight_by_difficulty"] is True
        assert result["config"]["system_prompt"] == "You are a strict evaluator."
        assert started == [SCENARIOS[0].id]
        assert finished == [SCENARIOS[0].id]
        assert not (tmp_path / "data" / "benchmarks.sqlite").exists()

    @pytest.mark.asyncio
    async def test_short_preset_resolves_core_scenarios_and_persists_when_enabled(
        self, monkeypatch, tmp_path
    ):
        from tool_eval_bench.application import service as service_module
        from tool_eval_bench.domain.scenarios import ScenarioResult, ScenarioStatus
        from tool_eval_bench.evals.scenarios import SCENARIOS
        from tool_eval_bench.runner.orchestrator import score_results

        monkeypatch.chdir(tmp_path)
        seen_ids: list[str] = []

        async def fake_run(adapter, *, scenarios, **kwargs):
            seen_ids[:] = [s.id for s in scenarios]
            results = [
                ScenarioResult(
                    scenario_id=s.id,
                    status=ScenarioStatus.PASS,
                    points=2,
                    summary="ok",
                    raw_log="trace",
                )
                for s in scenarios
            ]
            return score_results(results, scenarios)

        monkeypatch.setattr(service_module, "run_all_scenarios", fake_run)
        monkeypatch.setattr(
            service_module.BenchmarkService, "_adapter_for", lambda *a, **k: object()
        )

        result = await run_benchmark(
            model="test-model",
            base_url="http://localhost:8000",
            short=True,
            persist=True,
            output_dir=str(tmp_path / "runs"),
        )

        assert seen_ids == [s.id for s in SCENARIOS]
        assert result["total_scenarios"] == len(SCENARIOS)
        assert Path(result["report_path"]).exists()
        assert (tmp_path / "data" / "benchmarks.sqlite").exists()


# ---------------------------------------------------------------------------
# JSONL progress callbacks (from cli/bench.py)
# ---------------------------------------------------------------------------


class TestStderrProgress:
    """Test the JSONL progress event callbacks."""

    @pytest.mark.asyncio
    async def test_scenario_start_emits_jsonl(self, capsys):
        from tool_eval_bench.cli.bench import _stderr_progress_start

        scenario = MagicMock()
        scenario.id = "TC-01"
        scenario.title = "Test Scenario"
        scenario.category.value = "A"

        await _stderr_progress_start(scenario, 0, 69)

        captured = capsys.readouterr()
        line = captured.err.strip()
        event = json.loads(line)
        assert event["event"] == "scenario_start"
        assert event["scenario_id"] == "TC-01"
        assert event["index"] == 0
        assert event["total"] == 69

    @pytest.mark.asyncio
    async def test_scenario_result_emits_jsonl(self, capsys):
        from tool_eval_bench.cli.bench import _stderr_progress_result

        scenario = MagicMock()
        scenario.id = "TC-01"

        result = MagicMock()
        result.status.value = "pass"
        result.points = 2
        result.duration_seconds = 1.234

        await _stderr_progress_result(scenario, result, 0, 69)

        captured = capsys.readouterr()
        line = captured.err.strip()
        event = json.loads(line)
        assert event["event"] == "scenario_result"
        assert event["scenario_id"] == "TC-01"
        assert event["status"] == "pass"
        assert event["points"] == 2
        assert event["duration_seconds"] == 1.23


# ---------------------------------------------------------------------------
# _emit_json_output tests
# ---------------------------------------------------------------------------


class TestEmitJsonOutput:
    """Test JSON output to stdout and file."""

    def test_stdout_output(self, capsys):
        from tool_eval_bench.cli.bench import _emit_json_output

        data = {"run_id": "test", "scores": {"final_score": 50}}
        _emit_json_output(data)

        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["schema_version"] == OUTPUT_SCHEMA_VERSION
        assert parsed["run_id"] == "test"

    def test_file_output(self, tmp_path, capsys):
        from tool_eval_bench.cli.bench import _emit_json_output

        out_file = str(tmp_path / "result.json")
        data = {"run_id": "file-test", "scores": {"final_score": 75}}
        _emit_json_output(data, json_file=out_file)

        # Should NOT print to stdout
        captured = capsys.readouterr()
        assert captured.out == ""

        # Should write to file
        content = json.loads(Path(out_file).read_text(encoding="utf-8"))
        assert content["run_id"] == "file-test"
        assert content["schema_version"] == OUTPUT_SCHEMA_VERSION

        # Should emit benchmark_complete on stderr
        stderr_line = captured.err.strip()
        event = json.loads(stderr_line)
        assert event["event"] == "benchmark_complete"
        assert event["json_file"] == out_file

    def test_file_creates_parent_dirs(self, tmp_path):
        from tool_eval_bench.cli.bench import _emit_json_output

        out_file = str(tmp_path / "nested" / "deep" / "result.json")
        _emit_json_output({"scores": {}}, json_file=out_file)
        assert Path(out_file).exists()


# ---------------------------------------------------------------------------
# Headless model detection (P0 agent-friendliness)
# ---------------------------------------------------------------------------


class TestHeadlessModelDetection:
    """Test that --json mode auto-selects models without interactive prompts."""

    def test_headless_auto_selects_first_model(self, capsys):
        """When headless=True and multiple models are available, pick the first."""
        from io import StringIO

        from rich.console import Console

        from tool_eval_bench.cli.bench import _detect_model

        # Mock httpx response with 2 models
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"id": "model-a", "root": "org/model-a"},
                {"id": "model-b", "root": "org/model-b"},
            ]
        }
        mock_response.raise_for_status = MagicMock()

        console = Console(file=StringIO(), width=200, no_color=True)

        with patch("tool_eval_bench.cli.bench.asyncio") as mock_asyncio:
            mock_asyncio.run.side_effect = lambda coro: (coro.close(), (mock_response, False))[1]
            api_id, display = _detect_model(
                "http://localhost:8000",
                None,
                console,
                headless=True,
            )

        assert api_id == "model-a"
        assert display == "org/model-a"

        # Should emit JSONL event on stderr
        captured = capsys.readouterr()
        event = json.loads(captured.err.strip())
        assert event["event"] == "model_auto_selected"
        assert event["model"] == "model-a"
        assert event["total_available"] == 2
        assert "model-b" in event["available_models"]

    def test_headless_single_model_no_event(self, capsys):
        """Single model should auto-select without emitting model_auto_selected."""
        from io import StringIO

        from rich.console import Console

        from tool_eval_bench.cli.bench import _detect_model

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"id": "only-model"}]}
        mock_response.raise_for_status = MagicMock()

        console = Console(file=StringIO(), width=200, no_color=True)

        with patch("tool_eval_bench.cli.bench.asyncio") as mock_asyncio:
            mock_asyncio.run.side_effect = lambda coro: (coro.close(), (mock_response, False))[1]
            api_id, display = _detect_model(
                "http://localhost:8000",
                None,
                console,
                headless=True,
            )

        assert api_id == "only-model"
        # No model_auto_selected event for single model
        captured = capsys.readouterr()
        assert captured.err.strip() == ""


class TestHeadlessError:
    """Test structured error output for headless mode."""

    def test_emits_jsonl_and_exits(self, capsys):
        from tool_eval_bench.cli.bench import _headless_error

        with pytest.raises(SystemExit) as exc_info:
            _headless_error("connection_failed", "Server is down", exit_code=2)

        assert exc_info.value.code == 2

        captured = capsys.readouterr()
        event = json.loads(captured.err.strip())
        assert event["event"] == "error"
        assert event["error"] == "connection_failed"
        assert event["message"] == "Server is down"

    def test_exit_code_3_for_no_models(self, capsys):
        from tool_eval_bench.cli.bench import _headless_error

        with pytest.raises(SystemExit) as exc_info:
            _headless_error("no_models", "Empty model list", exit_code=3)

        assert exc_info.value.code == 3

    def test_default_exit_code_is_1(self, capsys):
        from tool_eval_bench.cli.bench import _headless_error

        with pytest.raises(SystemExit) as exc_info:
            _headless_error("unknown", "Something broke")

        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# Server auto-discovery
# ---------------------------------------------------------------------------


class TestServerDiscovery:
    """Test automatic inference server port scanning."""

    def test_discovers_vllm_on_8000(self, capsys):
        from tool_eval_bench.cli.bench import _discover_server

        with patch("tool_eval_bench.cli.server.asyncio") as mock_asyncio:
            # Inner _probe returns (url, backend, server_name, port)
            discovered = ("http://localhost:8000", "vllm", "vLLM", 8000)
            mock_asyncio.run.side_effect = lambda coro: (
                coro.close(),
                discovered,
            )[1]

            result = _discover_server(headless=True)

        assert result is not None
        base_url, backend = result
        assert base_url == "http://localhost:8000"
        assert backend == "vllm"

        # Headless mode should emit JSONL
        captured = capsys.readouterr()
        event = json.loads(captured.err.strip())
        assert event["event"] == "server_discovered"
        assert event["port"] == 8000
        assert event["server_type"] == "vLLM"

    def test_returns_none_when_no_server(self):
        from tool_eval_bench.cli.bench import _discover_server

        with patch("tool_eval_bench.cli.server.asyncio") as mock_asyncio:
            mock_asyncio.run.side_effect = lambda coro: (coro.close(), None)[1]

            result = _discover_server(headless=True)

        assert result is None

    def test_common_ports_list(self):
        from tool_eval_bench.cli.bench import _DISCOVERY_PORTS

        ports = [p for p, _, _ in _DISCOVERY_PORTS]
        # vLLM default
        assert 8000 in ports
        # llama.cpp / alt vLLM ports
        assert 8080 in ports
        assert 8081 in ports
        assert 8082 in ports
        # SGLang default
        assert 30000 in ports
        # LiteLLM default
        assert 4000 in ports


# ---------------------------------------------------------------------------
# Probe server readiness
# ---------------------------------------------------------------------------


class TestProbeServer:
    """Test the --probe readiness check."""

    def test_probe_success_exits_0(self, capsys):
        from io import StringIO

        from rich.console import Console

        from tool_eval_bench.cli.bench import _probe_server

        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {"data": [{"id": "test-model"}]}

        console = Console(file=StringIO(), width=200, no_color=True)

        with patch("tool_eval_bench.cli.bench.asyncio") as mock_asyncio:
            mock_asyncio.run.side_effect = lambda coro: (coro.close(), mock_resp)[1]
            with pytest.raises(SystemExit) as exc_info:
                _probe_server(console, "http://localhost:8000", None, headless=True)

        assert exc_info.value.code == 0

        captured = capsys.readouterr()
        event = json.loads(captured.err.strip())
        assert event["event"] == "probe_result"
        assert event["status"] == "ready"
        assert "test-model" in event["models"]

    def test_probe_failure_exits_1(self, capsys):
        from io import StringIO

        from rich.console import Console

        from tool_eval_bench.cli.bench import _probe_server

        console = Console(file=StringIO(), width=200, no_color=True)

        with patch("tool_eval_bench.cli.bench.asyncio") as mock_asyncio:

            def fail(coro):
                coro.close()
                raise Exception("Connection refused")

            mock_asyncio.run.side_effect = fail
            with pytest.raises(SystemExit) as exc_info:
                _probe_server(console, "http://localhost:8000", None, headless=True)

        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        event = json.loads(captured.err.strip())
        assert event["event"] == "probe_result"
        assert event["status"] == "failed"


# ---------------------------------------------------------------------------
# BenchmarkService persistence bypass (regression test)
# ---------------------------------------------------------------------------


class TestServicePersistence:
    """Verify that repo=None / reporter=None correctly skips persistence."""

    def test_none_repo_is_not_replaced(self):
        from tool_eval_bench.runner.service import BenchmarkService

        service = BenchmarkService(repo=None, reporter=None)
        assert service.repo is None
        assert service.reporter is None

    def test_default_repo_created_when_omitted(self, tmp_path, monkeypatch):
        from tool_eval_bench.runner.service import BenchmarkService
        from tool_eval_bench.storage.db import RunRepository

        monkeypatch.chdir(tmp_path)
        service = BenchmarkService()
        assert isinstance(service.repo, RunRepository)
        service.repo.close()


# ---------------------------------------------------------------------------
# Backend detection from response headers
# ---------------------------------------------------------------------------


class TestBackendDetection:
    """Test _detect_backend_from_response."""

    def test_detects_vllm_header(self):
        from tool_eval_bench.cli.server import (
            detect_backend_from_response as _detect_backend_from_response,
        )

        resp = MagicMock()
        resp.headers = {"server": "vllm/0.8.5"}
        backend, label = _detect_backend_from_response(resp, 8080)
        assert backend == "vllm"
        assert label == "vLLM"

    def test_detects_llamacpp_header(self):
        from tool_eval_bench.cli.server import (
            detect_backend_from_response as _detect_backend_from_response,
        )

        resp = MagicMock()
        resp.headers = {"server": "llama.cpp/server"}
        backend, label = _detect_backend_from_response(resp, 8080)
        assert backend == "llamacpp"
        assert label == "llama.cpp"

    def test_detects_sglang_header(self):
        from tool_eval_bench.cli.server import (
            detect_backend_from_response as _detect_backend_from_response,
        )

        resp = MagicMock()
        resp.headers = {"server": "sglang/0.4.6"}
        backend, label = _detect_backend_from_response(resp, 30000)
        assert backend == "sglang"
        assert label == "SGLang"

    def test_generic_header_does_not_use_port_hint(self):
        from tool_eval_bench.cli.server import (
            detect_backend_from_response as _detect_backend_from_response,
        )

        resp = MagicMock()
        resp.headers = {"server": "uvicorn"}  # Generic
        backend, label = _detect_backend_from_response(resp, 4000)
        assert backend == "unknown"
        assert label == "inference server"

    def test_unknown_port_returns_generic(self):
        from tool_eval_bench.cli.server import (
            detect_backend_from_response as _detect_backend_from_response,
        )

        resp = MagicMock()
        resp.headers = {}
        backend, label = _detect_backend_from_response(resp, 9999)
        assert backend == "unknown"
        assert label == "inference server"


# ---------------------------------------------------------------------------
# Async re-export in __init__.py
# ---------------------------------------------------------------------------


class TestAsyncReExport:
    """Verify that the top-level run_benchmark is properly async."""

    def test_reexport_is_coroutine_function(self):
        import inspect

        from tool_eval_bench import run_benchmark

        assert inspect.iscoroutinefunction(run_benchmark)


# ---------------------------------------------------------------------------
# Error constants (domain/errors.py)
# ---------------------------------------------------------------------------


class TestErrorConstants:
    """Verify the structured error taxonomy is consistent."""

    def test_codes_are_normalized_and_unique(self):
        from tool_eval_bench.domain import errors

        codes = [
            errors.CONNECTION_FAILED,
            errors.HTTP_ERROR,
            errors.DETECTION_FAILED,
            errors.INVALID_RESPONSE,
            errors.NO_MODELS,
            errors.NO_SERVER,
        ]
        for code in codes:
            assert isinstance(code, str)
            assert code == code.lower()
            assert " " not in code
        assert len(codes) == len(set(codes))


# ---------------------------------------------------------------------------
# --dry-run output
# ---------------------------------------------------------------------------


class TestDryRun:
    """Verify --dry-run produces correct scenario lists."""

    def test_dry_run_json_contains_all_scenarios(self):
        """Dry-run JSON output should list all 69 core scenarios."""
        import argparse

        from tool_eval_bench.cli.bench import _resolve_scenarios

        args = argparse.Namespace(
            short=False,
            hardmode=False,
            scenarios=None,
            categories=None,
        )
        scenarios = _resolve_scenarios(args)
        assert len(scenarios) == 69

    def test_dry_run_short_subset(self):
        """Dry-run with --short should list only 15 scenarios."""
        import argparse

        from tool_eval_bench.cli.bench import _resolve_scenarios

        args = argparse.Namespace(
            short=True,
            hardmode=False,
            scenarios=None,
            categories=None,
        )
        scenarios = _resolve_scenarios(args)
        assert len(scenarios) == 15

    def test_dry_run_category_filter(self):
        """Dry-run with --categories should filter correctly."""
        import argparse

        from tool_eval_bench.cli.bench import _resolve_scenarios

        args = argparse.Namespace(
            short=False,
            hardmode=False,
            scenarios=None,
            categories=["A"],
        )
        scenarios = _resolve_scenarios(args)
        assert all(s.category.value == "A" for s in scenarios)
        assert len(scenarios) >= 1


class TestReferenceDateValidation:
    def test_invalid_date_raises(self) -> None:
        """Invalid --reference-date should raise ValueError, not silently fall through."""
        import asyncio

        from tool_eval_bench.application.service import BenchmarkService

        service = BenchmarkService(
            repo=MagicMock(),
            reporter=MagicMock(),
        )

        with pytest.raises(ValueError, match="Invalid --reference-date"):
            asyncio.run(
                service.run_benchmark(
                    model="test",
                    backend="vllm",
                    base_url="http://localhost:9999",
                    reference_date="not-a-date",
                )
            )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("bad_kwargs", "match"),
    [
        ({"model": ""}, "model must be a non-empty string"),
        ({"model": "   "}, "model must be a non-empty string"),
        ({"base_url": ""}, "base_url must be a non-empty string"),
        ({"base_url": "ftp://localhost:8000"}, "base_url must use http or https"),
        ({"base_url": "http:///v1"}, "base_url is missing a host"),
        ({"backend": "not-a-real-backend"}, "Unsupported backend"),
        ({"concurrency": 0}, "concurrency must be at least 1"),
        ({"concurrency": -2}, "concurrency must be at least 1"),
        ({"concurrency": False}, "concurrency must be an integer"),
        ({"max_turns": True}, "max_turns must be an integer"),
        ({"seed": True}, "seed must be an integer"),
        ({"variant_seed": False}, "variant_seed must be an integer"),
        ({"timeout_seconds": 0}, "timeout_seconds must be a finite number greater than 0"),
        ({"timeout_seconds": -5.0}, "timeout_seconds must be a finite number greater than 0"),
        (
            {"timeout_seconds": float("nan")},
            "timeout_seconds must be a finite number greater than 0",
        ),
        (
            {"timeout_seconds": float("inf")},
            "timeout_seconds must be a finite number greater than 0",
        ),
        ({"temperature": -0.1}, "temperature must be a finite number >= 0"),
        ({"temperature": float("nan")}, "temperature must be a finite number >= 0"),
        ({"temperature": float("inf")}, "temperature must be a finite number >= 0"),
        ({"reference_date": "2026-3-5"}, "Invalid --reference-date"),
        ({"decision_judge_model": "judge-m"}, "requires a base URL"),
        ({"decision_judge": "recommended"}, "requires a base URL"),
        (
            {"decision_judge_base_url": "http://localhost:8090", "decision_judge_model": " "},
            "requires a model",
        ),
        ({"locale": "klingon"}, "Unsupported locale"),
        (
            {"scenarios": "TC-01", "locale": "de"},
            "None of the selected scenarios support --locale de",
        ),
    ],
)
async def test_api_run_benchmark_rejects_invalid_inputs_before_any_probe(
    monkeypatch: pytest.MonkeyPatch, bad_kwargs: dict[str, object], match: str
) -> None:
    import httpx

    from tool_eval_bench.evals.scenarios import ALL_SCENARIOS
    from tool_eval_bench.utils import metadata

    real_client = httpx.AsyncClient
    tc01 = next(s for s in ALL_SCENARIOS if s.id == "TC-01")
    tc02 = next(s for s in ALL_SCENARIOS if s.id == "TC-02")
    seen_requests: list[httpx.Request] = []

    def fail_on_network(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        return httpx.Response(500)

    monkeypatch.setattr(
        metadata.httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(fail_on_network), **kw),
    )

    overrides = dict(bad_kwargs)
    if overrides.get("scenarios") == "TC-01":
        overrides["scenarios"] = [tc01]

    defaults: dict[str, object] = {
        "model": "qwen",
        "base_url": "http://localhost:8000",
        "scenarios": [tc02],
        "persist": False,
    }
    with pytest.raises(ValueError, match=match):
        await run_benchmark(**{**defaults, **overrides})  # type: ignore[arg-type]

    assert seen_requests == []
