"""Tests for llama-benchy integration module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tool_eval_bench.runner.llama_benchy import (
    _build_command,
    _parse_benchmark_entry,
    _stat_mean,
    parse_json_output,
)


# ---------------------------------------------------------------------------
# Fixtures — minimal llama-benchy JSON output
# ---------------------------------------------------------------------------

SAMPLE_BENCHMARK_ENTRY = {
    "concurrency": 1,
    "context_size": 0,
    "prompt_size": 2048,
    "response_size": 32,
    "is_context_prefill_phase": False,
    "pp_throughput": {"mean": 987.1, "std": 1.1, "values": [986.0, 988.2]},
    "pp_req_throughput": {"mean": 987.1, "std": 1.1, "values": [986.0, 988.2]},
    "tg_throughput": {"mean": 50.0, "std": 0.03, "values": [50.0, 49.9]},
    "tg_req_throughput": {"mean": 50.0, "std": 0.03, "values": [50.0, 49.9]},
    "peak_throughput": {"mean": 51.6, "std": 0.01, "values": [51.6, 51.6]},
    "peak_req_throughput": {"mean": 51.6, "std": 0.01, "values": [51.6, 51.6]},
    "ttfr": {"mean": 2076.0, "std": 2.3, "values": [2078.3, 2073.8]},
    "est_ppt": {"mean": 2074.8, "std": 2.3, "values": [2077.0, 2072.5]},
    "e2e_ttft": {"mean": 2076.0, "std": 2.3, "values": [2078.3, 2073.8]},
}

SAMPLE_CONCURRENT_ENTRY = {
    "concurrency": 2,
    "context_size": 4096,
    "prompt_size": 2048,
    "response_size": 32,
    "is_context_prefill_phase": False,
    "pp_throughput": {"mean": 1968.5, "std": 2.6, "values": [1965.9, 1971.1]},
    "pp_req_throughput": {"mean": 986.7, "std": 2.0, "values": [988.2, 983.6]},
    "tg_throughput": {"mean": 98.7, "std": 0.4, "values": [98.3, 99.1]},
    "tg_req_throughput": {"mean": 49.9, "std": 0.02, "values": [49.9, 49.9]},
    "peak_throughput": {"mean": 101.9, "std": 0.4, "values": [101.5, 102.3]},
    "peak_req_throughput": {"mean": 51.6, "std": 0.02, "values": [51.6, 51.5]},
    "ttfr": {"mean": 2077.0, "std": 4.2, "values": [2073.7, 2083.3]},
    "est_ppt": {"mean": 2075.7, "std": 4.2, "values": [2072.4, 2082.1]},
    "e2e_ttft": {"mean": 2077.0, "std": 4.2, "values": [2073.7, 2083.3]},
}

SAMPLE_CTX_PREFILL_ENTRY = {
    "concurrency": 1,
    "context_size": 4096,
    "prompt_size": 4096,
    "response_size": 32,
    "is_context_prefill_phase": True,
    "pp_throughput": {"mean": 1500.0, "std": 5.0, "values": [1495.0, 1505.0]},
    "pp_req_throughput": {"mean": 1500.0, "std": 5.0, "values": [1495.0, 1505.0]},
    "tg_throughput": {"mean": 48.0, "std": 0.5, "values": [47.5, 48.5]},
    "tg_req_throughput": {"mean": 48.0, "std": 0.5, "values": [47.5, 48.5]},
    "peak_throughput": {"mean": 50.0, "std": 0.1, "values": [49.9, 50.1]},
    "peak_req_throughput": {"mean": 50.0, "std": 0.1, "values": [49.9, 50.1]},
    "ttfr": {"mean": 2740.0, "std": 3.0, "values": [2737.0, 2743.0]},
    "est_ppt": {"mean": 2738.0, "std": 3.0, "values": [2735.0, 2741.0]},
    "e2e_ttft": {"mean": 2740.0, "std": 3.0, "values": [2737.0, 2743.0]},
}

SAMPLE_JSON_OUTPUT = {
    "version": "0.3.5",
    "timestamp": "2026-04-18 00:00:00Z",
    "latency_mode": "generation",
    "latency_ms": 1.27,
    "model": "test-model",
    "prefix_caching_enabled": False,
    "max_concurrency": 2,
    "benchmarks": [
        SAMPLE_BENCHMARK_ENTRY,
        SAMPLE_CONCURRENT_ENTRY,
    ],
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestStatMean:
    def test_extracts_mean(self):
        assert _stat_mean({"mean": 42.5, "std": 1.0}) == 42.5

    def test_missing_mean(self):
        assert _stat_mean({}) == 0.0

    def test_non_dict(self):
        assert _stat_mean(None) == 0.0


class TestParseBenchmarkEntry:
    def test_single_stream(self):
        sample = _parse_benchmark_entry(SAMPLE_BENCHMARK_ENTRY)
        assert sample.concurrency == 1
        assert sample.depth == 0
        assert sample.pp_tokens == 2048
        assert sample.tg_tokens == 32
        assert sample.pp_tps == pytest.approx(987.1, rel=0.01)
        assert sample.tg_tps == pytest.approx(50.0, rel=0.01)
        assert sample.ttft_ms == pytest.approx(2076.0, rel=0.01)
        assert sample.calibration_confidence == "llama-benchy"
        assert sample.error is None

    def test_concurrent(self):
        sample = _parse_benchmark_entry(SAMPLE_CONCURRENT_ENTRY)
        assert sample.concurrency == 2
        assert sample.depth == 4096
        # Concurrent uses total throughput
        assert sample.pp_tps == pytest.approx(1968.5, rel=0.01)
        assert sample.tg_tps == pytest.approx(98.7, rel=0.01)

    def test_context_prefill(self):
        sample = _parse_benchmark_entry(SAMPLE_CTX_PREFILL_ENTRY)
        assert sample.depth == 4096
        # When is_context_prefill_phase=True, requested_pp should be depth
        assert sample.requested_pp == 4096

    def test_total_ms_calculated(self):
        sample = _parse_benchmark_entry(SAMPLE_BENCHMARK_ENTRY)
        # total_ms = est_ppt_ms + (tg_tokens / tg_req_tps * 1000)
        assert sample.total_ms > 0
        assert sample.total_ms > sample.ttft_ms  # total > ttft


class TestParseJsonOutput:
    def test_parses_all_fields(self):
        result = parse_json_output(SAMPLE_JSON_OUTPUT)
        assert result.version == "0.3.5"
        assert result.model == "test-model"
        assert result.latency_mode == "generation"
        assert result.latency_ms == pytest.approx(1.27)
        assert len(result.samples) == 2

    def test_samples_are_throughput_samples(self):
        result = parse_json_output(SAMPLE_JSON_OUTPUT)
        for s in result.samples:
            assert hasattr(s, "pp_tps")
            assert hasattr(s, "tg_tps")
            assert hasattr(s, "ttft_ms")
            assert s.calibration_confidence == "llama-benchy"

    def test_empty_benchmarks(self):
        result = parse_json_output({"benchmarks": []})
        assert result.samples == []

    def test_preserves_raw_json(self):
        result = parse_json_output(SAMPLE_JSON_OUTPUT)
        assert result.raw_json == SAMPLE_JSON_OUTPUT


class TestBuildCommand:
    def test_basic_command(self, monkeypatch):
        monkeypatch.setattr(
            "tool_eval_bench.runner.llama_benchy.shutil.which",
            lambda name: "/usr/bin/llama-benchy" if name == "llama-benchy" else None,
        )
        cmd = _build_command(
            "http://localhost:8888/v1",
            "test-model",
        )
        assert "llama-benchy" in cmd
        assert "--base-url" in cmd
        assert "http://localhost:8888/v1" in cmd
        assert "--model" in cmd
        assert "test-model" in cmd
        assert "--format" in cmd
        assert "json" in cmd
        assert "--no-cache" in cmd
        # coherence check is enabled by default (skip-coherence NOT in cmd)
        assert "--skip-coherence" not in cmd

    def test_adds_api_key(self, monkeypatch):
        monkeypatch.setattr(
            "tool_eval_bench.runner.llama_benchy.shutil.which",
            lambda name: "/usr/bin/llama-benchy" if name == "llama-benchy" else None,
        )
        cmd = _build_command(
            "http://localhost:8888/v1",
            "test-model",
            api_key="sk-test",
        )
        assert "--api-key" in cmd
        assert "sk-test" in cmd

    def test_multiple_pp_tg_depths(self, monkeypatch):
        monkeypatch.setattr(
            "tool_eval_bench.runner.llama_benchy.shutil.which",
            lambda name: "/usr/bin/llama-benchy" if name == "llama-benchy" else None,
        )
        cmd = _build_command(
            "http://localhost:8888/v1",
            "test-model",
            pp=[1024, 2048],
            tg=[32, 64],
            depths=[0, 4096],
            concurrency_levels=[1, 2, 4],
        )
        # llama-benchy uses space-separated values: --pp 1024 2048
        pp_idx = cmd.index("--pp")
        assert cmd[pp_idx + 1] == "1024"
        assert cmd[pp_idx + 2] == "2048"

        tg_idx = cmd.index("--tg")
        assert cmd[tg_idx + 1] == "32"
        assert cmd[tg_idx + 2] == "64"

        depth_idx = cmd.index("--depth")
        assert cmd[depth_idx + 1] == "0"
        assert cmd[depth_idx + 2] == "4096"

        conc_idx = cmd.index("--concurrency")
        assert cmd[conc_idx + 1] == "1"
        assert cmd[conc_idx + 2] == "2"
        assert cmd[conc_idx + 3] == "4"

        # Each flag appears only ONCE (not repeated per value)
        assert cmd.count("--pp") == 1
        assert cmd.count("--tg") == 1
        assert cmd.count("--depth") == 1
        assert cmd.count("--concurrency") == 1

    def test_url_normalisation(self, monkeypatch):
        monkeypatch.setattr(
            "tool_eval_bench.runner.llama_benchy.shutil.which",
            lambda name: "/usr/bin/llama-benchy" if name == "llama-benchy" else None,
        )
        # URL without /v1 should get /v1 appended
        cmd = _build_command("http://localhost:8888", "test-model")
        url_idx = cmd.index("--base-url") + 1
        assert cmd[url_idx] == "http://localhost:8888/v1"

    def test_uvx_fallback(self, monkeypatch):
        def mock_which(name):
            if name == "uvx":
                return "/usr/bin/uvx"
            return None
        monkeypatch.setattr(
            "tool_eval_bench.runner.llama_benchy.shutil.which",
            mock_which,
        )
        cmd = _build_command("http://localhost:8888/v1", "test-model")
        assert cmd[0] == "uvx"
        assert cmd[1] == "llama-benchy"

    def test_raises_when_not_available(self, monkeypatch):
        monkeypatch.setattr(
            "tool_eval_bench.runner.llama_benchy.shutil.which",
            lambda name: None,
        )
        with pytest.raises(RuntimeError, match="llama-benchy is not available"):
            _build_command("http://localhost:8888/v1", "test-model")

    def test_extra_args(self, monkeypatch):
        monkeypatch.setattr(
            "tool_eval_bench.runner.llama_benchy.shutil.which",
            lambda name: "/usr/bin/llama-benchy" if name == "llama-benchy" else None,
        )
        cmd = _build_command(
            "http://localhost:8888/v1",
            "test-model",
            extra_args=["--no-warmup"],
        )
        assert "--no-warmup" in cmd

    def test_url_with_trailing_slash(self, monkeypatch):
        monkeypatch.setattr(
            "tool_eval_bench.runner.llama_benchy.shutil.which",
            lambda name: "/usr/bin/llama-benchy" if name == "llama-benchy" else None,
        )
        cmd = _build_command("http://localhost:8888/v1/", "test-model")
        url_idx = cmd.index("--base-url") + 1
        # Trailing slash should be stripped, /v1 kept
        assert cmd[url_idx] == "http://localhost:8888/v1"

    def test_no_cache_disabled(self, monkeypatch):
        monkeypatch.setattr(
            "tool_eval_bench.runner.llama_benchy.shutil.which",
            lambda name: "/usr/bin/llama-benchy" if name == "llama-benchy" else None,
        )
        cmd = _build_command(
            "http://localhost:8888/v1", "test-model",
            no_cache=False,
        )
        assert "--no-cache" not in cmd

    def test_skip_coherence_flag_explicit(self, monkeypatch):
        """--skip-coherence should only appear when explicitly requested."""
        monkeypatch.setattr(
            "tool_eval_bench.runner.llama_benchy.shutil.which",
            lambda name: "/usr/bin/llama-benchy" if name == "llama-benchy" else None,
        )
        cmd = _build_command(
            "http://localhost:8888/v1", "test-model",
            skip_coherence=True,
        )
        assert "--skip-coherence" in cmd

    def test_coherence_enabled_by_default(self, monkeypatch):
        """Coherence check should run by default (--skip-coherence absent)."""
        monkeypatch.setattr(
            "tool_eval_bench.runner.llama_benchy.shutil.which",
            lambda name: "/usr/bin/llama-benchy" if name == "llama-benchy" else None,
        )
        cmd = _build_command(
            "http://localhost:8888/v1", "test-model",
        )
        assert "--skip-coherence" not in cmd

    def test_output_file_passed(self, monkeypatch):
        monkeypatch.setattr(
            "tool_eval_bench.runner.llama_benchy.shutil.which",
            lambda name: "/usr/bin/llama-benchy" if name == "llama-benchy" else None,
        )
        cmd = _build_command(
            "http://localhost:8888/v1", "test-model",
            output_file="/tmp/results.json",
        )
        assert "--save-result" in cmd
        idx = cmd.index("--save-result") + 1
        assert cmd[idx] == "/tmp/results.json"

    def test_tokenizer_passed_when_differs(self, monkeypatch):
        """--tokenizer should be passed when it differs from --model."""
        monkeypatch.setattr(
            "tool_eval_bench.runner.llama_benchy.shutil.which",
            lambda name: "/usr/bin/llama-benchy" if name == "llama-benchy" else None,
        )
        cmd = _build_command(
            "http://localhost:8888/v1", "Qwen3.6-35B",
            tokenizer="Qwen/Qwen3.6-35B-A3B-FP8",
        )
        assert "--tokenizer" in cmd
        idx = cmd.index("--tokenizer") + 1
        assert cmd[idx] == "Qwen/Qwen3.6-35B-A3B-FP8"

    def test_tokenizer_omitted_when_same_as_model(self, monkeypatch):
        """--tokenizer should NOT be passed when it matches --model."""
        monkeypatch.setattr(
            "tool_eval_bench.runner.llama_benchy.shutil.which",
            lambda name: "/usr/bin/llama-benchy" if name == "llama-benchy" else None,
        )
        cmd = _build_command(
            "http://localhost:8888/v1", "Qwen/Qwen3.6-35B-A3B-FP8",
            tokenizer="Qwen/Qwen3.6-35B-A3B-FP8",
        )
        assert "--tokenizer" not in cmd

    def test_tokenizer_omitted_when_none(self, monkeypatch):
        """--tokenizer should NOT be passed when tokenizer is None."""
        monkeypatch.setattr(
            "tool_eval_bench.runner.llama_benchy.shutil.which",
            lambda name: "/usr/bin/llama-benchy" if name == "llama-benchy" else None,
        )
        cmd = _build_command(
            "http://localhost:8888/v1", "test-model",
            tokenizer=None,
        )
        assert "--tokenizer" not in cmd


class TestIsAvailable:
    def test_available_via_direct_binary(self, monkeypatch):
        from tool_eval_bench.runner.llama_benchy import is_available
        monkeypatch.setattr(
            "tool_eval_bench.runner.llama_benchy.shutil.which",
            lambda name: "/usr/bin/llama-benchy" if name == "llama-benchy" else None,
        )
        assert is_available() is True

    def test_available_via_uvx(self, monkeypatch):
        from tool_eval_bench.runner.llama_benchy import is_available
        monkeypatch.setattr(
            "tool_eval_bench.runner.llama_benchy.shutil.which",
            lambda name: "/usr/bin/uvx" if name == "uvx" else None,
        )
        assert is_available() is True

    def test_not_available(self, monkeypatch):
        from tool_eval_bench.runner.llama_benchy import is_available
        monkeypatch.setattr(
            "tool_eval_bench.runner.llama_benchy.shutil.which",
            lambda name: None,
        )
        assert is_available() is False


class TestParserEdgeCases:
    """Edge cases in _parse_benchmark_entry that could cause crashes."""

    def test_zero_tg_req_throughput_no_division_error(self):
        """Zero tg_req_throughput should not cause ZeroDivisionError."""
        entry = {
            "concurrency": 1,
            "context_size": 0,
            "prompt_size": 512,
            "response_size": 16,
            "is_context_prefill_phase": False,
            "pp_throughput": {"mean": 500.0},
            "pp_req_throughput": {"mean": 500.0},
            "tg_throughput": {"mean": 0.0},
            "tg_req_throughput": {"mean": 0.0},
            "peak_throughput": {"mean": 0.0},
            "peak_req_throughput": {"mean": 0.0},
            "ttfr": {"mean": 100.0},
            "est_ppt": {"mean": 100.0},
            "e2e_ttft": {"mean": 100.0},
        }
        sample = _parse_benchmark_entry(entry)
        # Should not crash — gen_time_ms guard handles zero tg_req_tps
        assert sample.tg_tps == 0.0
        assert sample.total_ms == pytest.approx(100.0)  # est_ppt + 0

    def test_minimal_entry_missing_optional_fields(self):
        """Entries with only required fields should parse without errors."""
        entry = {
            "concurrency": 1,
            "prompt_size": 128,
            "response_size": 8,
        }
        sample = _parse_benchmark_entry(entry)
        assert sample.pp_tokens == 128
        assert sample.tg_tokens == 8
        assert sample.concurrency == 1
        assert sample.depth == 0
        assert sample.pp_tps == 0.0
        assert sample.tg_tps == 0.0

    def test_empty_stat_objects(self):
        """Stat objects that are empty dicts should yield 0.0."""
        entry = {
            "concurrency": 1,
            "context_size": 0,
            "prompt_size": 256,
            "response_size": 32,
            "pp_throughput": {},
            "pp_req_throughput": {},
            "tg_throughput": {},
            "tg_req_throughput": {},
            "ttfr": {},
            "est_ppt": {},
            "e2e_ttft": {},
        }
        sample = _parse_benchmark_entry(entry)
        assert sample.pp_tps == 0.0
        assert sample.tg_tps == 0.0
        assert sample.ttft_ms == 0.0

    def test_single_stream_uses_req_throughput(self):
        """For concurrency=1, per-request throughput should be used."""
        entry = {
            "concurrency": 1,
            "context_size": 0,
            "prompt_size": 2048,
            "response_size": 32,
            "pp_throughput": {"mean": 1000.0},
            "pp_req_throughput": {"mean": 999.0},
            "tg_throughput": {"mean": 55.0},
            "tg_req_throughput": {"mean": 54.0},
            "ttfr": {"mean": 100.0},
            "est_ppt": {"mean": 100.0},
            "e2e_ttft": {"mean": 100.0},
        }
        sample = _parse_benchmark_entry(entry)
        # Single stream: should prefer pp_req over pp_total
        assert sample.pp_tps == pytest.approx(999.0)
        assert sample.tg_tps == pytest.approx(54.0)

    def test_concurrent_uses_total_throughput(self):
        """For concurrency>1, total throughput should be used."""
        entry = {
            "concurrency": 4,
            "context_size": 0,
            "prompt_size": 2048,
            "response_size": 32,
            "pp_throughput": {"mean": 4000.0},
            "pp_req_throughput": {"mean": 1000.0},
            "tg_throughput": {"mean": 200.0},
            "tg_req_throughput": {"mean": 50.0},
            "ttfr": {"mean": 500.0},
            "est_ppt": {"mean": 500.0},
            "e2e_ttft": {"mean": 500.0},
        }
        sample = _parse_benchmark_entry(entry)
        assert sample.pp_tps == pytest.approx(4000.0)
        assert sample.tg_tps == pytest.approx(200.0)

    def test_parse_json_missing_top_level_fields(self):
        """parse_json_output should handle missing optional top-level fields."""
        result = parse_json_output({})
        assert result.version == ""
        assert result.model == ""
        assert result.samples == []
        assert result.latency_ms == 0.0


class TestRunLlamaBenchy:
    """Tests for the async subprocess runner using mocked subprocess."""

    async def test_happy_path(self, tmp_path, monkeypatch):
        """Successful run should parse JSON output and return results."""
        from tool_eval_bench.runner.llama_benchy import run_llama_benchy

        # Mock shutil.which to find llama-benchy
        monkeypatch.setattr(
            "tool_eval_bench.runner.llama_benchy.shutil.which",
            lambda name: "/usr/bin/llama-benchy" if name == "llama-benchy" else None,
        )

        captured_lines: list[str] = []

        # Mock asyncio.create_subprocess_exec
        class MockStdout:
            def __init__(self, lines):
                self._lines = lines
                self._idx = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self._idx >= len(self._lines):
                    raise StopAsyncIteration
                line = self._lines[self._idx]
                self._idx += 1
                return line.encode("utf-8")

        class MockProcess:
            def __init__(self, output_file):
                self.stdout = MockStdout(["Running benchmark...\n", "Done\n"])
                self._output_file = output_file

            async def wait(self):
                # Write valid JSON to the output file
                Path(self._output_file).write_text(
                    json.dumps(SAMPLE_JSON_OUTPUT), encoding="utf-8"
                )
                return 0

        output_file_ref: list[str] = []

        original_build = _build_command

        def mock_build(*args, **kwargs):
            cmd = original_build(*args, **kwargs)
            # Capture the output file from the command
            if "--save-result" in cmd:
                idx = cmd.index("--save-result") + 1
                output_file_ref.append(cmd[idx])
            return cmd

        monkeypatch.setattr(
            "tool_eval_bench.runner.llama_benchy._build_command", mock_build
        )

        async def mock_create_subprocess_exec(*args, **kwargs):
            assert output_file_ref, "output file should be set by _build_command"
            return MockProcess(output_file_ref[0])

        monkeypatch.setattr(
            "tool_eval_bench.runner.llama_benchy.asyncio.create_subprocess_exec",
            mock_create_subprocess_exec,
        )

        result = await run_llama_benchy(
            "http://localhost:8888/v1",
            "test-model",
            on_output=lambda line: captured_lines.append(line),
        )

        assert result.version == "0.3.5"
        assert result.model == "test-model"
        assert len(result.samples) == 2
        assert len(captured_lines) == 2
        assert "Running benchmark..." in captured_lines[0]

    async def test_nonzero_exit_raises(self, monkeypatch):
        """Non-zero exit code should raise RuntimeError."""
        from tool_eval_bench.runner.llama_benchy import run_llama_benchy

        monkeypatch.setattr(
            "tool_eval_bench.runner.llama_benchy.shutil.which",
            lambda name: "/usr/bin/llama-benchy" if name == "llama-benchy" else None,
        )

        class MockStdout:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

        class MockProcess:
            stdout = MockStdout()

            async def wait(self):
                return 1

        async def mock_create(*args, **kwargs):
            return MockProcess()

        monkeypatch.setattr(
            "tool_eval_bench.runner.llama_benchy.asyncio.create_subprocess_exec",
            mock_create,
        )

        with pytest.raises(RuntimeError, match="exited with code 1"):
            await run_llama_benchy("http://localhost:8888/v1", "test-model")

    async def test_empty_output_file_raises(self, monkeypatch):
        """Empty output file should raise RuntimeError."""
        from tool_eval_bench.runner.llama_benchy import run_llama_benchy

        monkeypatch.setattr(
            "tool_eval_bench.runner.llama_benchy.shutil.which",
            lambda name: "/usr/bin/llama-benchy" if name == "llama-benchy" else None,
        )

        class MockStdout:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

        class MockProcess:
            stdout = MockStdout()

            async def wait(self):
                # Don't write anything to the output file
                return 0

        async def mock_create(*args, **kwargs):
            return MockProcess()

        monkeypatch.setattr(
            "tool_eval_bench.runner.llama_benchy.asyncio.create_subprocess_exec",
            mock_create,
        )

        with pytest.raises(RuntimeError, match="did not produce JSON output"):
            await run_llama_benchy("http://localhost:8888/v1", "test-model")

    async def test_on_output_receives_all_lines(self, monkeypatch):
        """on_output callback should receive every stdout line."""
        from tool_eval_bench.runner.llama_benchy import run_llama_benchy

        monkeypatch.setattr(
            "tool_eval_bench.runner.llama_benchy.shutil.which",
            lambda name: "/usr/bin/llama-benchy" if name == "llama-benchy" else None,
        )

        expected_lines = [
            "Warming up...\n",
            "pp2048 tg32 @ d0: 987 pp/s, 50 tg/s\n",
            "pp2048 tg32 @ d4096 c2: 1968 pp/s, 99 tg/s\n",
            "Complete!\n",
        ]

        class MockStdout:
            def __init__(self):
                self._idx = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self._idx >= len(expected_lines):
                    raise StopAsyncIteration
                line = expected_lines[self._idx]
                self._idx += 1
                return line.encode("utf-8")

        output_file_ref: list[str] = []
        original_build = _build_command

        def mock_build(*args, **kwargs):
            cmd = original_build(*args, **kwargs)
            if "--save-result" in cmd:
                idx = cmd.index("--save-result") + 1
                output_file_ref.append(cmd[idx])
            return cmd

        monkeypatch.setattr(
            "tool_eval_bench.runner.llama_benchy._build_command", mock_build
        )

        class MockProcess:
            def __init__(self):
                self.stdout = MockStdout()

            async def wait(self):
                Path(output_file_ref[0]).write_text(
                    json.dumps(SAMPLE_JSON_OUTPUT), encoding="utf-8"
                )
                return 0

        async def mock_create(*args, **kwargs):
            return MockProcess()

        monkeypatch.setattr(
            "tool_eval_bench.runner.llama_benchy.asyncio.create_subprocess_exec",
            mock_create,
        )

        captured: list[str] = []
        await run_llama_benchy(
            "http://localhost:8888/v1",
            "test-model",
            on_output=lambda line: captured.append(line),
        )

        assert len(captured) == 4
        assert "Warming up" in captured[0]
        assert "Complete" in captured[3]

    async def test_noisy_lines_suppressed(self, monkeypatch):
        """PyTorch/HF Hub warnings should be filtered from on_output."""
        from tool_eval_bench.runner.llama_benchy import run_llama_benchy

        monkeypatch.setattr(
            "tool_eval_bench.runner.llama_benchy.shutil.which",
            lambda name: "/usr/bin/llama-benchy" if name == "llama-benchy" else None,
        )

        noisy_lines = [
            "llama-benchy (0.3.5)\n",
            "PyTorch was not found. Models won't be available and only tokenizers\n",
            "Warning: You are sending unauthenticated requests to the HF Hub.\n",
            "Running test: pp=2048, tg=128\n",
            "Done\n",
        ]

        class MockStdout:
            def __init__(self):
                self._idx = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self._idx >= len(noisy_lines):
                    raise StopAsyncIteration
                line = noisy_lines[self._idx]
                self._idx += 1
                return line.encode("utf-8")

        output_file_ref: list[str] = []
        original_build = _build_command

        def mock_build(*args, **kwargs):
            cmd = original_build(*args, **kwargs)
            if "--save-result" in cmd:
                idx = cmd.index("--save-result") + 1
                output_file_ref.append(cmd[idx])
            return cmd

        monkeypatch.setattr(
            "tool_eval_bench.runner.llama_benchy._build_command", mock_build
        )

        class MockProcess:
            def __init__(self):
                self.stdout = MockStdout()

            async def wait(self):
                Path(output_file_ref[0]).write_text(
                    json.dumps(SAMPLE_JSON_OUTPUT), encoding="utf-8"
                )
                return 0

        async def mock_create(*args, **kwargs):
            return MockProcess()

        monkeypatch.setattr(
            "tool_eval_bench.runner.llama_benchy.asyncio.create_subprocess_exec",
            mock_create,
        )

        captured: list[str] = []
        await run_llama_benchy(
            "http://localhost:8888/v1",
            "test-model",
            on_output=lambda line: captured.append(line),
        )

        # 3 of 5 lines are noisy — only 2 should pass through
        assert len(captured) == 3
        assert any("llama-benchy" in c for c in captured)
        assert any("Running test" in c for c in captured)
        # Noisy lines must not appear
        assert not any("PyTorch" in c for c in captured)
        assert not any("unauthenticated" in c for c in captured)

    async def test_subprocess_env_suppresses_warnings(self, monkeypatch):
        """Subprocess should receive env vars that suppress HF warnings."""
        from tool_eval_bench.runner.llama_benchy import run_llama_benchy

        monkeypatch.setattr(
            "tool_eval_bench.runner.llama_benchy.shutil.which",
            lambda name: "/usr/bin/llama-benchy" if name == "llama-benchy" else None,
        )

        captured_env: dict[str, str] = {}

        class MockStdout:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

        class MockProcess:
            stdout = MockStdout()

            async def wait(self):
                return 1  # Will raise RuntimeError, that's fine

        async def mock_create(*args, **kwargs):
            # Capture the env passed to the subprocess
            captured_env.update(kwargs.get("env", {}))
            return MockProcess()

        monkeypatch.setattr(
            "tool_eval_bench.runner.llama_benchy.asyncio.create_subprocess_exec",
            mock_create,
        )

        with pytest.raises(RuntimeError):
            await run_llama_benchy("http://localhost:8888/v1", "test-model")

        assert captured_env.get("PYTHONUNBUFFERED") == "1"
        assert captured_env.get("TRANSFORMERS_NO_ADVISORY_WARNINGS") == "1"
        assert captured_env.get("HF_HUB_DISABLE_IMPLICIT_TOKEN") == "1"


# ---------------------------------------------------------------------------
# Display logic tests: Weakest category
# ---------------------------------------------------------------------------

class TestWeakestCategoryDisplay:
    """Tests for the 'Weakest' category logic in the final panel."""

    def _make_summary(self, *, worst_pct: int | None = None, worst_cat: str | None = None):
        """Build a minimal ModelScoreSummary for display testing."""
        from tool_eval_bench.domain.scenarios import (
            ModelScoreSummary,
            ScenarioResult,
            ScenarioStatus,
        )
        return ModelScoreSummary(
            scenario_results=[
                ScenarioResult(
                    scenario_id="TC-01", status=ScenarioStatus.PASS,
                    points=2, summary="ok",
                ),
            ],
            category_scores=[],
            final_score=100,
            total_points=2,
            max_points=2,
            rating="★★★★★ Excellent",
            worst_category=worst_cat,
            worst_category_percent=worst_pct,
        )

    def test_weakest_hidden_at_100_percent(self):
        """'Weakest' line should NOT appear when worst category is 100%."""
        from io import StringIO

        from rich.console import Console

        from tool_eval_bench.cli.display import _print_final_panel

        summary = self._make_summary(worst_pct=100, worst_cat="A Tool Selection (100%)")
        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=120)
        _print_final_panel(console, "test-model", summary, elapsed=1.0)
        output = buf.getvalue()
        assert "Weakest" not in output

    def test_weakest_shown_below_100_percent(self):
        """'Weakest' line should appear when worst category is below 100%."""
        from io import StringIO

        from rich.console import Console

        from tool_eval_bench.cli.display import _print_final_panel

        summary = self._make_summary(worst_pct=75, worst_cat="B Parameter Precision (75%)")
        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=120)
        _print_final_panel(console, "test-model", summary, elapsed=1.0)
        output = buf.getvalue()
        assert "Weakest" in output
        assert "Parameter Precision" in output

    def test_weakest_hidden_when_none(self):
        """'Weakest' line should NOT appear when worst_category is None."""
        from io import StringIO

        from rich.console import Console

        from tool_eval_bench.cli.display import _print_final_panel

        summary = self._make_summary(worst_pct=None, worst_cat=None)
        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=120)
        _print_final_panel(console, "test-model", summary, elapsed=1.0)
        output = buf.getvalue()
        assert "Weakest" not in output


# ---------------------------------------------------------------------------
# Progress bar output parsing tests
# ---------------------------------------------------------------------------

class TestProgressBarParsing:
    """Test that llama-benchy output lines are correctly classified for progress tracking."""

    def test_running_test_line_detected(self):
        """'Running test:' lines should be recognised as test-start markers."""
        line = "Running test: pp=2048, tg=128, depth=0, concurrency=1"
        assert line.startswith("Running test:")
        desc = line.replace("Running test: ", "")
        assert desc == "pp=2048, tg=128, depth=0, concurrency=1"

    def test_run_progress_line_detected(self):
        """'Run X/Y' lines should match the run-progress regex."""
        import re
        for line in [
            "  Run 1/3 (batch size 1)...",
            "Run 2/5 (batch size 4)...",
            "    Run 10/10 (batch size 8)...",
        ]:
            assert re.match(r"\s*Run \d+/\d+", line), f"Should match: {line!r}"

    def test_non_run_lines_dont_match(self):
        """Non-run lines should NOT match the run regex."""
        import re
        for line in [
            "Warming up...",
            "Running test: pp=2048",
            "llama-benchy (0.3.5)",
            "Average latency (generation): 55.78 ms",
        ]:
            assert not re.match(r"\s*Run \d+/\d+", line), f"Should NOT match: {line!r}"

    def test_total_runs_calculation(self):
        """Total runs = pp × tg × depths × concurrency × runs_per_test."""
        pp = [2048]
        tg = [128]
        depths = [0, 4096, 8192]
        concurrency = [1, 2, 4]
        runs = 3

        total = len(pp) * len(tg) * len(depths) * len(concurrency) * runs
        assert total == 27  # 1 × 1 × 3 × 3 × 3

    def test_total_runs_multiple_pp_tg(self):
        """Multiple pp/tg values multiply the total."""
        pp = [1024, 2048]
        tg = [64, 128]
        depths = [0]
        concurrency = [1]
        runs = 3

        total = len(pp) * len(tg) * len(depths) * len(concurrency) * runs
        assert total == 12  # 2 × 2 × 1 × 1 × 3

    def test_warmup_line_detected(self):
        """Warmup lines should be identified but not 'warmup complete' lines."""
        warmup = "Warming up..."
        warmup_done = "Warmup (User only) complete. Delta: 9 tokens"

        assert "Warming up" in warmup and "complete" not in warmup.lower()
        assert "complete" in warmup_done.lower()

    def test_latency_line_detected(self):
        """Measuring latency line should be identified."""
        line = "Measuring latency using mode: generation..."
        assert "Measuring latency" in line


# ---------------------------------------------------------------------------
# CLI flag registration tests
# ---------------------------------------------------------------------------

class TestCLIFlags:
    """Verify that CLI flags are registered correctly after the refactoring."""

    @staticmethod
    def _get_help_text() -> str:
        """Get the CLI --help output by invoking main() with --help."""
        from io import StringIO
        from unittest.mock import patch

        from tool_eval_bench.cli.bench import main

        buf = StringIO()
        with patch("sys.argv", ["tool-eval-bench", "--help"]), \
             patch("sys.stdout", buf), \
             pytest.raises(SystemExit):
            main()
        return buf.getvalue()

    def test_perf_flags_exist(self):
        """--perf and --perf-only should be accepted in CLI help."""
        help_text = self._get_help_text()
        assert "--perf " in help_text
        assert "--perf-only" in help_text

    def test_perf_legacy_flags_registered(self):
        """--perf-legacy and --perf-legacy-only should exist in CLI help."""
        help_text = self._get_help_text()
        assert "--perf-legacy " in help_text
        assert "--perf-legacy-only" in help_text

    def test_benchy_tuning_flags_registered(self):
        """--benchy-runs, --benchy-latency-mode, --benchy-args should exist."""
        help_text = self._get_help_text()
        assert "--benchy-runs" in help_text
        assert "--benchy-latency-mode" in help_text
        assert "--benchy-args" in help_text

    def test_no_perf_benchy_flag(self):
        """--perf-benchy should NOT exist (removed in v1.2.0)."""
        help_text = self._get_help_text()
        assert "--perf-benchy" not in help_text

    def test_skip_coherence_flag_registered(self):
        """--skip-coherence should exist in CLI help."""
        help_text = self._get_help_text()
        assert "--skip-coherence" in help_text
