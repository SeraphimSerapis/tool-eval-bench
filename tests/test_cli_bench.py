"""Unit tests for cli/bench.py helper functions.

These tests cover pure/helper functions in the CLI module that do not require
a live inference server. They focus on argument parsing, scenario resolution,
backend detection, and small output helpers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tool_eval_bench.domain.models import DEFAULT_REQUEST_TIMEOUT_SECONDS
from tool_eval_bench.domain.scenarios import Category

# ---------------------------------------------------------------------------
# _resolve_scenarios
# ---------------------------------------------------------------------------


def _resolve_args(**kwargs) -> argparse.Namespace:
    defaults = {
        "short": False,
        "scenarios": None,
        "categories": None,
        "hardmode": False,
        "hardmode_only": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class TestResolveScenarios:
    def test_default_returns_all_scenarios(self) -> None:
        from tool_eval_bench.cli.bench import _resolve_scenarios
        from tool_eval_bench.evals.scenarios import ALL_SCENARIOS

        result = _resolve_scenarios(_resolve_args())
        assert len(result) == len(ALL_SCENARIOS)
        assert all(s.category != Category.P for s in result)

    def test_short_returns_core_scenarios(self) -> None:
        from tool_eval_bench.cli.bench import _resolve_scenarios
        from tool_eval_bench.evals.scenarios import SCENARIOS

        result = _resolve_scenarios(_resolve_args(short=True))
        assert len(result) == len(SCENARIOS)

    def test_hardmode_adds_hardmode_to_all(self) -> None:
        from tool_eval_bench.cli.bench import _resolve_scenarios
        from tool_eval_bench.evals.scenarios import ALL_SCENARIOS, HARDMODE_SCENARIOS

        result = _resolve_scenarios(_resolve_args(hardmode=True))
        assert len(result) == len(ALL_SCENARIOS) + len(HARDMODE_SCENARIOS)
        assert any(s.category == Category.P for s in result)

    def test_hardmode_adds_hardmode_to_short(self) -> None:
        from tool_eval_bench.cli.bench import _resolve_scenarios
        from tool_eval_bench.evals.scenarios import HARDMODE_SCENARIOS, SCENARIOS

        result = _resolve_scenarios(_resolve_args(short=True, hardmode=True))
        assert len(result) == len(SCENARIOS) + len(HARDMODE_SCENARIOS)

    def test_categories_filter(self) -> None:
        from tool_eval_bench.cli.bench import _resolve_scenarios

        result = _resolve_scenarios(_resolve_args(categories=["A", "K"]))
        assert all(s.category in {Category.A, Category.K} for s in result)
        assert len(result) > 0

    def test_categories_lowercase(self) -> None:
        from tool_eval_bench.cli.bench import _resolve_scenarios

        result = _resolve_scenarios(_resolve_args(categories=["a"]))
        assert all(s.category == Category.A for s in result)

    def test_scenarios_override_all(self) -> None:
        from tool_eval_bench.cli.bench import _resolve_scenarios

        result = _resolve_scenarios(_resolve_args(scenarios=["TC-01", "TC-02"], categories=["K"]))
        assert [s.id for s in result] == ["TC-01", "TC-02"]

    def test_explicit_hardmode_ids_opt_in_without_hardmode_flag(self) -> None:
        from tool_eval_bench.cli.bench import _resolve_scenarios

        result = _resolve_scenarios(_resolve_args(scenarios=["TC-85", "TC-88"]))

        assert [s.id for s in result] == ["TC-85", "TC-88"]
        assert all(s.category == Category.P for s in result)

    def test_explicit_ids_override_short_pool(self) -> None:
        from tool_eval_bench.cli.bench import _resolve_scenarios

        result = _resolve_scenarios(_resolve_args(short=True, scenarios=["TC-70"]))

        assert [s.id for s in result] == ["TC-70"]

    def test_unknown_explicit_id_fails_at_resolution(self) -> None:
        from tool_eval_bench.cli.bench import _resolve_scenarios

        with pytest.raises(ValueError, match=r"Unknown scenarios: NOPE-99"):
            _resolve_scenarios(_resolve_args(scenarios=["TC-01", "NOPE-99"]))

    def test_hardmode_only_stays_restrictive_for_explicit_ids(self) -> None:
        from tool_eval_bench.cli.bench import _resolve_scenarios

        with pytest.raises(ValueError, match=r"Unknown scenarios: TC-01"):
            _resolve_scenarios(_resolve_args(hardmode_only=True, scenarios=["TC-01"]))

    def test_hardmode_only_with_categories_filters_within_hardmode(self) -> None:
        from tool_eval_bench.cli.bench import _resolve_scenarios

        result = _resolve_scenarios(_resolve_args(hardmode_only=True, categories=["P"]))
        assert all(s.category == Category.P for s in result)
        assert len(result) > 0

    def test_hardmode_only_with_non_matching_category_returns_empty(self) -> None:
        from tool_eval_bench.cli.bench import _resolve_scenarios

        result = _resolve_scenarios(_resolve_args(hardmode_only=True, categories=["A"]))
        assert len(result) == 0


# ---------------------------------------------------------------------------
# _resolve_all_scenarios_for_ids
# ---------------------------------------------------------------------------


class TestResolveAllScenariosForIds:
    def test_resolves_known_ids(self) -> None:
        from tool_eval_bench.cli.bench import _resolve_all_scenarios_for_ids

        result = _resolve_all_scenarios_for_ids(["TC-01", "TC-02"])
        assert [s.id for s in result] == ["TC-01", "TC-02"]

    def test_ignores_unknown_ids(self) -> None:
        from tool_eval_bench.cli.bench import _resolve_all_scenarios_for_ids

        result = _resolve_all_scenarios_for_ids(["TC-01", "UNKNOWN-99"])
        assert [s.id for s in result] == ["TC-01"]

    def test_resolves_hardmode_ids(self) -> None:
        from tool_eval_bench.cli.bench import _resolve_all_scenarios_for_ids

        result = _resolve_all_scenarios_for_ids(["TC-70", "TC-84"])
        assert [s.id for s in result] == ["TC-70", "TC-84"]
        assert all(s.category == Category.P for s in result)

    def test_empty_list_returns_empty(self) -> None:
        from tool_eval_bench.cli.bench import _resolve_all_scenarios_for_ids

        result = _resolve_all_scenarios_for_ids([])
        assert result == []


class TestScenarioSelectorDispatch:
    @pytest.mark.parametrize(
        "argv",
        [
            ["--scenarios", "TC-85"],
            ["run", "--scenarios", "TC-85"],
        ],
    )
    def test_legacy_and_subcommand_parsers_preserve_hardmode_id(self, argv: list[str]) -> None:
        from tool_eval_bench.cli.bench import _make_parser
        from tool_eval_bench.cli.parser import parse_cli_args

        _, args = parse_cli_args(_make_parser, argv)

        assert args.scenarios == ["TC-85"]

    def test_run_rejects_unknown_id_before_model_discovery(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sys

        from tool_eval_bench.cli import dispatch

        monkeypatch.setattr(dispatch, "_load_dotenv", lambda: None)
        monkeypatch.setattr(
            dispatch,
            "_detect_model",
            lambda *args, **kwargs: pytest.fail("model discovery should not run"),
        )
        monkeypatch.setattr(
            dispatch,
            "_discover_server",
            lambda *args, **kwargs: pytest.fail("server discovery should not run"),
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "tool-eval-bench",
                "--scenarios",
                "NOPE-99",
            ],
        )

        with pytest.raises(SystemExit) as exc_info:
            dispatch.main()

        assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# _detect_backend_from_response
# ---------------------------------------------------------------------------


class TestDetectBackendFromResponse:
    def test_detects_vllm_from_server_header(self) -> None:
        from tool_eval_bench.cli.server import (
            detect_backend_from_response as _detect_backend_from_response,
        )

        resp = MagicMock()
        resp.headers = {"server": "vllm"}
        backend, label = _detect_backend_from_response(resp, 8000)
        assert backend == "vllm"
        assert label == "vLLM"

    def test_detects_sglang_from_server_header(self) -> None:
        from tool_eval_bench.cli.server import (
            detect_backend_from_response as _detect_backend_from_response,
        )

        resp = MagicMock()
        resp.headers = {"server": "sglang"}
        backend, label = _detect_backend_from_response(resp, 8000)
        assert backend == "vllm"
        assert label == "SGLang"

    def test_detects_llamacpp_from_server_header(self) -> None:
        from tool_eval_bench.cli.server import (
            detect_backend_from_response as _detect_backend_from_response,
        )

        resp = MagicMock()
        resp.headers = {"server": "llama.cpp"}
        backend, label = _detect_backend_from_response(resp, 8080)
        assert backend == "llamacpp"
        assert label == "llama.cpp"

    def test_falls_back_to_port_hint(self) -> None:
        from tool_eval_bench.cli.server import (
            detect_backend_from_response as _detect_backend_from_response,
        )

        resp = MagicMock()
        resp.headers = {}
        backend, label = _detect_backend_from_response(resp, 4000)
        assert backend == "litellm"
        assert label == "LiteLLM"

    def test_falls_back_to_generic_for_unknown_port(self) -> None:
        from tool_eval_bench.cli.server import (
            detect_backend_from_response as _detect_backend_from_response,
        )

        resp = MagicMock()
        resp.headers = {}
        backend, label = _detect_backend_from_response(resp, 12345)
        assert backend == "vllm"
        assert label == "inference server"


# ---------------------------------------------------------------------------
# _parse_int_list
# ---------------------------------------------------------------------------


class TestParseIntList:
    def test_comma_separated(self) -> None:
        from tool_eval_bench.cli.bench import _parse_int_list

        assert _parse_int_list("1,2,3") == [1, 2, 3]

    def test_space_separated(self) -> None:
        from tool_eval_bench.cli.bench import _parse_int_list

        assert _parse_int_list("10 20 30") == [10, 20, 30]

    def test_mixed_separators(self) -> None:
        from tool_eval_bench.cli.bench import _parse_int_list

        assert _parse_int_list("1, 2 3") == [1, 2, 3]

    def test_empty_returns_empty(self) -> None:
        from tool_eval_bench.cli.bench import _parse_int_list

        assert _parse_int_list("") == []

    def test_extra_whitespace_ignored(self) -> None:
        from tool_eval_bench.cli.bench import _parse_int_list

        assert _parse_int_list("  1 ,  2  ") == [1, 2]


# ---------------------------------------------------------------------------
# _parse_sweep_range
# ---------------------------------------------------------------------------


class TestParseSweepRange:
    def test_valid_range(self) -> None:
        from tool_eval_bench.cli.bench import _parse_sweep_range

        assert _parse_sweep_range("0.5-1.0") == (0.5, 1.0)

    def test_clamps_above_one(self) -> None:
        from tool_eval_bench.cli.bench import _parse_sweep_range

        assert _parse_sweep_range("0.5-1.5") == (0.5, 1.0)

    def test_rejects_single_value(self) -> None:
        from tool_eval_bench.cli.bench import _parse_sweep_range

        with pytest.raises(ValueError):
            _parse_sweep_range("0.5")

    def test_rejects_non_numeric(self) -> None:
        from tool_eval_bench.cli.bench import _parse_sweep_range

        with pytest.raises(ValueError):
            _parse_sweep_range("abc-def")

    def test_rejects_start_gte_end(self) -> None:
        from tool_eval_bench.cli.bench import _parse_sweep_range

        with pytest.raises(ValueError):
            _parse_sweep_range("0.8-0.8")

        with pytest.raises(ValueError):
            _parse_sweep_range("0.9-0.5")


# ---------------------------------------------------------------------------
# _redact_url
# ---------------------------------------------------------------------------


class TestRedactUrl:
    def test_redacts_ipv4_host(self) -> None:
        from tool_eval_bench.cli.bench import _redact_url

        assert _redact_url("http://192.168.1.5:8080") == "http://***:8080"

    def test_redacts_hostname(self) -> None:
        from tool_eval_bench.cli.bench import _redact_url

        assert _redact_url("http://my-server.local:8000") == "http://***:8000"


# ---------------------------------------------------------------------------
# _with_config_fingerprint
# ---------------------------------------------------------------------------


class TestWithConfigFingerprint:
    def test_adds_fingerprint(self) -> None:
        from tool_eval_bench.cli.bench import _with_config_fingerprint

        config = {"model": "test", "backend": "vllm"}
        result = _with_config_fingerprint(config)
        assert "config_fingerprint" in result
        assert result["model"] == "test"
        assert result["backend"] == "vllm"

    def test_fingerprint_is_deterministic(self) -> None:
        from tool_eval_bench.cli.bench import _with_config_fingerprint

        config = {"model": "test", "backend": "vllm"}
        fp1 = _with_config_fingerprint(config)["config_fingerprint"]
        fp2 = _with_config_fingerprint(config)["config_fingerprint"]
        assert fp1 == fp2


# ---------------------------------------------------------------------------
# _metadata_for_storage
# ---------------------------------------------------------------------------


class TestMetadataForStorage:
    def test_calls_to_dict(self) -> None:
        from tool_eval_bench.cli.bench import _metadata_for_storage

        ctx = MagicMock()
        ctx.to_dict.return_value = {"key": "value"}
        assert _metadata_for_storage(ctx) == {"key": "value"}

    def test_none_returns_empty(self) -> None:
        from tool_eval_bench.cli.bench import _metadata_for_storage

        assert _metadata_for_storage(None) == {}


# ---------------------------------------------------------------------------
# _headless_error
# ---------------------------------------------------------------------------


class TestHeadlessError:
    def test_emits_jsonl_and_exits(self, capsys) -> None:
        from tool_eval_bench.cli.bench import _headless_error

        with pytest.raises(SystemExit) as exc_info:
            _headless_error("NO_SERVER", "server not found", exit_code=3)

        assert exc_info.value.code == 3
        captured = capsys.readouterr()
        event = json.loads(captured.err.strip())
        assert event["event"] == "error"
        assert event["error"] == "NO_SERVER"
        assert event["message"] == "server not found"


# ---------------------------------------------------------------------------
# _make_parser
# ---------------------------------------------------------------------------


class TestMakeParser:
    def test_parser_has_expected_flags(self) -> None:
        from tool_eval_bench.cli.bench import _make_parser

        parser = _make_parser()
        args = parser.parse_args(["--model", "m", "--base-url", "http://localhost:8000"])
        assert args.model == "m"
        assert args.base_url == "http://localhost:8000"

    def test_version_flag(self, capsys: pytest.CaptureFixture[str]) -> None:
        from tool_eval_bench import __version__
        from tool_eval_bench.cli.bench import _make_parser

        parser = _make_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--version"])

        assert exc_info.value.code == 0
        assert f"tool-eval-bench {__version__}" in capsys.readouterr().out

    def test_defaults(self) -> None:
        from tool_eval_bench.cli.bench import _make_parser

        parser = _make_parser()
        args = parser.parse_args([])
        assert args.model is None
        assert args.base_url is None
        assert args.temperature == 0.0
        assert args.timeout == DEFAULT_REQUEST_TIMEOUT_SECONDS
        assert args.max_turns == 8
        assert args.trials == 1
        assert args.parallel == 1

    def test_plugin_only_flags(self) -> None:
        from tool_eval_bench.cli.bench import _make_parser

        parser = _make_parser()
        args = parser.parse_args(["--gsm8k-only", "--mmlu-only"])
        assert args.gsm8k_only is True
        assert args.mmlu_only is True

    def test_hardmode_flags(self) -> None:
        from tool_eval_bench.cli.bench import _make_parser

        parser = _make_parser()
        args = parser.parse_args(["--hardmode", "--hardmode-only"])
        assert args.hardmode is True
        assert args.hardmode_only is True

    def test_backend_kwargs_parsed_as_string(self) -> None:
        from tool_eval_bench.cli.bench import _make_parser

        parser = _make_parser()
        raw = '{"temperature": 0.5}'
        args = parser.parse_args(["--backend-kwargs", raw])
        assert args.backend_kwargs == raw

    def test_compare_report_subcommand(self) -> None:
        from tool_eval_bench.cli.bench import _make_parser

        parser = _make_parser()
        args = parser.parse_args(["compare-report", "a.md", "b.md", "-o", "out.html"])
        assert args.command == "compare-report"
        assert args.report_a == "a.md"
        assert args.report_b == "b.md"
        assert args.output == "out.html"
        assert args.kind == "auto"


# ---------------------------------------------------------------------------
# _emit_json_output
# ---------------------------------------------------------------------------


class TestEmitJsonOutput:
    def test_writes_to_file(self, tmp_path: Path) -> None:
        from tool_eval_bench.cli.bench import _emit_json_output

        out_file = tmp_path / "result.json"
        data = {"final_score": 87, "model": "test"}
        _emit_json_output(data, json_file=str(out_file))

        assert out_file.exists()
        parsed = json.loads(out_file.read_text(encoding="utf-8"))
        assert parsed["final_score"] == 87
        assert parsed["model"] == "test"

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        from tool_eval_bench.cli.bench import _emit_json_output

        out_file = tmp_path / "nested" / "dir" / "result.json"
        _emit_json_output({"final_score": 90}, json_file=str(out_file))
        assert out_file.exists()

    def test_prints_to_stdout(self, capsys) -> None:
        from tool_eval_bench.cli.bench import _emit_json_output

        _emit_json_output({"final_score": 75})
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["final_score"] == 75


# ---------------------------------------------------------------------------
# _persist_plugin_run
# ---------------------------------------------------------------------------


class TestPersistPluginRun:
    def test_persists_run(self) -> None:
        from tool_eval_bench.cli.bench import _persist_plugin_run
        from tool_eval_bench.storage.db import RunRepository

        run_data = {
            "run_id": "test-run-123",
            "model": "test-model",
            "benchmark_type": "gsm8k",
            "final_score": 0.85,
        }

        with patch.object(RunRepository, "__enter__", autospec=True) as mock_enter:
            mock_repo = MagicMock()
            mock_enter.return_value = mock_repo
            mock_enter.return_value.__exit__ = MagicMock(return_value=False)
            # Need to patch __exit__ on the class as well so the context manager works
            with patch.object(RunRepository, "__exit__", autospec=True) as mock_exit:
                mock_exit.return_value = False
                _persist_plugin_run(run_data)
                mock_repo.upsert_scenario_run.assert_called_once_with(run_data)


# ---------------------------------------------------------------------------
# Argument validation / mutual exclusion smoke tests
# ---------------------------------------------------------------------------


class TestArgumentValidation:
    def test_invalid_category_letter_allowed_at_parse_time(self) -> None:
        """Categories are not validated at parse time; resolution handles mismatches."""
        from tool_eval_bench.cli.bench import _make_parser

        parser = _make_parser()
        args = parser.parse_args(["--categories", "Z"])
        assert args.categories == ["Z"]
