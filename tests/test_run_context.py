"""Tests for RunContext, metadata collection, and report rendering (issue #6)."""

from __future__ import annotations

import json


# ---------------------------------------------------------------------------
# RunContext dataclass
# ---------------------------------------------------------------------------

class TestRunContext:
    """Tests for the RunContext dataclass."""

    def test_creation_minimal(self):
        from tool_eval_bench.domain.models import RunContext

        ctx = RunContext(
            tool_version="1.4.0",
            git_sha="abc123",
            hostname="test-host",
            platform_info="Linux-6.8",
            python_version="3.12.4",
            model="qwen3.6-27b",
            backend="vllm",
            base_url="http://***:8000",
        )
        assert ctx.tool_version == "1.4.0"
        assert ctx.temperature == 0.0  # default
        assert ctx.max_turns == 8  # default
        assert ctx.seed is None  # default
        assert ctx.engine_name is None  # tier 3 default

    def test_to_dict_strips_none(self):
        from tool_eval_bench.domain.models import RunContext

        ctx = RunContext(
            tool_version="1.4.0",
            git_sha=None,
            hostname="h",
            platform_info="p",
            python_version="3.12",
            model="m",
            backend="vllm",
            base_url="http://localhost",
        )
        d = ctx.to_dict()
        assert "git_sha" not in d
        assert "engine_name" not in d
        assert "server_model_id" not in d
        assert d["tool_version"] == "1.4.0"
        assert d["model"] == "m"

    def test_to_dict_includes_set_fields(self):
        from tool_eval_bench.domain.models import RunContext

        ctx = RunContext(
            tool_version="1.4.0",
            git_sha="abc123",
            hostname="h",
            platform_info="p",
            python_version="3.12",
            model="m",
            backend="vllm",
            base_url="http://localhost",
            engine_name="vLLM",
            engine_version="0.8.5",
            max_model_len=65536,
            quantization="AWQ",
        )
        d = ctx.to_dict()
        assert d["engine_name"] == "vLLM"
        assert d["engine_version"] == "0.8.5"
        assert d["max_model_len"] == 65536
        assert d["quantization"] == "AWQ"

    def test_to_dict_is_json_serializable(self):
        from tool_eval_bench.domain.models import RunContext

        ctx = RunContext(
            tool_version="1.4.0",
            git_sha="abc",
            hostname="h",
            platform_info="p",
            python_version="3.12",
            model="m",
            backend="vllm",
            base_url="http://localhost",
            extra_params={"temperature": 0.6, "top_p": 0.9},
        )
        # Should not raise
        serialized = json.dumps(ctx.to_dict())
        parsed = json.loads(serialized)
        assert parsed["extra_params"] == {"temperature": 0.6, "top_p": 0.9}

    def test_defaults_match_cli_defaults(self):
        """RunContext defaults should match CLI argparse defaults."""
        from tool_eval_bench.domain.models import RunContext

        ctx = RunContext(
            tool_version="1.0", git_sha=None, hostname="h",
            platform_info="p", python_version="3.12",
            model="m", backend="vllm", base_url="http://localhost",
        )
        assert ctx.temperature == 0.0
        assert ctx.max_turns == 8
        assert ctx.timeout_seconds == 60.0
        assert ctx.trials == 1
        assert ctx.parallel == 1
        assert ctx.error_rate == 0.0
        assert ctx.thinking_enabled is True


# ---------------------------------------------------------------------------
# URL redaction
# ---------------------------------------------------------------------------

class TestRedactUrl:
    """Tests for the shared redact_url utility."""

    def test_redacts_ip_with_port(self):
        from tool_eval_bench.utils.urls import redact_url

        assert redact_url("http://192.168.10.5:8080") == "http://***:8080"

    def test_redacts_hostname(self):
        from tool_eval_bench.utils.urls import redact_url

        assert redact_url("http://inference-01.local:8000/v1") == "http://***:8000/v1"

    def test_preserves_empty_string(self):
        from tool_eval_bench.utils.urls import redact_url

        assert redact_url("") == ""

    def test_preserves_no_host(self):
        from tool_eval_bench.utils.urls import redact_url

        assert redact_url("not-a-url") == "not-a-url"

    def test_cli_delegates_to_shared(self):
        """CLI _redact_url should produce the same result as the shared utility."""
        from tool_eval_bench.cli.bench import _redact_url
        from tool_eval_bench.utils.urls import redact_url

        url = "http://10.0.0.1:8000"
        assert _redact_url(url) == redact_url(url)


# ---------------------------------------------------------------------------
# Quantization guessing
# ---------------------------------------------------------------------------

class TestGuessQuantization:
    """Tests for the quantization inference heuristic."""

    def test_awq(self):
        from tool_eval_bench.utils.metadata import _guess_quantization

        assert _guess_quantization("Qwen/Qwen3-30B-A3B-AWQ") == "AWQ"

    def test_gptq(self):
        from tool_eval_bench.utils.metadata import _guess_quantization

        assert _guess_quantization("TheBloke/Llama-2-70B-GPTQ") == "GPTQ"

    def test_fp16(self):
        from tool_eval_bench.utils.metadata import _guess_quantization

        assert _guess_quantization("some-model-fp16") == "FP16"

    def test_autoround(self):
        from tool_eval_bench.utils.metadata import _guess_quantization

        assert _guess_quantization("Intel/gemma-4-31B-it-int4-AutoRound") == "INT4-AutoRound"

    def test_gguf_quant_level(self):
        from tool_eval_bench.utils.metadata import _guess_quantization

        result = _guess_quantization("some-model-Q4_K_M.gguf")
        assert result is not None
        assert "Q4_K" in result

    def test_none_for_plain_model(self):
        from tool_eval_bench.utils.metadata import _guess_quantization

        assert _guess_quantization("Qwen/Qwen3-30B-A3B") is None

    def test_none_for_none_input(self):
        from tool_eval_bench.utils.metadata import _guess_quantization

        assert _guess_quantization(None) is None


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

class TestRenderRunContext:
    """Tests for the Markdown report rendering of RunContext."""

    def _make_context(self, **overrides):
        from tool_eval_bench.domain.models import RunContext

        defaults = dict(
            tool_version="1.4.0",
            git_sha="abc123",
            hostname="inference-01",
            platform_info="Linux-6.8.0",
            python_version="3.12.4",
            model="qwen3.6-27b",
            backend="vllm",
            base_url="http://***:8000",
            temperature=0.0,
            max_turns=8,
            timeout_seconds=60.0,
            scenario_selector="all (69)",
        )
        defaults.update(overrides)
        return RunContext(**defaults)

    def test_run_context_section_present(self):
        from tool_eval_bench.storage.reports import _render_run_context

        ctx = self._make_context()
        lines = _render_run_context(ctx)
        md = "\n".join(lines)
        assert "## Run Context" in md

    def test_parameters_in_table(self):
        from tool_eval_bench.storage.reports import _render_run_context

        ctx = self._make_context(temperature=0.6, seed=42)
        lines = _render_run_context(ctx)
        md = "\n".join(lines)
        assert "| Temperature | 0.6 |" in md
        assert "| Seed | 42 |" in md

    def test_seed_shows_dash_when_none(self):
        from tool_eval_bench.storage.reports import _render_run_context

        ctx = self._make_context(seed=None)
        lines = _render_run_context(ctx)
        md = "\n".join(lines)
        assert "| Seed | — |" in md

    def test_engine_section_with_info(self):
        from tool_eval_bench.storage.reports import _render_run_context

        ctx = self._make_context(
            engine_name="vLLM", engine_version="0.8.5",
            max_model_len=65536, quantization="AWQ",
        )
        lines = _render_run_context(ctx)
        md = "\n".join(lines)
        assert "## Inference Engine" in md
        assert "vLLM 0.8.5" in md
        assert "65,536" in md
        assert "AWQ" in md

    def test_environment_section_without_engine(self):
        from tool_eval_bench.storage.reports import _render_run_context

        ctx = self._make_context()  # no engine info
        lines = _render_run_context(ctx)
        md = "\n".join(lines)
        assert "## Environment" in md
        assert "## Inference Engine" not in md
        assert "inference-01" in md

    def test_model_root_shown_when_different(self):
        from tool_eval_bench.storage.reports import _render_run_context

        ctx = self._make_context(
            server_model_root="Qwen/Qwen3.6-27B-Instruct",
        )
        lines = _render_run_context(ctx)
        md = "\n".join(lines)
        assert "Model (Root)" in md
        assert "Qwen/Qwen3.6-27B-Instruct" in md

    def test_model_root_hidden_when_same(self):
        from tool_eval_bench.storage.reports import _render_run_context

        ctx = self._make_context(
            server_model_root="qwen3.6-27b",  # same as model
        )
        lines = _render_run_context(ctx)
        md = "\n".join(lines)
        assert "Model (Root)" not in md

    def test_context_pressure_shown(self):
        from tool_eval_bench.storage.reports import _render_run_context

        ctx = self._make_context(context_pressure=0.8)
        lines = _render_run_context(ctx)
        md = "\n".join(lines)
        assert "Context Pressure" in md
        assert "80%" in md

    def test_thinking_disabled(self):
        from tool_eval_bench.storage.reports import _render_run_context

        ctx = self._make_context(thinking_enabled=False)
        lines = _render_run_context(ctx)
        md = "\n".join(lines)
        assert "| Thinking | disabled |" in md

    def test_extra_params_rendered(self):
        from tool_eval_bench.storage.reports import _render_run_context

        ctx = self._make_context(extra_params={"top_p": 0.9})
        lines = _render_run_context(ctx)
        md = "\n".join(lines)
        assert "Extra Params" in md
        assert "top_p" in md


# ---------------------------------------------------------------------------
# History context extraction
# ---------------------------------------------------------------------------

class TestHistoryContextExtraction:
    """Tests for the history context summary helpers."""

    def test_extract_context_summary_full(self):
        from tool_eval_bench.cli.history import _extract_context_summary

        run = {
            "metadata": {
                "tool_version": "1.4.0",
                "backend": "vllm",
                "engine_name": "vLLM",
                "engine_version": "0.8.5",
                "quantization": "AWQ",
                "temperature": 0.6,
            },
            "config": {},
        }
        result = _extract_context_summary(run)
        assert "v1.4.0" in result
        assert "vllm" in result
        assert "vLLM 0.8.5" in result
        assert "AWQ" in result
        assert "t=0.6" in result

    def test_extract_context_summary_empty_metadata(self):
        from tool_eval_bench.cli.history import _extract_context_summary

        run = {"metadata": {}, "config": {"backend": "vllm"}}
        result = _extract_context_summary(run)
        assert "vllm" in result

    def test_extract_context_summary_no_metadata(self):
        """Old runs without metadata should return empty string."""
        from tool_eval_bench.cli.history import _extract_context_summary

        run = {"config": {}}
        result = _extract_context_summary(run)
        assert result == ""

    def test_extract_context_summary_default_temp_hidden(self):
        """Temperature 0.0 (default) should not be shown."""
        from tool_eval_bench.cli.history import _extract_context_summary

        run = {"metadata": {"temperature": 0.0, "tool_version": "1.4.0"}, "config": {}}
        result = _extract_context_summary(run)
        assert "t=" not in result

    def test_extract_context_panel_full(self):
        from tool_eval_bench.cli.history import _extract_context_panel

        run = {
            "metadata": {
                "tool_version": "1.4.0",
                "git_sha": "abc123",
                "engine_name": "vLLM",
                "engine_version": "0.8.5",
                "max_model_len": 65536,
                "quantization": "AWQ",
                "server_model_root": "Qwen/Qwen3.6-27B-Instruct",
                "model": "qwen3.6-27b",
                "temperature": 0.0,
                "hostname": "inference-01",
            },
            "config": {"model": "qwen3.6-27b"},
        }
        lines = _extract_context_panel(run)
        text = "\n".join(lines)
        assert "v1.4.0" in text
        assert "vLLM 0.8.5" in text
        assert "65,536" in text
        assert "AWQ" in text
        assert "Qwen/Qwen3.6-27B-Instruct" in text
        assert "inference-01" in text

    def test_extract_context_panel_old_run(self):
        """Old runs should return empty list gracefully."""
        from tool_eval_bench.cli.history import _extract_context_panel

        run = {"metadata": {}, "config": {}}
        lines = _extract_context_panel(run)
        assert lines == []


# ---------------------------------------------------------------------------
# JSON repair for malformed tool-call arguments
# ---------------------------------------------------------------------------

class TestRepairJsonStr:
    """Tests for _repair_json_str (vLLM 400 resilience)."""

    def test_valid_json_passthrough(self):
        from tool_eval_bench.runner.orchestrator import _repair_json_str

        s = '{"city": "London", "units": "celsius"}'
        assert _repair_json_str(s) == s

    def test_empty_string(self):
        from tool_eval_bench.runner.orchestrator import _repair_json_str

        assert _repair_json_str("") == "{}"
        assert _repair_json_str("  ") == "{}"

    def test_none_like(self):
        from tool_eval_bench.runner.orchestrator import _repair_json_str

        assert _repair_json_str("") == "{}"

    def test_unterminated_string(self):
        """Gemma 4 failure case: string cut off mid-value."""
        from tool_eval_bench.runner.orchestrator import _repair_json_str
        import json

        broken = '{"city": "San Francisco, CA", "date": "2026-04-22", "query": "weather for tom'
        repaired = _repair_json_str(broken)
        # Must be valid JSON
        parsed = json.loads(repaired)
        assert isinstance(parsed, dict)

    def test_missing_closing_brace(self):
        from tool_eval_bench.runner.orchestrator import _repair_json_str
        import json

        broken = '{"city": "London"'
        repaired = _repair_json_str(broken)
        parsed = json.loads(repaired)
        assert parsed["city"] == "London"

    def test_missing_closing_bracket_and_brace(self):
        from tool_eval_bench.runner.orchestrator import _repair_json_str
        import json

        broken = '{"tags": ["urgent", "important"'
        repaired = _repair_json_str(broken)
        parsed = json.loads(repaired)
        assert isinstance(parsed, dict)

    def test_deeply_truncated_falls_back(self):
        """Completely unsalvageable JSON should return '{}'."""
        from tool_eval_bench.runner.orchestrator import _repair_json_str
        import json

        broken = '{"key": "val", "nested": {"inner": ['
        repaired = _repair_json_str(broken)
        # Must at least be valid JSON
        parsed = json.loads(repaired)
        assert isinstance(parsed, dict)

    def test_assistant_message_uses_repair(self):
        """Verify _assistant_message sanitizes arguments through repair."""
        from tool_eval_bench.adapters.base import ChatCompletionResult, ProviderToolCall
        from tool_eval_bench.runner.orchestrator import _assistant_message
        import json

        result = ChatCompletionResult(
            content="",
            tool_calls=[
                ProviderToolCall(
                    id="tc_1",
                    name="get_weather",
                    arguments_str='{"city": "Paris',  # broken
                ),
            ],
        )
        msg = _assistant_message(result)
        args_str = msg["tool_calls"][0]["function"]["arguments"]
        # Must be valid JSON
        parsed = json.loads(args_str)
        assert isinstance(parsed, dict)

