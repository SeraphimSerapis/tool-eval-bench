"""Reports and comparisons must not leak secrets or execute injected markup.

Markdown reports are the shared artifact: people post them, mail them, and feed
them to ``compare --report``. That makes them untrusted input for the HTML
generator, and makes their persisted metadata a place where an endpoint URL (or
credentials embedded in it) must not appear.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tool_eval_bench.cli.helpers import with_config_fingerprint
from tool_eval_bench.compare_reports.summary import generate_html as generate_summary_html
from tool_eval_bench.compare_reports.summary import parse_summary
from tool_eval_bench.compare_reports.tool_eval import generate_html as generate_tool_eval_html
from tool_eval_bench.compare_reports.tool_eval import parse_md
from tool_eval_bench.domain.models import BenchmarkConfig
from tool_eval_bench.utils.metadata import collect_run_metadata
from tool_eval_bench.utils.urls import (
    endpoint_identity,
    is_same_origin,
    metrics_request_target,
    redact_url,
    validate_http_url,
)

PAYLOAD = '<script>alert("xss")</script>'


def _tool_eval_report(model_name: str, scenario_id: str, summary: str) -> str:
    return f"""# Tool-Call Benchmark — {model_name}

- **Run ID**: `{PAYLOAD}`
- **Date**: `2026-07-04T00:00:00+00:00`
- **tool-eval-bench**: `{PAYLOAD}`
- **Final Score**: **80** / 100
- **Total Points**: 80 / 100
- **Rating**: {PAYLOAD}

| Field | Value |
|---|---|
| **Backend** | {PAYLOAD} |
| **Model (API)** | `{model_name}` |
| **Temperature** | {PAYLOAD} |
| **Thinking** | {PAYLOAD} |

## Category Scores

| Category | Earned | Max | % |
|---|---|---|---|
| {PAYLOAD} | 8 | 10 | 80 |

## Scenario Results

| ID | Title | Status | Points | Summary |
|---|---|---|---|---|
| {scenario_id} | {PAYLOAD} | ❌ | 0 | {summary} |
| {scenario_id}-P | {PAYLOAD} | ⚠️ | 1 | {summary} |

## Performance by Difficulty

| Tier | Earned | Max | % |
|---|---|---|---|
| {PAYLOAD} | 4 | 5 | 80 |
"""


class TestHtmlComparisonEscaping:
    @pytest.fixture
    def malicious_reports(self, tmp_path: Path) -> tuple[Path, Path]:
        a = tmp_path / "a.md"
        b = tmp_path / "b.md"
        a.write_text(
            _tool_eval_report(f"Model{PAYLOAD}", f"TC{PAYLOAD}", PAYLOAD), encoding="utf-8"
        )
        b.write_text(_tool_eval_report("Clean Model", "TC-02", "clean"), encoding="utf-8")
        return a, b

    def test_tool_eval_comparison_never_emits_raw_injected_markup(
        self, malicious_reports: tuple[Path, Path], tmp_path: Path
    ) -> None:
        a, b = malicious_reports
        out = tmp_path / "out.html"

        generate_tool_eval_html(parse_md(str(a)), parse_md(str(b)), str(out))

        html = out.read_text(encoding="utf-8")
        assert PAYLOAD not in html
        assert "<script>alert" not in html
        # The content is still present, just inert.
        assert "&lt;script&gt;" in html

    def test_both_orderings_are_escaped(
        self, malicious_reports: tuple[Path, Path], tmp_path: Path
    ) -> None:
        """The generator sorts by score, so winner and runner-up paths both matter."""
        a, b = malicious_reports
        out = tmp_path / "out.html"

        generate_tool_eval_html(parse_md(str(b)), parse_md(str(a)), str(out))

        assert PAYLOAD not in out.read_text(encoding="utf-8")

    def test_summary_comparison_escapes_injected_model_names(self, tmp_path: Path) -> None:
        report = f"""# Cross-Trial Summary — Model{PAYLOAD}

- **Run ID**: `{PAYLOAD}`
- **Date**: `2026-07-04T00:00:00+00:00`
- **tool-eval-bench**: `{PAYLOAD}`
- **Trials**: 3
- **Mean Score**: 80.0
"""
        a = tmp_path / "a.md"
        b = tmp_path / "b.md"
        a.write_text(report, encoding="utf-8")
        b.write_text(report.replace(PAYLOAD, "clean"), encoding="utf-8")
        out = tmp_path / "out.html"

        generate_summary_html(parse_summary(str(a)), parse_summary(str(b)), str(out))

        html = out.read_text(encoding="utf-8")
        assert PAYLOAD not in html
        assert "<script>alert" not in html

    def test_report_without_a_date_does_not_crash(self, tmp_path: Path) -> None:
        undated = _tool_eval_report("A", "TC-01", "x").replace(
            "- **Date**: `2026-07-04T00:00:00+00:00`\n", ""
        )
        a = tmp_path / "a.md"
        b = tmp_path / "b.md"
        a.write_text(undated, encoding="utf-8")
        b.write_text(undated, encoding="utf-8")
        out = tmp_path / "out.html"

        generate_tool_eval_html(parse_md(str(a)), parse_md(str(b)), str(out))

        assert out.exists()

    def test_quote_characters_cannot_break_out_of_an_attribute(self, tmp_path: Path) -> None:
        payload = '" onmouseover="alert(1)'
        a = tmp_path / "a.md"
        b = tmp_path / "b.md"
        a.write_text(_tool_eval_report(f"M{payload}", "TC-01", "x"), encoding="utf-8")
        b.write_text(_tool_eval_report("Clean", "TC-02", "y"), encoding="utf-8")
        out = tmp_path / "out.html"

        generate_tool_eval_html(parse_md(str(a)), parse_md(str(b)), str(out))

        html = out.read_text(encoding="utf-8")
        assert payload not in html
        assert "&quot; onmouseover=&quot;" in html


class TestPersistedMetadataRedaction:
    @pytest.mark.asyncio
    async def test_endpoint_host_is_not_persisted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import tool_eval_bench.utils.metadata as metadata_module

        async def no_probe(base_url: str, api_key: str | None) -> dict:
            return {}

        monkeypatch.setattr(metadata_module, "_probe_models", no_probe)

        metadata = await collect_run_metadata(
            BenchmarkConfig(
                model="m",
                backend="vllm",
                base_url="http://10.1.2.3:8080/v1",
                api_key="sk-secret",
            )
        )

        assert "10.1.2.3" not in str(metadata)
        assert metadata["config"]["base_url"] == "http://***:8080/v1"

    @pytest.mark.asyncio
    async def test_credentials_in_the_url_are_not_persisted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import tool_eval_bench.utils.metadata as metadata_module

        async def no_probe(base_url: str, api_key: str | None) -> dict:
            return {}

        monkeypatch.setattr(metadata_module, "_probe_models", no_probe)

        metadata = await collect_run_metadata(
            BenchmarkConfig(
                model="m",
                backend="vllm",
                base_url="https://user:hunter2@endpoint.internal/v1",
            )
        )

        assert "hunter2" not in str(metadata)
        assert "endpoint.internal" not in str(metadata)

    def test_redact_url_drops_userinfo_and_host(self) -> None:
        assert redact_url("https://user:pw@host.internal:8443/v1") == "https://***:8443/v1"

    def test_redact_url_drops_query_and_fragment_credentials(self) -> None:
        redacted = redact_url("https://user:pw@host.internal:8443/v1?token=secret#private")

        assert redacted == "https://***:8443/v1"
        assert "secret" not in redacted

    def test_endpoint_identity_ignores_credentials_but_separates_hosts(self) -> None:
        first = endpoint_identity("https://user:one@host.internal/v1?token=one")
        same = endpoint_identity("https://user:two@host.internal:443/v1?token=two")
        other = endpoint_identity("https://other.internal/v1")

        assert first == same
        assert first != other
        assert "host.internal" not in first

    def test_plugin_config_drops_url_credentials_and_keeps_endpoint_identity(self) -> None:
        original = {"base_url": "https://user:secret@one.internal:8443/v1?token=leak", "model": "m"}
        first = with_config_fingerprint(original)
        same_endpoint_new_token = with_config_fingerprint(
            {"base_url": "https://other:new-secret@one.internal:8443/v1?token=fresh", "model": "m"}
        )
        other_endpoint = with_config_fingerprint(
            {"base_url": "https://user:secret@two.internal:8443/v1", "model": "m"}
        )

        assert first["base_url"] == "https://***:8443/v1"
        assert "secret" not in str(first)
        assert "one.internal" not in str(first)
        assert original["base_url"].endswith("token=leak")
        assert first["config_fingerprint"] == same_endpoint_new_token["config_fingerprint"]
        assert first["config_fingerprint"] != other_endpoint["config_fingerprint"]


class TestMetricsUrlCredentialScope:
    def test_default_target_carries_the_token(self) -> None:
        url, headers = metrics_request_target("http://host:8000/v1", None, "sk-secret")
        assert url == "http://host:8000/metrics"
        assert headers == {"Authorization": "Bearer sk-secret"}

    def test_same_origin_override_carries_the_token(self) -> None:
        _, headers = metrics_request_target(
            "http://host:8000/v1", "http://host:8000/proxy/metrics", "sk-secret"
        )
        assert headers == {"Authorization": "Bearer sk-secret"}

    def test_cross_host_override_does_not_leak_the_token(self) -> None:
        """A --metrics-url on another host must not receive the endpoint's key."""
        url, headers = metrics_request_target(
            "http://host:8000/v1", "http://someone-else.example/metrics", "sk-secret"
        )
        assert url == "http://someone-else.example/metrics"
        assert headers == {}

    def test_cross_port_override_does_not_leak_the_token(self) -> None:
        _, headers = metrics_request_target(
            "http://host:8000/v1", "http://host:9090/metrics", "sk-secret"
        )
        assert headers == {}

    def test_scheme_downgrade_does_not_leak_the_token(self) -> None:
        _, headers = metrics_request_target("https://host/v1", "http://host/metrics", "sk-secret")
        assert headers == {}

    def test_no_token_means_no_header_either_way(self) -> None:
        _, headers = metrics_request_target("http://host:8000/v1", "http://other/metrics", None)
        assert headers == {}

    @pytest.mark.parametrize(
        "bad", ["file:///etc/passwd", "gopher://host/", "/relative/metrics", "notaurl"]
    )
    def test_non_http_metrics_url_is_rejected(self, bad: str) -> None:
        with pytest.raises(ValueError, match="--metrics-url"):
            metrics_request_target("http://host:8000/v1", bad, None)


class TestUrlGuards:
    @pytest.mark.parametrize(
        "a,b",
        [
            ("http://host/x", "http://host:80/y"),
            ("https://host/x", "https://HOST:443/y"),
            ("http://host:8000/v1", "http://host:8000/metrics"),
        ],
    )
    def test_same_origin_matches(self, a: str, b: str) -> None:
        assert is_same_origin(a, b)

    @pytest.mark.parametrize(
        "a,b",
        [
            ("http://host/x", "https://host/x"),
            ("http://a/x", "http://b/x"),
            ("http://host:1/x", "http://host:2/x"),
            ("notaurl", "http://host/x"),
        ],
    )
    def test_same_origin_rejects(self, a: str, b: str) -> None:
        assert not is_same_origin(a, b)

    def test_validate_http_url_returns_input_when_valid(self) -> None:
        assert validate_http_url("https://host:8443/v1") == "https://host:8443/v1"

    def test_validate_http_url_error_does_not_echo_the_host(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            validate_http_url("ftp://secret.internal/x", what="--metrics-url")
        assert "secret.internal" not in str(excinfo.value)
