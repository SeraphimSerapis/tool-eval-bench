"""Credential redaction in the logged llama-benchy command line.

Deliberately not marked ``integration``: ``tests/test_llama_benchy.py`` is
excluded from the default suite because its cases need the ``[perf]`` extra,
and a security regression guard must not live in a file most runs skip. The
module under test keeps llama-benchy a soft dependency — it shells out and
parses JSON, never importing it — so these cases run anywhere.

These assert on what the redactor returns rather than routing a fake credential
through ``logger.info``. A test that logs one trips
``py/clear-text-logging-sensitive-data`` itself, because CodeQL cannot tell a
sanitized string from a raw one at the call site. The call site is covered
anyway: that query is exactly what caught the original leak, and it re-clears
only while ``_redact_command`` stays in front of the logger.
"""

from __future__ import annotations

import pytest

from tool_eval_bench.runner.llama_benchy import (
    _redact_argument,
    _redact_command,
    _strip_url_credentials,
)

SECRET = "hunter2"


class TestStripUrlCredentials:
    def test_userinfo_is_removed(self) -> None:
        result = _strip_url_credentials(f"https://admin:{SECRET}@gpu.internal:8000/v1")
        assert result == "https://gpu.internal:8000/v1"
        assert SECRET not in result
        assert "admin" not in result

    def test_username_only_is_removed(self) -> None:
        assert _strip_url_credentials("https://token@host/v1") == "https://host/v1"

    def test_query_is_dropped(self) -> None:
        # A query string is the other place a key travels.
        result = _strip_url_credentials("http://host/v1?api_key=leaked")
        assert result == "http://host/v1"
        assert "leaked" not in result

    def test_fragment_is_dropped(self) -> None:
        assert _strip_url_credentials("http://host/v1#tok") == "http://host/v1"

    def test_ordinary_url_survives_intact(self) -> None:
        # Host and port must survive: the log exists to show which server ran.
        assert _strip_url_credentials("http://localhost:8000/v1") == "http://localhost:8000/v1"

    def test_default_port_url_survives(self) -> None:
        assert _strip_url_credentials("https://api.example.com/v1") == "https://api.example.com/v1"

    def test_ipv6_literal_keeps_its_brackets(self) -> None:
        # urlsplit strips the brackets; without restoring them the address and
        # the port run together into something unparsable.
        assert _strip_url_credentials("http://[::1]:8000/v1") == "http://[::1]:8000/v1"

    def test_ipv6_literal_with_credentials(self) -> None:
        result = _strip_url_credentials(f"https://admin:{SECRET}@[2001:db8::1]:8443/v1")
        assert result == "https://[2001:db8::1]:8443/v1"
        assert SECRET not in result

    def test_malformed_authority_fails_closed(self) -> None:
        # Withhold the whole value rather than guess which part was the secret.
        assert _strip_url_credentials(f"http://admin:{SECRET}@host:notaport/v1") == (
            "<unparsable-url>"
        )

    def test_hostless_value_is_returned_unchanged(self) -> None:
        assert _strip_url_credentials("http://") == "http://"


class TestRedactArgument:
    def test_bare_url_argument_is_stripped(self) -> None:
        assert SECRET not in _redact_argument(f"https://u:{SECRET}@host/v1")

    def test_flag_equals_url_is_stripped(self) -> None:
        result = _redact_argument(f"--base-url=https://u:{SECRET}@host/v1")
        assert result == "--base-url=https://host/v1"

    def test_non_url_argument_is_untouched(self) -> None:
        for arg in ("--pp", "2048", "--latency-mode", "generation", "--no-cache"):
            assert _redact_argument(arg) == arg

    def test_non_http_scheme_is_untouched(self) -> None:
        # Only http(s) is rewritten; a file path with '=' must survive as-is.
        assert _redact_argument("--save-result=/tmp/out.json") == "--save-result=/tmp/out.json"


class TestRedactCommand:
    def _command(self, base_url: str) -> list[str]:
        return [
            "llama-benchy",
            "--base-url",
            base_url,
            "--model",
            "test-model",
            "--api-key",
            "sk-test-secret",
            "--pp",
            "2048",
        ]

    def test_url_password_is_not_logged(self) -> None:
        """Regression for the CodeQL clear-text-logging alert.

        ``--api-key`` was redacted while credentials embedded in ``--base-url``
        went to the log verbatim.
        """
        rendered = _redact_command(self._command(f"https://admin:{SECRET}@gpu.internal:8000/v1"))
        assert SECRET not in rendered
        assert "sk-test-secret" not in rendered
        assert "gpu.internal:8000" in rendered

    def test_api_key_redaction_still_works(self) -> None:
        rendered = _redact_command(self._command("http://localhost:8000/v1"))
        assert "sk-test-secret" not in rendered
        assert "<redacted>" in rendered

    def test_both_api_key_spellings_are_redacted(self) -> None:
        rendered = _redact_command(
            ["llama-benchy", "--api-key", "sk-one", "--api-key=sk-two"],
        )
        assert "sk-one" not in rendered and "sk-two" not in rendered

    def test_ordinary_command_is_unchanged(self) -> None:
        rendered = _redact_command(
            ["llama-benchy", "--base-url", "http://localhost:8000/v1", "--pp", "2048"]
        )
        assert rendered == "llama-benchy --base-url http://localhost:8000/v1 --pp 2048"

    def test_credentials_in_passthrough_extra_args_are_stripped(self) -> None:
        # --benchy-args forwards arbitrary flags; a URL there leaks the same way.
        rendered = _redact_command(
            [
                "llama-benchy",
                "--base-url",
                "http://localhost:8000/v1",
                "--proxy",
                f"https://u:{SECRET}@proxy.internal:3128",
            ]
        )
        assert SECRET not in rendered

    def test_api_key_value_that_looks_like_a_url_is_still_redacted(self) -> None:
        rendered = _redact_command(["llama-benchy", "--api-key", f"https://{SECRET}@x/"])
        assert SECRET not in rendered
        assert "<redacted>" in rendered

    @pytest.mark.parametrize(
        "base_url",
        [
            "http://localhost:8000/v1",
            "https://api.example.com/v1",
            "http://[::1]:8000/v1",
        ],
    )
    def test_uncredentialed_urls_stay_readable(self, base_url: str) -> None:
        assert base_url in _redact_command(self._command(base_url))
