"""The HuggingFace retry ladder, which is the path users actually hit.

Rate limiting is the documented common failure for a first-use download, and
`_fetch_with_retry` is what stands between a 429 and a failed benchmark. None
of it was covered.
"""

from __future__ import annotations

import json
import urllib.error
from email.message import Message
from typing import Any

import pytest

from tool_eval_bench.plugins import hf_utils

URL = "https://datasets-server.huggingface.co/rows?dataset=test"


class _Body:
    """Stands in for the file-like object `urlopen` returns."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._raw = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> _Body:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _http_error(code: int, *, retry_after: str | None = None) -> urllib.error.HTTPError:
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError(URL, code, "boom", headers, None)


@pytest.fixture
def urlopen(monkeypatch: pytest.MonkeyPatch):
    """Script `urlopen`'s outcomes in order, and record every sleep."""
    slept: list[float] = []
    monkeypatch.setattr(hf_utils.time, "sleep", slept.append)

    def script(outcomes: list[Any]) -> list[float]:
        remaining = list(outcomes)

        def fake(url: str, timeout: int | None = None) -> Any:
            outcome = remaining.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return _Body(outcome)

        monkeypatch.setattr(hf_utils.urllib.request, "urlopen", fake)
        return slept

    return script


def test_a_first_try_success_does_not_sleep(urlopen) -> None:
    slept = urlopen([{"rows": []}])

    assert hf_utils._fetch_with_retry(URL) == {"rows": []}
    assert slept == []


def test_a_429_is_retried_and_then_succeeds(urlopen) -> None:
    slept = urlopen([_http_error(429), {"ok": True}])

    assert hf_utils._fetch_with_retry(URL) == {"ok": True}
    assert slept == [2.0], "the first retry should use the initial backoff"


def test_retry_after_overrides_the_backoff(urlopen) -> None:
    """A server that says how long to wait is more accurate than our guess."""
    slept = urlopen([_http_error(429, retry_after="7"), {"ok": True}])

    hf_utils._fetch_with_retry(URL)

    assert slept == [7.0]


def test_backoff_doubles_and_is_capped(urlopen) -> None:
    slept = urlopen([_http_error(429)] * 4 + [{"ok": True}])

    hf_utils._fetch_with_retry(URL, max_retries=8)

    assert slept == [2.0, 4.0, 8.0, 16.0]
    assert max(slept) <= hf_utils._MAX_BACKOFF_S


def test_a_server_error_is_retried(urlopen) -> None:
    slept = urlopen([_http_error(503), {"ok": True}])

    assert hf_utils._fetch_with_retry(URL) == {"ok": True}
    assert slept == [2.0]


def test_a_client_error_is_not_retried(urlopen) -> None:
    """A 404 will not become a 200 by waiting."""
    slept = urlopen([_http_error(404)])

    with pytest.raises(urllib.error.HTTPError):
        hf_utils._fetch_with_retry(URL)
    assert slept == []


def test_a_network_error_is_retried(urlopen) -> None:
    slept = urlopen([TimeoutError("timed out"), {"ok": True}])

    assert hf_utils._fetch_with_retry(URL) == {"ok": True}
    assert slept == [2.0]


def test_exhausting_the_retries_raises_the_last_error(urlopen) -> None:
    urlopen([_http_error(429)] * 3)

    with pytest.raises(urllib.error.HTTPError) as caught:
        hf_utils._fetch_with_retry(URL, max_retries=3)

    assert caught.value.code == 429


def test_the_caller_is_told_about_every_retry(urlopen) -> None:
    """The CLI turns these into the "retrying" line a waiting user sees."""
    urlopen([_http_error(429, retry_after="3"), _http_error(500), OSError("reset"), {"ok": True}])
    seen: list[tuple[int, int, float]] = []

    hf_utils._fetch_with_retry(URL, max_retries=6, on_retry=lambda *args: seen.append(args))

    assert [attempt for attempt, _, _ in seen] == [1, 2, 3]
    # The backoff keeps doubling underneath a server-dictated wait, so the
    # ladder stays conservative once an endpoint has started refusing.
    assert [wait for _, _, wait in seen] == [3.0, 4.0, 8.0]
    assert {total for _, total, _ in seen} == {6}
