"""Redaction of server URLs written to shareable output.

Lives in the domain layer so storage, which may import nothing else, can apply
it to anything a caller hands a report writer.
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit, urlunsplit


def redact_url(url: str) -> str:
    """Mask the authority and remove query credentials from a persisted URL.

    e.g. http://192.168.10.5:8080 → http://***:8080

    Fails closed: an authority urllib cannot parse, such as an unclosed IPv6
    bracket, is withheld entirely rather than raising. Callers often redact
    inside an ``except`` handler, where a second exception would lose the first.
    """
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
    except ValueError:
        return f"{url.partition(':')[0].lower()}://***"
    if not hostname:
        return urlunsplit((parsed.scheme, "", parsed.path, "", ""))
    try:
        port = parsed.port
    except ValueError:
        port = None
    redacted_netloc = "***" if port is None else f"***:{port}"
    return urlunsplit((parsed.scheme, redacted_netloc, parsed.path, "", ""))


# Stops at whitespace, quotes, backticks, angle brackets and commas, which is
# where URLs end in error messages and comma-joined lists.
_URL_IN_TEXT = re.compile(r"https?://[^\s'\"`<>,]+", re.IGNORECASE)
_TRAILING_PUNCTUATION = ".;:!?"
_OPENER = {")": "(", "]": "[", "}": "{"}


def _redact_match(match: re.Match[str]) -> str:
    # Prose punctuation and an unbalanced closing bracket belong to the
    # sentence, not the URL: "(see http://h/v1)." keeps its ")." outside.
    url, tail = match.group(), ""
    while url:
        last = url[-1]
        unbalanced = last in _OPENER and url.count(last) > url.count(_OPENER[last])
        if last not in _TRAILING_PUNCTUATION and not unbalanced:
            break
        url, tail = url[:-1], last + tail
    return redact_url(url) + tail


def redact_urls(text: str) -> str:
    """Apply :func:`redact_url` to every http(s) URL inside free text.

    For error messages, not model output: httpx's ``HTTPStatusError`` quotes the
    full request URL, userinfo and query included, and that message ends up in
    traces, reports and stored scores.

    A URL ends at the first comma so comma-joined URLs redact separately. The
    trade-off: a query or userinfo secret that itself contains a comma leaves
    everything after that comma in the text.
    """
    return _URL_IN_TEXT.sub(_redact_match, text)
