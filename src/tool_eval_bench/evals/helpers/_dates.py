"""ISO date/time matching and benchmark reference-date helpers."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import date, datetime, timedelta, tzinfo
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from tool_eval_bench.domain.scenarios import ScenarioState
from tool_eval_bench.domain.tools import BENCHMARK_REFERENCE_DATE
from tool_eval_bench.evals.helpers._text import as_str

_ISO_OFFSET = r"(?:[Zz]|[+-](?:[01]\d|2[0-3]):?[0-5]\d)"
_ISO_TIME_TAIL = rf"(?:[01]\d|2[0-3]):[0-5]\d(?::[0-5]\d(?:\.\d+)?)?(?:{_ISO_OFFSET})?"


def datetime_matches(value: Any, date: str, time: str) -> bool:
    """Flexible ISO 8601 datetime match that accepts any valid timezone representation.

    Matches strings like:
     - "2026-03-21T08:00:00"          (naive)
     - "2026-03-21T08:00:00Z"         (UTC)
     - "2026-03-21T08:00:00+01:00"    (CET)
     - "2026-03-21T08:00:00-05:00"    (EST)
     - "2026-03-21 08:00"             (space-separated)

    The ``date`` and ``time`` arguments are the expected local date and time
    components (e.g. date="2026-03-21", time="08:00").  Only the date and
    hour:minute portion of ``value`` is compared — valid seconds and timezone
    offsets are accepted so that timezone-aware models are not penalised,
    while malformed suffixes (invalid seconds, out-of-range offsets, trailing
    garbage) are rejected.
    """
    s = as_str(value).strip()
    m = re.fullmatch(
        rf"(\d{{4}}-\d{{2}}-\d{{2}})[Tt ](\d{{2}}:\d{{2}})"
        rf"(?::[0-5]\d(?:\.\d+)?)?(?:{_ISO_OFFSET})?",
        s,
    )
    if not m:
        return False
    return m.group(1) == date and m.group(2) == time[:5]


def date_matches(value: Any, date: str) -> bool:
    """Check that a value names the given ISO date (YYYY-MM-DD), optionally with a valid time."""
    if not date:
        return False
    s = as_str(value).strip()
    return bool(re.fullmatch(rf"{re.escape(date)}(?:[Tt ]{_ISO_TIME_TAIL})?", s))


def time_matches(value: Any, expected: str) -> bool:
    """Return whether a ``time`` argument names ``expected`` as ``HH:MM``.

    Accepts an optional valid seconds field and an explicit UTC offset, because
    "14:00", "14:00:00" and "14:00+01:00" all schedule the same moment. Two
    scenarios creating the same event through the same tool disagreed on this,
    and the stricter one reported the seconds suffix as a timezone error.
    """
    if not expected:
        return False
    return bool(
        re.fullmatch(
            rf"{re.escape(expected)}(?::[0-5]\d(?:\.\d+)?)?(?:{_ISO_OFFSET})?",
            as_str(value).strip(),
        )
    )


_WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


def reference_date(state: ScenarioState) -> date:
    """The date the model was told is "today" for this run.

    ``--reference-date`` reaches the model through the system prompt and reaches
    the evaluator through ``state.meta``.  Any scenario whose prompt says
    "tomorrow" or "next Tuesday" must derive its expected date from here rather
    than hard-coding one, or the flag silently grades against a date the model
    was never given.  ``BENCHMARK_REFERENCE_DATE`` is the fallback for states
    built directly in tests.
    """
    raw = as_str(state.meta.get("reference_date")).strip() or BENCHMARK_REFERENCE_DATE
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return date.fromisoformat(BENCHMARK_REFERENCE_DATE)


def days_after_reference(state: ScenarioState, days: int) -> str:
    """ISO date ``days`` after the reference date. ``days=1`` is "tomorrow"."""
    return (reference_date(state) + timedelta(days=days)).isoformat()


def next_weekday_after_reference(state: ScenarioState, weekday: str, *, offset: int = 0) -> str:
    """ISO date of the next ``weekday`` strictly after the reference date.

    "Next Monday" on a Friday is three days out; on a Monday it is seven, never
    zero.  ``offset`` shifts the result in days, for prompts that name a weekday
    and then move the event ("next Tuesday", corrected to Wednesday).
    """
    try:
        target = _WEEKDAYS[weekday.strip().lower()]
    except KeyError:
        raise ValueError(f"Unknown weekday {weekday!r}") from None
    ref = reference_date(state)
    ahead = (target - ref.weekday()) % 7 or 7
    return (ref + timedelta(days=ahead + offset)).isoformat()


def utc_offset_aliases(
    iso_date: str,
    timezone: str,
    *,
    zone_info_cls: Callable[[str], tzinfo] = ZoneInfo,
) -> set[str]:
    """Offset spellings that are correct for ``timezone`` on ``iso_date``.

    Europe/Berlin is UTC+1 in winter and UTC+2 in summer, so a scenario that
    accepts "cet" as a synonym for "europe/berlin" is only right for part of the
    year.  Deriving the aliases from the date keeps the timezone check honest
    when ``--reference-date`` moves the event across a DST transition.
    """
    winter, summer = ("cet", "utc+1"), ("cest", "utc+2")
    try:
        moment = datetime.fromisoformat(f"{iso_date}T12:00:00").replace(
            tzinfo=zone_info_cls(timezone)
        )
    except (ValueError, ZoneInfoNotFoundError):
        return {*winter, *summer}
    offset = moment.utcoffset()
    if offset == timedelta(hours=1):
        return set(winter)
    if offset == timedelta(hours=2):
        return set(summer)
    return set()
