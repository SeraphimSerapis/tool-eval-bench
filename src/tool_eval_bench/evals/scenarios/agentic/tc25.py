"""TC-25 — Cross-Reference Prior Results."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from tool_eval_bench.domain.scenarios import (
    Category,
    ScenarioDefinition,
    ScenarioDisplayDetail,
    ScenarioEvaluation,
    ScenarioState,
    ToolCallRecord,
)
from tool_eval_bench.evals.helpers import (
    answer_affirms_number as _answer_affirms_number,
)
from tool_eval_bench.evals.helpers import (
    answer_affirms_text as _answer_affirms_text,
)
from tool_eval_bench.evals.helpers import (
    as_str as _as_str,
)
from tool_eval_bench.evals.helpers import counted_calls as _counted_calls
from tool_eval_bench.evals.helpers import (
    days_after_reference as _days_after_reference,
)
from tool_eval_bench.evals.helpers import (
    fail_eval as _fail,
)
from tool_eval_bench.evals.helpers import first_counted as _first_counted
from tool_eval_bench.evals.helpers import forbid_unrequested_side_effects
from tool_eval_bench.evals.helpers import (
    generic_tool_fallback_simple as _generic_tool_fallback,
)
from tool_eval_bench.evals.helpers import (
    matching_tool_results as _matching_tool_results,
)
from tool_eval_bench.evals.helpers import (
    negated_before as _negated_before,
)
from tool_eval_bench.evals.helpers import (
    normalize as _normalize,
)
from tool_eval_bench.evals.helpers import (
    partial_eval as _partial,
)
from tool_eval_bench.evals.helpers import (
    pass_eval as _pass,
)
from tool_eval_bench.evals.helpers import (
    positive_argument_contains as _positive_argument_contains,
)
from tool_eval_bench.evals.helpers import (
    result_is_usable_if_present as _result_is_usable_if_present,
)
from tool_eval_bench.evals.helpers import (
    strip_markdown_emphasis as _strip_markdown_emphasis,
)
from tool_eval_bench.evals.helpers import (
    with_noise as _noise,
)


def _tc25_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_weather":
        return _noise(
            {
                "location": "Berlin",
                "temperature": 5,
                "units": "celsius",
                "condition": "Clear",
                "humidity": 45,
            },
            "get_weather",
        )
    if call.name == "set_reminder":
        return _noise({"status": "created", "reminder_id": "rem_552"}, "set_reminder")
    return _generic_tool_fallback(call)


_TC25_DONT_FORGET_RE = re.compile(r"\b(?:don't|do\s+not|never)\s+forget\b", re.IGNORECASE)
_TC25_DENIES_REMINDER = re.compile(
    r"\b(?:(?:did|have|had|will|would)\s+not|didn't|haven't|hadn't|won't|not\s+going\s+to)\s+"
    r"(?:set|create|add|schedule)\s+(?:a\s+|the\s+|any\s+)?(?:coat\s+)?reminder\b"
    r"|\bno\s+(?:coat\s+)?reminder\s+(?:was|has\s+been|is|will\s+be)\s+(?:set|created|added|scheduled)\b"
    r"|\b(?:skip(?:ped|ping)?|omitt(?:ed|ing)?|withheld|did\s+not\s+need|didn't\s+need|no\s+need\s+for)\s+"
    r"(?:the\s+|a\s+|any\s+)?(?:coat\s+)?reminder\b"
)
_TC25_TEMP_MENTION = re.compile(
    r"(?<![\d.])(-?\d+(?:\.\d+)?)\s*(?:degrees?\s*)?(?:°\s*(?P<short>[cf])\b|(?P<long>celsius|fahrenheit)\b)",
    re.IGNORECASE,
)
_TC25_THRESHOLD_PREFIX = re.compile(
    r"\b(?:below|under|less\s+than|at\s+or\s+below|threshold(?:\s+of)?|if(?:\s+it(?:'s|\s+is))?\s+(?:below|under))\s*$",
    re.IGNORECASE,
)
_TC25_ABOVE_THRESHOLD = re.compile(
    r"\b(?:above|over|warmer\s+than|higher\s+than|not\s+(?:below|under))\s+10(?:\.0+)?\s*(?:°\s*c\b|degrees?\s*c(?:elsius)?\b|celsius\b)?",
    re.IGNORECASE,
)


def _tc25_mentions_coat(message: str) -> bool:
    """Accept reminder phrasing like 'Don't forget to bring a coat' while rejecting 'no coat'."""
    normalized = _TC25_DONT_FORGET_RE.sub("remember", _normalize(_strip_markdown_emphasis(message)))
    return _answer_affirms_text(normalized, "coat")


def _tc25_answer_contradicts(final_answer: str) -> bool:
    """Return True when the final answer contradicts the 5°C reading or reminder creation."""
    answer = _normalize(_strip_markdown_emphasis(final_answer))
    if not answer:
        return False
    if _TC25_DENIES_REMINDER.search(answer):
        return True
    for match in _TC25_ABOVE_THRESHOLD.finditer(answer):
        if not _negated_before(
            answer[max(0, match.start() - 120) : match.start()],
            answer[match.end() : match.end() + 40],
        ):
            return True
    if _answer_affirms_number(answer, "5") or _answer_affirms_number(answer, "41"):
        return False
    for match in _TC25_TEMP_MENTION.finditer(answer):
        prefix = answer[max(0, match.start() - 120) : match.start()]
        suffix = answer[match.end() : match.end() + 40]
        if _negated_before(prefix, suffix):
            continue
        value = float(match.group(1))
        unit = (match.group("short") or match.group("long") or "").lower()
        is_fahrenheit = unit.startswith("f")
        if is_fahrenheit:
            if abs(value - 41.0) < 0.5:
                continue
            if abs(value - 50.0) < 0.5 and _TC25_THRESHOLD_PREFIX.search(prefix):
                continue
            return True
        if abs(value - 5.0) < 0.5:
            continue
        if abs(value - 10.0) < 0.5 and _TC25_THRESHOLD_PREFIX.search(prefix):
            continue
        return True
    return False


def _tc25_is_tomorrow_morning(datetime_value: Any, state: ScenarioState) -> bool:
    """Return whether ``datetime_value`` schedules the reminder for tomorrow morning."""
    dt_str = _as_str(datetime_value).strip().lower()
    if not dt_str:
        return False
    if re.search(r"\btomorrow\s+morning\b", dt_str) and not re.search(
        r"\b(?:not|never|day\s+after|the\s+day\s+after)\s+tomorrow\b", dt_str
    ):
        return True
    parse_str = dt_str[:-1] if dt_str.endswith("z") else dt_str
    try:
        target = datetime.fromisoformat(parse_str)
    except ValueError:
        return False
    return target.date().isoformat() == _days_after_reference(state, 1) and 5 <= target.hour < 12


@forbid_unrequested_side_effects({"set_reminder": 1})
def _tc25_eval(state: ScenarioState) -> ScenarioEvaluation:
    """Two-part request: 'What's the weather in Berlin? Also, if it's below 10°C, set a reminder to bring a coat.'

    The model should call get_weather ONCE, see 5°C < 10°C, then set the reminder.
    It should NOT call get_weather twice.
    """
    weather_calls = [c for c in state.tool_calls if c.name == "get_weather"]
    reminder_calls = [c for c in state.tool_calls if c.name == "set_reminder"]

    if len(weather_calls) == 0:
        return _fail("Did not check the weather.")

    berlin_calls = [
        call
        for call in weather_calls
        if _positive_argument_contains(call.arguments.get("location"), "berlin")
    ]
    if not berlin_calls:
        return _fail("Checked the wrong location for the conditional weather request.")

    # Check if the weather tool actually returned data or was error-injected
    weather_results = _matching_tool_results(state, berlin_calls[-1])
    weather_got_error = not _result_is_usable_if_present(state, berlin_calls[-1])

    if weather_got_error:
        if reminder_calls:
            return _fail(
                "Set a coat reminder despite get_weather returning an error with no temperature."
            )
        # The model never received the temperature, so it cannot evaluate the
        # conditional. Credit retry attempts and graceful behavior.
        if len(weather_calls) > 1:
            return _partial("Weather tool failed, retried, good recovery attempt.")
        return _partial(
            "Weather tool returned an error, so the temperature condition could not be evaluated."
        )

    if len(_counted_calls(weather_calls)) > 1:
        return _partial("Called get_weather multiple times instead of reusing the first result.")

    recorded_temperatures: list[float] = []
    for result in weather_results:
        if isinstance(result.result, dict):
            temperature = result.result.get("temperature")
            if isinstance(temperature, (int, float)) and not isinstance(temperature, bool):
                recorded_temperatures.append(temperature)
    if recorded_temperatures and not any(temperature < 10 for temperature in recorded_temperatures):
        if reminder_calls:
            return _fail(
                "Set a coat reminder even though the recorded temperature was not below 10°C."
            )
        return _pass("Correctly skipped the reminder because the temperature was not below 10°C.")

    if len(reminder_calls) == 0:
        return _fail("Checked weather (5°C) but did not set the coat reminder.")

    first_weather = _counted_calls(weather_calls)[0]
    if any(call.turn <= first_weather.turn for call in reminder_calls):
        return _fail("Set the reminder before receiving the weather result.")

    coat_call = _first_counted(
        reminder_calls,
        lambda call: _tc25_mentions_coat(_as_str(call.arguments.get("message"))),
    )
    if coat_call is not None:
        if not _result_is_usable_if_present(state, coat_call):
            return _partial("The coat reminder call returned an explicit error.")
        if not _tc25_is_tomorrow_morning(coat_call.arguments.get("datetime"), state):
            return _partial(
                "Set a coat reminder, but the datetime was not tomorrow morning.",
                f"Got datetime={coat_call.arguments.get('datetime')!r}.",
            )
        if len(_counted_calls(state.tool_calls)) != 2:
            return _partial("Completed the conditional request with unnecessary additional calls.")
        if _tc25_answer_contradicts(state.final_answer):
            return _partial(
                "Set the coat reminder, but the final answer contradicted the weather reading "
                "or reminder action."
            )
        return _pass("Checked weather once, recognized 5°C < 10°C, set coat reminder.")
    return _partial("Set a reminder but didn't mention bringing a coat.")


SCENARIO = ScenarioDefinition(
    id="TC-25",
    title="Cross-Reference Prior Results",
    category=Category.I,
    user_message="What's the weather in Berlin? Also, if it's below 10°C, set a reminder to bring a coat tomorrow morning.",
    description="Should call get_weather once, then conditionally set_reminder.",
    handle_tool_call=_tc25_handle,
    evaluate=_tc25_eval,
    difficulty=2,
    dependencies=(("get_weather", "set_reminder"),),
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it calls get_weather once, sees 5°C < 10, sets coat reminder.",
    "Fail if it calls get_weather twice or skips the reminder.",
)
