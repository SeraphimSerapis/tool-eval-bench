"""Autonomous planning & creative composition scenarios.

TC-51 to TC-56: 6 scenarios in Categories M (Autonomous Planning)
and N (Creative Composition) that test whether models can decompose
goals into tool chains without step-by-step guidance, and combine
tools in non-obvious ways.

TC-61: Async polling scenario (Category C expansion).
TC-62: 5-turn deep research (Category I expansion).
TC-63: Accumulating constraints (Category I expansion).
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
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
    as_str as _as_str,
)
from tool_eval_bench.evals.helpers import (
    as_str_list as _as_str_list,
)
from tool_eval_bench.evals.helpers import (
    asks_for_clarification as _asks_clarification,
)
from tool_eval_bench.evals.helpers import (
    fail_eval as _fail,
)
from tool_eval_bench.evals.helpers import (
    includes_text as _includes_text,
)
from tool_eval_bench.evals.helpers import (
    matching_tool_results as _matching_tool_results,
)
from tool_eval_bench.evals.helpers import (
    parse_math_expression as _parse_math_expression,
)
from tool_eval_bench.evals.helpers import (
    partial_eval as _partial,
)
from tool_eval_bench.evals.helpers import (
    pass_eval as _pass,
)
from tool_eval_bench.evals.helpers import (
    result_is_usable_if_present as _result_is_usable_if_present,
)
from tool_eval_bench.evals.helpers import (
    tool_calls_by_name as _tool_calls_by_name,
)
from tool_eval_bench.evals.helpers import (
    with_noise as _noise,
)


def _result_matches_if_present(
    state: ScenarioState,
    call: ToolCallRecord,
    predicate: Any,
) -> bool:
    """Validate an explicit result while preserving synthetic trace support.

    Direct evaluator tests and imported historical traces may contain tool
    calls without result records.  Runtime traces always have results, so an
    explicit result must be both successful and consistent with the fixture
    the scenario promises.  If the trace has stable IDs, never borrow a
    same-named result from another call.
    """
    exact_results = [result for result in state.tool_results if result.call_id == call.id]
    known_call_ids = {candidate.id for candidate in state.tool_calls}
    if exact_results and any(result.name == call.name for result in exact_results):
        results = exact_results
    elif exact_results and all(result.name in {"", "unknown"} for result in exact_results):
        # Older synthetic tests attached a bare payload to call_0 without a
        # tool name.  It is evidence for the first call only when the caller
        # can identify that name, otherwise leave it unknown.
        results = []
    elif exact_results and all(
        any(candidate.name == result.name for candidate in state.tool_calls)
        for result in exact_results
    ):
        # Legacy fixtures sometimes default every result call_id to call_0.
        # If the named result belongs to another recorded call, do not let it
        # masquerade as this call's payload.
        results = []
    elif any(
        result.call_id in known_call_ids and result.name == call.name
        for result in state.tool_results
    ):
        return False
    else:
        results = _matching_tool_results(state, call)
    if not results:
        return True
    return _result_is_usable_if_present(state, call) and any(
        predicate(result.result) for result in results
    )


def _result_has_status(payload: Any, status: str, identifier: str | None = None) -> bool:
    if not isinstance(payload, dict):
        return False
    if str(payload.get("status", "")).strip().lower() != status:
        return False
    return identifier is None or identifier in str(payload)


def _has_unexpected_tools(state: ScenarioState, allowed: set[str]) -> bool:
    return any(call.name not in allowed for call in state.tool_calls)


def _call_index(state: ScenarioState, target: ToolCallRecord) -> int:
    """Find a call by identity, not dataclass equality, for dependency checks."""
    return next(index for index, call in enumerate(state.tool_calls) if call is target)


# ===================================================================
# TC-51: Goal-Level Planning
# Category M — Autonomous Planning
# ===================================================================

_TC51_ENGINEERING = frozenset({"alice@company.com", "bob@company.com", "carol@company.com"})
_UNRELATED_UNIVERSAL_MUTATIONS = frozenset({"set_reminder", "run_code"})


def _tc51_contacts_result_is_engineering(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    contacts = payload.get("results")
    if not isinstance(contacts, list):
        return False
    emails = {
        _as_str(contact.get("email")).strip().lower()
        for contact in contacts
        if isinstance(contact, dict)
    }
    return _TC51_ENGINEERING <= emails


def _tc51_event_result_is_created(payload: Any) -> bool:
    return _result_has_status(payload, "created", "evt_tc51")


def _tc51_email_result_is_sent(payload: Any) -> bool:
    return _result_has_status(payload, "sent")


def _tc51_friday(state: ScenarioState) -> str:
    """Return the date meant by ``this Friday`` for the benchmark date."""
    raw = _as_str(state.meta.get("reference_date")).strip() or "2026-03-20"
    try:
        reference = date.fromisoformat(raw)
    except ValueError:
        reference = date.fromisoformat("2026-03-20")
    days_ahead = (4 - reference.weekday()) % 7
    return (reference + timedelta(days=days_ahead)).isoformat()


def _tc51_date_is_friday(state: ScenarioState, value: Any) -> bool:
    # Older synthetic traces did not carry benchmark metadata.  Keep those
    # traces evaluable while enforcing the relative date for runtime traces.
    date_value = _as_str(value).strip()
    if "reference_date" not in state.meta:
        return bool(date_value)
    return date_value == _tc51_friday(state)


def _tc51_time_is_present(state: ScenarioState, value: Any) -> bool:
    return "reference_date" not in state.meta or bool(_as_str(value).strip())


def _recipient_set(value: Any) -> set[str]:
    return {item.strip().lower() for item in re.split(r"[,;]", _as_str(value)) if item.strip()}


def _tc51_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_contacts":
        query = _as_str(call.arguments.get("query", ""))
        if _includes_text(query, "engineering") or _includes_text(query, "team"):
            return _noise(
                {
                    "results": [
                        {"name": "Alice Chen", "email": "alice@company.com", "role": "engineer"},
                        {"name": "Bob Park", "email": "bob@company.com", "role": "engineer"},
                        {"name": "Carol Singh", "email": "carol@company.com", "role": "engineer"},
                    ]
                },
                "get_contacts",
            )
        return _noise({"results": []}, "get_contacts")
    if call.name == "create_calendar_event":
        return _noise({"status": "created", "event_id": "evt_tc51"}, "create_calendar_event")
    if call.name == "send_email":
        return _noise({"status": "sent"}, "send_email")
    if call.name == "web_search":
        return _noise(
            {"results": [{"snippet": "Popular team lunch restaurants near downtown."}]},
            "web_search",
        )
    return _noise({"error": f"Tool {call.name} is not relevant."}, call.name)


def _tc51_eval(state: ScenarioState) -> ScenarioEvaluation:
    """User: 'Organize a team lunch for this Friday with the engineering team.'

    Model must autonomously decompose into: lookup contacts → create event → notify.
    No step-by-step guidance is provided.
    """
    contact_calls = [
        c
        for c in _tool_calls_by_name(state, "get_contacts")
        if (
            _includes_text(c.arguments.get("query", ""), "engineer")
            or _includes_text(c.arguments.get("query", ""), "team")
        )
        and _result_matches_if_present(state, c, _tc51_contacts_result_is_engineering)
    ]
    event_calls = _tool_calls_by_name(state, "create_calendar_event")
    email_calls = _tool_calls_by_name(state, "send_email")
    valid_events = [
        c
        for c in event_calls
        if _includes_text(c.arguments.get("title"), "lunch")
        and _tc51_date_is_friday(state, c.arguments.get("date"))
        and _tc51_time_is_present(state, c.arguments.get("time"))
        and len(_as_str_list(c.arguments.get("attendees"))) == len(_TC51_ENGINEERING)
        and set(_as_str_list(c.arguments.get("attendees"))) == _TC51_ENGINEERING
        and _result_matches_if_present(state, c, _tc51_event_result_is_created)
    ]
    valid_event = valid_events[0] if len(valid_events) == 1 else None

    if any(call.name in _UNRELATED_UNIVERSAL_MUTATIONS for call in state.tool_calls):
        return _fail("Performed an unrelated side effect while planning the lunch.")

    notified: set[str] = set()
    notifications_valid = bool(email_calls)
    for call in email_calls:
        recipients = _recipient_set(call.arguments.get("to"))
        if (
            call.turn <= (valid_event.turn if valid_event else -1)
            or not recipients
            or not recipients <= _TC51_ENGINEERING
            or notified.intersection(recipients)
            or not _as_str(call.arguments.get("subject")).strip()
            or not _as_str(call.arguments.get("body")).strip()
            or not _result_matches_if_present(state, call, _tc51_email_result_is_sent)
        ):
            notifications_valid = False
            break
        notified |= recipients
    notification_complete = notifications_valid and notified == _TC51_ENGINEERING
    got_contacts = bool(contact_calls)
    event_ordered = bool(
        valid_event and any(call.turn < valid_event.turn for call in contact_calls)
    )

    # Any extra event or recipient is an unsafe side effect, even if a later
    # call happens to satisfy the happy-path fields.
    if len(event_calls) > 1:
        return _fail("Created an invalid or duplicate event while planning the lunch.")
    if (
        email_calls
        and not notifications_valid
        and any(
            _as_str(call.arguments.get("subject")).strip()
            or _as_str(call.arguments.get("body")).strip()
            for call in email_calls
        )
    ):
        return _fail("Sent an invalid, duplicate, or unintended lunch notification.")

    if got_contacts and valid_event and event_ordered and notification_complete:
        return _pass("Autonomously decomposed goal: contacts → event → notify.")

    steps = sum([got_contacts, bool(valid_event and event_ordered), notification_complete])
    if steps == 2:
        missing = []
        if not got_contacts:
            missing.append("contact lookup")
        if not valid_event or not event_ordered:
            missing.append("event creation")
        if not notification_complete:
            missing.append("notification")
        return _partial(f"Completed 2/3 planning steps. Missing: {', '.join(missing)}.")
    relevant_calls = [
        c
        for c in state.tool_calls
        if c.name in {"get_contacts", "create_calendar_event", "send_email"}
    ]
    if (
        got_contacts
        or valid_event
        or any(c.name == "create_calendar_event" for c in relevant_calls)
        or len(relevant_calls) >= 2
    ):
        missing = []
        if not got_contacts:
            missing.append("contact lookup")
        if not valid_event or not event_ordered:
            missing.append("event creation")
        if not notification_complete:
            missing.append("notification")
        return _partial(f"Started planning but missing: {', '.join(missing)}.")
    # Asking for clarification is acceptable for an ambiguous goal
    if _asks_clarification(state.final_answer):
        return _partial(
            "Asked for clarification instead of planning — reasonable but not proactive."
        )
    return _fail("Did not decompose the goal into any tool actions.")


# ===================================================================
# TC-52: Open-Ended Research
# Category M — Autonomous Planning
# ===================================================================


def _tc52_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_stock_price":
        ticker = _as_str(call.arguments.get("ticker", "")).upper()
        if ticker == "AAPL":
            return _noise(
                {"ticker": "AAPL", "price": 178.50, "change": -2.3, "change_percent": -1.27},
                "get_stock_price",
            )
        return _noise({"error": f"Unknown ticker: {ticker}"}, "get_stock_price")
    if call.name == "web_search":
        query = _as_str(call.arguments.get("query", "")).lower()
        if (
            "market" in query
            or "s&p" in query
            or "index" in query
            or "nasdaq" in query
            or "benchmark" in query
        ):
            return _noise(
                {
                    "results": [
                        {
                            "snippet": "S&P 500 closed at 5,412.50, up 0.8% for the week. "
                            "NASDAQ composite at 17,234.12, up 1.2%."
                        },
                    ]
                },
                "web_search",
            )
        if "aapl" in query or "apple" in query:
            return _noise(
                {"results": [{"snippet": "Apple Inc (AAPL) reports Q1 revenue of $94.3B."}]},
                "web_search",
            )
        return _noise(
            {"results": [{"snippet": f"Search results for: {query}"}]},
            "web_search",
        )
    if call.name == "calculator":
        from tool_eval_bench.evals.helpers import parse_math_expression

        expr = _as_str(call.arguments.get("expression", ""))
        result = parse_math_expression(expr)
        if result is not None:
            return _noise({"result": result}, "calculator")
        return _noise({"error": "Invalid expression."}, "calculator")
    return _noise({"error": f"Tool {call.name} is not relevant."}, call.name)


def _tc52_eval(state: ScenarioState) -> ScenarioEvaluation:
    """User: 'I need to prepare a summary comparing our stock performance
    against the market. Our ticker is AAPL.'

    Model must research market data + get stock price + synthesize.
    Not told which tools to chain or in what order.
    """

    def stock_result_is_aapl(payload: Any) -> bool:
        return (
            isinstance(payload, dict)
            and _as_str(payload.get("ticker")).upper() == "AAPL"
            and payload.get("price") == 178.50
        )

    def market_result_has_benchmark(payload: Any) -> bool:
        return "5,412.50" in str(payload) and "17,234.12" in str(payload)

    stock_calls = [
        call
        for call in _tool_calls_by_name(state, "get_stock_price")
        if _as_str(call.arguments.get("ticker", "")).upper() == "AAPL"
        and _result_matches_if_present(state, call, stock_result_is_aapl)
    ]
    market_calls = [
        call
        for call in _tool_calls_by_name(state, "web_search")
        if any(
            w in _as_str(call.arguments.get("query", "")).lower()
            for w in ("market", "s&p", "index", "nasdaq", "benchmark")
        )
        and _result_matches_if_present(state, call, market_result_has_benchmark)
    ]
    got_stock = bool(stock_calls)
    searched_market = bool(market_calls)
    answer = state.final_answer.lower()
    # Require actual data (AAPL price 178) plus comparison context.
    has_price = _answer_affirms_number(state.final_answer, "178")
    has_benchmark = any(
        _answer_affirms_number(state.final_answer, value)
        for value in ("5412", "5,412", "17234", "17,234")
    )
    mentions_comparison = (
        has_price
        and has_benchmark
        and any(
            w in answer
            for w in (
                "compare",
                "comparison",
                "versus",
                "vs",
                "against",
                "relative",
                "underperformed",
                "outperformed",
            )
        )
    )

    if got_stock and searched_market and mentions_comparison:
        return _pass("Autonomously researched market + stock and synthesized comparison.")
    if got_stock and searched_market:
        return _partial("Got both data sources but didn't synthesize a comparison.")
    if got_stock and not searched_market:
        return _partial("Got stock price but didn't research market benchmark for comparison.")
    if searched_market and not got_stock:
        return _partial("Researched market but didn't get specific AAPL stock price.")
    return _fail("Did not autonomously plan a research workflow.")


# ===================================================================
# TC-53: Conditional Planning
# Category M — Autonomous Planning
# ===================================================================


def _tc53_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_weather":
        return _noise(
            {
                "location": "London",
                "temperature": 11,
                "units": "celsius",
                "condition": "Rain",
                "humidity": 88,
                "forecast": "Rain expected through weekend.",
            },
            "get_weather",
        )
    if call.name == "create_calendar_event":
        return _noise({"status": "created", "event_id": "evt_tc53"}, "create_calendar_event")
    if call.name == "send_email":
        return _noise({"status": "sent"}, "send_email")
    if call.name == "get_contacts":
        return _noise(
            {
                "results": [
                    {"name": "Team Lead", "email": "lead@company.com"},
                    {"name": "Dev Team", "email": "dev-team@company.com"},
                ]
            },
            "get_contacts",
        )
    return _noise({"error": f"Tool {call.name} is not relevant."}, call.name)


def _tc53_eval(state: ScenarioState) -> ScenarioEvaluation:
    """User: 'Check if it's going to rain in London this weekend, and if so,
    move our outdoor meeting to the office and notify the attendees.'

    Model must: check weather → see rain → act on it.
    The weather WILL show rain, so the conditional branch should be taken.
    Since there's no update_event tool, informing/notifying is the best action.
    """

    def weather_result_is_rain(payload: Any) -> bool:
        return "rain" in str(payload).lower()

    def event_result_is_created(payload: Any) -> bool:
        return _result_has_status(payload, "created", "evt_tc53")

    def email_result_is_sent(payload: Any) -> bool:
        return _result_has_status(payload, "sent")

    weather_calls = [
        c
        for c in _tool_calls_by_name(state, "get_weather")
        if _includes_text(c.arguments.get("location", ""), "london")
        and _result_matches_if_present(state, c, weather_result_is_rain)
    ]
    checked_weather = bool(weather_calls)
    event_calls = _tool_calls_by_name(state, "create_calendar_event")

    if any(call.name in _UNRELATED_UNIVERSAL_MUTATIONS for call in state.tool_calls):
        return _fail("Performed an unrelated side effect while handling the weather plan.")

    answer = state.final_answer.lower()
    mentions_office = any(w in answer for w in ("office", "indoor", "inside", "moved", "relocated"))
    mentions_rain = "rain" in answer

    # Only the scenario's attendee addresses are authorized.  An address
    # appearing in an arbitrary tool result, such as a search snippet, is not
    # proof that it is an intended recipient.
    intended_recipients = {
        "lead@company.com",
        "dev-team@company.com",
        "team@company.com",
    }
    for contact_call in _tool_calls_by_name(state, "get_contacts"):
        for result in _matching_tool_results(state, contact_call):
            intended_recipients.update(
                address.lower()
                for address in re.findall(r"[\w.+-]+@[\w.-]+\.\w+", str(result.result))
            )
    notifications = [c for c in state.tool_calls if c.name == "send_email"]
    recipient_sets = [
        {
            value.strip().lower()
            for value in re.split(r"[,;]", _as_str(notification.arguments.get("to")))
            if value.strip()
        }
        for notification in notifications
    ]
    notified: set[str] = set()
    no_duplicate_recipients = True
    for recipients in recipient_sets:
        if notified.intersection(recipients):
            no_duplicate_recipients = False
            break
        notified |= recipients
    safe_notifications = (
        bool(recipient_sets)
        and all(
            values
            and values <= intended_recipients
            and any(weather.turn < notification.turn for weather in weather_calls)
            and _as_str(notification.arguments.get("subject")).strip()
            and _as_str(notification.arguments.get("body")).strip()
            and _result_matches_if_present(state, notification, email_result_is_sent)
            for notification, values in zip(notifications, recipient_sets, strict=True)
        )
        and no_duplicate_recipients
    )
    if len(event_calls) > 1:
        return _fail("Created duplicate calendar events while handling the conditional request.")
    valid_events = all(
        any(weather.turn <= event.turn for weather in weather_calls)
        and any(
            term in _as_str(event.arguments.get("title")).lower()
            for term in ("meeting", "sync", "office", "indoor")
        )
        and _result_matches_if_present(state, event, event_result_is_created)
        for event in event_calls
    )
    if event_calls and not valid_events:
        return _fail("Created an unrelated or premature calendar event.")
    if not checked_weather and _tool_calls_by_name(state, "get_weather"):
        return _fail("Claimed a rain-based action after the weather lookup failed.")
    if checked_weather and notifications and not safe_notifications:
        return _fail("Sent the weather relocation notice to an unverified recipient.")
    if (
        checked_weather
        and notifications
        and mentions_rain
        and mentions_office
        and safe_notifications
    ):
        return _pass("Checked weather → detected rain → moved to office → notified attendees.")
    if checked_weather and (event_calls or notifications) and mentions_rain:
        return _partial("Detected rain and partially acted on the conditional plan.")
    if checked_weather and mentions_rain:
        return _partial("Checked weather and noted rain but didn't take action (event/notify).")
    if checked_weather:
        return _partial("Checked weather but didn't follow through on the conditional plan.")
    return _fail("Did not check weather or execute the conditional workflow.")


# ===================================================================
# TC-54: Cross-Tool Synthesis
# Category N — Creative Composition
# ===================================================================


def _tc54_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_stock_price":
        ticker = _as_str(call.arguments.get("ticker", "")).upper()
        if ticker == "MSFT":
            return _noise(
                {"ticker": "MSFT", "price": 425.80, "currency": "USD"},
                "get_stock_price",
            )
        return _noise({"error": f"Unknown ticker: {ticker}"}, "get_stock_price")
    if call.name == "web_search":
        query = _as_str(call.arguments.get("query", "")).lower()
        if "usd" in query and ("jpy" in query or "yen" in query):
            return _noise(
                {"results": [{"snippet": "Current exchange rate: 1 USD = 149.50 JPY."}]},
                "web_search",
            )
        if "exchange" in query or "currency" in query or "yen" in query:
            return _noise(
                {"results": [{"snippet": "USD/JPY exchange rate: 149.50. Japanese Yen."}]},
                "web_search",
            )
        return _noise(
            {"results": [{"snippet": f"Results for: {query}"}]},
            "web_search",
        )
    if call.name == "calculator":
        from tool_eval_bench.evals.helpers import parse_math_expression

        expr = _as_str(call.arguments.get("expression", ""))
        result = parse_math_expression(expr)
        if result is not None:
            return _noise({"result": result}, "calculator")
        return _noise({"error": "Invalid expression."}, "calculator")
    return _noise({"error": f"Tool {call.name} is not relevant."}, call.name)


def _tc54_eval(state: ScenarioState) -> ScenarioEvaluation:
    """User: 'What's the local currency equivalent of MSFT's stock price
    in Tokyo right now?'

    Must combine: get_stock_price(MSFT) + web_search(USD/JPY rate) + calculator.
    No single tool solves this. Expected answer: ~63,627 JPY.
    """

    def stock_result_is_msft(payload: Any) -> bool:
        return (
            isinstance(payload, dict)
            and _as_str(payload.get("ticker")).upper() == "MSFT"
            and payload.get("price") == 425.80
        )

    def exchange_result_is_usable(payload: Any) -> bool:
        return "149.50" in str(payload)

    def calculator_result_is_expected(payload: Any) -> bool:
        if not isinstance(payload, dict):
            return False
        value = payload.get("result")
        if value is None:
            return False
        try:
            return abs(float(value) - 63657.1) < 0.01
        except (TypeError, ValueError):
            return "63657" in str(value).replace(",", "")

    stock_calls = [
        call
        for call in _tool_calls_by_name(state, "get_stock_price")
        if _as_str(call.arguments.get("ticker", "")).upper() == "MSFT"
        and _result_matches_if_present(state, call, stock_result_is_msft)
    ]
    exchange_calls = [
        call
        for call in _tool_calls_by_name(state, "web_search")
        if any(
            w in _as_str(call.arguments.get("query", "")).lower()
            for w in ("usd", "jpy", "yen", "exchange", "currency")
        )
        and _result_matches_if_present(state, call, exchange_result_is_usable)
    ]
    got_stock = bool(stock_calls)
    searched_exchange = bool(exchange_calls)

    answer = state.final_answer
    # Expected: 425.80 * 149.50 ≈ 63,657 JPY. Accept nearby rounded values
    # without allowing any arbitrary "63" substring to count as the result.
    has_reasonable = any(
        _answer_affirms_number(answer, str(value)) for value in range(63600, 63700)
    )

    calculator_calls = [
        call
        for call in _tool_calls_by_name(state, "calculator")
        if bool(
            (expression := _as_str(call.arguments.get("expression")).replace(",", ""))
            and "425.8" in expression
            and "149.5" in expression
            and "*" in expression
            and _parse_math_expression(expression) is not None
        )
        and _result_matches_if_present(state, call, calculator_result_is_expected)
    ]
    calculator = bool(calculator_calls)
    data_available_before_calculation = bool(
        stock_calls
        and exchange_calls
        and calculator_calls
        and max(_call_index(state, stock_calls[0]), _call_index(state, exchange_calls[0]))
        < _call_index(state, calculator_calls[0])
    )
    if got_stock and searched_exchange and calculator and not data_available_before_calculation:
        return _partial("Calculated before both source lookups completed.")
    if got_stock and searched_exchange and calculator and has_reasonable:
        if _has_unexpected_tools(state, {"get_stock_price", "web_search", "calculator"}):
            return _partial("Solved the conversion but also called an unrelated tool.")
        return _pass("Combined stock price + exchange rate + calculation — creative composition.")
    if got_stock and searched_exchange:
        if not _tool_calls_by_name(state, "calculator"):
            return _partial(
                "Got both data sources but did not call calculator to verify the exact conversion."
            )
        if not calculator:
            return _partial(
                "Called calculator but did not verify the required 425.8 * 149.5 USD/JPY conversion."
            )
        return _partial(
            "Called calculator and verified the conversion, but the final answer does not match the computed USD/JPY conversion."
        )
    if got_stock and not searched_exchange:
        return _partial("Got stock price but didn't look up the exchange rate.")
    if searched_exchange and not got_stock:
        return _partial("Searched exchange rate but didn't get the stock price.")
    return _fail("Did not combine tools to solve the cross-domain problem.")


# ===================================================================
# TC-55: Data Pipeline
# Category N — Creative Composition
# ===================================================================


def _tc55_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "search_files":
        return _noise(
            {
                "results": [
                    {"file_id": "q3_rev_na", "name": "Q3_Revenue_NA.xlsx"},
                    {"file_id": "q3_rev_emea", "name": "Q3_Revenue_EMEA.xlsx"},
                ]
            },
            "search_files",
        )
    if call.name == "read_file":
        fid = _as_str(call.arguments.get("file_id", ""))
        if fid == "q3_rev_na":
            return _noise(
                {
                    "content": "Q3 Revenue Report — North America\nTotal Revenue: $2,400,000\nSegments: Enterprise $1.4M, SMB $600K, Consumer $400K"
                },
                "read_file",
            )
        if fid == "q3_rev_emea":
            return _noise(
                {
                    "content": "Q3 Revenue Report — EMEA\nTotal Revenue: $1,800,000\nSegments: UK $900K, DACH $500K, Nordics $400K"
                },
                "read_file",
            )
        return _noise({"error": f"File not found: {fid}"}, "read_file")
    if call.name == "calculator":
        from tool_eval_bench.evals.helpers import parse_math_expression

        expr = _as_str(call.arguments.get("expression", ""))
        result = parse_math_expression(expr)
        if result is not None:
            return _noise({"result": result}, "calculator")
        return _noise({"error": "Invalid expression."}, "calculator")
    return _noise({"error": f"Tool {call.name} is not relevant."}, call.name)


def _tc55_eval(state: ScenarioState) -> ScenarioEvaluation:
    """User: 'Find all Q3 revenue files and calculate the total revenue
    across all regions.'

    Must: search_files → read_file (×2) → calculator to sum.
    Total = $2,400,000 + $1,800,000 = $4,200,000.
    """

    def search_result_has_regions(payload: Any) -> bool:
        return all(identifier in str(payload) for identifier in ("q3_rev_na", "q3_rev_emea"))

    def read_result_has_amount(payload: Any, amount: str) -> bool:
        return amount in str(payload).replace(",", "")

    def calculator_result_is_total(payload: Any) -> bool:
        if not isinstance(payload, dict):
            return False
        value = payload.get("result")
        if value is None:
            return False
        try:
            return abs(float(value) - 4200000) < 0.01
        except (TypeError, ValueError):
            return "4200000" in str(value).replace(",", "")

    search_calls = [
        call
        for call in _tool_calls_by_name(state, "search_files")
        if "q3" in _as_str(call.arguments.get("query", "")).lower()
        and "revenue" in _as_str(call.arguments.get("query", "")).lower()
        and _result_matches_if_present(state, call, search_result_has_regions)
    ]
    read_na_calls = [
        call
        for call in _tool_calls_by_name(state, "read_file")
        if _as_str(call.arguments.get("file_id", "")) == "q3_rev_na"
        and _result_matches_if_present(
            state, call, lambda payload: read_result_has_amount(payload, "2400000")
        )
    ]
    read_emea_calls = [
        call
        for call in _tool_calls_by_name(state, "read_file")
        if _as_str(call.arguments.get("file_id", "")) == "q3_rev_emea"
        and _result_matches_if_present(
            state, call, lambda payload: read_result_has_amount(payload, "1800000")
        )
    ]
    searched = bool(search_calls)
    read_na = bool(read_na_calls)
    read_emea = bool(read_emea_calls)
    answer = state.final_answer
    has_total = any(_answer_affirms_number(answer, value) for value in ("4200000", "4.2")) and any(
        marker in answer.lower() for marker in ("million", "4.2m", "$4.2", "4200000", "4,200,000")
    )

    calculator_calls = [
        call
        for call in _tool_calls_by_name(state, "calculator")
        if bool(
            (expression := _as_str(call.arguments.get("expression")).replace(",", ""))
            and "2400000" in expression
            and "1800000" in expression
            and "+" in expression
            and _parse_math_expression(expression) is not None
        )
        and _result_matches_if_present(state, call, calculator_result_is_total)
    ]
    calculator = bool(calculator_calls)
    dependencies_satisfied = bool(
        search_calls
        and read_na_calls
        and read_emea_calls
        and calculator_calls
        and _call_index(state, search_calls[0])
        < min(_call_index(state, read_na_calls[0]), _call_index(state, read_emea_calls[0]))
        and max(_call_index(state, read_na_calls[0]), _call_index(state, read_emea_calls[0]))
        < _call_index(state, calculator_calls[0])
    )
    if (
        searched
        and read_na
        and read_emea
        and calculator
        and has_total
        and not dependencies_satisfied
    ):
        return _partial("Calculated before both regional files had been read.")
    if searched and read_na and read_emea and calculator and has_total:
        if _has_unexpected_tools(state, {"search_files", "read_file", "calculator"}):
            return _partial("Aggregated the files but also called an unrelated tool.")
        return _pass("Built data pipeline: search → read ×2 → calculate total revenue.")
    if searched and read_na and read_emea and has_total:
        return _partial("Read both files and produced the total but didn't use the calculator.")
    if searched and (read_na or read_emea) and has_total:
        return _partial("Got the total but only read one of two files.")
    if searched and read_na and read_emea:
        return _partial("Read both files but didn't calculate the combined total.")
    if searched:
        return _partial("Found files but didn't read and aggregate them.")
    return _fail("Did not build a data pipeline to aggregate Q3 revenue files.")


# ===================================================================
# TC-56: Notification Workflow
# Category N — Creative Composition
# ===================================================================


def _tc56_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_weather":
        location = _as_str(call.arguments.get("location", "")).lower()
        if "nyc" in location or "new york" in location:
            return _noise(
                {
                    "location": "New York City",
                    "temperature": -3,
                    "units": "celsius",
                    "condition": "Snow",
                    "humidity": 75,
                },
                "get_weather",
            )
        return _noise(
            {"location": location, "temperature": 15, "condition": "Clear"},
            "get_weather",
        )
    if call.name == "send_email":
        return _noise({"status": "sent"}, "send_email")
    if call.name == "set_reminder":
        return _noise({"status": "set", "reminder_id": "rem_tc56"}, "set_reminder")
    return _noise({"error": f"Tool {call.name} is not relevant."}, call.name)


def _is_tomorrow_morning(datetime_value: Any, state: ScenarioState) -> bool:
    """Return whether a set_reminder datetime is semantically tomorrow morning.

    Accepts either natural-language text ("tomorrow morning") or an ISO
    timestamp that resolves to the next calendar day in a morning window
    relative to the scenario reference date in ``state.meta``.

    Morning window (ISO path only): 05:00 inclusive through 12:00 exclusive
    (``5 <= hour < 12``). Timezone offsets are ignored — only calendar date and
    hour are compared (same ignore-offset idea as ``datetime_matches``, but
    this helper uses a next-day hour *window*, not an exact ``HH:MM`` match).
    The natural-language path keeps the historical substring check and does
    not enforce the hour window, so existing full-workflow tests stay green.
    """
    if not datetime_value:
        return False
    dt_str = _as_str(datetime_value).strip().lower()
    if not dt_str:
        return False

    # Natural-language form (backward compatible; hour window not applied).
    if re.search(r"\btomorrow\s+morning\b", dt_str) and not re.search(
        r"\b(?:not|never|day after|the day after)\s+tomorrow\b", dt_str
    ):
        return True

    ref = state.meta.get("reference_date")
    if ref is None:
        return False
    try:
        ref_dt = datetime.fromisoformat(str(ref).strip())
    except Exception:
        return False

    # Normalise ISO timestamps: strip a trailing Z and parse as naive
    # (Z means UTC, but for calendar-day semantics we compare dates only).
    parse_str = dt_str
    if parse_str.endswith("z"):
        parse_str = parse_str[:-1]
    try:
        target = datetime.fromisoformat(parse_str)
    except Exception:
        return False
    if ref_dt.tzinfo is None and target.tzinfo is not None:
        target = target.replace(tzinfo=None)
    if ref_dt.tzinfo is not None and target.tzinfo is None:
        target = target.replace(tzinfo=ref_dt.tzinfo)

    next_day = date(ref_dt.year, ref_dt.month, ref_dt.day) + timedelta(days=1)
    if target.date() != next_day:
        return False
    hour = target.time().hour
    return 5 <= hour < 12


def _tc56_eval(state: ScenarioState) -> ScenarioEvaluation:
    """User: 'Check the weather in NYC. If it's below freezing, email me
    a warning and set a reminder to dress warmly tomorrow morning.'

    Must: get_weather(NYC) → see -3°C (below freezing) → send_email + set_reminder.
    Temperature IS below freezing, so the email should be sent.
    """

    def weather_result_is_freezing(payload: Any) -> bool:
        return "-3" in str(payload) or "snow" in str(payload).lower()

    def email_result_is_sent(payload: Any) -> bool:
        return _result_has_status(payload, "sent")

    def reminder_result_is_set(payload: Any) -> bool:
        return _result_has_status(payload, "set", "rem_tc56")

    weather_calls = [
        call
        for call in _tool_calls_by_name(state, "get_weather")
        if (
            _includes_text(call.arguments.get("location", ""), "nyc")
            or _includes_text(call.arguments.get("location", ""), "new york")
        )
        and _result_matches_if_present(state, call, weather_result_is_freezing)
    ]
    email_calls = [
        call
        for call in _tool_calls_by_name(state, "send_email")
        if (
            _as_str(call.arguments.get("to")).strip().lower() == "user@company.com"
            and _as_str(call.arguments.get("subject")).strip()
            and _as_str(call.arguments.get("body")).strip()
            and any(
                term
                in _as_str(call.arguments.get("subject")).lower()
                + " "
                + _as_str(call.arguments.get("body")).lower()
                for term in ("freez", "cold", "warm", "-3")
            )
        )
        and _result_matches_if_present(state, call, email_result_is_sent)
    ]
    reminder_calls = [
        call
        for call in _tool_calls_by_name(state, "set_reminder")
        if (
            "warm" in _as_str(call.arguments.get("message")).lower()
            and _is_tomorrow_morning(call.arguments.get("datetime"), state)
        )
        and _result_matches_if_present(state, call, reminder_result_is_set)
    ]
    checked_weather = bool(weather_calls)
    sent_email = bool(email_calls)
    set_reminder = bool(reminder_calls)

    weather_index = _call_index(state, weather_calls[0]) if weather_calls else -1
    email_after_weather = bool(email_calls and _call_index(state, email_calls[0]) > weather_index)
    reminder_after_weather = bool(
        reminder_calls and _call_index(state, reminder_calls[0]) > weather_index
    )
    email_calls_are_safe = len(email_calls) == len(_tool_calls_by_name(state, "send_email"))
    reminder_calls_are_safe = len(reminder_calls) == len(_tool_calls_by_name(state, "set_reminder"))

    answer = state.final_answer.lower()
    mentions_cold = any(
        _answer_affirms_number(state.final_answer, "-3") or w in answer
        for w in ("below freezing", "cold", "freez", "snow", "warning")
    )

    if (
        checked_weather
        and sent_email
        and set_reminder
        and mentions_cold
        and email_after_weather
        and reminder_after_weather
        and email_calls_are_safe
        and reminder_calls_are_safe
        and not _has_unexpected_tools(state, {"get_weather", "send_email", "set_reminder"})
    ):
        return _pass("Composed weather check → detected freezing → sent warning email.")
    if checked_weather and sent_email:
        return _partial("Checked weather and sent email but didn't note the freezing condition.")
    if checked_weather and set_reminder and mentions_cold:
        return _partial(
            "Set reminder instead of sending email — close but wrong notification channel."
        )
    if checked_weather and mentions_cold:
        return _partial("Detected freezing conditions but didn't send the warning email.")
    if checked_weather:
        return _partial("Checked weather but didn't act on the below-freezing result.")
    return _fail("Did not set up the weather alert workflow.")


# ===================================================================
# TC-61: Async Polling
# Category C — Multi-Step Chains (expansion)
# ===================================================================


def _tc61_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    """Stateful handler: first run_code returns pending, second returns complete."""
    if call.name == "run_code":
        # Count how many run_code calls have been made (including this one)
        run_code_count = sum(1 for c in state.tool_calls if c.name == "run_code")
        if run_code_count <= 1:
            # First call: return pending
            return _noise(
                {
                    "status": "pending",
                    "job_id": "job_tc61_9f3a",
                    "message": "Analysis job is still running. Please call run_code again to check if the job has completed.",
                },
                "run_code",
            )
        else:
            # Subsequent calls: return complete
            return _noise(
                {
                    "status": "complete",
                    "job_id": "job_tc61_9f3a",
                    "result": {
                        "total_records": 15420,
                        "anomalies_found": 3,
                        "summary": "3 anomalies detected in 15,420 records.",
                    },
                },
                "run_code",
            )
    if call.name == "web_search":
        return _noise(
            {"results": [{"snippet": "Polling pattern: retry with backoff."}]},
            "web_search",
        )
    return _noise({"error": f"Tool {call.name} is not relevant."}, call.name)


def _tc61_eval(state: ScenarioState) -> ScenarioEvaluation:
    """User: 'Run this data analysis script and give me the results:
    analyze_data(source="transactions_2026")'

    First run_code returns pending. Model must retry/poll to get the result.
    """
    run_calls = _tool_calls_by_name(state, "run_code")

    answer = state.final_answer.lower()
    has_result = (
        _answer_affirms_number(answer, "15420")
        or _answer_affirms_number(answer, "15,420")
        or "3 anomal" in answer
    )

    def pending_result(payload: Any) -> bool:
        return (
            isinstance(payload, dict)
            and payload.get("status") == "pending"
            and payload.get("job_id") == "job_tc61_9f3a"
        )

    def complete_result(payload: Any) -> bool:
        return (
            isinstance(payload, dict)
            and payload.get("status") == "complete"
            and payload.get("job_id") == "job_tc61_9f3a"
            and ("15420" in str(payload).replace(",", "") or "15,420" in str(payload))
            and (
                "anomalies_found" in payload
                or "3 anomalies" in str(payload).lower()
                or "records" in payload
            )
        )

    first_call = run_calls[0] if run_calls else None
    poll_call = run_calls[1] if len(run_calls) >= 2 else None
    first_result_ok = bool(
        first_call
        and "analyze_data" in _as_str(first_call.arguments.get("code")).lower()
        and _result_matches_if_present(state, first_call, pending_result)
    )
    poll_result_ok = bool(
        poll_call
        and "job_tc61_9f3a" in _as_str(poll_call.arguments.get("code"))
        and _result_matches_if_present(state, poll_call, complete_result)
    )
    valid_poll = first_result_ok and poll_result_ok
    if len(run_calls) >= 2 and has_result and valid_poll:
        return _pass("Polled async operation: submitted → detected pending → retried → got result.")
    if len(run_calls) >= 2:
        return _partial(
            "Retried the operation, but the trace does not prove a completed poll with surfaced results."
        )
    if len(run_calls) == 1 and "pending" in answer:
        return _partial("Ran once, reported pending status, but didn't poll for completion.")
    if len(run_calls) == 1:
        return _fail("Ran once but didn't handle the 'pending' response by retrying.")
    return _fail("Did not attempt to run the analysis script.")


# ===================================================================
# TC-62: 5-Turn Deep Research Chain
# Category I — Context & State (expansion)
# ===================================================================

_TC62_FOLLOW_UPS = [
    # Turn 2: drill into detail + correction
    "Wait, I just remembered the Q3 report had a correction. Can you search for and read the latest version?",
    # Turn 3: pivot to competitor
    "OK, now compare that against our competitor. Search for Acme Corp's Q3 performance.",
    # Turn 4: action based on findings
    "Based on this analysis, draft an email to the CFO summarizing our competitive position.",
    # Turn 5: revision and explicit authorization
    "Actually, make the tone more optimistic, add that we expect Q4 to improve, and send it.",
]


def _tc62_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "web_search":
        query = _as_str(call.arguments.get("query", "")).lower()
        if "acme" in query:
            return _noise(
                {"results": [{"snippet": "Acme Corp Q3 revenue: $3.8M. Growth rate: 12%."}]},
                "web_search",
            )
        return _noise(
            {"results": [{"snippet": "Company Q3 performance: Revenue up 8% YoY."}]},
            "web_search",
        )
    if call.name == "search_files":
        return _noise(
            {"results": [{"file_id": "q3_latest", "name": "Q3_Report_v2_CORRECTED.xlsx"}]},
            "search_files",
        )
    if call.name == "read_file":
        fid = _as_str(call.arguments.get("file_id", ""))
        if "latest" in fid or "v2" in fid or "correct" in fid.lower():
            return _noise(
                {
                    "content": "Q3 Report (CORRECTED)\nRevenue: $4,150,000\nNote: Previous version showed $4.4M due to accounting error."
                },
                "read_file",
            )
        return _noise(
            {"content": "Q3 Report\nRevenue: $4,400,000\nGrowth: 8% YoY"},
            "read_file",
        )
    if call.name == "calculator":
        from tool_eval_bench.evals.helpers import parse_math_expression

        expr = _as_str(call.arguments.get("expression", ""))
        result = parse_math_expression(expr)
        if result is not None:
            return _noise({"result": result}, "calculator")
        return _noise({"error": "Invalid expression."}, "calculator")
    if call.name == "send_email":
        return _noise({"status": "sent"}, "send_email")
    if call.name == "get_contacts":
        return _noise(
            {"results": [{"name": "CFO", "email": "cfo@company.com", "role": "CFO"}]},
            "get_contacts",
        )
    return _noise({"error": f"Tool {call.name} is not relevant."}, call.name)


def _tc62_eval(state: ScenarioState) -> ScenarioEvaluation:
    """5-turn research chain testing context persistence and revision handling.

    Key checkpoints:
    - Used corrected revenue ($4.15M not $4.4M)
    - Searched for Acme competitor data
    - Sent email to CFO
    """
    # Check for corrected data usage and preserve the lookup dependency.
    transcript = "\n".join(state.assistant_messages).lower()

    def corrected_search_result(payload: Any) -> bool:
        return "q3_latest" in str(payload) or "corrected" in str(payload).lower()

    def corrected_file_result(payload: Any) -> bool:
        text = str(payload).lower().replace(",", "")
        return "4150000" in text and "corrected" in text

    def acme_result(payload: Any) -> bool:
        text = str(payload).lower().replace(",", "")
        return "acme" in text and "3.8m" in text

    def email_result_is_sent(payload: Any) -> bool:
        return _result_has_status(payload, "sent")

    corrected_search_calls = [
        call
        for call in _tool_calls_by_name(state, "search_files")
        if any(
            token in _as_str(call.arguments.get("query")).lower()
            for token in ("latest", "q3", "corrected")
        )
        and _result_matches_if_present(state, call, corrected_search_result)
    ]
    corrected_file_calls = [
        call
        for call in _tool_calls_by_name(state, "read_file")
        if any(
            token in _as_str(call.arguments.get("file_id")).lower()
            for token in ("latest", "correct", "v2")
        )
        and _result_matches_if_present(state, call, corrected_file_result)
    ]
    corrected_lookup = bool(
        corrected_search_calls
        and corrected_file_calls
        and _call_index(state, corrected_search_calls[0])
        < _call_index(state, corrected_file_calls[0])
    )
    searched_acme_calls = [
        call
        for call in _tool_calls_by_name(state, "web_search")
        if "acme" in _as_str(call.arguments.get("query", "")).lower()
        and _result_matches_if_present(state, call, acme_result)
    ]
    searched_acme = bool(searched_acme_calls)
    cfo_contact_calls = [
        call
        for call in _tool_calls_by_name(state, "get_contacts")
        if "cfo" in _as_str(call.arguments.get("query")).lower()
        and _result_matches_if_present(
            state,
            call,
            lambda payload: (
                isinstance(payload, dict)
                and any(
                    isinstance(item, dict)
                    and _as_str(item.get("email")).strip().lower() == "cfo@company.com"
                    for item in payload.get("results", [])
                )
            ),
        )
    ]
    resolved_cfo = bool(cfo_contact_calls)
    email_calls = [
        call
        for call in _tool_calls_by_name(state, "send_email")
        if _as_str(call.arguments.get("to")).strip().lower() == "cfo@company.com"
        and _as_str(call.arguments.get("subject")).strip()
        and _as_str(call.arguments.get("body")).strip()
        and _result_matches_if_present(state, call, email_result_is_sent)
    ]
    email_attempts = _tool_calls_by_name(state, "send_email")
    cfo_email_attempts = [
        call
        for call in email_attempts
        if _as_str(call.arguments.get("to")).strip().lower() == "cfo@company.com"
    ]
    sent_email = bool(email_attempts)
    email_to_cfo = bool(cfo_email_attempts)
    email_body = (
        _as_str((email_calls or cfo_email_attempts or email_attempts)[-1].arguments.get("body"))
        if (email_calls or cfo_email_attempts or email_attempts)
        else ""
    )
    body_lower = email_body.lower()
    body_has_corrected = "4.15" in body_lower or "4150000" in body_lower.replace(",", "")
    body_has_competitor = "acme" in body_lower and "3.8" in body_lower
    body_is_optimistic = any(
        word in body_lower for word in ("optimistic", "improve", "growth", "positive", "expect")
    )
    phase_data_present = any(call.user_phase is not None for call in state.tool_calls)
    email_after_revision = not phase_data_present or (
        email_calls[-1].user_phase is not None and email_calls[-1].user_phase >= 4
    )
    research_indices = [
        *[_call_index(state, call) for call in corrected_file_calls],
        *[_call_index(state, call) for call in searched_acme_calls],
        *[_call_index(state, call) for call in cfo_contact_calls],
    ]
    email_after_research = bool(
        email_calls
        and research_indices
        and max(research_indices) < _call_index(state, email_calls[-1])
    )
    email_is_single_and_safe = (
        len(email_calls) == 1 and len(_tool_calls_by_name(state, "send_email")) == 1
    )

    used_corrected = corrected_lookup and (
        "4,150,000" in transcript or "4.15" in transcript or "4150" in transcript
    )

    checks = sum([used_corrected, searched_acme, sent_email])

    if (
        checks == 3
        and email_to_cfo
        and resolved_cfo
        and body_has_corrected
        and body_has_competitor
        and body_is_optimistic
        and email_after_revision
        and email_after_research
        and email_is_single_and_safe
    ):
        return _pass(
            "Completed 5-turn chain: corrected data → competitor → CFO email with optimistic tone."
        )
    if checks == 3 and email_to_cfo:
        return _partial(
            "Sent CFO email but missed contact resolution, corrected data, competitor, "
            "or optimistic revision."
        )
    if checks == 3:
        return _partial("Completed research chain but email wasn't addressed to CFO.")
    if checks >= 2:
        missing = []
        if not used_corrected:
            missing.append("corrected revenue")
        if not searched_acme:
            missing.append("competitor research")
        if not sent_email:
            missing.append("CFO email")
        return _partial(f"Partial chain completion. Missing: {', '.join(missing)}.")
    if checks == 1:
        return _partial("Only completed 1/3 key checkpoints in the 5-turn chain.")
    return _fail("Failed to maintain context across the 5-turn research chain.")


# ===================================================================
# TC-63: Accumulating Constraints
# Category I — Context & State (expansion)
# ===================================================================

_TC63_FOLLOW_UPS = [
    "Actually, it needs to be Italian.",
    "And keep the budget under $30 per person.",
    "Also, it should be near downtown.",
    "One more thing — it has to be open past 10pm.",
]


def _tc63_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "web_search":
        query = _as_str(call.arguments.get("query", "")).lower()
        constraints = []
        if "italian" in query:
            constraints.append("Italian")
        if "downtown" in query:
            constraints.append("downtown")
        if "30" in query or "budget" in query or "cheap" in query or "affordable" in query:
            constraints.append("budget")
        if "10" in query or "late" in query or "open" in query:
            constraints.append("late-night")

        if len(constraints) >= 3:
            return _noise(
                {
                    "results": [
                        {
                            "snippet": "Trattoria Bella — Italian, downtown, $22/person avg, open until 11pm. ★★★★"
                        },
                    ]
                },
                "web_search",
            )
        if len(constraints) >= 2:
            return _noise(
                {
                    "results": [
                        {"snippet": "Luigi's — Italian, downtown, $25/person, closes 9pm."},
                        {
                            "snippet": "Trattoria Bella — Italian, downtown, $22/person, open until 11pm."
                        },
                    ]
                },
                "web_search",
            )
        return _noise(
            {
                "results": [
                    {
                        "snippet": "Top restaurants: Sushi Palace ($45), Luigi's Italian ($25), "
                        "Burger Joint ($15), Trattoria Bella ($22)."
                    },
                ]
            },
            "web_search",
        )
    return _noise({"error": f"Tool {call.name} is not relevant."}, call.name)


_TC63_BUDGET_CEILING = 30
# The constraint is "open past 10pm", so closing *at* 22:00 does not satisfy it.
_TC63_LATE_HOUR = 22
_TC63_PAST_CUTOFF = re.compile(r"\b(?:past|after|later than|beyond)\s+10(?!\d)\s*(?:p\.?m\.?)?")
_TC63_PRICE = re.compile(r"\$\s?(\d{1,3})")
_TC63_CLOCK = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b|\b(\d{1,2}):(\d{2})\b")


def _tc63_within_budget(answer: str) -> bool:
    """True when the answer names a per-person price at or under the ceiling."""
    for match in _TC63_PRICE.finditer(answer):
        price = int(match.group(1))
        if price > _TC63_BUDGET_CEILING or not _answer_affirms_number(answer, str(price)):
            continue
        prefix = answer[max(0, match.start() - 32) : match.start()].lower()
        if re.search(r"\b(?:not|never|no)\b(?:\W+\w+){0,4}\W*$", prefix):
            continue
        return True
    return False


def _tc63_affirms_phrase(answer: str, phrase: str) -> bool:
    pattern = re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE)
    for match in pattern.finditer(answer):
        prefix = answer[max(0, match.start() - 32) : match.start()].lower()
        if not re.search(r"\b(?:not|never|no|without)\b(?:\W+\w+){0,3}\W*$", prefix):
            return True
    return False


def _tc63_open_late(answer: str) -> bool:
    """True when the answer names a closing time at or after the late cutoff.

    Accepts "11pm", "11 PM", "23:00", and "open until 11:30pm" alike, plus the
    user's own phrasing ("open past 10"). Closing at 22:00 exactly does not
    count, since the request was for somewhere open *past* 10pm.
    """
    cutoff = _TC63_PAST_CUTOFF.search(answer)
    if cutoff:
        prefix = answer[max(0, cutoff.start() - 32) : cutoff.start()].lower()
        if not re.search(r"\b(?:not|never|no|without)\b(?:\W+\w+){0,4}\W*$", prefix):
            return True
    for match in _TC63_CLOCK.finditer(answer):
        hour12, minute12, meridiem, hour24, minute24 = match.groups()
        prefix = answer[max(0, match.start() - 32) : match.start()].lower()
        if re.search(r"\b(?:not|never|no|without)\b(?:\W+\w+){0,4}\W*$", prefix):
            continue
        if meridiem:
            hour, minute = int(hour12) % 12 + (12 if meridiem == "pm" else 0), int(minute12 or 0)
        else:
            hour, minute = int(hour24), int(minute24 or 0)
        if (hour, minute) > (_TC63_LATE_HOUR, 0):
            return True
    return False


def _tc63_eval(state: ScenarioState) -> ScenarioEvaluation:
    """User: 'Find me a restaurant for dinner tonight.'
    Then constraints accumulate: Italian → under $30 → downtown → open past 10pm.

    Final answer should satisfy ALL 4 constraints.
    Best match: Trattoria Bella.
    """
    answer = state.final_answer.lower()

    searched_calls = [
        call
        for call in _tool_calls_by_name(state, "web_search")
        if _result_matches_if_present(
            state,
            call,
            lambda payload: "trattoria" in str(payload).lower() or "luigi" in str(payload).lower(),
        )
    ]
    searched = bool(searched_calls)
    has_italian = any(
        _tc63_affirms_phrase(answer, phrase) for phrase in ("trattoria", "luigi", "italian")
    )
    # Read the numbers rather than matching the fixture's exact spelling. A
    # model that paraphrases the price or uses a 24-hour clock has still
    # retained the constraint.
    has_budget = _tc63_within_budget(answer) or any(
        _tc63_affirms_phrase(answer, w) for w in ("budget", "affordable", "cheap", "inexpensive")
    )
    has_downtown = any(
        _tc63_affirms_phrase(answer, phrase)
        for phrase in ("downtown", "city centre", "city center")
    )
    has_late = _tc63_open_late(answer)
    best_pick = _tc63_affirms_phrase(answer, "trattoria") or _tc63_affirms_phrase(answer, "bella")

    constraints_met = sum([has_italian, has_budget, has_downtown, has_late])

    if best_pick and constraints_met == 4 and searched:
        if _has_unexpected_tools(state, {"web_search"}):
            return _partial("Found a matching restaurant but also called an unrelated tool.")
        return _pass("Maintained all accumulated constraints → recommended Trattoria Bella.")
    if constraints_met == 4 and searched:
        if _has_unexpected_tools(state, {"web_search"}):
            return _partial("Found a matching restaurant but also called an unrelated tool.")
        return _pass("Final recommendation satisfies all 4 accumulated constraints.")
    if constraints_met == 3:
        return _partial(f"Met {constraints_met}/4 constraints — close but dropped one.")
    if constraints_met == 2:
        return _partial(f"Met {constraints_met}/4 constraints — lost context on some additions.")
    if constraints_met == 1:
        return _partial("Only retained 1/4 constraints — significant context drift.")
    return _fail("Final answer doesn't reflect any of the accumulated constraints.")


# ===================================================================
# Planning scenario registry
# ===================================================================

PLANNING_SCENARIOS: list[ScenarioDefinition] = [
    # Category M — Autonomous Planning
    ScenarioDefinition(
        id="TC-51",
        title="Goal-Level Planning",
        category=Category.M,
        user_message="Organize a team lunch for this Friday with the engineering team.",
        description="Autonomously decompose goal into contacts → event → notify workflow.",
        handle_tool_call=_tc51_handle,
        evaluate=_tc51_eval,
        difficulty=4,
    ),
    ScenarioDefinition(
        id="TC-52",
        title="Open-Ended Research",
        category=Category.M,
        user_message="I need to prepare a summary comparing our stock performance against the market. Our ticker is AAPL.",
        description="Autonomously research market data + stock price and synthesize comparison.",
        handle_tool_call=_tc52_handle,
        evaluate=_tc52_eval,
        difficulty=4,
    ),
    ScenarioDefinition(
        id="TC-53",
        title="Conditional Planning",
        category=Category.M,
        user_message="Check if it's going to rain in London this weekend, and if so, move our outdoor meeting to the office and notify the attendees.",
        description="Execute conditional workflow: weather check → branch on rain → act.",
        handle_tool_call=_tc53_handle,
        evaluate=_tc53_eval,
        difficulty=4,
    ),
    # Category N — Creative Composition
    ScenarioDefinition(
        id="TC-54",
        title="Cross-Tool Synthesis",
        category=Category.N,
        user_message="What's the local currency equivalent of MSFT's stock price in Tokyo right now?",
        description="Combine stock price + exchange rate lookup + calculation.",
        handle_tool_call=_tc54_handle,
        evaluate=_tc54_eval,
        difficulty=4,
    ),
    ScenarioDefinition(
        id="TC-55",
        title="Data Pipeline",
        category=Category.N,
        user_message="Find all Q3 revenue files and calculate the total revenue across all regions.",
        description="Build pipeline: search → read ×2 → calculate aggregate.",
        handle_tool_call=_tc55_handle,
        evaluate=_tc55_eval,
        difficulty=4,
    ),
    ScenarioDefinition(
        id="TC-56",
        title="Notification Workflow",
        category=Category.N,
        user_message="Check the weather in NYC. If it's below freezing, email me at user@company.com with a warning and set a reminder to dress warmly tomorrow morning.",
        description="Compose weather check → conditional → email notification.",
        handle_tool_call=_tc56_handle,
        evaluate=_tc56_eval,
        difficulty=3,
    ),
    # Category C expansion — Async Polling
    ScenarioDefinition(
        id="TC-61",
        title="Async Polling",
        category=Category.C,
        user_message='Run this data analysis script and give me the results: analyze_data(source="transactions_2026")',
        description="Handle async tool response: submit → detect pending → poll → surface result.",
        handle_tool_call=_tc61_handle,
        evaluate=_tc61_eval,
        difficulty=3,
    ),
    # Category I expansion — Deep Multi-Turn
    ScenarioDefinition(
        id="TC-62",
        title="5-Turn Research Chain",
        category=Category.I,
        user_message="Can you help me put together a competitive analysis report? Start by looking up our latest quarterly performance.",
        description="5-turn research chain with data correction, competitor pivot, and revision.",
        handle_tool_call=_tc62_handle,
        evaluate=_tc62_eval,
        follow_up_messages=_TC62_FOLLOW_UPS,
        # The attainable reference path contains dependent search/read rounds,
        # a competitor lookup, a draft turn, contact resolution, delivery, and
        # a final response. The default eight turns cannot reach authorization.
        max_turns_override=14,
        difficulty=4,
    ),
    ScenarioDefinition(
        id="TC-63",
        title="Accumulating Constraints",
        category=Category.I,
        user_message="Find me a restaurant for dinner tonight.",
        description="Maintain 4 constraints accumulated across 5 turns.",
        handle_tool_call=_tc63_handle,
        evaluate=_tc63_eval,
        follow_up_messages=_TC63_FOLLOW_UPS,
        difficulty=4,
    ),
]


PLANNING_DISPLAY_DETAILS: dict[str, ScenarioDisplayDetail] = {
    "TC-51": ScenarioDisplayDetail(
        "Pass if it autonomously decomposes: contacts → calendar event → email notification.",
        "Fail if it doesn't break down the goal into tool actions.",
    ),
    "TC-52": ScenarioDisplayDetail(
        "Pass if it gets AAPL stock price AND researches market benchmark, then synthesizes.",
        "Fail if it doesn't autonomously plan the research workflow.",
    ),
    "TC-53": ScenarioDisplayDetail(
        "Pass if it checks weather → detects rain → moves meeting to office → notifies.",
        "Fail if it ignores the conditional or doesn't act on the rain result.",
    ),
    "TC-54": ScenarioDisplayDetail(
        "Pass if it combines stock price + exchange rate to calculate JPY equivalent.",
        "Fail if it doesn't creatively combine multiple tools.",
    ),
    "TC-55": ScenarioDisplayDetail(
        "Pass if it searches → reads both revenue files → calculates total ($4.2M).",
        "Fail if it doesn't build the multi-read data pipeline.",
    ),
    "TC-56": ScenarioDisplayDetail(
        "Pass if it checks NYC weather → detects freezing → sends warning email.",
        "Fail if it doesn't compose weather check with notification.",
    ),
    "TC-61": ScenarioDisplayDetail(
        "Pass if it submits → detects 'pending' → polls again → surfaces the result.",
        "Fail if it doesn't retry after receiving the pending status.",
    ),
    "TC-62": ScenarioDisplayDetail(
        "Pass if it handles all 5 turns: research → correct data → competitor → CFO email.",
        "Fail if it loses context or ignores the correction/revision.",
    ),
    "TC-63": ScenarioDisplayDetail(
        "Pass if final recommendation satisfies all 4 accumulated constraints.",
        "Fail if it forgets earlier constraints as new ones are added.",
    ),
}
