"""Autonomous planning & creative composition scenarios.

TC-51 to TC-56: 6 scenarios in Categories M (Autonomous Planning)
and N (Creative Composition) that test whether models can decompose
goals into tool chains without step-by-step guidance, and combine
tools in non-obvious ways.

TC-61: Async polling scenario (Category C expansion).
TC-62: 8-turn deep research (Category I expansion).
TC-63: Accumulating constraints (Category I expansion).
"""

from __future__ import annotations

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
    as_str as _as_str,
    includes_text as _includes_text,
    answer_contains_number as _answer_contains_number,
    has_tool_call as _has_tool_call,
    asks_for_clarification as _asks_clarification,
    with_noise as _noise,
    pass_eval as _pass,
    partial_eval as _partial,
    fail_eval as _fail,
)


# ===================================================================
# TC-51: Goal-Level Planning
# Category M — Autonomous Planning
# ===================================================================

def _tc51_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_contacts":
        query = _as_str(call.arguments.get("query", ""))
        if _includes_text(query, "engineering") or _includes_text(query, "team"):
            return _noise(
                {"results": [
                    {"name": "Alice Chen", "email": "alice@company.com", "role": "engineer"},
                    {"name": "Bob Park", "email": "bob@company.com", "role": "engineer"},
                    {"name": "Carol Singh", "email": "carol@company.com", "role": "engineer"},
                ]},
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
    got_contacts = _has_tool_call(
        state, "get_contacts",
        lambda c: _includes_text(c.arguments.get("query", ""), "engineer")
        or _includes_text(c.arguments.get("query", ""), "team"),
    )
    created_event = _has_tool_call(state, "create_calendar_event")
    sent_notification = _has_tool_call(state, "send_email")

    steps = sum([got_contacts, created_event, sent_notification])

    if steps == 3:
        return _pass("Autonomously decomposed goal: contacts → event → notify.")
    if steps == 2:
        missing = []
        if not got_contacts:
            missing.append("contact lookup")
        if not created_event:
            missing.append("event creation")
        if not sent_notification:
            missing.append("notification")
        return _partial(f"Completed 2/3 planning steps. Missing: {', '.join(missing)}.")
    if got_contacts or created_event:
        return _partial("Started planning but didn't complete the full workflow.")
    # Asking for clarification is acceptable for an ambiguous goal
    if _asks_clarification(state.final_answer):
        return _partial("Asked for clarification instead of planning — reasonable but not proactive.")
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
        if "market" in query or "s&p" in query or "index" in query or "nasdaq" in query:
            return _noise(
                {"results": [
                    {"snippet": "S&P 500 closed at 5,412.50, up 0.8% for the week. "
                     "NASDAQ composite at 17,234.12, up 1.2%."},
                ]},
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
    got_stock = _has_tool_call(
        state, "get_stock_price",
        lambda c: _as_str(c.arguments.get("ticker", "")).upper() == "AAPL",
    )
    searched_market = _has_tool_call(
        state, "web_search",
        lambda c: any(
            w in _as_str(c.arguments.get("query", "")).lower()
            for w in ("market", "s&p", "index", "nasdaq", "benchmark")
        ),
    )
    answer = state.final_answer.lower()
    mentions_comparison = any(
        w in answer for w in ("178", "aapl", "s&p", "market", "compare", "outperform", "underperform")
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
            {"location": "London", "temperature": 11, "units": "celsius",
             "condition": "Rain", "humidity": 88, "forecast": "Rain expected through weekend."},
            "get_weather",
        )
    if call.name == "create_calendar_event":
        return _noise({"status": "created", "event_id": "evt_tc53"}, "create_calendar_event")
    if call.name == "send_email":
        return _noise({"status": "sent"}, "send_email")
    if call.name == "get_contacts":
        return _noise(
            {"results": [
                {"name": "Team Lead", "email": "lead@company.com"},
                {"name": "Dev Team", "email": "dev-team@company.com"},
            ]},
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
    checked_weather = _has_tool_call(
        state, "get_weather",
        lambda c: _includes_text(c.arguments.get("location", ""), "london"),
    )
    created_event = _has_tool_call(state, "create_calendar_event")
    sent_email = _has_tool_call(state, "send_email")

    answer = state.final_answer.lower()
    mentions_office = any(w in answer for w in ("office", "indoor", "inside", "moved", "relocated"))
    mentions_rain = "rain" in answer

    if checked_weather and created_event and sent_email and mentions_office:
        return _pass("Checked weather → detected rain → moved to office → notified attendees.")
    if checked_weather and sent_email and mentions_rain:
        return _pass("Checked weather → detected rain → notified attendees about the move.")
    if checked_weather and mentions_rain and mentions_office:
        return _pass("Checked weather → detected rain → recommended moving to office.")
    if checked_weather and (created_event or sent_email) and mentions_rain:
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
    got_stock = _has_tool_call(
        state, "get_stock_price",
        lambda c: _as_str(c.arguments.get("ticker", "")).upper() == "MSFT",
    )
    searched_exchange = _has_tool_call(
        state, "web_search",
        lambda c: any(
            w in _as_str(c.arguments.get("query", "")).lower()
            for w in ("usd", "jpy", "yen", "exchange", "currency")
        ),
    )


    answer = state.final_answer
    # Expected: 425.80 * 149.50 ≈ 63,657
    has_reasonable = _answer_contains_number(answer, "63") or _answer_contains_number(answer, "636")

    if got_stock and searched_exchange and has_reasonable:
        return _pass("Combined stock price + exchange rate + calculation — creative composition.")
    if got_stock and searched_exchange:
        return _partial("Got both data sources but final answer may be imprecise.")
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
            {"results": [
                {"file_id": "q3_rev", "name": "Q3_Revenue.xlsx"},
                {"file_id": "q3_exp", "name": "Q3_Expenses.xlsx"},
            ]},
            "search_files",
        )
    if call.name == "read_file":
        fid = _as_str(call.arguments.get("file_id", ""))
        if fid == "q3_rev":
            return _noise(
                {"content": "Q3 Revenue Report\nTotal Revenue: $2,400,000\nRegions: NA $1.2M, EU $800K, APAC $400K"},
                "read_file",
            )
        if fid == "q3_exp":
            return _noise(
                {"content": "Q3 Expense Report\nTotal Expenses: $1,800,000\nCategories: Payroll $1M, Infrastructure $500K, Marketing $300K"},
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
    """User: 'Find all files related to Q3 and calculate the total budget
    across all of them.'

    Must: search_files → read_file (×2) → calculator to sum.
    Total = $2,400,000 + $1,800,000 = $4,200,000.
    """
    searched = _has_tool_call(state, "search_files")
    read_rev = _has_tool_call(
        state, "read_file",
        lambda c: _as_str(c.arguments.get("file_id", "")) == "q3_rev",
    )
    read_exp = _has_tool_call(
        state, "read_file",
        lambda c: _as_str(c.arguments.get("file_id", "")) == "q3_exp",
    )
    answer = state.final_answer
    has_total = (
        _answer_contains_number(answer, "4200000")
        or _answer_contains_number(answer, "4.2")
        or _includes_text(answer, "$4,200,000")
        or _includes_text(answer, "$4.2M")
        or _includes_text(answer, "$4.2 million")
    )

    if searched and read_rev and read_exp and has_total:
        return _pass("Built data pipeline: search → read ×2 → calculate total.")
    if searched and (read_rev or read_exp) and has_total:
        return _partial("Got the total but only read one of two files.")
    if searched and read_rev and read_exp:
        return _partial("Read both files but didn't calculate the combined total.")
    if searched:
        return _partial("Found files but didn't read and aggregate them.")
    return _fail("Did not build a data pipeline to aggregate Q3 files.")


# ===================================================================
# TC-56: Notification Workflow
# Category N — Creative Composition
# ===================================================================

def _tc56_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_weather":
        location = _as_str(call.arguments.get("location", "")).lower()
        if "nyc" in location or "new york" in location:
            return _noise(
                {"location": "New York City", "temperature": -3, "units": "celsius",
                 "condition": "Snow", "humidity": 75},
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


def _tc56_eval(state: ScenarioState) -> ScenarioEvaluation:
    """User: 'Check the weather in NYC. If it's below freezing, email me
    a warning and set a reminder to dress warmly tomorrow morning.'

    Must: get_weather(NYC) → see -3°C (below freezing) → send_email + set_reminder.
    Temperature IS below freezing, so the email should be sent.
    """
    checked_weather = _has_tool_call(
        state, "get_weather",
        lambda c: _includes_text(c.arguments.get("location", ""), "nyc")
        or _includes_text(c.arguments.get("location", ""), "new york"),
    )
    sent_email = _has_tool_call(state, "send_email")
    set_reminder = _has_tool_call(state, "set_reminder")

    answer = state.final_answer.lower()
    mentions_cold = any(w in answer for w in ("below freezing", "cold", "-3", "freez", "snow", "warning"))

    if checked_weather and sent_email and mentions_cold:
        return _pass("Composed weather check → detected freezing → sent warning email.")
    if checked_weather and sent_email:
        return _partial("Checked weather and sent email but didn't note the freezing condition.")
    if checked_weather and set_reminder and mentions_cold:
        return _partial("Set reminder instead of sending email — close but wrong notification channel.")
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
                {"status": "pending", "job_id": "job_tc61_9f3a",
                 "message": "Analysis job is still running. Please call run_code again to check if the job has completed."},
                "run_code",
            )
        else:
            # Subsequent calls: return complete
            return _noise(
                {"status": "complete", "job_id": "job_tc61_9f3a",
                 "result": {"total_records": 15420, "anomalies_found": 3,
                            "summary": "3 anomalies detected in 15,420 records."}},
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
    from tool_eval_bench.evals.helpers import tool_calls_by_name
    run_calls = tool_calls_by_name(state, "run_code")

    answer = state.final_answer.lower()
    has_result = (
        _answer_contains_number(answer, "15420")
        or _answer_contains_number(answer, "15,420")
        or "3 anomal" in answer
        or "anomalies" in answer
    )

    if len(run_calls) >= 2 and has_result:
        return _pass("Polled async operation: submitted → detected pending → retried → got result.")
    if len(run_calls) >= 2:
        return _partial("Retried the operation but didn't surface the final result clearly.")
    if len(run_calls) == 1 and "pending" in answer:
        return _partial("Ran once, reported pending status, but didn't poll for completion.")
    if len(run_calls) == 1:
        return _fail("Ran once but didn't handle the 'pending' response by retrying.")
    return _fail("Did not attempt to run the analysis script.")


# ===================================================================
# TC-62: 8-Turn Deep Research Chain
# Category I — Context & State (expansion)
# ===================================================================

_TC62_FOLLOW_UPS = [
    # Turn 2: drill into detail + correction
    "Wait, I just remembered the Q3 report had a correction. Can you search for and read the latest version?",
    # Turn 3: pivot to competitor
    "OK, now compare that against our competitor. Search for Acme Corp's Q3 performance.",
    # Turn 4: action based on findings
    "Based on this analysis, draft an email to the CFO summarizing our competitive position.",
    # Turn 5: revision
    "Actually, make the tone more optimistic and add that we expect Q4 to improve.",
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
                {"content": "Q3 Report (CORRECTED)\nRevenue: $4,150,000\nNote: Previous version showed $4.4M due to accounting error."},
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
    """6-turn research chain testing context persistence and revision handling.

    Key checkpoints:
    - Used corrected revenue ($4.15M not $4.4M)
    - Searched for Acme competitor data
    - Sent email to CFO
    """
    # Check for corrected data usage
    transcript = "\n".join(state.assistant_messages).lower()

    used_corrected = "4,150,000" in transcript or "4.15" in transcript or "4150" in transcript
    searched_acme = _has_tool_call(
        state, "web_search",
        lambda c: "acme" in _as_str(c.arguments.get("query", "")).lower(),
    )
    sent_email = _has_tool_call(state, "send_email")
    email_to_cfo = _has_tool_call(
        state, "send_email",
        lambda c: _includes_text(c.arguments.get("to", ""), "cfo"),
    )

    checks = sum([used_corrected, searched_acme, sent_email])

    if checks == 3 and email_to_cfo:
        # Check for optimistic tone in email body
        email_calls = [c for c in state.tool_calls if c.name == "send_email"]
        if email_calls:
            body = _as_str(email_calls[-1].arguments.get("body", ""))
            optimistic = any(w in body.lower() for w in ("optimistic", "improve", "growth", "positive", "expect"))
            if optimistic:
                return _pass("Completed 8-turn chain: corrected data → competitor → CFO email with optimistic tone.")
            return _partial("Sent CFO email but missed the optimistic tone revision from turn 7.")
        return _partial("Completed research but email details unclear.")
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
        return _partial("Only completed 1/3 key checkpoints in the 8-turn chain.")
    return _fail("Failed to maintain context across the 8-turn research chain.")


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
                {"results": [
                    {"snippet": "Trattoria Bella — Italian, downtown, $22/person avg, open until 11pm. ★★★★"},
                ]},
                "web_search",
            )
        if len(constraints) >= 2:
            return _noise(
                {"results": [
                    {"snippet": "Luigi's — Italian, downtown, $25/person, closes 9pm."},
                    {"snippet": "Trattoria Bella — Italian, downtown, $22/person, open until 11pm."},
                ]},
                "web_search",
            )
        return _noise(
            {"results": [
                {"snippet": "Top restaurants: Sushi Palace ($45), Luigi's Italian ($25), "
                 "Burger Joint ($15), Trattoria Bella ($22)."},
            ]},
            "web_search",
        )
    return _noise({"error": f"Tool {call.name} is not relevant."}, call.name)


def _tc63_eval(state: ScenarioState) -> ScenarioEvaluation:
    """User: 'Find me a restaurant for dinner tonight.'
    Then constraints accumulate: Italian → under $30 → downtown → open past 10pm.

    Final answer should satisfy ALL 4 constraints.
    Best match: Trattoria Bella.
    """
    answer = state.final_answer.lower()

    has_italian = "italian" in answer or "trattoria" in answer or "luigi" in answer
    has_budget = any(w in answer for w in ("$22", "$25", "under $30", "budget", "affordable"))
    has_downtown = "downtown" in answer
    has_late = any(w in answer for w in ("11pm", "10pm", "late", "open past", "11 pm", "10 pm"))
    best_pick = "trattoria" in answer or "bella" in answer

    constraints_met = sum([has_italian, has_budget, has_downtown, has_late])

    if best_pick and constraints_met >= 3:
        return _pass("Maintained all accumulated constraints → recommended Trattoria Bella.")
    if constraints_met == 4:
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
        id="TC-51", title="Goal-Level Planning", category=Category.M,
        user_message="Organize a team lunch for this Friday with the engineering team.",
        description="Autonomously decompose goal into contacts → event → notify workflow.",
        handle_tool_call=_tc51_handle, evaluate=_tc51_eval,
    ),
    ScenarioDefinition(
        id="TC-52", title="Open-Ended Research", category=Category.M,
        user_message="I need to prepare a summary comparing our stock performance against the market. Our ticker is AAPL.",
        description="Autonomously research market data + stock price and synthesize comparison.",
        handle_tool_call=_tc52_handle, evaluate=_tc52_eval,
    ),
    ScenarioDefinition(
        id="TC-53", title="Conditional Planning", category=Category.M,
        user_message="Check if it's going to rain in London this weekend, and if so, move our outdoor meeting to the office and notify the attendees.",
        description="Execute conditional workflow: weather check → branch on rain → act.",
        handle_tool_call=_tc53_handle, evaluate=_tc53_eval,
    ),
    # Category N — Creative Composition
    ScenarioDefinition(
        id="TC-54", title="Cross-Tool Synthesis", category=Category.N,
        user_message="What's the local currency equivalent of MSFT's stock price in Tokyo right now?",
        description="Combine stock price + exchange rate lookup + calculation.",
        handle_tool_call=_tc54_handle, evaluate=_tc54_eval,
    ),
    ScenarioDefinition(
        id="TC-55", title="Data Pipeline", category=Category.N,
        user_message="Find all files related to Q3 and calculate the total budget across all of them.",
        description="Build pipeline: search → read ×2 → calculate aggregate.",
        handle_tool_call=_tc55_handle, evaluate=_tc55_eval,
    ),
    ScenarioDefinition(
        id="TC-56", title="Notification Workflow", category=Category.N,
        user_message="Check the weather in NYC. If it's below freezing, email me a warning and set a reminder to dress warmly tomorrow morning.",
        description="Compose weather check → conditional → email notification.",
        handle_tool_call=_tc56_handle, evaluate=_tc56_eval,
    ),
    # Category C expansion — Async Polling
    ScenarioDefinition(
        id="TC-61", title="Async Polling", category=Category.C,
        user_message='Run this data analysis script and give me the results: analyze_data(source="transactions_2026")',
        description="Handle async tool response: submit → detect pending → poll → surface result.",
        handle_tool_call=_tc61_handle, evaluate=_tc61_eval,
    ),
    # Category I expansion — Deep Multi-Turn
    ScenarioDefinition(
        id="TC-62", title="6-Turn Research Chain", category=Category.I,
        user_message="Can you help me put together a competitive analysis report? Start by looking up our latest quarterly performance.",
        description="6-turn research chain with data correction, competitor pivot, and revision.",
        handle_tool_call=_tc62_handle, evaluate=_tc62_eval,
        follow_up_messages=_TC62_FOLLOW_UPS,
    ),
    ScenarioDefinition(
        id="TC-63", title="Accumulating Constraints", category=Category.I,
        user_message="Find me a restaurant for dinner tonight.",
        description="Maintain 4 constraints accumulated across 5 turns.",
        handle_tool_call=_tc63_handle, evaluate=_tc63_eval,
        follow_up_messages=_TC63_FOLLOW_UPS,
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
        "Pass if it searches → reads both files → calculates total ($4.2M).",
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
        "Pass if it handles all 6 turns: research → correct data → competitor → CFO email.",
        "Fail if it loses context or ignores the correction/revision.",
    ),
    "TC-63": ScenarioDisplayDetail(
        "Pass if final recommendation satisfies all 4 accumulated constraints.",
        "Fail if it forgets earlier constraints as new ones are added.",
    ),
}
