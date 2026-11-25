"""TC-52 — Open-Ended Research."""

from __future__ import annotations

import re
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
    fail_eval as _fail,
)
from tool_eval_bench.evals.helpers import forbid_unrequested_side_effects
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
    strip_markdown_emphasis as _strip_markdown_emphasis,
)
from tool_eval_bench.evals.helpers import (
    tool_calls_by_name as _tool_calls_by_name,
)
from tool_eval_bench.evals.helpers import (
    with_noise as _noise,
)
from tool_eval_bench.evals.scenarios.planning._shared import (
    _result_matches_if_present,
)


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
        if any(
            term in query
            for term in (
                "market",
                "s&p",
                "sp500",
                "index",
                "indices",
                "nasdaq",
                "benchmark",
                "dow",
            )
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
    return _noise({"error": f"Tool {call.name} is not available."}, call.name)


_TC52_BENCHMARK_NAME = re.compile(
    r"\b(?:s&p\s*500|s&p|sp500|nasdaq|market|index|indices|benchmark|dow(?:\s+jones)?)\b", re.I
)
_TC52_SENTENCE_CHAR = r"(?:[^.!?\n]|(?<=\d)\.(?=\d))"
_TC52_SAME_CLAUSE_CHAR = r"(?:[^.!?\n,;]|(?<=\d)\.(?=\d)|\([^)\n]{0,40}\))"
_TC52_WRONG_DIRECTION = re.compile(
    r"\b(?:aapl|apple)\b"
    r"(?:(?!\b(?:while|whereas|but|however|though|although|s&p|sp500|nasdaq|market|benchmark|index|indices|dow)\b)"
    rf"{_TC52_SAME_CLAUSE_CHAR}){{0,80}}?"
    r"\b(?P<verb>outperform\w*|out-perform\w*|beat(?:s|ing)?|"
    r"outpac\w*|ahead\s+of|stronger\s+than|better\s+than)\b"
    rf"{_TC52_SENTENCE_CHAR}{{0,50}}?\b(?:market|s&p|sp500|nasdaq|index|indices|benchmark|broad|dow)\b",
    re.I,
)
# Benchmark-first or object-less outperformance ("Compared to the benchmark,
# AAPL ($178.50) outperformed.") must stay clause-local so "AAPL is down, while
# the market outperformed" is not blamed on AAPL.
_TC52_WRONG_DIRECTION_BARE = re.compile(
    r"\b(?:aapl|apple)\b"
    r"(?:(?!\b(?:while|whereas|but|however|though|although|s&p|sp500|nasdaq|market|benchmark|index|indices|dow)\b)"
    rf"{_TC52_SAME_CLAUSE_CHAR}){{0,60}}?"
    r"\b(?P<verb>outperform\w*|out-perform\w*|outpac\w*)\b",
    re.I,
)
_TC52_BENCHMARK_TRAILED_AAPL = re.compile(
    r"\b(?:s&p\s*500|s&p|sp500|nasdaq|market|index|indices|benchmark|dow(?:\s+jones)?)\b"
    r"(?:(?!\b(?:while|whereas|but|however|though|although|aapl|apple)\b)"
    rf"{_TC52_SAME_CLAUSE_CHAR}){{0,60}}?"
    r"\b(?P<verb>under-?perform\w*|lag(?:s|ged|ging)?|trail(?:s|ed|ing)?|fell\s+behind|behind)\b"
    rf"{_TC52_SAME_CLAUSE_CHAR}{{0,40}}?\b(?:aapl|apple)\b",
    re.I,
)
_TC52_FAILED_TO_NEGATOR = re.compile(
    r"\b(?:fail(?:s|ed|ing)?|unable|struggl(?:e[ds]?|ing)|short\s+of)\s+to\s+(?:\w+\s+){0,2}$",
    re.I,
)
_TC52_AAPL_POSITIVE = re.compile(
    r"\b(?:aapl|apple)\b"
    r"(?:(?!\b(?:while|whereas|but|however|though|although|s&p|sp500|nasdaq|market|benchmark|index|indices|dow|compared|versus|vs\.?|against)\b)"
    r"(?:[^.!?\n,;]|(?<=\d)\.(?=\d))){0,50}?"
    r"(?P<sign>\+\s*\d+(?:\.\d+)?\s*%|\b(?:up|rose|gained|climbed|advanced|increased|rallied)\s+(?:by\s+)?\d+(?:\.\d+)?\s*%)",
    re.I,
)
_TC52_BENCHMARK_NEGATIVE = re.compile(
    r"\b(?:s&p\s*500|s&p|sp500|nasdaq|dow(?:\s+jones)?|market|benchmark)\b"
    r"(?:(?!\b(?:while|whereas|but|however|though|although|aapl|apple|compared|versus|vs\.?|against)\b)"
    r"(?:[^.!?\n,;]|(?<=\d)\.(?=\d))){0,50}?"
    r"(?P<sign>-\s*\d+(?:\.\d+)?\s*%|\b(?:down|fell|dropped|declined)\s+(?:by\s+)?\d+(?:\.\d+)?\s*%)",
    re.I,
)


def _tc52_is_negated_verb(plain: str, verb_start: int, verb_end: int) -> bool:
    prefix = plain[max(0, verb_start - 120) : verb_start]
    return _negated_before(prefix, plain[verb_end : verb_end + 40]) or bool(
        _TC52_FAILED_TO_NEGATOR.search(prefix)
    )


def _tc52_claims_outperformance(answer: str) -> bool:
    plain = _strip_markdown_emphasis(answer)
    for pattern in (
        _TC52_WRONG_DIRECTION,
        _TC52_WRONG_DIRECTION_BARE,
        _TC52_BENCHMARK_TRAILED_AAPL,
    ):
        for match in pattern.finditer(plain):
            verb_start, verb_end = match.span("verb")
            if not _tc52_is_negated_verb(plain, verb_start, verb_end):
                return True
    for pattern in (_TC52_AAPL_POSITIVE, _TC52_BENCHMARK_NEGATIVE):
        for match in pattern.finditer(plain):
            sign_start, sign_end = match.span("sign")
            if not _tc52_is_negated_verb(plain, sign_start, sign_end):
                return True
    return False


_TC52_PUNT = re.compile(
    r"\b(?:no\s+idea|cannot\s+compare|can't\s+compare|unable\s+to\s+compare|"
    r"not\s+sure\s+how\s+(?:it|aapl|apple)\s+compares)\b",
    re.I,
)
_TC52_UNDERPERFORMANCE = re.compile(
    r"\b(?:under-?perform\w*|lag(?:s|ged|ging)?|trail(?:s|ed|ing)?|behind|"
    r"weaker\s+than|worse\s+than|lower\s+than|"
    r"below\s+(?:the\s+)?(?:market|index|indices|benchmark|s&p|sp500|nasdaq|dow)|"
    r"(?:fail(?:s|ed|ing)?\s+to|unable\s+to|did\s+not|didn't)\s+"
    r"(?:out-?perform\w*|beat|outpace|keep\s+up\s+with|match))\b",
    re.I,
)
_TC52_COMPARISON_WORD = re.compile(
    r"\b(?:compare[ds]?|comparing|comparison|versus|vs\.?|against|relative|"
    r"whereas|while|contrast\w*)\b",
    re.I,
)
_TC52_AAPL_DOWN = re.compile(
    r"\b(?:down|fell|dropped|declined|negative|slipped|dipped|lower|lost|retreated)\b|-\s*1\.27\s*%",
    re.I,
)
_TC52_MARKET_UP = re.compile(
    r"\b(?:up|rose|gained|positive|climbed|advanced|increased|higher|rallied|edged\s+up)\b"
    r"|\+\s*(?:0\.8|1\.2)\s*%",
    re.I,
)


@forbid_unrequested_side_effects({"run_code": None})
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
            for w in (
                "market",
                "s&p",
                "sp500",
                "index",
                "indices",
                "nasdaq",
                "benchmark",
                "dow",
            )
        )
        and _result_matches_if_present(state, call, market_result_has_benchmark)
    ]
    got_stock = bool(stock_calls)
    searched_market = bool(market_calls)
    answer_raw = state.final_answer
    answer = _normalize(answer_raw)
    # Require actual AAPL price plus benchmark data (either index level or
    # percentage change paired with a benchmark name), and a valid comparison
    # that does not invert direction or punt.
    has_price = any(_answer_affirms_number(answer_raw, value) for value in ("178.5", "178"))
    has_benchmark_level = any(
        _answer_affirms_number(answer_raw, value)
        for value in ("5412.5", "5412", "17234.12", "17234")
    )
    has_benchmark_pct = bool(_TC52_BENCHMARK_NAME.search(answer)) and any(
        _answer_affirms_number(answer_raw, value) for value in ("0.8", "1.2")
    )
    has_benchmark = has_benchmark_level or has_benchmark_pct
    wrong_direction = _tc52_claims_outperformance(answer_raw)
    punted = bool(_TC52_PUNT.search(answer))
    has_contrast = bool(
        _TC52_UNDERPERFORMANCE.search(answer)
        or (_TC52_AAPL_DOWN.search(answer) and _TC52_MARKET_UP.search(answer))
        or _TC52_COMPARISON_WORD.search(answer)
    )
    mentions_comparison = (
        has_price and has_benchmark and has_contrast and not wrong_direction and not punted
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


SCENARIO = ScenarioDefinition(
    id="TC-52",
    title="Open-Ended Research",
    category=Category.M,
    user_message="I need to prepare a summary comparing our stock performance against the market. Our ticker is AAPL.",
    description="Autonomously research market data + stock price and synthesize comparison.",
    handle_tool_call=_tc52_handle,
    evaluate=_tc52_eval,
    difficulty=4,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it gets AAPL stock price AND researches market benchmark, then synthesizes.",
    "Fail if it doesn't autonomously plan the research workflow.",
)
