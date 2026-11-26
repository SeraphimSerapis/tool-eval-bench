"""Safe arithmetic parser, mock tool fallbacks, and noise enrichment."""

from __future__ import annotations

import ast
import math
import operator
import re
from typing import Any

from tool_eval_bench.domain.scenarios import ToolCallRecord
from tool_eval_bench.evals.helpers._text import as_str

# ---------------------------------------------------------------------------
# Safe math expression parser (replaces eval())
# ---------------------------------------------------------------------------

_BINARY_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
}

_UNARY_OPS: dict[type, Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_node(node: ast.expr) -> float:
    """Recursively evaluate an AST node containing only arithmetic."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return _BINARY_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


def parse_math_expression(expression: str) -> float | None:
    """Safely evaluate a simple arithmetic expression.

    Uses AST parsing instead of eval() for safety. Supports +, -, *, /, %
    with parentheses and numeric literals.
    """
    sanitized = re.sub(r"(?<=\d)_(?=\d)", "", expression.replace(",", "")).strip()
    if not re.match(r"^[\d\s()+\-*/.%]+$", sanitized):
        return None
    try:
        tree = ast.parse(sanitized, mode="eval")
        result = _eval_node(tree.body)
        if isinstance(result, (int, float)):
            value = float(result)
            return value if math.isfinite(value) else None
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Mock tool fallback
# ---------------------------------------------------------------------------


def generic_tool_fallback(call: ToolCallRecord) -> Any:
    """Default mock response for tools not relevant to a scenario."""
    if call.name == "calculator":
        result = parse_math_expression(as_str(call.arguments.get("expression", "")))
        payload = {"error": "Invalid expression."} if result is None else {"result": result}
        return with_noise(payload, call.name)
    if call.name == "web_search":
        return with_noise(
            {
                "results": [
                    {"snippet": f"Search results for {as_str(call.arguments.get('query', ''))}"}
                ]
            },
            call.name,
        )
    if call.name == "run_code":
        return with_noise(
            {"error": "Code execution is not available in this environment."},
            call.name,
        )
    return with_noise(
        {"error": f"Tool {call.name} is not available."},
        call.name,
    )


def generic_tool_fallback_simple(call: ToolCallRecord) -> Any:
    """Simpler fallback that always returns an error (for scenarios that don't need calc/search)."""
    return with_noise({"error": f"Tool {call.name} is not available."}, call.name)


def with_noise(payload: Any, tool_name: str) -> Any:
    """Wrap a mock tool result with realistic payload noise.

    Applied automatically to make mock responses look like real API responses
    with extra metadata fields, timestamps, and nested objects.
    """
    from tool_eval_bench.evals.noise import enrich_payload

    if not isinstance(payload, dict):
        return payload
    return enrich_payload(tool_name, payload)
