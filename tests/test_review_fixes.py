"""Edge-case regression tests for math expression parsing and payload noise."""

from __future__ import annotations

from tool_eval_bench.evals.helpers import parse_math_expression
from tool_eval_bench.evals.noise import _seed_from_payload, enrich_payload


class TestMathParserEdgeCases:
    """Boundary and edge case tests for the AST-based safe math parser."""

    def test_empty_string(self) -> None:
        assert parse_math_expression("") is None

    def test_dot_only(self) -> None:
        assert parse_math_expression(".") is None

    def test_division_by_zero(self) -> None:
        # Should raise ZeroDivisionError internally, return None
        assert parse_math_expression("1/0") is None

    def test_very_long_number(self) -> None:
        result = parse_math_expression("9" * 100)
        assert result is not None
        assert result == float("9" * 100)

    def test_nested_parens(self) -> None:
        result = parse_math_expression("((((1+2))))")
        assert result == 3.0

    def test_floating_point_precision(self) -> None:
        result = parse_math_expression("0.1 + 0.2")
        assert result is not None
        assert abs(result - 0.3) < 1e-10

    def test_negative_number(self) -> None:
        result = parse_math_expression("-5")
        assert result == -5.0

    def test_modulo(self) -> None:
        result = parse_math_expression("10 % 3")
        assert result == 1.0

    def test_complex_expression(self) -> None:
        result = parse_math_expression("(2 + 3) * 4 - 6 / 2")
        assert result == 17.0

    def test_rejects_function_calls(self) -> None:
        assert parse_math_expression("__import__('os')") is None

    def test_rejects_attribute_access(self) -> None:
        assert parse_math_expression("a.b") is None

    def test_rejects_string_literals(self) -> None:
        assert parse_math_expression("'hello'") is None

    def test_rejects_comparison_operators(self) -> None:
        assert parse_math_expression("1 < 2") is None

    def test_commas_stripped(self) -> None:
        # 1,000 → 1000
        result = parse_math_expression("1,000 + 500")
        assert result == 1500.0


class TestNoiseVariation:
    """Verify that different payloads produce different noise values."""

    def test_different_weather_payloads_get_different_ids(self) -> None:
        p1 = enrich_payload("get_weather", {"temperature": 8, "condition": "overcast"})
        p2 = enrich_payload("get_weather", {"temperature": 22, "condition": "sunny"})

        # Same tool, different data → different request IDs
        assert p1["request_id"] != p2["request_id"]
        assert p1["station_id"] != p2["station_id"]

    def test_same_payload_produces_same_noise(self) -> None:
        """Determinism: identical input → identical output."""
        p1 = enrich_payload("get_weather", {"temperature": 8, "condition": "overcast"})
        p2 = enrich_payload("get_weather", {"temperature": 8, "condition": "overcast"})
        assert p1 == p2

    def test_search_payloads_vary(self) -> None:
        p1 = enrich_payload("web_search", {"results": [{"snippet": "result a"}]})
        p2 = enrich_payload("web_search", {"results": [{"snippet": "result b"}]})
        assert p1["request_id"] != p2["request_id"]

    def test_error_payloads_vary(self) -> None:
        e1 = enrich_payload("unknown_tool", {"error": "Tool X not relevant"})
        e2 = enrich_payload("unknown_tool_2", {"error": "Tool Y not relevant"})
        assert e1["request_id"] != e2["request_id"]
        assert e1["trace_id"] != e2["trace_id"]

    def test_seed_determinism(self) -> None:
        s1 = _seed_from_payload({"key": "value"}, "salt")
        s2 = _seed_from_payload({"key": "value"}, "salt")
        assert s1 == s2

    def test_seed_varies_with_salt(self) -> None:
        s1 = _seed_from_payload({"key": "value"}, "a")
        s2 = _seed_from_payload({"key": "value"}, "b")
        assert s1 != s2
