"""Tests for GSM8K answer evaluator."""

from __future__ import annotations

import pytest

from tool_eval_bench.plugins.gsm8k.evaluator import (
    evaluate_answer,
    extract_answer,
)


class TestExtractAnswer:
    """Test numeric answer extraction from model responses."""

    def test_standard_marker(self):
        """The standard #### N format."""
        text = "Some reasoning steps...\n#### 42"
        num, method = extract_answer(text)
        assert num == 42
        assert method == "marker"

    def test_marker_with_comma(self):
        """#### 70,000 — commas should be stripped."""
        text = "The profit is 70,000 dollars.\n#### 70,000"
        num, method = extract_answer(text)
        assert num == 70_000
        assert method == "marker"

    def test_marker_with_decimal(self):
        """#### 7450.4 — decimals should work."""
        text = "2% of 372520 is 7450.4\n#### 7450.4"
        num, method = extract_answer(text)
        assert num == pytest.approx(7450.4)
        assert method == "marker"

    def test_marker_negative(self):
        """#### -5 — negative numbers."""
        text = "The loss was 5 dollars.\n#### -5"
        num, method = extract_answer(text)
        assert num == -5
        assert method == "marker"

    def test_answer_is_pattern(self):
        """'the answer is N' fallback."""
        text = "After all the math, the answer is 18."
        num, method = extract_answer(text)
        assert num == 18
        assert method == "answer_pattern"

    def test_answer_equals_pattern(self):
        """'answer = N' pattern."""
        text = "answer = 42"
        num, method = extract_answer(text)
        assert num == 42
        assert method == "answer_pattern"

    def test_answer_colon_pattern(self):
        """'answer: N' pattern."""
        text = "Final answer: 99"
        num, method = extract_answer(text)
        assert num == 99
        assert method == "answer_pattern"

    def test_last_number_fallback(self):
        """When no marker or pattern, use the last standalone number."""
        text = "First we get 100, then subtract 30, giving us 70"
        num, method = extract_answer(text)
        assert num == 70
        assert method == "last_number"

    def test_no_number(self):
        """No extractable number at all."""
        text = "I cannot solve this problem."
        num, method = extract_answer(text)
        assert num is None
        assert method == "none"

    def test_dollar_sign(self):
        """Dollar sign in answer."""
        text = "She makes $18 every day.\n#### 18"
        num, method = extract_answer(text)
        assert num == 18
        assert method == "marker"

    def test_multiline_marker(self):
        """Marker on its own line with surrounding text."""
        text = "Step 1: 16 - 3 - 4 = 9\nStep 2: 9 * 2 = 18\n#### 18\n"
        num, method = extract_answer(text)
        assert num == 18
        assert method == "marker"

    def test_marker_takes_priority_over_answer_pattern(self):
        """#### marker should win over 'the answer is' pattern."""
        text = "The answer is 999, but actually\n#### 42"
        num, method = extract_answer(text)
        assert num == 42
        assert method == "marker"

    def test_empty_string(self):
        """Empty string should return None."""
        num, method = extract_answer("")
        assert num is None
        assert method == "none"

    def test_only_whitespace(self):
        """Whitespace-only should return None."""
        num, method = extract_answer("   \n\n  ")
        assert num is None
        assert method == "none"

    def test_dollar_in_answer_pattern(self):
        """'the answer is $18' — dollar sign in answer pattern."""
        text = "Therefore, the answer is $18."
        num, method = extract_answer(text)
        assert num == 18
        assert method == "answer_pattern"


class TestEvaluateAnswer:
    """Test answer comparison logic."""

    def test_exact_integer_match(self):
        """Exact integer match should be correct."""
        result = evaluate_answer("#### 42", 42.0)
        assert result.correct is True
        assert result.extracted_answer == 42.0

    def test_integer_mismatch(self):
        """Wrong integer should fail."""
        result = evaluate_answer("#### 43", 42.0)
        assert result.correct is False

    def test_float_close_enough(self):
        """Floats within tolerance should pass."""
        result = evaluate_answer("#### 7450.4", 7450.4)
        assert result.correct is True

    def test_float_too_far(self):
        """Floats outside tolerance should fail."""
        result = evaluate_answer("#### 7500", 7450.4)
        assert result.correct is False

    def test_no_answer_extracted(self):
        """When nothing can be extracted, result is incorrect."""
        result = evaluate_answer("I don't know", 42.0)
        assert result.correct is False
        assert result.extracted_answer is None
        assert result.extraction_method == "none"

    def test_comma_separated_number(self):
        """Large numbers with commas should be parsed correctly."""
        result = evaluate_answer("#### 70,000", 70000.0)
        assert result.correct is True

    def test_zero_answer(self):
        """Zero should match zero."""
        result = evaluate_answer("#### 0", 0.0)
        assert result.correct is True

    def test_negative_answer(self):
        """Negative numbers should match."""
        result = evaluate_answer("#### -15", -15.0)
        assert result.correct is True

    def test_extraction_method_recorded(self):
        """Extraction method should be recorded in result."""
        result = evaluate_answer("#### 42", 42.0)
        assert result.extraction_method == "marker"

        result2 = evaluate_answer("the answer is 42", 42.0)
        assert result2.extraction_method == "answer_pattern"

    def test_ground_truth_always_recorded(self):
        """Ground truth should always be in the result, even on failure."""
        result = evaluate_answer("no number here", 99.0)
        assert result.ground_truth == 99.0


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        # A unit or percent sign must not split the number.
        ("She saves 25%.", 25.0),
        ("It took 15km", 15.0),
        # A markdown bullet is not a minus sign.
        ("Total per hour:\n- 18", 18.0),
        # The last "the answer is" wins, as a correction should.
        ("Let me check if the answer is 12. No, recount: the answer is 15.", 15.0),
        # Negative ground truths exist in GSM8K.
        ("The answer is -10 degrees, a drop of 10 from 0.", -10.0),
        # A markdown heading does not hide the real marker.
        ("#### Step 1\n5*3=15\n#### 15\nHope this helps with 2 things", 15.0),
    ],
)
def test_extraction_takes_the_final_whole_answer(response: str, expected: float) -> None:
    value, _method = extract_answer(response)
    assert value == expected


def test_first_parseable_marker_wins_over_a_hallucinated_exemplar() -> None:
    # The model answered, then kept going and invented the next exemplar.
    response = "3 * 6 = 18\n#### 18\n\nQ: Tom has 40 apples...\nA: 40 + 2 = 42\n#### 42"
    assert extract_answer(response) == (18.0, "marker")
