"""Tests for IFEval constraint checkers and evaluator."""

from __future__ import annotations

import pytest

from tool_eval_bench.plugins.ifeval.checkers import (
    available_checkers,
    check_instruction,
)
from tool_eval_bench.plugins.ifeval.evaluator import (
    evaluate_prompt,
)

# ---------------------------------------------------------------------------
# Length constraints
# ---------------------------------------------------------------------------


class TestLengthConstraints:
    """Test word, sentence, and paragraph count checks."""

    def test_number_words_at_least_pass(self):
        response = " ".join(["word"] * 100)
        assert check_instruction(
            "length_constraints:number_words", response, {"num_words": 50, "relation": "at least"}
        )

    def test_number_words_at_least_fail(self):
        response = "just three words"
        assert not check_instruction(
            "length_constraints:number_words", response, {"num_words": 50, "relation": "at least"}
        )

    def test_number_words_at_most(self):
        response = "only five words here now"
        assert check_instruction(
            "length_constraints:number_words", response, {"num_words": 10, "relation": "at most"}
        )

    def test_number_words_exactly(self):
        response = "one two three"
        assert check_instruction(
            "length_constraints:number_words", response, {"num_words": 3, "relation": "exactly"}
        )

    def test_number_sentences(self):
        response = "First sentence. Second sentence. Third sentence."
        assert check_instruction(
            "length_constraints:number_sentences",
            response,
            {"num_sentences": 3, "relation": "at least"},
        )

    def test_number_paragraphs(self):
        response = "Paragraph one.\n***\nParagraph two.\n***\nParagraph three."
        assert check_instruction(
            "length_constraints:number_paragraphs", response, {"num_paragraphs": 3}
        )

    def test_number_paragraphs_is_an_exact_count(self):
        response = "Paragraph one.\n***\nParagraph two.\n***\nParagraph three."
        assert not check_instruction(
            "length_constraints:number_paragraphs", response, {"num_paragraphs": 2}
        )
        assert not check_instruction(
            "length_constraints:number_paragraphs", response, {"num_paragraphs": 4}
        )

    def test_number_paragraphs_ignores_blank_lines_without_divider(self):
        response = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        assert not check_instruction(
            "length_constraints:number_paragraphs", response, {"num_paragraphs": 3}
        )

    def test_number_paragraphs_empty_middle_chunk_fails(self):
        response = "One.\n***\n***\nTwo."
        assert not check_instruction(
            "length_constraints:number_paragraphs", response, {"num_paragraphs": 2}
        )

    def test_number_paragraphs_ignores_leading_and_trailing_divider(self):
        response = "***\nOne.\n***\nTwo.\n***"
        assert check_instruction(
            "length_constraints:number_paragraphs", response, {"num_paragraphs": 2}
        )

    def test_number_paragraphs_fail(self):
        response = "Just one paragraph."
        assert not check_instruction(
            "length_constraints:number_paragraphs", response, {"num_paragraphs": 3}
        )

    def test_nth_paragraph_first_word(self):
        response = "Hello world.\n\nWelcome back.\n\nGoodbye now."
        assert check_instruction(
            "length_constraints:nth_paragraph_first_word",
            response,
            {"nth_paragraph": 2, "first_word": "welcome"},
        )

    def test_nth_paragraph_first_word_strips_quotes_and_punctuation(self):
        response = '"Weekend, then.\n\nTwo.\n\nThree.\n\nFour."'
        kwargs = {"num_paragraphs": 4, "nth_paragraph": 1, "first_word": "weekend"}
        assert check_instruction("length_constraints:nth_paragraph_first_word", response, kwargs)

    def test_nth_paragraph_first_word_enforces_paragraph_count(self):
        kwargs = {"num_paragraphs": 4, "nth_paragraph": 1, "first_word": "weekend"}
        assert not check_instruction(
            "length_constraints:nth_paragraph_first_word", "Weekend is here.\n\nTwo.", kwargs
        )


# ---------------------------------------------------------------------------
# Keyword constraints
# ---------------------------------------------------------------------------


class TestOfficialCounting:
    """Word and bullet counts match the reference implementation's tokenization."""

    @pytest.mark.parametrize(
        ("response", "kwargs", "expected"),
        [
            # "don't" is two \w+ tokens, so 48 + 3 = 51 words.
            ("word " * 48 + "don't stop", {"num_words": 51, "relation": "at least"}, True),
            ("well-known " * 25, {"num_words": 50, "relation": "at least"}, True),
            # Markdown punctuation is not a word.
            ("- * - * " * 10 + "a b c", {"num_words": 20, "relation": "less than"}, True),
            ("word " * 300 + "\u2014 " * 5, {"num_words": 303, "relation": "less than"}, True),
        ],
    )
    def test_number_words(self, response: str, kwargs: dict, expected: bool) -> None:
        assert check_instruction("length_constraints:number_words", response, kwargs) is expected

    @pytest.mark.parametrize(
        ("response", "expected"),
        [
            ("\u2022 one\n\u2022 two", False),
            ("+ one\n+ two", False),
            # A line opening with "*text*" counts as a bullet, so this has three.
            ("*Note*: first\n* one\n* two", False),
            # A "---" rule counts as a dash bullet.
            ("* one\n* two\n---\n", False),
            ("-one\n-two", True),
            ("* one\n- two", True),
        ],
    )
    def test_bullet_lists(self, response: str, expected: bool) -> None:
        kwargs = {"num_bullets": 2}
        assert (
            check_instruction("detectable_format:number_bullet_lists", response, kwargs) is expected
        )


class TestKeywordConstraints:
    """Test keyword existence, frequency, and forbidden words."""

    def test_keywords_existence_pass(self):
        response = "The cat sat on the mat"
        assert check_instruction("keywords:existence", response, {"keywords": ["cat", "mat"]})

    def test_keywords_existence_fail(self):
        response = "The dog ran in the park"
        assert not check_instruction("keywords:existence", response, {"keywords": ["cat", "mat"]})

    def test_keywords_frequency(self):
        response = "hello hello hello world"
        assert check_instruction(
            "keywords:frequency",
            response,
            {"keyword": "hello", "frequency": 3, "relation": "at least"},
        )

    def test_keywords_frequency_fail(self):
        response = "hello world"
        assert not check_instruction(
            "keywords:frequency",
            response,
            {"keyword": "hello", "frequency": 3, "relation": "at least"},
        )

    def test_forbidden_words_pass(self):
        response = "The cat sat on the mat"
        assert check_instruction(
            "keywords:forbidden_words", response, {"forbidden_words": ["dog", "bird"]}
        )

    def test_forbidden_words_fail(self):
        response = "The dog ran in the park"
        assert not check_instruction(
            "keywords:forbidden_words", response, {"forbidden_words": ["dog", "bird"]}
        )

    def test_forbidden_words_match_whole_words_only(self):
        kwargs = {"forbidden_words": ["no", "can"]}
        assert check_instruction("keywords:forbidden_words", "I know nothing about Canada.", kwargs)
        assert not check_instruction("keywords:forbidden_words", "No, I CAN'T.", kwargs)

    def test_letter_frequency(self):
        response = "aaaaabbb"
        assert check_instruction(
            "keywords:letter_frequency",
            response,
            {"letter": "a", "let_frequency": 5, "let_relation": "at least"},
        )

    def test_letter_frequency_fail(self):
        response = "aab"
        assert not check_instruction(
            "keywords:letter_frequency",
            response,
            {"letter": "a", "let_frequency": 5, "let_relation": "at least"},
        )


# ---------------------------------------------------------------------------
# Format constraints
# ---------------------------------------------------------------------------


class TestFormatConstraints:
    """Test format-related checks."""

    def test_highlighted_sections(self):
        response = "Here is *section one* and *section two* and *section three*."
        assert check_instruction(
            "detectable_format:number_highlighted_sections", response, {"num_highlights": 3}
        )

    def test_highlighted_sections_fail(self):
        response = "Here is *section one* only."
        assert not check_instruction(
            "detectable_format:number_highlighted_sections", response, {"num_highlights": 3}
        )

    def test_bullet_lists(self):
        response = "Items:\n- item one\n- item two\n- item three"
        assert check_instruction(
            "detectable_format:number_bullet_lists", response, {"num_bullets": 3}
        )

    def test_bullet_lists_reject_extra_bullets(self):
        response = "Items:\n- item one\n- item two\n- item three\n- item four"
        assert not check_instruction(
            "detectable_format:number_bullet_lists", response, {"num_bullets": 3}
        )

    def test_placeholders(self):
        response = "Name: [name], Address: [address], Phone: [phone]"
        assert check_instruction(
            "detectable_format:number_placeholders", response, {"num_placeholders": 3}
        )

    def test_json_format_valid(self):
        response = '{"key": "value", "num": 42}'
        assert check_instruction("detectable_format:json_format", response, {})

    def test_json_format_with_fences(self):
        response = '```json\n{"key": "value"}\n```'
        assert check_instruction("detectable_format:json_format", response, {})

    def test_json_format_invalid(self):
        response = "This is not JSON at all"
        assert not check_instruction("detectable_format:json_format", response, {})

    def test_title(self):
        response = "<<My Title>>\n\nContent here."
        assert check_instruction("detectable_format:title", response, {})

    def test_title_after_a_long_first_line(self):
        response = " ".join(["word"] * 20) + "\n<<Press>>\nBody"
        assert check_instruction("detectable_format:title", response, {})

    def test_title_requires_double_angular_brackets(self):
        assert not check_instruction("detectable_format:title", "My Title\n\nContent.", {})
        assert not check_instruction("detectable_format:title", "# My Title\n\nContent.", {})
        assert not check_instruction("detectable_format:title", "<<  >>\nContent.", {})

    def test_multiple_sections(self):
        response = "SECTION 1\nContent.\nSECTION 2\nMore content.\nSECTION 3\nEnd."
        kwargs = {"num_sections": 3, "section_spliter": "SECTION"}
        assert check_instruction("detectable_format:multiple_sections", response, kwargs)

    def test_multiple_sections_does_not_count_the_preamble(self):
        response = "Intro\nSECTION 1 a\nSECTION 2 b"
        kwargs = {"num_sections": 3, "section_spliter": "SECTION"}
        assert not check_instruction("detectable_format:multiple_sections", response, kwargs)

    def test_multiple_sections_without_splitter_fails_closed(self):
        response = "# Section 1\nContent.\n# Section 2\nMore content."
        assert not check_instruction(
            "detectable_format:multiple_sections", response, {"num_sections": 2}
        )


# ---------------------------------------------------------------------------
# Punctuation constraints
# ---------------------------------------------------------------------------


class TestPunctuationConstraints:
    """Test punctuation checks."""

    def test_no_comma_pass(self):
        response = "This sentence has no commas at all."
        assert check_instruction("punctuation:no_comma", response, {})

    def test_no_comma_fail(self):
        response = "This sentence has commas, right here."
        assert not check_instruction("punctuation:no_comma", response, {})


# ---------------------------------------------------------------------------
# Start/end constraints
# ---------------------------------------------------------------------------


class TestStartEndConstraints:
    """Test start/end phrase checks."""

    def test_end_phrase_pass(self):
        response = "Some text. Is there anything else I can help with?"
        assert check_instruction(
            "startend:end_checker",
            response,
            {"end_phrase": "Is there anything else I can help with?"},
        )

    def test_end_phrase_fail(self):
        response = "Some text here."
        assert not check_instruction("startend:end_checker", response, {"end_phrase": "The end."})

    def test_end_phrase_ignores_case_and_wrapping_quotes(self):
        kwargs = {"end_phrase": "What would happen to human next?"}
        response = '"ALL CAPS. WHAT WOULD HAPPEN TO HUMAN NEXT?"'
        assert check_instruction("startend:end_checker", response, kwargs)

    def test_quotation_double(self):
        response = '"This is a quoted response"'
        assert check_instruction("startend:quotation", response, {})

    def test_quotation_requires_straight_double_quotes(self):
        assert not check_instruction("startend:quotation", "'single quoted'", {})
        assert not check_instruction("startend:quotation", "\u201ccurly quoted\u201d", {})
        assert not check_instruction("startend:quotation", '"', {})

    def test_quotation_fail(self):
        response = "This is not quoted"
        assert not check_instruction("startend:quotation", response, {})


# ---------------------------------------------------------------------------
# Case constraints
# ---------------------------------------------------------------------------


class TestCaseConstraints:
    """Test case transformation checks."""

    def test_lowercase_pass(self):
        response = "this is all lowercase 123"
        assert check_instruction("change_case:english_lowercase", response, {})

    def test_lowercase_fail(self):
        response = "This Has Uppercase"
        assert not check_instruction("change_case:english_lowercase", response, {})

    def test_case_checkers_fail_on_empty_and_letterless_responses(self):
        for iid in ("change_case:english_lowercase", "change_case:english_capital"):
            assert not check_instruction(iid, "", {})
            assert not check_instruction(iid, "123 ...", {})

    def test_capital_requires_all_capital_letters(self):
        assert check_instruction("change_case:english_capital", "THIS IS ALL CAPS 123", {})

    def test_capital_rejects_title_case(self):
        assert not check_instruction("change_case:english_capital", "Every Word Is Capitalized", {})

    def test_english_uppercase_is_not_an_ifeval_instruction(self):
        assert "change_case:english_uppercase" not in available_checkers()


# ---------------------------------------------------------------------------
# Combination / misc constraints
# ---------------------------------------------------------------------------


class TestCombinationConstraints:
    """Test combination and misc checks."""

    def test_repeat_prompt_pass(self):
        prompt = "Tell me a story about a cat."
        response = f"{prompt.upper()} Here is my story..."
        assert check_instruction(
            "combination:repeat_prompt", response, {"prompt_to_repeat": prompt}
        )

    def test_repeat_prompt_must_come_first(self):
        prompt = "Tell me a story about a cat."
        response = f"You asked: {prompt} Here is my story..."
        assert not check_instruction(
            "combination:repeat_prompt", response, {"prompt_to_repeat": prompt}
        )

    def test_repeat_prompt_fail(self):
        response = "Here is a story about a dog."
        assert not check_instruction(
            "combination:repeat_prompt",
            response,
            {"prompt_to_repeat": "Tell me a story about a cat."},
        )

    def test_two_responses(self):
        response = "Response one here.\n******\nResponse two here."
        assert check_instruction("combination:two_responses", response, {})

    def test_two_responses_fail(self):
        response = "Just one response."
        assert not check_instruction("combination:two_responses", response, {})

    @pytest.mark.parametrize(
        "response",
        [
            "One.\n---\nTwo.",  # wrong separator
            "Same\n******\nSame",  # identical responses
            "A\n******\nB\n******\nC",  # three responses
            "A\n******\n******\nB",  # empty middle response
        ],
    )
    def test_two_responses_rejects_non_official_shapes(self, response):
        assert not check_instruction("combination:two_responses", response, {})

    def test_postscript_pass(self):
        response = "Main content here.\n\nP.S. Don't forget to check."
        assert check_instruction("detectable_content:postscript", response, {})

    def test_postscript_fail(self):
        response = "Main content only."
        assert not check_instruction("detectable_content:postscript", response, {})

    def test_postscript_requires_the_requested_marker(self):
        assert check_instruction(
            "detectable_content:postscript",
            "Main content.\n\nP.P.S. Extra note.",
            {"postscript_marker": "P.P.S."},
        )
        assert not check_instruction(
            "detectable_content:postscript",
            "Main content.\n\nP.S. Extra note.",
            {"postscript_marker": "P.P.S"},
        )

    def test_postscript_may_appear_on_any_line(self):
        # The earlier "final line only" rule rejected a sign-off after the postscript.
        assert check_instruction(
            "detectable_content:postscript",
            "body\nP.S. Call me.\nBest, Tim",
            {"postscript_marker": "P.S."},
        )

    def test_postscript_is_case_insensitive(self):
        # Paired with english_lowercase in the dataset, so "p.s." must count.
        assert check_instruction(
            "detectable_content:postscript",
            "some lowercase text.\n\np.p.s remember this.",
            {"postscript_marker": "P.P.S"},
        )

    def test_postscript_can_be_final_line_of_a_constrained_paragraph(self):
        response = (
            "First paragraph.\n\nSecond paragraph.\n\nbonding begins here.\nP.P.S Final note."
        )
        assert check_instruction(
            "detectable_content:postscript",
            response,
            {"postscript_marker": "P.P.S"},
        )

    def test_language_english(self):
        response = "This is a response in English."
        assert check_instruction("language:response_language", response, {"language": "en"})

    def test_language_unsupported_code_fails_closed(self):
        assert not check_instruction("language:response_language", "garbage", {"language": "xx"})

    def test_language_script_mismatch_fails(self):
        assert not check_instruction("language:response_language", "garbage", {"language": "kn"})

    def test_language_latin_script_alone_is_not_enough(self):
        assert not check_instruction(
            "language:response_language",
            "This is an ordinary English response with several words.",
            {"language": "de"},
        )
        assert check_instruction(
            "language:response_language",
            "Das ist eine deutsche Antwort und die Aussage ist klar.",
            {"language": "de"},
        )

    def test_language_kannada_script_passes(self):
        assert check_instruction("language:response_language", "ಕನ್ನಡದಲ್ಲಿ ಉತ್ತರ", {"language": "kn"})

    def test_constrained_response_uses_prompt_options(self):
        prompt = (
            "Choose from the following: ('My answer is yes.', 'My answer is no.', "
            "'My answer is maybe.')"
        )
        assert check_instruction(
            "detectable_format:constrained_response",
            "My answer is yes.",
            {"prompt": prompt},
        )
        assert not check_instruction(
            "detectable_format:constrained_response", "banana", {"prompt": prompt}
        )

    def test_constrained_response_uses_structured_options(self):
        assert check_instruction(
            "detectable_format:constrained_response",
            "I choose red.",
            {"allowed_responses": ["red", "blue"]},
        )
        assert not check_instruction(
            "detectable_format:constrained_response",
            "red and blue",
            {"allowed_responses": ["red", "blue"]},
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestCheckerRegistry:
    """Test the checker registry."""

    def test_all_checkers_registered(self):
        checkers = available_checkers()
        assert len(checkers) >= 20  # We have ~25 checkers

    def test_unknown_checker_raises(self):
        with pytest.raises(KeyError):
            check_instruction("nonexistent:checker", "test", {})


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class TestEvaluatePrompt:
    """Test prompt-level evaluation."""

    def test_all_pass(self):
        response = "THIS IS ALL UPPERCASE."
        result = evaluate_prompt(
            response,
            ["change_case:english_capital"],
            [{}],
        )
        assert result.prompt_pass is True
        assert result.instructions_passed == 1
        assert result.instructions_total == 1

    def test_partial_pass(self):
        response = "this has no commas but is lowercase."
        result = evaluate_prompt(
            response,
            ["punctuation:no_comma", "change_case:english_capital"],
            [{}, {}],
        )
        assert result.prompt_pass is False  # Not all passed
        assert result.instructions_passed == 1  # no_comma passed
        assert result.instructions_total == 2

    def test_all_fail(self):
        response = "this has commas, and is lowercase."
        result = evaluate_prompt(
            response,
            ["punctuation:no_comma", "change_case:english_capital"],
            [{}, {}],
        )
        assert result.prompt_pass is False
        assert result.instructions_passed == 0

    def test_unknown_instruction_fails_closed(self):
        """Unknown instructions must not inflate a benchmark score."""
        response = "Any response."
        result = evaluate_prompt(
            response,
            ["unknown:instruction"],
            [{}],
        )
        assert result.prompt_pass is False
        assert result.instructions_passed == 0
        assert result.instruction_results[0].error is not None

    def test_prompt_context_is_available_to_constrained_checker(self):
        result = evaluate_prompt(
            "banana",
            ["detectable_format:constrained_response"],
            [{}],
            prompt="Answer with one of the following phrases: My answer is yes. or My answer is no.",
        )
        assert result.prompt_pass is False

    def test_empty_instructions(self):
        result = evaluate_prompt("Any response.", [], [])
        assert result.prompt_pass is True
        assert result.instructions_total == 0

    def test_none_kwargs_filtered(self):
        """None values in kwargs should be filtered out."""
        response = " ".join(["word"] * 100)
        result = evaluate_prompt(
            response,
            ["length_constraints:number_words"],
            [{"num_words": 50, "relation": "at least", "num_sentences": None}],
        )
        assert result.prompt_pass is True


# ---------------------------------------------------------------------------
# Rating
# ---------------------------------------------------------------------------


class TestIFEvalRating:
    """Test IFEval rating function."""

    def test_excellent(self):
        from tool_eval_bench.plugins.ifeval.plugin import _rating_for_accuracy

        assert "Excellent" in _rating_for_accuracy(90)

    def test_poor(self):
        from tool_eval_bench.plugins.ifeval.plugin import _rating_for_accuracy

        assert "Poor" in _rating_for_accuracy(20)


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


class TestReportRendering:
    """Test Markdown report section generation for IFEval."""

    def _make_result(self, passed: int = 8, total: int = 10):
        from tool_eval_bench.domain.plugin import BenchmarkResult
        from tool_eval_bench.plugins.ifeval.plugin import _rating_for_accuracy

        accuracy = passed / total * 100
        items = []
        for i in range(total):
            is_pass = i < passed
            items.append(
                {
                    "key": i,
                    "prompt": f"Write a response that contains at least {i + 10} words.",
                    "prompt_pass": is_pass,
                    "instructions_passed": 1 if is_pass else 0,
                    "instructions_total": 1,
                    "instruction_details": [
                        {
                            "id": "length_constraints:number_words",
                            "passed": is_pass,
                            "error": None if is_pass else "Expected at least 10 words",
                        }
                    ],
                    "model_response": "A sufficiently long response." if is_pass else "Short.",
                }
            )
        return BenchmarkResult(
            plugin_name="ifeval",
            score=accuracy,
            score_label=f"Prompt: {accuracy:.1f}%",
            rating=_rating_for_accuracy(accuracy),
            details={
                "prompts_passed": passed,
                "total": total,
                "prompt_accuracy": round(accuracy, 2),
                "instruction_accuracy": round(accuracy, 2),
                "instructions_passed": passed,
                "instructions_total": total,
                "dataset_size": 541,
                "constraint_types": {
                    "length_constraints:number_words": {
                        "passed": passed,
                        "total": total,
                        "accuracy": round(accuracy, 1),
                    }
                },
            },
            item_results=items,
            metadata={"dataset": "google/IFEval"},
            duration_seconds=30.0,
            total_tokens=5000,
        )

    def test_report_has_accuracy(self):
        from tool_eval_bench.plugins.ifeval.plugin import IFEvalPlugin

        plugin = IFEvalPlugin()
        lines = plugin.render_report_section(self._make_result(8, 10))
        text = "\n".join(lines)
        assert "80.0%" in text

    def test_report_has_error_analysis(self):
        from tool_eval_bench.plugins.ifeval.plugin import IFEvalPlugin

        plugin = IFEvalPlugin()
        lines = plugin.render_report_section(self._make_result(5, 10))
        text = "\n".join(lines)
        assert "Error Analysis" in text
        assert "Constraint violations" in text

    def test_report_shows_all_failures(self):
        from tool_eval_bench.plugins.ifeval.plugin import IFEvalPlugin

        plugin = IFEvalPlugin()
        lines = plugin.render_report_section(self._make_result(0, 25))
        text = "\n".join(lines)
        assert "25 total" in text
        assert "| 24 |" in text

    def test_report_collapsible_for_large_failure_sets(self):
        from tool_eval_bench.plugins.ifeval.plugin import IFEvalPlugin

        plugin = IFEvalPlugin()
        lines = plugin.render_report_section(self._make_result(0, 35))
        text = "\n".join(lines)
        assert "<details>" in text
        assert "Show all 35 failures" in text
        assert "</details>" in text

    def test_report_has_prompt_text(self):
        from tool_eval_bench.plugins.ifeval.plugin import IFEvalPlugin

        plugin = IFEvalPlugin()
        lines = plugin.render_report_section(self._make_result(5, 10))
        text = "\n".join(lines)
        assert "Prompt (excerpt)" in text

    def test_report_has_failed_constraints(self):
        from tool_eval_bench.plugins.ifeval.plugin import IFEvalPlugin

        plugin = IFEvalPlugin()
        lines = plugin.render_report_section(self._make_result(5, 10))
        text = "\n".join(lines)
        assert "Failed Constraints" in text
        assert "length_constraints:number_words" in text

    def test_report_has_detailed_failure_samples(self):
        from tool_eval_bench.plugins.ifeval.plugin import IFEvalPlugin

        plugin = IFEvalPlugin()
        lines = plugin.render_report_section(self._make_result(5, 10))
        text = "\n".join(lines)
        assert "Detailed Failure Samples" in text
        assert "**Prompt:**" in text
        assert "**Model response:**" in text

    def test_report_no_failures_when_perfect(self):
        from tool_eval_bench.plugins.ifeval.plugin import IFEvalPlugin

        plugin = IFEvalPlugin()
        lines = plugin.render_report_section(self._make_result(10, 10))
        text = "\n".join(lines)
        assert "Failed Prompts" not in text
        assert "Error Analysis" not in text
