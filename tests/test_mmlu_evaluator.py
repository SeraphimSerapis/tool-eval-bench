"""Tests for MMLU evaluator — answer extraction and comparison."""

from __future__ import annotations

from tool_eval_bench.plugins.mmlu.dataset import (
    CATEGORIES,
    SUBJECT_CATEGORIES,
    MMLUItem,
)
from tool_eval_bench.plugins.mmlu.evaluator import (
    evaluate_answer,
    extract_answer,
)
from tool_eval_bench.plugins.mmlu.prompts import (
    build_messages,
)

# ---------------------------------------------------------------------------
# extract_answer
# ---------------------------------------------------------------------------


class TestExtractAnswer:
    """Test multiple-choice letter extraction."""

    def test_exact_single_letter(self):
        assert extract_answer("B") == ("B", "exact")

    def test_exact_lowercase(self):
        assert extract_answer("c") == ("C", "exact")

    def test_exact_with_period(self):
        assert extract_answer("A.") == ("A", "exact")

    def test_exact_with_paren(self):
        assert extract_answer("(D)") == ("D", "exact")

    def test_the_answer_is(self):
        assert extract_answer("The answer is B") == ("B", "answer_pattern")

    def test_answer_is_with_colon(self):
        assert extract_answer("After analysis, the answer is: C") == ("C", "answer_pattern")

    def test_answer_colon(self):
        assert extract_answer("Answer: D") == ("D", "answer_pattern")

    def test_answer_colon_with_paren(self):
        assert extract_answer("Answer: (A)") == ("A", "answer_pattern")

    def test_first_standalone_letter(self):
        assert extract_answer(
            "I think the best choice here is definitely B because of the reasons stated."
        ) == ("B", "first_letter")

    def test_empty_string(self):
        assert extract_answer("") == (None, "none")

    def test_no_valid_letter(self):
        assert extract_answer("I'm not sure about this question") == (None, "none")

    def test_whitespace_only(self):
        assert extract_answer("   ") == (None, "none")

    def test_multiline_response(self):
        resp = "Let me think step by step.\nThe answer is A."
        assert extract_answer(resp) == ("A", "answer_pattern")

    def test_long_explanation_with_answer(self):
        resp = (
            "This is a complex problem. First, we need to consider X. "
            "Then, applying theorem Y, we get Z. Therefore, the answer is D."
        )
        assert extract_answer(resp) == ("D", "answer_pattern")

    def test_final_choice_wins_over_earlier_option_list(self):
        resp = "The options are A, B, C, and D. I choose D."
        assert extract_answer(resp) == ("D", "answer_pattern")

    def test_described_option_does_not_override_later_explicit_answer(self):
        resp = "Option A is plausible, but the answer is D."
        assert extract_answer(resp) == ("D", "answer_pattern")

    def test_last_standalone_letter_wins_without_answer_cue(self):
        resp = "I considered A, then B, before settling on D"
        assert extract_answer(resp) == ("D", "last_letter")


# ---------------------------------------------------------------------------
# evaluate_answer
# ---------------------------------------------------------------------------


class TestEvaluateAnswer:
    """Test answer evaluation against ground truth."""

    def test_correct_answer(self):
        result = evaluate_answer("B", 1)  # B = index 1
        assert result.correct is True
        assert result.extracted_answer == "B"
        assert result.ground_truth_letter == "B"

    def test_incorrect_answer(self):
        result = evaluate_answer("A", 2)  # Expected C
        assert result.correct is False
        assert result.extracted_answer == "A"
        assert result.ground_truth_letter == "C"

    def test_no_answer_extracted(self):
        result = evaluate_answer("I don't know", 0)
        assert result.correct is False
        assert result.extracted_answer is None

    def test_all_indices(self):
        for i, letter in enumerate("ABCD"):
            result = evaluate_answer(letter, i)
            assert result.correct is True
            assert result.ground_truth_letter == letter


# ---------------------------------------------------------------------------
# MMLUItem
# ---------------------------------------------------------------------------


class TestMMLUItem:
    """Test MMLUItem properties."""

    def test_answer_letter(self):
        item = MMLUItem(0, "Q?", "math", ["a", "b", "c", "d"], 2)
        assert item.answer_letter == "C"

    def test_category_known(self):
        item = MMLUItem(0, "Q?", "abstract_algebra", ["a", "b", "c", "d"], 0)
        assert item.category == "STEM"

    def test_category_unknown(self):
        item = MMLUItem(0, "Q?", "unknown_subject", ["a", "b", "c", "d"], 0)
        assert item.category == "Other"

    def test_to_dict(self):
        item = MMLUItem(5, "What is?", "virology", ["a", "b", "c", "d"], 3)
        d = item.to_dict()
        assert d["index"] == 5
        assert d["answer_letter"] == "D"
        assert d["subject"] == "virology"


# ---------------------------------------------------------------------------
# Subject categories
# ---------------------------------------------------------------------------


class TestSubjectCategories:
    """Test subject → category mapping."""

    def test_all_categories_present(self):
        assert "STEM" in CATEGORIES
        assert "Humanities" in CATEGORIES
        assert "Social Sciences" in CATEGORIES
        assert "Other" in CATEGORIES

    def test_57_subjects(self):
        assert len(SUBJECT_CATEGORIES) == 57

    def test_stem_subjects(self):
        stem = [s for s, c in SUBJECT_CATEGORIES.items() if c == "STEM"]
        assert "abstract_algebra" in stem
        assert "machine_learning" in stem
        assert len(stem) == 19

    def test_humanities_subjects(self):
        humanities = [s for s, c in SUBJECT_CATEGORIES.items() if c == "Humanities"]
        assert "philosophy" in humanities
        assert "world_religions" in humanities

    def test_social_sciences(self):
        ss = [s for s, c in SUBJECT_CATEGORIES.items() if c == "Social Sciences"]
        assert "sociology" in ss
        assert "econometrics" in ss


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


class TestBuildMessages:
    """Test prompt construction."""

    def test_zero_shot(self):
        item = MMLUItem(0, "What is 2+2?", "elementary_mathematics", ["3", "4", "5", "6"], 1)
        msgs = build_messages(item, n_shots=0)
        assert len(msgs) == 2  # system + user
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert "A." in msgs[1]["content"]
        assert "B." in msgs[1]["content"]

    def test_few_shot(self):
        item = MMLUItem(0, "Q?", "abstract_algebra", ["a", "b", "c", "d"], 0)
        examples = [
            MMLUItem(i, f"Ex {i}?", "abstract_algebra", ["x", "y", "z", "w"], i % 4)
            for i in range(5)
        ]
        msgs = build_messages(item, few_shot_examples=examples, n_shots=3)
        # system + 3*(user+assistant) + user = 8
        assert len(msgs) == 8
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"

    def test_system_prompt_includes_subject(self):
        item = MMLUItem(0, "Q?", "machine_learning", ["a", "b", "c", "d"], 0)
        msgs = build_messages(item)
        assert "Machine Learning" in msgs[0]["content"]

    def test_no_few_shot_examples(self):
        item = MMLUItem(0, "Q?", "virology", ["a", "b", "c", "d"], 0)
        msgs = build_messages(item, few_shot_examples=None, n_shots=5)
        assert len(msgs) == 2  # system + user only


# ---------------------------------------------------------------------------
# Rating
# ---------------------------------------------------------------------------


class TestRating:
    """Test rating function."""

    def test_excellent(self):
        from tool_eval_bench.plugins.mmlu.plugin import _rating_for_accuracy

        assert "Excellent" in _rating_for_accuracy(90)

    def test_good(self):
        from tool_eval_bench.plugins.mmlu.plugin import _rating_for_accuracy

        assert "Good" in _rating_for_accuracy(75)

    def test_poor(self):
        from tool_eval_bench.plugins.mmlu.plugin import _rating_for_accuracy

        assert "Poor" in _rating_for_accuracy(20)


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


class TestReportRendering:
    """Test Markdown report section generation for MMLU."""

    def _make_result(self, correct: int = 8, total: int = 10):
        from tool_eval_bench.domain.plugin import BenchmarkResult
        from tool_eval_bench.plugins.mmlu.plugin import _rating_for_accuracy

        accuracy = correct / total * 100
        items = []
        for i in range(total):
            is_correct = i < correct
            items.append(
                {
                    "index": i,
                    "subject": "abstract_algebra",
                    "category": "STEM",
                    "question": f"What is the order of the group Z_{i + 2}?",
                    "correct": is_correct,
                    "ground_truth": "B" if is_correct else "C",
                    "extracted_answer": "B" if is_correct else None,
                    "extraction_method": "exact" if is_correct else "none",
                    "model_response": "The answer is B" if is_correct else "I think...",
                }
            )
        return BenchmarkResult(
            plugin_name="mmlu",
            score=accuracy,
            score_label=f"{accuracy:.1f}% ({correct}/{total})",
            rating=_rating_for_accuracy(accuracy),
            details={
                "correct": correct,
                "total": total,
                "accuracy": round(accuracy, 2),
                "n_shots": 5,
                "dataset_size": 14042,
                "categories": {
                    "STEM": {"correct": correct, "total": total, "accuracy": round(accuracy, 1)},
                },
                "extraction_methods": {
                    "exact": correct,
                    "none": total - correct,
                },
            },
            item_results=items,
            metadata={"dataset": "cais/mmlu"},
            duration_seconds=60.0,
            total_tokens=10000,
        )

    def test_report_has_accuracy(self):
        from tool_eval_bench.plugins.mmlu.plugin import MMLUPlugin

        plugin = MMLUPlugin()
        lines = plugin.render_report_section(self._make_result(8, 10))
        text = "\n".join(lines)
        assert "80.0%" in text
        assert "8/10" in text

    def test_report_has_error_analysis(self):
        from tool_eval_bench.plugins.mmlu.plugin import MMLUPlugin

        plugin = MMLUPlugin()
        lines = plugin.render_report_section(self._make_result(5, 10))
        text = "\n".join(lines)
        assert "Error Analysis" in text
        assert "No answer extracted" in text

    def test_report_shows_all_failures(self):
        from tool_eval_bench.plugins.mmlu.plugin import MMLUPlugin

        plugin = MMLUPlugin()
        lines = plugin.render_report_section(self._make_result(0, 25))
        text = "\n".join(lines)
        assert "25 total" in text
        assert "| 24 |" in text  # Last failure visible

    def test_report_collapsible_for_large_failure_sets(self):
        from tool_eval_bench.plugins.mmlu.plugin import MMLUPlugin

        plugin = MMLUPlugin()
        lines = plugin.render_report_section(self._make_result(0, 35))
        text = "\n".join(lines)
        assert "<details>" in text
        assert "Show all 35 failures" in text
        assert "</details>" in text

    def test_report_has_question_and_response_columns(self):
        from tool_eval_bench.plugins.mmlu.plugin import MMLUPlugin

        plugin = MMLUPlugin()
        lines = plugin.render_report_section(self._make_result(5, 10))
        text = "\n".join(lines)
        assert "Question (excerpt)" in text
        assert "Response (excerpt)" in text

    def test_report_has_detailed_failure_samples(self):
        from tool_eval_bench.plugins.mmlu.plugin import MMLUPlugin

        plugin = MMLUPlugin()
        lines = plugin.render_report_section(self._make_result(5, 10))
        text = "\n".join(lines)
        assert "Detailed Failure Samples" in text
        assert "**Question:**" in text
        assert "**Model response:**" in text

    def test_report_no_failures_when_perfect(self):
        from tool_eval_bench.plugins.mmlu.plugin import MMLUPlugin

        plugin = MMLUPlugin()
        lines = plugin.render_report_section(self._make_result(10, 10))
        text = "\n".join(lines)
        assert "Failed Questions" not in text
        assert "Error Analysis" not in text
