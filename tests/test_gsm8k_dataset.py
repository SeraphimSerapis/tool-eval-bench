"""Tests for GSM8K dataset loader, prompts, rating, and report rendering."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tool_eval_bench.domain.plugin import BenchmarkResult
from tool_eval_bench.plugins.gsm8k.dataset import (
    GSM8KItem,
    _extract_ground_truth,
    _load_from_cache,
    _save_to_cache,
)
from tool_eval_bench.plugins.gsm8k.plugin import GSM8KPlugin, _rating_for_accuracy
from tool_eval_bench.plugins.gsm8k.prompts import FEW_SHOT_EXAMPLES, SYSTEM_PROMPT, build_messages

# ---------------------------------------------------------------------------
# Dataset: _extract_ground_truth
# ---------------------------------------------------------------------------


class TestExtractGroundTruth:
    """Test ground truth extraction from raw GSM8K answers."""

    def test_simple_integer(self):
        assert _extract_ground_truth("Some steps\n#### 18") == 18.0

    def test_with_commas(self):
        assert _extract_ground_truth("Work...\n#### 70,000") == 70_000.0

    def test_negative(self):
        assert _extract_ground_truth("Loss\n#### -5") == -5.0

    def test_decimal(self):
        assert _extract_ground_truth("Result\n#### 3.14") == pytest.approx(3.14)

    def test_no_marker_raises(self):
        with pytest.raises(ValueError, match="No #### pattern"):
            _extract_ground_truth("No marker here")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="No #### pattern"):
            _extract_ground_truth("")


# ---------------------------------------------------------------------------
# Dataset: cache round-trip
# ---------------------------------------------------------------------------


class TestCacheRoundTrip:
    """Test save and load from JSONL cache file."""

    def test_round_trip(self, tmp_path: Path):
        items = [
            GSM8KItem(index=0, question="Q1?", raw_answer="A1\n#### 5", ground_truth=5.0),
            GSM8KItem(index=1, question="Q2?", raw_answer="A2\n#### 10", ground_truth=10.0),
            GSM8KItem(
                index=2, question="Q3 with émojis 🎉?", raw_answer="A3\n#### 42", ground_truth=42.0
            ),
        ]
        cache_file = tmp_path / "test.jsonl"
        _save_to_cache(cache_file, items)

        # Verify file format
        lines = cache_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3
        first = json.loads(lines[0])
        assert first["index"] == 0
        assert first["question"] == "Q1?"
        assert first["ground_truth"] == 5.0

        # Round-trip
        loaded = _load_from_cache(cache_file)
        assert len(loaded) == 3
        assert loaded[0].index == 0
        assert loaded[0].question == "Q1?"
        assert loaded[0].ground_truth == 5.0
        assert loaded[2].question == "Q3 with émojis 🎉?"

    def test_empty_lines_skipped(self, tmp_path: Path):
        cache_file = tmp_path / "test.jsonl"
        cache_file.write_text(
            '{"index":0,"question":"Q","raw_answer":"A\\n#### 1","ground_truth":1}\n'
            "\n"
            '{"index":1,"question":"Q2","raw_answer":"A\\n#### 2","ground_truth":2}\n'
            "\n",
            encoding="utf-8",
        )
        loaded = _load_from_cache(cache_file)
        assert len(loaded) == 2

    def test_empty_file(self, tmp_path: Path):
        cache_file = tmp_path / "test.jsonl"
        cache_file.write_text("", encoding="utf-8")
        loaded = _load_from_cache(cache_file)
        assert loaded == []


# ---------------------------------------------------------------------------
# Dataset: GSM8KItem.to_dict
# ---------------------------------------------------------------------------


class TestGSM8KItem:
    def test_to_dict(self):
        item = GSM8KItem(
            index=5, question="How many?", raw_answer="Steps\n#### 7", ground_truth=7.0
        )
        d = item.to_dict()
        assert d == {"index": 5, "question": "How many?", "ground_truth": 7.0}
        # raw_answer should NOT be in to_dict (it's verbose)
        assert "raw_answer" not in d


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


class TestPrompts:
    """Test prompt construction."""

    def test_system_prompt_exists(self):
        assert len(SYSTEM_PROMPT) > 0
        assert "math" in SYSTEM_PROMPT.lower()

    def test_few_shot_count(self):
        assert len(FEW_SHOT_EXAMPLES) == 8

    def test_few_shot_format(self):
        for ex in FEW_SHOT_EXAMPLES:
            assert "question" in ex
            assert "answer" in ex
            assert "####" in ex["answer"], f"Missing #### in: {ex['answer'][:50]}"

    def test_build_messages_zero_shot(self):
        msgs = build_messages("What is 2+2?", n_shots=0)
        assert len(msgs) == 2  # system + user
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "What is 2+2?"

    def test_build_messages_8_shot(self):
        msgs = build_messages("What is 2+2?", n_shots=8)
        # system + 8*(user+assistant) + user = 1 + 16 + 1 = 18
        assert len(msgs) == 18
        assert msgs[0]["role"] == "system"
        assert msgs[-1]["role"] == "user"
        assert msgs[-1]["content"] == "What is 2+2?"
        # Check alternation
        for i in range(1, 17, 2):
            assert msgs[i]["role"] == "user"
            assert msgs[i + 1]["role"] == "assistant"

    def test_build_messages_3_shot(self):
        msgs = build_messages("Q?", n_shots=3)
        # system + 3*(user+assistant) + user = 1 + 6 + 1 = 8
        assert len(msgs) == 8

    def test_build_messages_capped_at_available(self):
        """Requesting more shots than available should just use all 8."""
        msgs = build_messages("Q?", n_shots=100)
        assert len(msgs) == 18  # same as 8-shot

    def test_build_messages_rejects_negative_n_shots(self):
        with pytest.raises(ValueError, match="n_shots"):
            build_messages("Q?", n_shots=-1)


# ---------------------------------------------------------------------------
# Rating
# ---------------------------------------------------------------------------


class TestRating:
    """Test accuracy to rating mapping."""

    def test_excellent(self):
        assert "★★★★★" in _rating_for_accuracy(95.0)
        assert "★★★★★" in _rating_for_accuracy(90.0)

    def test_good(self):
        rating = _rating_for_accuracy(80.0)
        assert "★★★★" in rating
        assert "★★★★★" not in rating

    def test_adequate(self):
        rating = _rating_for_accuracy(65.0)
        assert "★★★" in rating

    def test_weak(self):
        rating = _rating_for_accuracy(45.0)
        assert "★★" in rating

    def test_poor(self):
        rating = _rating_for_accuracy(20.0)
        assert "★ " in rating
        assert "Poor" in rating

    def test_boundary_90(self):
        assert "Excellent" in _rating_for_accuracy(90.0)
        assert "Good" in _rating_for_accuracy(89.9)

    def test_boundary_75(self):
        assert "Good" in _rating_for_accuracy(75.0)
        assert "Adequate" in _rating_for_accuracy(74.9)

    def test_zero(self):
        assert "Poor" in _rating_for_accuracy(0.0)

    def test_hundred(self):
        assert "Excellent" in _rating_for_accuracy(100.0)


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


class TestReportRendering:
    """Test Markdown report section generation."""

    def _make_result(self, correct: int = 8, total: int = 10) -> BenchmarkResult:
        accuracy = correct / total * 100
        items = []
        for i in range(total):
            is_correct = i < correct
            items.append(
                {
                    "index": i,
                    "correct": is_correct,
                    "ground_truth": float(i + 1),
                    "extracted_answer": float(i + 1) if is_correct else None,
                    "extraction_method": "marker" if is_correct else "none",
                    "model_response": f"Some response for {i}",
                }
            )
        return BenchmarkResult(
            plugin_name="gsm8k",
            score=accuracy,
            score_label=f"{accuracy:.1f}% ({correct}/{total})",
            rating=_rating_for_accuracy(accuracy),
            details={
                "correct": correct,
                "total": total,
                "accuracy": round(accuracy, 2),
                "n_shots": 8,
                "dataset_size": 1319,
            },
            item_results=items,
            metadata={"dataset": "openai/gsm8k"},
            duration_seconds=45.0,
            total_tokens=5000,
        )

    def test_report_has_accuracy(self):
        plugin = GSM8KPlugin()
        result = self._make_result(8, 10)
        lines = plugin.render_report_section(result)
        text = "\n".join(lines)
        assert "80.0%" in text
        assert "8/10" in text

    def test_report_has_rating(self):
        plugin = GSM8KPlugin()
        result = self._make_result(8, 10)
        lines = plugin.render_report_section(result)
        text = "\n".join(lines)
        assert "★★★★" in text

    def test_report_has_extraction_methods(self):
        plugin = GSM8KPlugin()
        result = self._make_result(8, 10)
        lines = plugin.render_report_section(result)
        text = "\n".join(lines)
        assert "Extraction Methods" in text
        assert "marker" in text

    def test_report_has_failures(self):
        plugin = GSM8KPlugin()
        result = self._make_result(5, 10)
        lines = plugin.render_report_section(result)
        text = "\n".join(lines)
        assert "Failed Questions" in text
        assert "5 total" in text

    def test_report_no_failures_section_when_perfect(self):
        plugin = GSM8KPlugin()
        result = self._make_result(10, 10)
        lines = plugin.render_report_section(result)
        text = "\n".join(lines)
        assert "Failed Questions" not in text

    def test_report_has_tokens(self):
        plugin = GSM8KPlugin()
        result = self._make_result()
        lines = plugin.render_report_section(result)
        text = "\n".join(lines)
        assert "5,000" in text

    def test_report_has_duration(self):
        plugin = GSM8KPlugin()
        result = self._make_result()
        lines = plugin.render_report_section(result)
        text = "\n".join(lines)
        assert "45.0s" in text

    def test_report_shows_all_failures(self):
        """All failures should be shown (no cap), with collapsible details for >30."""
        plugin = GSM8KPlugin()
        result = self._make_result(correct=0, total=30)
        lines = plugin.render_report_section(result)
        text = "\n".join(lines)
        assert "30 total" in text
        # All 30 failures should appear in the table (index 0-29)
        assert "| 29 |" in text

    def test_report_collapsible_for_large_failure_sets(self):
        """Failures > 30 should be wrapped in <details>."""
        plugin = GSM8KPlugin()
        result = self._make_result(correct=0, total=35)
        lines = plugin.render_report_section(result)
        text = "\n".join(lines)
        assert "<details>" in text
        assert "Show all 35 failures" in text
        assert "</details>" in text

    def test_report_has_error_analysis(self):
        """Reports with failures should include an Error Analysis section."""
        plugin = GSM8KPlugin()
        result = self._make_result(correct=5, total=10)
        lines = plugin.render_report_section(result)
        text = "\n".join(lines)
        assert "Error Analysis" in text
        assert "No answer extracted" in text

    def test_report_has_detailed_failure_samples(self):
        """Reports should include detailed failure deep-dives."""
        plugin = GSM8KPlugin()
        result = self._make_result(correct=5, total=10)
        lines = plugin.render_report_section(result)
        text = "\n".join(lines)
        assert "Detailed Failure Samples" in text
        assert "**Question:**" in text
        assert "**Model response:**" in text


class TestDatasetDownload:
    """Dataset conversion, HF REST API download, and local cache loading."""

    def test_rows_to_items_parses_commas_and_skips_unmarked_rows(self) -> None:
        from tool_eval_bench.plugins.gsm8k.dataset import _rows_to_items

        items = _rows_to_items(
            [
                {"question": "q", "answer": "work #### 1,200"},
                {"question": "bad", "answer": "no marker"},
            ]
        )
        assert len(items) == 1
        assert items[0].ground_truth == 1200.0

    def test_rest_download_and_load_dataset_cache(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import httpx

        from tool_eval_bench.plugins import hf_utils
        from tool_eval_bench.plugins.gsm8k import dataset as gsm

        monkeypatch.setattr(hf_utils, "load_via_datasets_lib", lambda *a, **k: None)

        class Response:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, object]:
                return {
                    "num_rows_total": 2,
                    "rows": [
                        {"row_idx": 0, "row": {"question": "q", "answer": "work #### 2"}},
                        {"row_idx": 1, "row": {"question": "bad", "answer": "none"}},
                    ],
                }

        class Client:
            def __init__(self, **kwargs: object) -> None:
                pass

            def __enter__(self) -> Client:
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def get(self, url: str) -> Response:
                return Response()

        monkeypatch.setattr(httpx, "Client", Client)
        progress: list[tuple[int, int]] = []
        items, method = gsm._download_dataset(on_progress=lambda a, b: progress.append((a, b)))
        assert method == "rest_api"
        assert len(items) == 1 and items[0].ground_truth == 2
        assert progress == [(2, 2)]

        cache = tmp_path / "gsm8k" / "test.jsonl"
        monkeypatch.setattr(gsm, "_CACHE_DIR", cache.parent)
        monkeypatch.setattr(gsm, "_CACHE_FILE", cache)
        monkeypatch.setattr(gsm, "_find_cache_file", lambda: cache)
        assert gsm.load_dataset(force_download=True)[0].ground_truth == 2
        assert gsm.load_dataset()[0].ground_truth == 2
