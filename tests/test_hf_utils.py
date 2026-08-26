"""Tests for hf_utils — HuggingFace download utilities.

Covers:
- Partial cache file I/O (_count_lines, _append_rows_to_file, _read_rows_from_file)
- Resume logic in download_rows_paginated (via mocked network)
- load_via_datasets_lib import fallback
- load_via_datasets_lib with mock datasets library
- Progress callback firing
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

# -- Partial cache helpers --


def test_count_lines_empty(tmp_path: Path) -> None:
    """Empty file returns 0."""
    from tool_eval_bench.plugins.hf_utils import _count_lines

    f = tmp_path / "empty.jsonl"
    f.write_text("", encoding="utf-8")
    assert _count_lines(f) == 0


def test_count_lines_with_data(tmp_path: Path) -> None:
    """Counts only non-empty lines."""
    from tool_eval_bench.plugins.hf_utils import _count_lines

    f = tmp_path / "data.jsonl"
    f.write_text('{"a":1}\n\n{"b":2}\n', encoding="utf-8")
    assert _count_lines(f) == 2


def test_append_and_read_rows(tmp_path: Path) -> None:
    """Append rows then read them back."""
    from tool_eval_bench.plugins.hf_utils import _append_rows_to_file, _read_rows_from_file

    f = tmp_path / "rows.jsonl"
    f.write_text("", encoding="utf-8")  # create empty

    _append_rows_to_file(f, [{"x": 1}, {"x": 2}])
    _append_rows_to_file(f, [{"x": 3}])

    rows = _read_rows_from_file(f)
    assert len(rows) == 3
    assert rows[0] == {"x": 1}
    assert rows[2] == {"x": 3}


def test_read_rows_skips_blank_lines(tmp_path: Path) -> None:
    """Blank lines in JSONL are silently skipped."""
    from tool_eval_bench.plugins.hf_utils import _read_rows_from_file

    f = tmp_path / "rows.jsonl"
    f.write_text('{"a":1}\n\n\n{"b":2}\n\n', encoding="utf-8")
    rows = _read_rows_from_file(f)
    assert len(rows) == 2


# -- load_via_datasets_lib --


def test_load_via_datasets_lib_import_error() -> None:
    """Returns None when datasets library is not installed."""
    from tool_eval_bench.plugins.hf_utils import load_via_datasets_lib

    # The library is not installed in test venv, so should return None
    result = load_via_datasets_lib("openai/gsm8k", "main", "test")
    assert result is None


def test_load_via_datasets_lib_with_mock() -> None:
    """When datasets is available, converts rows to dicts."""
    from tool_eval_bench.plugins.hf_utils import load_via_datasets_lib

    # Create a mock dataset
    mock_items = [
        {"question": "What is 2+2?", "answer": "#### 4"},
        {"question": "What is 3+3?", "answer": "#### 6"},
        {"question": "What is 1+1?", "answer": "#### 2"},
    ]

    mock_ds = MagicMock()
    mock_ds.__len__ = MagicMock(return_value=3)
    mock_ds.__iter__ = MagicMock(return_value=iter(mock_items))

    mock_load = MagicMock(return_value=mock_ds)

    with patch.dict("sys.modules", {"datasets": MagicMock(load_dataset=mock_load)}):
        progress_calls: list[tuple[int, int]] = []
        result = load_via_datasets_lib(
            "openai/gsm8k",
            "main",
            "test",
            on_progress=lambda d, t: progress_calls.append((d, t)),
        )

    assert result is not None
    assert len(result) == 3
    assert result[0]["question"] == "What is 2+2?"
    # Final progress should be (3, 3)
    assert progress_calls[-1] == (3, 3)


def test_load_via_datasets_lib_exception_returns_none() -> None:
    """If datasets.load_dataset raises, returns None (fallback to REST)."""
    from tool_eval_bench.plugins.hf_utils import load_via_datasets_lib

    mock_load = MagicMock(side_effect=RuntimeError("Connection failed"))

    with patch.dict("sys.modules", {"datasets": MagicMock(load_dataset=mock_load)}):
        result = load_via_datasets_lib("cais/mmlu", "all", "test")

    assert result is None


# -- download_rows_paginated (mocked network) --


def _mock_fetch_factory(pages: list[list[dict]], total: int):
    """Create a _fetch_with_retry mock that returns paginated results."""
    call_count = 0

    def mock_fetch(url, *, max_retries=5, on_retry=None):
        nonlocal call_count
        if "info?" in url:
            return {"dataset_info": {"splits": {"test": {"num_examples": total}}}}
        # Paginated rows request
        page_idx = min(call_count, len(pages) - 1)
        rows = [{"row_idx": i, "row": r} for i, r in enumerate(pages[page_idx])]
        call_count += 1
        return {"rows": rows}

    return mock_fetch


def test_download_rows_paginated_basic() -> None:
    """Downloads all rows in a single page."""
    from tool_eval_bench.plugins.hf_utils import download_rows_paginated

    mock = _mock_fetch_factory(
        pages=[[{"q": "a"}, {"q": "b"}, {"q": "c"}]],
        total=3,
    )

    with (
        patch("tool_eval_bench.plugins.hf_utils._fetch_with_retry", mock),
        patch(
            "tool_eval_bench.plugins.hf_utils.get_dataset_info",
            return_value={"dataset_info": {"splits": {"test": {"num_examples": 3}}}},
        ),
    ):
        rows = download_rows_paginated("ds", "cfg", "test", page_size=10)

    assert len(rows) == 3
    assert rows[0] == {"q": "a"}


def test_download_rows_paginated_with_partial_resume(tmp_path: Path) -> None:
    """Resume from partial cache file."""
    from tool_eval_bench.plugins.hf_utils import download_rows_paginated

    partial = tmp_path / "test.partial.jsonl"
    # Write 2 already-downloaded rows
    partial.write_text(
        json.dumps({"q": "a"}) + "\n" + json.dumps({"q": "b"}) + "\n", encoding="utf-8"
    )

    # Mock should only be called for the remaining 1 row
    mock = _mock_fetch_factory(
        pages=[[{"q": "c"}]],
        total=3,
    )

    with (
        patch("tool_eval_bench.plugins.hf_utils._fetch_with_retry", mock),
        patch(
            "tool_eval_bench.plugins.hf_utils.get_dataset_info",
            return_value={"dataset_info": {"splits": {"test": {"num_examples": 3}}}},
        ),
    ):
        rows = download_rows_paginated(
            "ds",
            "cfg",
            "test",
            page_size=10,
            partial_path=partial,
        )

    # Should have all 3 rows (2 from cache + 1 new)
    assert len(rows) == 3


def test_download_rows_paginated_complete_partial(tmp_path: Path) -> None:
    """When partial cache is already complete, returns without network calls."""
    from tool_eval_bench.plugins.hf_utils import download_rows_paginated

    partial = tmp_path / "test.partial.jsonl"
    partial.write_text(
        json.dumps({"q": "a"})
        + "\n"
        + json.dumps({"q": "b"})
        + "\n"
        + json.dumps({"q": "c"})
        + "\n",
        encoding="utf-8",
    )

    # The mock_fetch should NOT be called — partial cache is already complete

    with patch(
        "tool_eval_bench.plugins.hf_utils.get_dataset_info",
        return_value={"dataset_info": {"splits": {"test": {"num_examples": 3}}}},
    ):
        rows = download_rows_paginated(
            "ds",
            "cfg",
            "test",
            page_size=10,
            partial_path=partial,
        )

    assert len(rows) == 3


def test_download_rows_paginated_progress_callback() -> None:
    """Progress callback is called with correct counts."""
    from tool_eval_bench.plugins.hf_utils import download_rows_paginated

    mock = _mock_fetch_factory(
        pages=[[{"q": "a"}, {"q": "b"}]],
        total=2,
    )

    calls: list[tuple[int, int]] = []

    with (
        patch("tool_eval_bench.plugins.hf_utils._fetch_with_retry", mock),
        patch(
            "tool_eval_bench.plugins.hf_utils.get_dataset_info",
            return_value={"dataset_info": {"splits": {"test": {"num_examples": 2}}}},
        ),
    ):
        download_rows_paginated(
            "ds",
            "cfg",
            "test",
            page_size=10,
            on_progress=lambda d, t: calls.append((d, t)),
        )

    assert (2, 2) in calls


# -- Dataset loader integration (datasets lib path) --


def test_gsm8k_rows_to_items() -> None:
    """GSM8K _rows_to_items converts raw dicts correctly."""
    from tool_eval_bench.plugins.gsm8k.dataset import _rows_to_items

    rows = [
        {"question": "What is 2+2?", "answer": "The answer is #### 4"},
        {"question": "What is 3+3?", "answer": "The answer is #### 6"},
    ]
    items = _rows_to_items(rows)
    assert len(items) == 2
    assert items[0].question == "What is 2+2?"
    assert items[0].ground_truth == 4.0
    assert items[1].ground_truth == 6.0


def test_gsm8k_rows_to_items_skips_bad_rows() -> None:
    """Rows without #### pattern are skipped."""
    from tool_eval_bench.plugins.gsm8k.dataset import _rows_to_items

    rows = [
        {"question": "Good", "answer": "#### 42"},
        {"question": "Bad", "answer": "no answer marker here"},
    ]
    items = _rows_to_items(rows)
    assert len(items) == 1


def test_mmlu_rows_to_items() -> None:
    """MMLU _rows_to_items converts raw dicts correctly."""
    from tool_eval_bench.plugins.mmlu.dataset import _rows_to_items

    rows = [
        {"question": "Q1", "subject": "physics", "choices": ["A", "B", "C", "D"], "answer": 2},
    ]
    items = _rows_to_items(rows)
    assert len(items) == 1
    assert items[0].question == "Q1"
    assert items[0].subject == "physics"
    assert items[0].answer == 2


def test_ifeval_rows_to_items() -> None:
    """IFEval _rows_to_items converts raw dicts correctly."""
    from tool_eval_bench.plugins.ifeval.dataset import _rows_to_items

    rows = [
        {
            "key": 42,
            "prompt": "Write a story",
            "instruction_id_list": ["length:min_words"],
            "kwargs": [{"min_words": 100}],
        },
    ]
    items = _rows_to_items(rows)
    assert len(items) == 1
    assert items[0].key == 42
    assert items[0].prompt == "Write a story"
