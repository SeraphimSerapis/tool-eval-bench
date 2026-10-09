"""Scoring contracts shared by the accuracy plugins (GSM8K, MMLU, IFEval, needle).

These run the real plugins against small preloaded datasets.  The rejected
request cases go through the real ``OpenAICompatibleAdapter`` over an
``httpx.MockTransport``, because the bug lived in how the adapter's soft
``[server error N]`` result reached the graders.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from tool_eval_bench.adapters.openai_compat import OpenAICompatibleAdapter
from tool_eval_bench.domain.adapters import ChatCompletionResult
from tool_eval_bench.domain.plugin import final_answer_text
from tool_eval_bench.plugins.gsm8k.dataset import GSM8KItem
from tool_eval_bench.plugins.gsm8k.plugin import GSM8KPlugin
from tool_eval_bench.plugins.ifeval.dataset import IFEvalItem
from tool_eval_bench.plugins.ifeval.plugin import IFEvalPlugin
from tool_eval_bench.plugins.mmlu.dataset import SUBJECT_CATEGORIES, MMLUItem
from tool_eval_bench.plugins.mmlu.plugin import MMLUPlugin, resolve_subjects, select_items
from tool_eval_bench.plugins.needle.haystack import build_cases
from tool_eval_bench.plugins.needle.plugin import NeedlePlugin

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rejecting_adapter() -> OpenAICompatibleAdapter:
    """A real adapter whose server answers every request with HTTP 401."""

    def always_401(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "Invalid API key 401"}})

    adapter = OpenAICompatibleAdapter()
    adapter._client = httpx.AsyncClient(transport=httpx.MockTransport(always_401))
    return adapter


class _FixedAdapter:
    """Returns the same ``ChatCompletionResult`` for every request."""

    def __init__(self, result: ChatCompletionResult) -> None:
        self.result = result

    async def chat_completion(self, **kwargs: Any) -> ChatCompletionResult:
        return self.result


def _truncated(reasoning: str) -> ChatCompletionResult:
    """A turn the token budget cut off mid-reasoning, before any content."""
    return ChatCompletionResult(content="", reasoning=reasoning, finish_reason="length")


def _mmlu_item(index: int, subject: str, answer: int = 0) -> MMLUItem:
    return MMLUItem(index, f"q{index}", subject, ["w", "x", "y", "z"], answer)


def _mmlu_pool(counts: dict[str, int]) -> list[MMLUItem]:
    """Items grouped by subject, in the subject-sorted layout of ``cais/mmlu``."""
    items: list[MMLUItem] = []
    for subject, n in counts.items():
        items.extend(_mmlu_item(len(items), subject) for _ in range(n))
    return items


_IFEVAL_ITEMS = [
    IFEvalItem(1, "Write a haiku without commas.", ["punctuation:no_comma"], [{}]),
    IFEvalItem(
        2,
        "Answer in under 50 words.",
        ["length_constraints:number_words"],
        [{"num_words": 50, "relation": "less than"}],
    ),
]


# ---------------------------------------------------------------------------
# Rejected requests are errors, never graded answers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRejectedRequests:
    async def test_gsm8k(self) -> None:
        # Ground truth 401 would match the "[server error 401]" text if graded.
        items = [GSM8KItem(0, "How many?", "#### 401", 401.0)]
        result = await GSM8KPlugin().run(
            _rejecting_adapter(), model="m", base_url="http://x", n_shots=0, _preloaded_items=items
        )
        assert result.details["errors"] == 1
        assert result.details["correct"] == 0
        assert result.details["status"] == "incomplete"

    async def test_mmlu(self) -> None:
        items = [_mmlu_item(0, "anatomy")]
        result = await MMLUPlugin().run(
            _rejecting_adapter(),
            model="m",
            base_url="http://x",
            n_shots=0,
            _preloaded_items={"test": items, "dev": []},
        )
        assert result.details["errors"] == 1
        assert result.details["correct"] == 0
        assert result.item_results[0]["is_error"] is True

    async def test_ifeval(self) -> None:
        # The error body has no commas and is under 50 words, so grading it
        # would pass both prompts.
        result = await IFEvalPlugin().run(
            _rejecting_adapter(), model="m", base_url="http://x", _preloaded_items=_IFEVAL_ITEMS
        )
        assert result.details["errors"] == 2
        assert result.details["prompts_passed"] == 0

    async def test_needle(self) -> None:
        cases = build_cases([1024], [0.5], seed=1)
        result = await NeedlePlugin().run(
            _rejecting_adapter(), model="m", base_url="http://x", cases=cases
        )
        assert result.details["errors"] == 1
        assert result.details["retrieved"] == 0


# ---------------------------------------------------------------------------
# Truncated turns
# ---------------------------------------------------------------------------


class TestFinalAnswerText:
    def test_content_wins(self) -> None:
        result = ChatCompletionResult(content="B", reasoning="A", finish_reason="length")
        assert final_answer_text(result) == "B"

    def test_reasoning_stands_in_when_generation_finished(self) -> None:
        result = ChatCompletionResult(content="", reasoning="#### 4", finish_reason="stop")
        assert final_answer_text(result) == "#### 4"

    def test_unknown_finish_reason_keeps_the_fallback(self) -> None:
        result = ChatCompletionResult(content="", reasoning="#### 4", finish_reason=None)
        assert final_answer_text(result) == "#### 4"

    def test_empty_content_cut_off_by_length_is_truncated(self) -> None:
        assert final_answer_text(_truncated("so far 12")) is None

    def test_whitespace_content_cut_off_by_length_is_truncated(self) -> None:
        result = ChatCompletionResult(content="\n", reasoning="so far 12", finish_reason="length")
        assert final_answer_text(result) is None

    def test_whitespace_content_falls_back_to_reasoning(self) -> None:
        result = ChatCompletionResult(content=" \n", reasoning="#### 4", finish_reason="stop")
        assert final_answer_text(result) == "#### 4"


@pytest.mark.asyncio
class TestTruncatedTurns:
    async def test_gsm8k_does_not_grade_unfinished_reasoning(self) -> None:
        # The stray "12" in the unfinished reasoning matches the ground truth.
        items = [GSM8KItem(0, "How many?", "#### 12", 12.0)]
        result = await GSM8KPlugin().run(
            _FixedAdapter(_truncated("First, 12 apples, then")),
            model="m",
            base_url="http://x",
            n_shots=0,
            _preloaded_items=items,
        )
        assert result.details["correct"] == 0
        assert result.details["truncated"] == 1
        assert result.details["errors"] == 0
        row = result.item_results[0]
        assert row["truncated"] is True
        assert row["extraction_method"] == "truncated"
        report = "\n".join(GSM8KPlugin().render_report_section(result))
        assert "**Truncated**: 1" in report

    async def test_mmlu_does_not_grade_unfinished_reasoning(self) -> None:
        items = [_mmlu_item(0, "anatomy", answer=0)]
        result = await MMLUPlugin().run(
            _FixedAdapter(_truncated("Option A looks")),
            model="m",
            base_url="http://x",
            n_shots=0,
            _preloaded_items={"test": items, "dev": []},
        )
        assert result.details["correct"] == 0
        assert result.details["truncated"] == 1
        assert result.item_results[0]["truncated"] is True
        report = "\n".join(MMLUPlugin().render_report_section(result))
        assert "**Truncated**: 1" in report

    async def test_ifeval_fails_every_instruction(self) -> None:
        result = await IFEvalPlugin().run(
            _FixedAdapter(_truncated("no commas here")),
            model="m",
            base_url="http://x",
            _preloaded_items=_IFEVAL_ITEMS,
        )
        assert result.details["prompts_passed"] == 0
        assert result.details["truncated"] == 2
        assert result.details["instructions_passed"] == 0
        assert result.details["instructions_total"] == 2
        assert result.details["constraint_types"]["punctuation:no_comma"]["passed"] == 0
        report = "\n".join(IFEvalPlugin().render_report_section(result))
        assert "**Truncated**: 2" in report

    async def test_finished_reasoning_is_still_graded(self) -> None:
        items = [GSM8KItem(0, "How many?", "#### 12", 12.0)]
        result = await GSM8KPlugin().run(
            _FixedAdapter(
                ChatCompletionResult(content="", reasoning="#### 12", finish_reason="stop")
            ),
            model="m",
            base_url="http://x",
            n_shots=0,
            _preloaded_items=items,
        )
        assert result.details["correct"] == 1
        assert result.details["truncated"] == 0


# ---------------------------------------------------------------------------
# Empty selections fail instead of saving 0/0
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestEmptySelection:
    async def test_gsm8k(self) -> None:
        with pytest.raises(ValueError, match="no questions"):
            await GSM8KPlugin().run(
                _FixedAdapter(ChatCompletionResult(content="#### 1")),
                model="m",
                base_url="http://x",
                _preloaded_items=[],
            )

    async def test_ifeval(self) -> None:
        with pytest.raises(ValueError, match="no prompts"):
            await IFEvalPlugin().run(
                _FixedAdapter(ChatCompletionResult(content="ok")),
                model="m",
                base_url="http://x",
                _preloaded_items=[],
            )

    async def test_mmlu_filter_matching_nothing(self) -> None:
        items = [_mmlu_item(0, "anatomy")]
        with pytest.raises(ValueError, match="no questions"):
            await MMLUPlugin().run(
                _FixedAdapter(ChatCompletionResult(content="A")),
                model="m",
                base_url="http://x",
                subjects=["philosophy"],
                _preloaded_items={"test": items, "dev": []},
            )

    async def test_mmlu_unknown_subject_fails_before_loading(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tool_eval_bench.plugins.mmlu import plugin as mmlu_plugin

        def fail_download(*args: Any, **kwargs: Any) -> list[MMLUItem]:
            raise AssertionError("an unknown subject must not trigger a download")

        monkeypatch.setattr(mmlu_plugin, "load_dataset", fail_download)
        with pytest.raises(ValueError, match="anatomyy"):
            await MMLUPlugin().run(
                _FixedAdapter(ChatCompletionResult(content="A")),
                model="m",
                base_url="http://x",
                subjects=["anatomyy"],
            )


# ---------------------------------------------------------------------------
# MMLU subject resolution and stratified sampling
# ---------------------------------------------------------------------------


class TestResolveSubjects:
    def test_category_matches_in_any_case(self) -> None:
        stem = {s for s, cat in SUBJECT_CATEGORIES.items() if cat == "STEM"}
        assert resolve_subjects(["stem"]) == stem
        assert resolve_subjects(["STEM"]) == stem

    def test_exact_subject(self) -> None:
        assert resolve_subjects(["anatomy"]) == {"anatomy"}

    def test_subject_matches_in_any_case(self) -> None:
        assert resolve_subjects(["Anatomy"]) == {"anatomy"}

    def test_unknown_name_lists_the_valid_choices(self) -> None:
        with pytest.raises(ValueError) as exc:
            resolve_subjects(["anatomy", "astrology"])
        message = str(exc.value)
        assert "astrology" in message
        assert "STEM" in message
        assert "anatomy" in message.split("Subjects:")[1]


class TestSelectItems:
    _COUNTS = {"abstract_algebra": 100, "anatomy": 135, "astronomy": 152, "virology": 166}

    def test_quotas_are_proportional_across_every_subject(self) -> None:
        pool = _mmlu_pool(self._COUNTS)
        selected = select_items(pool, None, 50)
        per_subject = {s: sum(1 for it in selected if it.subject == s) for s in self._COUNTS}
        # Exact shares of 50 over 553: 9.04, 12.21, 13.74, 15.01.
        assert per_subject == {
            "abstract_algebra": 9,
            "anatomy": 12,
            "astronomy": 14,
            "virology": 15,
        }
        assert len(selected) == 50

    def test_takes_the_first_items_of_each_subject_in_dataset_order(self) -> None:
        pool = _mmlu_pool(self._COUNTS)
        selected = select_items(pool, None, 50)
        assert [it.index for it in selected] == sorted(it.index for it in selected)
        anatomy = [it.index for it in selected if it.subject == "anatomy"]
        assert anatomy == list(range(100, 112))

    def test_remainder_ties_go_to_the_alphabetically_first_subject(self) -> None:
        pool = _mmlu_pool({"anatomy": 1, "virology": 1})
        assert [it.subject for it in select_items(pool, None, 1)] == ["anatomy"]

    def test_is_deterministic(self) -> None:
        pool = _mmlu_pool(self._COUNTS)
        assert select_items(pool, None, 37) == select_items(pool, None, 37)

    def test_filter_applies_before_the_limit(self) -> None:
        pool = _mmlu_pool(self._COUNTS)
        selected = select_items(pool, ["anatomy", "virology"], 10)
        assert {it.subject for it in selected} == {"anatomy", "virology"}
        assert len(selected) == 10

    @pytest.mark.parametrize("limit", [0, 553, 1000])
    def test_no_limit_or_a_limit_past_the_pool_keeps_everything(self, limit: int) -> None:
        pool = _mmlu_pool(self._COUNTS)
        assert select_items(pool, None, limit) == pool

    @pytest.mark.asyncio
    async def test_plugin_records_the_sampling(self) -> None:
        pool = _mmlu_pool({"anatomy": 4, "virology": 4})
        result = await MMLUPlugin().run(
            _FixedAdapter(ChatCompletionResult(content="A")),
            model="m",
            base_url="http://x",
            n_shots=0,
            limit=4,
            _preloaded_items={"test": pool, "dev": []},
        )
        assert result.details["total"] == 4
        assert result.details["sampling"] == "stratified"
        assert {r["subject"] for r in result.item_results} == {"anatomy", "virology"}


# ---------------------------------------------------------------------------
# Run identity: shuffle seed, dataset revision, item hash
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRunIdentity:
    _ITEMS = [GSM8KItem(i, f"q{i}", f"#### {i}", float(i)) for i in range(20)]

    async def _run(self, **kwargs: Any) -> Any:
        return await GSM8KPlugin().run(
            _FixedAdapter(ChatCompletionResult(content="#### 0")),
            model="m",
            base_url="http://x",
            n_shots=0,
            limit=5,
            _preloaded_items=self._ITEMS,
            **kwargs,
        )

    async def test_unseeded_shuffle_records_the_seed_it_drew(self) -> None:
        first = await self._run(shuffle=True)
        seed = first.details["shuffle_seed"]
        assert isinstance(seed, int)
        # Replaying the recorded seed selects the same questions.
        replay = await self._run(shuffle=True, seed=seed)
        assert replay.details["items_sha256"] == first.details["items_sha256"]
        assert [r["index"] for r in replay.item_results] == [r["index"] for r in first.item_results]

    async def test_seeded_shuffle_records_the_given_seed(self) -> None:
        result = await self._run(shuffle=True, seed=42)
        assert result.details["shuffle_seed"] == 42

    async def test_unshuffled_run_has_no_shuffle_seed(self) -> None:
        result = await self._run()
        assert result.details["shuffle_seed"] is None

    async def test_item_hash_tracks_the_graded_items(self) -> None:
        a = await self._run()
        b = await self._run()
        c = await self._run(shuffle=True, seed=3)
        assert a.details["items_sha256"] == b.details["items_sha256"]
        assert a.details["items_sha256"] != c.details["items_sha256"]

    async def test_mmlu_item_hash_covers_the_few_shot_exemplars(self) -> None:
        test = [_mmlu_item(0, "anatomy")]

        async def run(dev: list[MMLUItem]) -> Any:
            return await MMLUPlugin().run(
                _FixedAdapter(ChatCompletionResult(content="A")),
                model="m",
                base_url="http://x",
                n_shots=1,
                _preloaded_items={"test": test, "dev": dev},
            )

        a = await run([_mmlu_item(100, "anatomy")])
        b = await run([_mmlu_item(101, "anatomy")])
        assert a.details["items_sha256"] != b.details["items_sha256"]

    async def test_preloaded_data_without_a_manifest_has_unknown_revision(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        monkeypatch.chdir(tmp_path)
        result = await self._run()
        assert result.details["dataset_revision"] == "unknown"


def test_fresh_download_records_the_pinned_revision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    from tool_eval_bench.plugins.gsm8k import dataset as gsm8k_dataset

    monkeypatch.chdir(tmp_path)
    items = [GSM8KItem(0, "q0", "#### 1", 1.0)]
    monkeypatch.setattr(gsm8k_dataset, "_download_dataset", lambda **_kw: (items, "datasets"))
    gsm8k_dataset.load_dataset()
    assert gsm8k_dataset.dataset_revision() == gsm8k_dataset._REVISION


# ---------------------------------------------------------------------------
# CLI: config identity and up-front validation
# ---------------------------------------------------------------------------


def _parse(tmp_path: Any, *flags: str) -> Any:
    from tool_eval_bench.cli.legacy_parser import _make_parser

    return _make_parser().parse_args([*flags, "--output-dir", str(tmp_path)])


def test_cli_rejects_an_unknown_mmlu_subject_before_downloading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    from rich.console import Console

    from tool_eval_bench.cli import plugin_runners

    def fail_download(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("an unknown subject must not trigger a download")

    monkeypatch.setattr(plugin_runners, "load_dataset_with_progress", fail_download)
    args = _parse(tmp_path, "--mmlu-only", "--mmlu-subjects", "anatomy,anatomyy")
    console = Console(record=True, width=400)
    with pytest.raises(SystemExit) as exited:
        plugin_runners._run_mmlu_benchmark(
            console, "m", "M", "http://x/v1", None, args, output_dir=str(tmp_path)
        )
    assert exited.value.code == 1
    assert "anatomyy" in console.export_text()


def test_cli_rejects_an_empty_mmlu_subject_list_before_downloading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    from rich.console import Console

    from tool_eval_bench.cli import plugin_runners

    def fail_download(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("an empty subject list must not trigger a download")

    monkeypatch.setattr(plugin_runners, "load_dataset_with_progress", fail_download)
    args = _parse(tmp_path, "--mmlu-only", "--mmlu-subjects", " , ")
    console = Console(record=True, width=400)
    with pytest.raises(SystemExit) as exited:
        plugin_runners._run_mmlu_benchmark(
            console, "m", "M", "http://x/v1", None, args, output_dir=str(tmp_path)
        )
    assert exited.value.code == 1
    assert "at least one" in console.export_text()


def test_cli_normalises_mmlu_subjects_in_the_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    from rich.console import Console

    from tool_eval_bench.adapters import factory
    from tool_eval_bench.application import run_queries
    from tool_eval_bench.cli import plugin_runners
    from tool_eval_bench.domain.plugin import BenchmarkResult

    async def fake_run(self: Any, adapter: Any, **kwargs: Any) -> BenchmarkResult:
        details = {"total": 1, "correct": 1, "errors": 0, "answered": 1}
        return BenchmarkResult("mmlu", 100.0, "100%", "★★★★★ Excellent", details=details)

    monkeypatch.setattr(MMLUPlugin, "run", fake_run)
    pool = [_mmlu_item(0, "anatomy")]
    monkeypatch.setattr(plugin_runners, "load_dataset_with_progress", lambda *a, **k: pool)
    monkeypatch.setattr(factory, "build_adapter", lambda *a, **k: object())
    persisted: list[dict[str, Any]] = []
    monkeypatch.setattr(run_queries, "persist_run", persisted.append)

    for flag in ("STEM", "stem, STEM"):
        args = _parse(tmp_path, "--mmlu-only", "--mmlu-subjects", flag)
        plugin_runners._run_mmlu_benchmark(
            Console(record=True), "m", "M", "http://x/v1", None, args, output_dir=str(tmp_path)
        )

    configs = [row["config"] for row in persisted]
    assert [c["subjects"] for c in configs] == ["stem", "stem"]
    assert configs[0]["config_fingerprint"] == configs[1]["config_fingerprint"]


def test_cli_config_separates_runs_by_extra_params(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    from rich.console import Console

    from tool_eval_bench.adapters import factory
    from tool_eval_bench.application import run_queries
    from tool_eval_bench.cli import plugin_runners
    from tool_eval_bench.domain.plugin import BenchmarkResult

    async def fake_run(self: Any, adapter: Any, **kwargs: Any) -> BenchmarkResult:
        details = {"total": 1, "correct": 1, "errors": 0, "answered": 1, "shuffle_seed": None}
        return BenchmarkResult("gsm8k", 100.0, "100%", "★★★★★ Excellent", details=details)

    monkeypatch.setattr(GSM8KPlugin, "run", fake_run)
    monkeypatch.setattr(plugin_runners, "load_dataset_with_progress", lambda *a, **k: [object()])
    monkeypatch.setattr(factory, "build_adapter", lambda *a, **k: object())
    persisted: list[dict[str, Any]] = []
    monkeypatch.setattr(run_queries, "persist_run", persisted.append)

    args = _parse(tmp_path, "--gsm8k-only")
    for extra in (None, {"top_k": 20}):
        plugin_runners._run_gsm8k_benchmark(
            Console(record=True),
            "m",
            "M",
            "http://x/v1",
            None,
            args,
            extra_params=extra,
            output_dir=str(tmp_path),
        )

    configs = [row["config"] for row in persisted]
    assert [c["extra_params"] for c in configs] == [None, {"top_k": 20}]
    assert configs[0]["config_fingerprint"] != configs[1]["config_fingerprint"]
