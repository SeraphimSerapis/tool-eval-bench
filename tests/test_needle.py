"""Needle-in-a-haystack: case construction, placement, grading, and scoring."""

from __future__ import annotations

from typing import Any

import pytest

from tool_eval_bench.cli.plugin_runners import _needle_context_lengths, _needle_depths
from tool_eval_bench.domain.filler import build_haystack_text
from tool_eval_bench.plugins.needle.haystack import (
    NeedleCase,
    _insert_at_depth,
    build_cases,
    build_needle_messages,
    grade_response,
)
from tool_eval_bench.plugins.needle.plugin import NeedlePlugin, _effective_context
from tool_eval_bench.plugins.registry import available_plugins, get_plugin


def _case(**overrides: Any) -> NeedleCase:
    answer = overrides.pop("answer", "K7QM-2XPD-9WLR")
    defaults: dict[str, Any] = {
        "context_tokens": 2048,
        "depth_percent": 0.5,
        "needle": f"The maintenance passphrase for the Helios relay is {answer}.",
        "question": "What is the maintenance passphrase for the Helios relay?",
        "answer": answer,
    }
    return NeedleCase(**{**defaults, **overrides})


def _cases(depths: tuple[float, ...], *, context_tokens: int = 1024) -> list[NeedleCase]:
    """Cases that differ by answer as well as depth, so a stub can tell them apart."""
    return [
        _case(context_tokens=context_tokens, depth_percent=d, answer=f"ANSWER-{i}")
        for i, d in enumerate(depths)
    ]


# ---------------------------------------------------------------------------
# Haystack text
# ---------------------------------------------------------------------------


class TestHaystackText:
    def test_seeded_builds_are_reproducible(self) -> None:
        assert build_haystack_text(1024, seed=7) == build_haystack_text(1024, seed=7)

    def test_different_seeds_diverge(self) -> None:
        # Two cells sharing a token prefix would let a prefix cache answer the
        # second one, which is the failure this benchmark exists to avoid.
        assert build_haystack_text(1024, seed=1) != build_haystack_text(1024, seed=2)

    def test_length_tracks_the_token_target(self) -> None:
        short = build_haystack_text(512, seed=3)
        long = build_haystack_text(4096, seed=3)
        assert len(long) > len(short) * 4

    @pytest.mark.parametrize("target", [0, -1])
    def test_non_positive_target_is_empty(self, target: int) -> None:
        assert build_haystack_text(target, seed=1) == ""


# ---------------------------------------------------------------------------
# Case grid
# ---------------------------------------------------------------------------


class TestBuildCases:
    def test_grid_covers_every_length_and_depth(self) -> None:
        cases = build_cases([1024, 2048], [0.0, 0.5, 1.0], seed=1)
        assert len(cases) == 6
        assert {c.context_tokens for c in cases} == {1024, 2048}
        assert {c.depth_percent for c in cases} == {0.0, 0.5, 1.0}

    def test_ordered_length_major(self) -> None:
        # An interrupted run should still have covered every depth at the
        # lengths it reached, rather than every length at one depth.
        cases = build_cases([1024, 2048], [0.0, 1.0], seed=1)
        assert [c.context_tokens for c in cases] == [1024, 1024, 2048, 2048]

    def test_answers_are_unique_per_cell(self) -> None:
        cases = build_cases([1024, 2048, 4096], [0.0, 0.5, 1.0], seed=42)
        assert len({c.answer for c in cases}) == len(cases)

    def test_needle_states_its_answer(self) -> None:
        for case in build_cases([1024], [0.0, 0.25, 0.5, 0.75, 1.0], seed=9):
            assert case.answer in case.needle

    def test_seeded_grid_is_reproducible(self) -> None:
        first = build_cases([1024], [0.0, 1.0], seed=5)
        second = build_cases([1024], [0.0, 1.0], seed=5)
        assert [c.answer for c in first] == [c.answer for c in second]

    def test_empty_axes_produce_no_cases(self) -> None:
        assert build_cases([], [0.5], seed=1) == []
        assert build_cases([1024], [], seed=1) == []

    def test_cell_id_identifies_the_cell(self) -> None:
        assert _case(context_tokens=8192, depth_percent=0.25).cell_id == "8K@25%"


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------


class TestPlacement:
    HAYSTACK = "One. Two. Three. Four. Five. Six. Seven. Eight. Nine. Ten."

    @pytest.mark.parametrize("depth", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_needle_is_always_present(self, depth: float) -> None:
        placed = _insert_at_depth(self.HAYSTACK, "NEEDLE.", depth)
        assert "NEEDLE." in placed

    def test_depth_zero_places_the_needle_first(self) -> None:
        assert _insert_at_depth(self.HAYSTACK, "NEEDLE.", 0.0).startswith("NEEDLE.")

    def test_depth_one_places_the_needle_last(self) -> None:
        assert _insert_at_depth(self.HAYSTACK, "NEEDLE.", 1.0).rstrip().endswith("NEEDLE.")

    def test_deeper_needles_sit_further_in(self) -> None:
        shallow = _insert_at_depth(self.HAYSTACK, "NEEDLE.", 0.2).index("NEEDLE.")
        deep = _insert_at_depth(self.HAYSTACK, "NEEDLE.", 0.8).index("NEEDLE.")
        assert shallow < deep

    def test_haystack_survives_placement(self) -> None:
        placed = _insert_at_depth(self.HAYSTACK, "NEEDLE.", 0.5)
        for sentence in ("One.", "Five.", "Ten."):
            assert sentence in placed

    def test_single_sentence_haystack_still_places(self) -> None:
        assert "NEEDLE." in _insert_at_depth("Only one sentence.", "NEEDLE.", 0.5)

    def test_messages_carry_the_needle_and_question(self) -> None:
        case = _case()
        messages = build_needle_messages(case, seed=11)
        assert messages[0]["role"] == "system"
        user = messages[1]["content"]
        assert case.needle in user
        assert case.question in user
        assert user.rstrip().endswith(case.question)


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------


class TestGrading:
    def test_exact_answer_passes(self) -> None:
        assert grade_response(_case(), "K7QM-2XPD-9WLR")

    def test_answer_in_a_sentence_passes(self) -> None:
        # Retrieval, not instruction following: the fact was found.
        assert grade_response(_case(), "The passphrase is K7QM-2XPD-9WLR.")

    def test_case_and_punctuation_differences_pass(self) -> None:
        assert grade_response(_case(), "k7qm 2xpd 9wlr")

    def test_wrong_answer_fails(self) -> None:
        assert not grade_response(_case(), "K7QM-2XPD-9WLX")

    def test_refusal_fails(self) -> None:
        assert not grade_response(_case(), "The document does not mention a passphrase.")

    @pytest.mark.parametrize("response", ["", "   "])
    def test_empty_response_fails(self, response: str) -> None:
        assert not grade_response(_case(), response)

    def test_numeric_answer_is_not_matched_by_a_substring(self) -> None:
        # "4821" must not be credited to a response that only says "482".
        assert not grade_response(_case(answer="4821"), "The count was 482.")


# ---------------------------------------------------------------------------
# Grid geometry
# ---------------------------------------------------------------------------


class TestGridGeometry:
    def test_depths_span_both_ends(self) -> None:
        assert _needle_depths(5) == [0.0, 0.25, 0.5, 0.75, 1.0]

    def test_single_depth_probes_the_middle(self) -> None:
        assert _needle_depths(1) == [0.5]

    def test_lengths_are_ascending_and_fit_the_window(self) -> None:
        lengths = _needle_context_lengths(32768, 4)
        assert lengths == sorted(lengths)
        assert len(lengths) == 4
        # Every haystack must leave room for the prompt and the answer.
        assert lengths[-1] < 32768

    def test_tiny_window_yields_one_length(self) -> None:
        assert len(_needle_context_lengths(2048, 4)) == 1


# ---------------------------------------------------------------------------
# Effective context
# ---------------------------------------------------------------------------


class TestEffectiveContext:
    def test_largest_fully_retrieved_length_wins(self) -> None:
        lengths = [1024, 4096, 16384]
        by_length = {"1024": 100.0, "4096": 100.0, "16384": 60.0}
        assert _effective_context(lengths, by_length) == 4096

    def test_none_when_every_length_missed(self) -> None:
        assert _effective_context([1024], {"1024": 80.0}) is None

    def test_a_gap_does_not_hide_a_larger_pass(self) -> None:
        # Retrieval is not always monotonic; report the largest length that
        # actually passed rather than stopping at the first dip.
        lengths = [1024, 4096, 16384]
        by_length = {"1024": 100.0, "4096": 40.0, "16384": 100.0}
        assert _effective_context(lengths, by_length) == 16384


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------


class _StubResponse:
    def __init__(self, content: str) -> None:
        self.content = content
        self.reasoning = None
        self.prompt_tokens = 100
        self.completion_tokens = 10


class _StubAdapter:
    """Answers correctly except for the cells named in *miss*."""

    def __init__(self, *, miss: set[str] | None = None, raises: set[str] | None = None) -> None:
        self.miss = miss or set()
        self.raises = raises or set()
        self.calls = 0

    async def chat_completion(self, **kwargs: Any) -> _StubResponse:
        self.calls += 1
        user = kwargs["messages"][1]["content"]
        answer = user.split("relay is ")[1].split(".")[0] if "relay is " in user else ""
        for cell in self.raises:
            if cell in user:
                raise RuntimeError("server exploded")
        if any(marker in user for marker in self.miss):
            return _StubResponse("I could not find it.")
        return _StubResponse(answer)


@pytest.mark.asyncio
class TestNeedlePlugin:
    async def test_all_needles_retrieved_scores_100(self) -> None:
        cases = _cases((0.0, 1.0))
        result = await NeedlePlugin().run(
            _StubAdapter(),
            model="m",
            base_url="http://x",
            cases=cases,
        )
        assert result.score == 100.0
        assert result.details["retrieved"] == 2
        assert result.details["effective_context"] == 1024

    async def test_missed_needle_lowers_the_score(self) -> None:
        cases = _cases((0.0, 1.0))
        adapter = _StubAdapter(miss={cases[1].answer})
        result = await NeedlePlugin().run(adapter, model="m", base_url="http://x", cases=cases)
        assert result.score == 50.0
        assert result.details["effective_context"] is None

    async def test_request_failure_counts_as_a_miss_not_a_crash(self) -> None:
        cases = _cases((0.0, 1.0))
        result = await NeedlePlugin().run(
            _StubAdapter(raises={cases[0].answer}),
            model="m",
            base_url="http://x",
            cases=cases,
        )
        assert result.details["errors"] == 1
        assert result.details["retrieved"] == 1
        assert result.details["incomplete"] is True
        assert result.details["completion_rate"] == 50.0

    async def test_empty_grid_returns_a_zero_result(self) -> None:
        result = await NeedlePlugin().run(_StubAdapter(), model="m", base_url="http://x", cases=[])
        assert result.score == 0.0
        assert result.details["total"] == 0

    async def test_zero_concurrency_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="concurrency"):
            await NeedlePlugin().run(
                _StubAdapter(),
                model="m",
                base_url="http://x",
                cases=[_case()],
                concurrency=0,
            )

    async def test_progress_is_reported_once_per_cell(self) -> None:
        cases = _cases((0.0, 0.5, 1.0))
        seen: list[int] = []

        async def on_progress(current: int, total: int, info: dict) -> None:
            seen.append(current)

        await NeedlePlugin().run(
            _StubAdapter(),
            model="m",
            base_url="http://x",
            cases=cases,
            on_progress=on_progress,
        )
        assert sorted(seen) == [1, 2, 3]

    async def test_report_section_renders_the_grid(self) -> None:
        cases = _cases((0.0, 1.0))
        plugin = NeedlePlugin()
        result = await plugin.run(
            _StubAdapter(miss={cases[1].answer}),
            model="m",
            base_url="http://x",
            cases=cases,
            context_size=4096,
        )
        report = "\n".join(plugin.render_report_section(result))
        assert "Needle in a Haystack" in report
        assert "Retrieval Grid" in report
        assert "Missed Needles" in report
        assert cases[1].answer in report


# ---------------------------------------------------------------------------
# CLI runner
# ---------------------------------------------------------------------------


def _needle_args(**overrides: Any) -> Any:
    import argparse

    defaults: dict[str, Any] = {
        "needle_depths": 2,
        "needle_lengths": 2,
        "context_size": 8192,
        "metrics_url": None,
        "seed": 1,
        "parallel": 2,
        "temperature": 0.0,
        "timeout": 1.0,
        "format": None,
    }
    return argparse.Namespace(**{**defaults, **overrides})


class TestNeedleRunner:
    def test_run_persists_and_reports(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
        from rich.console import Console

        from tool_eval_bench.cli import plugin_runners
        from tool_eval_bench.domain.plugin import BenchmarkResult

        async def fake_run(self: Any, adapter: Any, *, on_progress: Any = None, **kw: Any) -> Any:
            await on_progress(1, 2, {"cell_id": "1K@0%", "found": True, "model_response": "ok"})
            await on_progress(2, 2, {"cell_id": "5K@100%", "found": False, "model_response": "no"})
            return BenchmarkResult(
                "needle",
                50,
                "50%",
                "★★ Weak",
                details={
                    "total": 2,
                    "retrieved": 1,
                    "errors": 0,
                    "answered": 2,
                    "completion_rate": 100.0,
                    "accuracy": 50.0,
                    "context_size": 8192,
                    "context_lengths": [1024, 5120],
                    "depths": [0.0, 1.0],
                    "effective_context": 1024,
                },
                item_results=[
                    {"context_tokens": 1024, "depth_percent": 0.0, "found": True},
                    {"context_tokens": 5120, "depth_percent": 1.0, "found": False},
                ],
                duration_seconds=1,
                total_tokens=10,
            )

        monkeypatch.setattr(NeedlePlugin, "run", fake_run)
        monkeypatch.setattr(NeedlePlugin, "render_report_section", lambda self, r: ["report"])
        monkeypatch.setattr(plugin_runners, "_with_config_fingerprint", lambda value: value)
        monkeypatch.setattr(plugin_runners, "_metadata_for_storage", lambda value: {})
        persisted: list[dict] = []
        monkeypatch.setattr(plugin_runners, "_persist_plugin_run", persisted.append)

        console = Console(record=True, width=180)
        plugin_runners._run_needle_benchmark(
            console, "m", "Display", "url", None, _needle_args(), output_dir=str(tmp_path)
        )

        assert [item["run_type"] for item in persisted] == ["needle"]
        output = console.export_text()
        assert "Retrieval Accuracy" in output
        assert "Effective context" in output
        assert "Retrieval grid" in output

    def test_undetectable_context_size_stops_before_running(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from rich.console import Console

        from tool_eval_bench.cli import plugin_runners
        from tool_eval_bench.runner import context_pressure

        async def no_context(*a: Any, **k: Any) -> None:
            return None

        monkeypatch.setattr(context_pressure, "detect_context_size", no_context)

        console = Console(record=True, width=180)
        plugin_runners._run_needle_benchmark(
            console, "m", "Display", "url", None, _needle_args(context_size=None)
        )

        assert "Could not auto-detect the context window" in console.export_text()


# ---------------------------------------------------------------------------
# Flat invocation
# ---------------------------------------------------------------------------


def _parse(argv: list[str]) -> Any:
    from tool_eval_bench.cli.legacy_parser import make_parser
    from tool_eval_bench.cli.parser import parse_cli_args

    _, args = parse_cli_args(make_parser, argv)
    return args


def _dispatch(args: Any) -> tuple[list[str], bool]:
    """Return which plugins ran, and whether the CLI stopped before scenarios."""
    from rich.console import Console

    from tool_eval_bench.cli.plugin_runners import run_selected_plugins

    calls: list[str] = []

    def runner(name: str) -> Any:
        def run(*_a: Any, **_k: Any) -> None:
            calls.append(name)

        return run

    stopped = run_selected_plugins(
        Console(),
        "model",
        "display",
        "http://server",
        None,
        args,
        runners={n: runner(n) for n in ("gsm8k", "mmlu", "ifeval", "needle")},
        extra_params=None,
        output_dir=None,
        run_context=None,
    )
    return calls, stopped


class TestFlatInvocation:
    """`--needle` must work without the `bench` subcommand, the way `--perf` does."""

    def test_bare_flag_needs_no_subcommand(self) -> None:
        args = _parse(["--needle"])
        assert args.needle is True
        assert _dispatch(args) == (["needle"], False)

    def test_chains_with_the_other_top_level_flags(self) -> None:
        args = _parse(["--hardmode", "--seed", "42", "--perf", "--needle"])
        assert (args.hardmode, args.seed, args.perf, args.needle) == (True, 42, True, True)
        # --perf hands its samples on rather than stopping, so needle still runs
        # and the tool-call scenarios still follow it.
        assert _dispatch(args) == (["needle"], False)

    def test_grid_flags_work_flat(self) -> None:
        args = _parse(["--needle", "--needle-depths", "8", "--needle-lengths", "6"])
        assert (args.needle_depths, args.needle_lengths) == (8, 6)

    def test_needle_only_stops_before_tool_scenarios(self) -> None:
        assert _dispatch(_parse(["--needle-only"])) == (["needle"], True)

    def test_runs_last_in_the_stable_plugin_order(self) -> None:
        calls, _ = _dispatch(_parse(["--needle", "--gsm8k", "--ifeval"]))
        assert calls == ["gsm8k", "ifeval", "needle"]

    def test_bench_subcommand_remains_equivalent(self) -> None:
        flat = _parse(["--needle", "--seed", "42"])
        via_bench = _parse(["bench", "--needle", "--seed", "42"])
        assert (flat.needle, flat.seed) == (via_bench.needle, via_bench.seed)

    def test_defaults_off(self) -> None:
        args = _parse(["--short"])
        assert args.needle is False and args.needle_only is False
        assert _dispatch(args) == ([], False)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_needle_is_a_registered_plugin(self) -> None:
        assert "needle" in available_plugins()

    def test_registry_returns_the_plugin(self) -> None:
        assert get_plugin("needle").name == "needle"
