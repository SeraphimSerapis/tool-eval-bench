"""Canonical form of a system prompt override.

The same words must produce the same run whichever entry point supplied them, so
these rules are enforced in one place and asserted here directly.
"""

from __future__ import annotations

import pytest

from tool_eval_bench.utils.system_prompt import MAX_SYSTEM_PROMPT_BYTES, normalize_system_prompt


def test_plain_text_is_returned_unchanged() -> None:
    assert normalize_system_prompt("You are a strict evaluator.") == "You are a strict evaluator."


def test_trailing_whitespace_is_dropped() -> None:
    """A file ends in a newline, an inline flag does not: one cohort, not two."""
    assert normalize_system_prompt("Be strict.\n") == "Be strict."
    assert normalize_system_prompt("Be strict.  \n\n") == "Be strict."


def test_leading_byte_order_mark_is_dropped() -> None:
    assert normalize_system_prompt("\ufeffBe strict.") == "Be strict."


def test_internal_whitespace_is_preserved() -> None:
    """Only the edges are normalized; the prompt's own shape is the user's."""
    assert normalize_system_prompt("  Rules:\n  - one\n  - two\n") == "  Rules:\n  - one\n  - two"


@pytest.mark.parametrize("blank", ["", "   ", "\n\t ", "\ufeff  \n"])
def test_blank_prompts_are_rejected(blank: str) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        normalize_system_prompt(blank)


def test_prompt_at_the_limit_is_accepted() -> None:
    assert len(normalize_system_prompt("x" * MAX_SYSTEM_PROMPT_BYTES).encode("utf-8")) == (
        MAX_SYSTEM_PROMPT_BYTES
    )


def test_oversized_prompt_is_rejected() -> None:
    """The text is persisted in the run config and metadata, so it is bounded."""
    with pytest.raises(ValueError, match="exceeds"):
        normalize_system_prompt("x" * (MAX_SYSTEM_PROMPT_BYTES + 1))


def test_the_limit_counts_utf8_bytes_not_characters() -> None:
    """A multi-byte prompt that fits in characters can still blow the byte budget."""
    with pytest.raises(ValueError, match="exceeds"):
        normalize_system_prompt("\u00e9" * MAX_SYSTEM_PROMPT_BYTES)


# ---------------------------------------------------------------------------
# CLI flag resolution, file I/O errors, mode gating, and resume mismatch
# ---------------------------------------------------------------------------


def _prompt_parser_and_args(argv: list[str]):
    from tool_eval_bench.cli.legacy_parser import _make_parser

    parser = _make_parser()
    return parser, parser.parse_args(argv)


def _resolved_prompt(argv: list[str]) -> str | None:
    from tool_eval_bench.cli.dispatch import _resolve_system_prompt

    parser, args = _prompt_parser_and_args(argv)
    _resolve_system_prompt(args, parser)
    return args.system_prompt


def _prompt_error(argv: list[str]) -> str:
    """Return the message argparse printed before exiting on these flags."""
    import contextlib
    import io

    from tool_eval_bench.cli.dispatch import _resolve_system_prompt

    parser, args = _prompt_parser_and_args(argv)
    stderr = io.StringIO()
    with pytest.raises(SystemExit), contextlib.redirect_stderr(stderr):
        _resolve_system_prompt(args, parser)
    return stderr.getvalue()


def test_system_prompt_file_is_resolved_and_normalized(tmp_path) -> None:
    """The file variant lands on args.system_prompt in canonical form."""
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("You are a strict contract evaluator.\n", encoding="utf-8")

    assert _resolved_prompt(["--system-prompt-file", str(prompt_file)]) == (
        "You are a strict contract evaluator."
    )
    assert _resolved_prompt(["--system-prompt", "Be terse."]) == "Be terse."
    assert _resolved_prompt([]) is None


def test_same_text_from_file_or_inline_is_one_cohort(tmp_path) -> None:
    """A file ends in a newline and an inline flag does not; the run is the same."""
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("Be strict.\n", encoding="utf-8")

    assert _resolved_prompt(["--system-prompt-file", str(prompt_file)]) == _resolved_prompt(
        ["--system-prompt", "Be strict."]
    )


def test_utf8_bom_is_stripped_from_a_prompt_file(tmp_path) -> None:
    """A BOM is invisible to the reader but would otherwise split the cohort."""
    prompt_file = tmp_path / "bom.txt"
    prompt_file.write_bytes(b"\xef\xbb\xbfBe strict.")

    assert _resolved_prompt(["--system-prompt-file", str(prompt_file)]) == "Be strict."


def test_system_prompt_flags_are_mutually_exclusive(tmp_path) -> None:
    """Both flags together is an error for being both, not for a missing file."""
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("Be strict.\n", encoding="utf-8")

    message = _prompt_error(["--system-prompt", "a", "--system-prompt-file", str(prompt_file)])

    assert "mutually exclusive" in message


@pytest.mark.parametrize("blank", ["", "   ", "\n\t "])
def test_blank_inline_system_prompt_is_rejected(blank: str) -> None:
    assert "must not be empty" in _prompt_error(["--system-prompt", blank])


def test_blank_system_prompt_file_is_rejected(tmp_path) -> None:
    prompt_file = tmp_path / "blank.txt"
    prompt_file.write_text("   \n", encoding="utf-8")

    assert "must not be empty" in _prompt_error(["--system-prompt-file", str(prompt_file)])


def test_missing_system_prompt_file_is_a_parse_error(tmp_path) -> None:
    message = _prompt_error(["--system-prompt-file", str(tmp_path / "absent.txt")])

    assert "Cannot read" in message


def test_empty_system_prompt_file_path_is_a_parse_error() -> None:
    """An empty path is a mistake; silently running the built-in prompt hides it."""
    assert "Cannot read" in _prompt_error(["--system-prompt-file", ""])


def test_non_utf8_system_prompt_file_is_a_parse_error(tmp_path) -> None:
    """Latin-1 bytes must surface as a CLI error, not a UnicodeDecodeError traceback."""
    prompt_file = tmp_path / "latin1.txt"
    prompt_file.write_bytes(b"caf\xe9 au lait")

    assert "not valid UTF-8" in _prompt_error(["--system-prompt-file", str(prompt_file)])


def test_oversized_system_prompt_file_is_a_parse_error(tmp_path) -> None:
    """The text is persisted in the run config, so it is bounded."""
    prompt_file = tmp_path / "huge.txt"
    prompt_file.write_text("x" * (MAX_SYSTEM_PROMPT_BYTES + 1), encoding="utf-8")

    assert "larger than" in _prompt_error(["--system-prompt-file", str(prompt_file)])


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(["--perf-only"], id="perf-only"),
        pytest.param(["--gsm8k-only"], id="plugin-only"),
        pytest.param(["--spec-bench"], id="spec-bench-alone"),
        pytest.param(["--skip-tool-eval", "--perf"], id="skip-tool-eval"),
        pytest.param(["--spec-live"], id="spec-live"),
        pytest.param(["--decision-live"], id="decision-live"),
    ],
)
def test_override_is_dropped_on_modes_that_run_no_scenarios(argv: list[str]) -> None:
    """Nothing would receive the prompt, so it must not reach the report either."""
    from rich.console import Console

    from tool_eval_bench.cli.dispatch import _drop_unused_system_prompt

    _, args = _prompt_parser_and_args([*argv, "--system-prompt", "Be strict."])
    console = Console(record=True)

    _drop_unused_system_prompt(args, console)

    assert args.system_prompt is None
    assert "ignored" in console.export_text()


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param([], id="plain-run"),
        pytest.param(["--perf"], id="perf-plus-scenarios"),
        pytest.param(["--gsm8k"], id="plugin-plus-scenarios"),
        pytest.param(["--context-pressure-sweep", "0.5-1.0"], id="pressure-sweep"),
        pytest.param(
            ["--context-pressure-sweep", "0.5-1.0", "--skip-tool-eval"],
            id="pressure-sweep-with-skip-tool-eval",
        ),
        pytest.param(["--dry-run"], id="dry-run"),
        pytest.param(["--probe"], id="probe"),
    ],
)
def test_override_is_kept_on_modes_that_run_scenarios(argv: list[str]) -> None:
    from rich.console import Console

    from tool_eval_bench.cli.dispatch import _drop_unused_system_prompt

    _, args = _prompt_parser_and_args([*argv, "--system-prompt", "Be strict."])
    console = Console(record=True)

    _drop_unused_system_prompt(args, console)

    assert args.system_prompt == "Be strict."
    assert console.export_text().strip() == ""


def test_dropped_override_warning_stays_off_stdout_in_json_mode() -> None:
    """--json output is parsed by scripts; the warning goes to the log instead."""
    from rich.console import Console

    from tool_eval_bench.cli.dispatch import _drop_unused_system_prompt

    _, args = _prompt_parser_and_args(["--perf-only", "--json", "--system-prompt", "Be strict."])
    console = Console(record=True)

    _drop_unused_system_prompt(args, console)

    assert args.system_prompt is None
    assert console.export_text().strip() == ""


def test_resume_config_mismatches_cover_system_prompt() -> None:
    from tool_eval_bench.cli.dispatch import _resume_config_mismatches
    from tool_eval_bench.cli.legacy_parser import _make_parser
    from tool_eval_bench.domain.scenarios import (
        Category,
        ScenarioDefinition,
        ScenarioEvaluation,
        ScenarioStatus,
    )

    scenario = ScenarioDefinition(
        id="TC-01",
        title="t",
        category=Category.A,
        user_message="u",
        description="d",
        handle_tool_call=lambda state, call: {},
        evaluate=lambda state: ScenarioEvaluation(ScenarioStatus.PASS, 2, "ok"),
    )
    args = _make_parser().parse_args(["--system-prompt", "Be terse."])
    previous = {"system_prompt": None}
    mismatches = _resume_config_mismatches(
        previous,
        model="m",
        backend="vllm",
        base_url="http://localhost:8000",
        scenarios=[scenario],
        args=args,
        extra_params=None,
        scenario_packs=None,
        context_pressure=None,
    )
    assert "system_prompt" in mismatches
