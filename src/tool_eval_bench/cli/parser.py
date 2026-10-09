"""Subcommand parsing with permanent legacy CLI compatibility.

The runtime still consumes the mature flat-parser namespace.  This module
translates the discoverable subcommand surface into that namespace so new and
legacy invocations share exactly the same validation and dispatch paths.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

from tool_eval_bench.cli.command_registry import (
    COMMAND_REGISTRY,
    COMMAND_SPECS,
    KNOWN_COMMANDS,
    PLUGIN_FLAG_STEMS,
)


def _flag_action(source: argparse.Action) -> str | None:
    """Return ``store_true``/``store_false`` for a boolean flag, else ``None``.

    Read from the action's public ``nargs`` and ``const`` rather than by
    ``isinstance`` against ``argparse._StoreTrueAction``. Those classes are
    private and absent from ``argparse.__all__``; ``Action.const`` is part of
    the documented interface and says the same thing.
    """
    if source.nargs != 0 or not isinstance(source.const, bool):
        return None
    return "store_true" if source.const else "store_false"


def _copy_legacy_options(parser: argparse.ArgumentParser, destinations: tuple[str, ...]) -> None:
    """Copy selected options from the canonical flat parser into focused help.

    Reads ``legacy._actions`` because argparse exposes no public way to
    enumerate a parser's options. That one private access is why the flag
    surface is pinned by ``tests/snapshots/legacy_cli.json``.
    """
    from tool_eval_bench.cli.legacy_parser import make_parser as make_legacy_parser

    legacy = make_legacy_parser()
    wanted = set(destinations)
    for source in legacy._actions:
        if source.dest not in wanted or not source.option_strings:
            continue
        kwargs: dict = {
            "dest": source.dest,
            "default": source.default,
            "help": source.help,
            "required": source.required,
        }
        flag_action = _flag_action(source)
        if flag_action is not None:
            kwargs["action"] = flag_action
        else:
            kwargs.update(
                {
                    "type": source.type,
                    "choices": source.choices,
                    "metavar": source.metavar,
                }
            )
            if source.nargs is not None:
                kwargs["nargs"] = source.nargs
            if source.const is not None:
                kwargs["const"] = source.const
        parser.add_argument(*source.option_strings, **kwargs)


def _command_help(command: str, *, include_legacy_options: bool = True) -> argparse.ArgumentParser:
    """Return a focused help parser for *command*."""
    spec = COMMAND_REGISTRY[command]
    parser = argparse.ArgumentParser(
        prog=f"tool-eval-bench {command}", description=spec.description
    )
    if command == "plugin":
        parser.add_argument("benchmark", choices=spec.choices)
        _add_plugin_options(parser)
    elif command == "compare":
        parser.add_argument("--report", action="store_true")
        parser.add_argument("left", help="Run ID, or Markdown report with --report")
        parser.add_argument("right", help="Run ID, or Markdown report with --report")
        parser.add_argument("-o", "--output", help="Required with --report")
        parser.add_argument("--kind", choices=("auto", "summary", "tool-eval"), default="auto")
    elif command == "export":
        parser.add_argument("--format", choices=("csv", "json"), default="json")
        parser.add_argument("-o", "--output")
    elif command == "resume":
        parser.add_argument("run_id")
    elif command == "compare-report":
        parser.add_argument("report_a", help="First Markdown report")
        parser.add_argument("report_b", help="Second Markdown report")
        parser.add_argument("-o", "--output", required=True, help="Output HTML path")
        parser.add_argument("--kind", choices=("auto", "summary", "tool-eval"), default="auto")
    if include_legacy_options:
        _copy_legacy_options(parser, spec.help_dests)
    return parser


def make_parser() -> argparse.ArgumentParser:
    """Build the discoverable top-level subcommand parser.

    Runtime parsing uses :func:`parse_cli_args`; this parser is intentionally
    small and exists for concise top-level discovery and integrations.
    """
    parser = argparse.ArgumentParser(
        prog="tool-eval-bench",
        description="Evaluate OpenAI-compatible models with tool and accuracy benchmarks",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    for spec in COMMAND_SPECS:
        subparsers.add_parser(spec.name, help=spec.description, add_help=False)
    return parser


def _add_plugin_options(parser: argparse.ArgumentParser) -> None:
    """The ``plugin`` subcommand's own options, translated by :func:`_plugin_argv`."""
    parser.add_argument("--shots", type=int, metavar="N")
    parser.add_argument("--limit", type=int, metavar="N")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--subjects", metavar="LIST")
    parser.add_argument("--depths", type=int, metavar="N")
    parser.add_argument("--lengths", type=int, metavar="N")


class _Token(str):
    """A distinct ``str`` object, so the token argparse picks can be found by identity."""


def _take_positional(
    argv: list[str], command: str, dest: str, *, plugin_options: bool = False
) -> tuple[str | None, list[str]]:
    """Split the token argparse assigns to positional *dest* out of *argv*.

    The locating parser has every legacy option (plus the plugin options) with
    its arity and nothing else, so it never takes an option's value for *dest*:
    ``plugin --base-url http://x gsm8k`` finds ``gsm8k``. Types and choices are
    left out; the real parse reports bad values. argparse reports values, not
    positions, and the same string may also be an option value
    (``--label r-1 r-1``), so each token is wrapped in its own ``str`` object
    and the assigned value is matched by identity; argparse passes an untyped
    positional's string through unchanged.
    """
    from tool_eval_bench.cli.legacy_parser import make_parser as make_legacy_parser

    locator = argparse.ArgumentParser(prog=f"tool-eval-bench {command}", add_help=False)
    # Same private access as _copy_legacy_options, pinned by the same snapshot.
    for source in make_legacy_parser()._actions:
        if not source.option_strings:
            continue
        if source.nargs == 0:
            locator.add_argument(*source.option_strings, dest=source.dest, action="store_true")
        else:
            locator.add_argument(*source.option_strings, dest=source.dest, nargs=source.nargs)
    if plugin_options:
        _add_plugin_options(locator)
    locator.add_argument(dest, nargs="?")
    tokens = [_Token(value) for value in argv]
    namespace, _ = locator.parse_known_args(tokens)
    value = getattr(namespace, dest)
    for index, token in enumerate(tokens):
        if token is value:
            return argv[index], argv[:index] + argv[index + 1 :]
    return None, argv


def _plugin_argv(argv: list[str]) -> list[str]:
    parser = _command_help("plugin", include_legacy_options=False)
    benchmark, rest = _take_positional(argv, "plugin", "benchmark", plugin_options=True)
    if benchmark is None:
        parser.error("the following arguments are required: benchmark")
    args, remainder = parser.parse_known_args([benchmark, *rest])
    prefix = "--" + PLUGIN_FLAG_STEMS[args.benchmark].replace("_", "-")
    translated = [f"{prefix}-only"]
    if args.shots is not None:
        if args.benchmark == "ifeval":
            parser.error("--shots is not supported by the IFEval plugin")
        if args.benchmark == "decision":
            parser.error("--shots is not supported by the decision plugin")
        translated.extend((f"{prefix}-shots", str(args.shots)))
    if args.limit is not None:
        if args.benchmark == "needle":
            parser.error("--limit is not supported by the needle plugin; use --lengths/--depths")
        if args.benchmark == "decision":
            parser.error("--limit is not supported by the decision plugin")
        translated.extend((f"{prefix}-limit", str(args.limit)))
    if args.shuffle:
        if args.benchmark != "gsm8k":
            parser.error("--shuffle is only supported by the GSM8K plugin")
        translated.append("--gsm8k-shuffle")
    if args.subjects is not None:
        if args.benchmark != "mmlu":
            parser.error("--subjects is only supported by the MMLU plugin")
        translated.extend(("--mmlu-subjects", args.subjects))
    for name, value in (("depths", args.depths), ("lengths", args.lengths)):
        if value is None:
            continue
        if args.benchmark != "needle":
            parser.error(f"--{name} is only supported by the needle plugin")
        translated.extend((f"--needle-{name}", str(value)))
    return translated + remainder


def _compare_argv(argv: list[str]) -> list[str]:
    parser = _command_help("compare")
    args = parser.parse_args(argv)
    if not args.report:
        if args.output:
            parser.error("--output requires --report")
        if args.kind != "auto":
            parser.error("--kind requires --report")
        return ["--compare", args.left, args.right]
    if not args.output:
        parser.error("--output is required with --report")
    return [
        "compare-report",
        args.left,
        args.right,
        "--output",
        args.output,
        "--kind",
        args.kind,
    ]


def _export_argv(argv: list[str]) -> list[str]:
    parser = _command_help("export")
    args = parser.parse_args(argv)
    translated = ["--export", args.format]
    if args.output:
        translated.extend(("--export-output", args.output))
    return translated


def translate_argv(argv: Sequence[str]) -> list[str]:
    """Translate a subcommand invocation into legacy flat-parser arguments."""
    values = list(argv)
    if not values or values[0] not in KNOWN_COMMANDS:
        return values

    command, rest = values[0], values[1:]
    if any(value in {"-h", "--help"} for value in rest):
        _command_help(command).parse_args(["--help"])

    spec = COMMAND_REGISTRY[command]
    if spec.translation == "passthrough":
        return rest
    if spec.translation == "prefix":
        return [*spec.legacy_prefix, *rest]
    if spec.translation == "plugin":
        return _plugin_argv(rest)
    if spec.translation == "compare":
        return _compare_argv(rest)
    if spec.translation == "alias":
        return values
    if spec.translation == "export":
        return _export_argv(rest)
    if spec.translation == "resume":
        run_id, remaining = _take_positional(rest, "resume", "run_id")
        if run_id is None:
            _command_help("resume").error("the following arguments are required: run_id")
        return ["--resume", run_id, *remaining]
    raise AssertionError(f"Unhandled command: {command}")


def parse_cli_args(
    legacy_parser_factory: Callable[[], argparse.ArgumentParser],
    argv: Sequence[str] | None = None,
) -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    """Parse new or legacy arguments into the legacy runtime namespace."""
    parser = legacy_parser_factory()
    translated = translate_argv(sys.argv[1:] if argv is None else argv)
    return parser, parser.parse_args(translated)
