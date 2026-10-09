"""Which benchmark modes an invocation runs, read from the parsed flags.

``main()`` routes through the modes in a fixed order and stops at the first one
that finishes the invocation. These helpers mirror that routing so argument
validation and ``--dry-run`` can reason about it without importing dispatch.
"""

from __future__ import annotations

import argparse

from tool_eval_bench.cli.command_registry import PLUGIN_FLAG_STEMS


def any_plugin_selected(args: argparse.Namespace) -> bool:
    """Whether any accuracy plugin runs, through ``--<plugin>`` or ``--<plugin>-only``."""
    return any(
        getattr(args, stem) or getattr(args, f"{stem}_only") for stem in PLUGIN_FLAG_STEMS.values()
    )


def sends_system_prompt(args: argparse.Namespace) -> bool:
    """Whether this invocation runs tool-call scenarios, the override's only consumer.

    Also gates the checks that only matter when scenarios run, such as rejecting
    an empty scenario selection.

    Mirrors the routing in ``main()``: ``--perf-only``, ``--spec-live``, a lone
    ``--spec-bench``, any ``--<plugin>-only`` run, and ``--skip-tool-eval`` all
    stop before the scenarios; a context-pressure sweep is a scenario run.

    Probe and the storage commands are not considered here: they record
    nothing, so an unused flag is inert, and they already accept the rest of
    the run-control group in silence.
    """
    if args.spec_live or args.decision_live or args.perf_only:
        return False
    other_benchmarks = args.perf or any_plugin_selected(args)
    if args.spec_bench and (args.skip_tool_eval or not other_benchmarks):
        return False
    if args.context_pressure_sweep is not None:
        return True
    if args.skip_tool_eval:
        return False
    return not any(getattr(args, f"{stem}_only") for stem in PLUGIN_FLAG_STEMS.values())


def _selected_plugin_flags(args: argparse.Namespace) -> list[str]:
    flags: list[str] = []
    for stem in PLUGIN_FLAG_STEMS.values():
        flag = "--" + stem.replace("_", "-")
        if getattr(args, stem):
            flags.append(flag)
        if getattr(args, f"{stem}_only"):
            flags.append(f"{flag}-only")
    return flags


def mode_conflict(args: argparse.Namespace) -> str | None:
    """Describe a combination of mode flags where one mode would never run.

    ``main()`` stops at the first mode that finishes the invocation, so each of
    these used to exit 0 with a requested mode silently skipped. Returns None
    when every requested mode runs. The rules are listed in
    ``docs/cli-reference.md`` under "Combining modes".
    """
    plugins = _selected_plugin_flags(args)
    monitors = [
        flag
        for flag, selected in (
            ("--spec-live", args.spec_live),
            ("--decision-live", args.decision_live),
        )
        if selected
    ]
    benchmarks = [
        flag
        for flag, selected in (
            ("--perf", args.perf),
            ("--perf-only", args.perf_only),
            ("--spec-bench", args.spec_bench),
            ("--context-pressure-sweep", args.context_pressure_sweep is not None),
        )
        if selected
    ] + plugins

    others = monitors[1:] + benchmarks
    if monitors and others:
        return f"{monitors[0]} is a live monitor and cannot be combined with {others[0]}"

    if args.perf_only:
        dropped = [flag for flag in benchmarks if flag not in {"--perf", "--perf-only"}]
        if dropped:
            return (
                f"--perf-only runs throughput alone and cannot be combined with {dropped[0]}; "
                "use --perf to run both"
            )

    if args.context_pressure_sweep is not None:
        if args.spec_bench:
            return "--context-pressure-sweep cannot be combined with --spec-bench"
        if plugins:
            return f"--context-pressure-sweep cannot be combined with {plugins[0]}"

    only = [flag for flag in plugins if flag.endswith("-only")]
    if only:
        stem = only[0].removesuffix("-only")
        other = next((flag for flag in plugins if flag.removesuffix("-only") != stem), None)
        if other is not None:
            return (
                f"{only[0]} runs one plugin alone and cannot be combined with {other}; "
                "use the plain plugin flags with --skip-tool-eval to run several"
            )
    return None
