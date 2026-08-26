"""Shared live progress display for accuracy benchmark plugins.

GSM8K, MMLU, and IFEval each rendered their own byte-identical copy of this
Rich layout: a progress bar, a running tally line, and a line showing the item
that just finished.  The layout and the correct/wrong/error accounting are the
same for every item-scored benchmark, so they live here once.

What stays with each plugin is the wording: how a tally reads, and how a single
finished item is summarised.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from types import TracebackType
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


@dataclass
class ProgressTally:
    """Running correct/wrong/error counts over scored items."""

    correct: int = 0
    wrong: int = 0
    errors: int = 0

    def record(self, item_info: dict[str, Any]) -> None:
        """Count one finished item, classified the way every plugin does."""
        if item_info.get("is_error"):
            self.errors += 1
        elif item_info.get("correct"):
            self.correct += 1
        else:
            self.wrong += 1

    @property
    def processed(self) -> int:
        """Items counted so far."""
        return self.correct + self.wrong + self.errors

    @property
    def accuracy(self) -> float:
        """Percentage correct over everything counted, errors included."""
        return (self.correct / self.processed * 100) if self.processed else 0.0


def status_icon(item_info: dict[str, Any], *, key: str = "correct") -> str:
    """Return the markup icon for a finished item's outcome.

    *key* names the field carrying the pass/fail verdict; IFEval scores whole
    prompts under ``prompt_pass`` rather than ``correct``.
    """
    if item_info.get("is_error"):
        return "[yellow]⚠[/]"
    if item_info.get(key, False):
        return "[green]✓[/]"
    return "[red]✗[/]"


def truncate(text: str | None, limit: int = 90) -> str:
    """Flatten and clip an item's prompt so it fits one status line."""
    flat = (text or "").replace("\n", " ").strip()
    return flat[: limit - 3] + "…" if len(flat) > limit else flat


def tally_line(
    tally: ProgressTally, *, rate: float, unit: str = "q/min", accent: str = "magenta"
) -> str:
    """Render the shared tally line: counts, accuracy, and throughput.

    *accent* is the colour of the accuracy figure, which each benchmark keys to
    its own panel border.
    """
    parts = [
        f"  [bold green]✓ {tally.correct}[/]",
        f"[bold red]✗ {tally.wrong}[/]",
    ]
    if tally.errors > 0:
        parts.append(f"[bold yellow]⚠ {tally.errors}[/]")
    parts += [
        "[dim]│[/]",
        f"[bold {accent}]{tally.accuracy:.1f}%[/] accuracy",
        "[dim]│[/]",
        f"[dim]{rate:.1f} {unit}[/]",
    ]
    return "  ".join(parts)


class PluginProgressDisplay:
    """The live progress layout shared by the accuracy plugins.

    Used as a context manager.  ``advance`` is driven from a plugin's
    ``on_progress`` callback; ``finish`` replaces the tally with the final
    figures once the run completes.
    """

    def __init__(self, console: Console, *, total: int, description: str = "Evaluating…") -> None:
        self._console = console
        self._total = total
        self._description = description
        self.tally = ProgressTally()
        self._started = time.monotonic()

        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[bold]{task.percentage:>3.0f}%[/]"),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("[dim]eta[/]"),
            TimeRemainingColumn(),
            console=console,
        )
        self._stats_text = TextColumn("")
        self._stats_progress = Progress(self._stats_text, console=console)
        self._detail_text = TextColumn("")
        self._detail_progress = Progress(self._detail_text, console=console)
        self._group = Group(self.progress, self._stats_progress, self._detail_progress)
        self._live = Live(self._group, console=console, refresh_per_second=4)
        self._task: Any = None

    def __enter__(self) -> PluginProgressDisplay:
        self._stats_progress.add_task("", total=None)
        self._detail_progress.add_task("", total=None)
        self._live.__enter__()
        self._task = self.progress.add_task(self._description, total=self._total)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._live.__exit__(exc_type, exc, tb)

    @property
    def elapsed(self) -> float:
        """Seconds since the display was constructed."""
        return time.monotonic() - self._started

    def rate_per_minute(self, current: int) -> float:
        """Items per minute so far."""
        elapsed = self.elapsed
        return current / elapsed * 60 if elapsed > 0 else 0.0

    def advance(self, current: int, total: int, *, stats: str, detail: str = "") -> None:
        """Move the bar and replace both status lines."""
        self._stats_text.text_format = stats
        self._detail_text.text_format = detail
        self.progress.update(self._task, completed=current, total=total)

    def finish(self, *, completed: int, stats: str, description: str = "[green]✓ Complete") -> None:
        """Show the final counts and mark the bar complete."""
        self.progress.update(self._task, completed=completed, description=description)
        self._stats_text.text_format = stats
        self._detail_text.text_format = ""


def final_tally_line(
    *,
    correct: int,
    wrong: int,
    errors: int,
    score: float,
    rate: float,
    unit: str = "q/min",
    accent: str = "magenta",
) -> str:
    """Render the completed-run tally.

    Spacing differs from :func:`tally_line` (wider separators, and the word
    "errors" spelled out), because this line replaces the live tally once the
    bar is full rather than updating in place.
    """
    parts = f"  [bold green]✓ {correct}[/]  [bold red]✗ {wrong}[/]  "
    if errors > 0:
        parts += f"[bold yellow]⚠ {errors} errors[/]  "
    parts += (
        f"[dim]│[/]  [bold {accent}]{score:.1f}%[/] accuracy  [dim]│[/]  [dim]{rate:.1f} {unit}[/]"
    )
    return parts
