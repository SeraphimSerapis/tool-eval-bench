"""The ``--json`` output contract: stdout is one envelope, stderr is JSON lines.

Under ``--json``, ``main`` swaps its console for a :class:`HeadlessConsole`,
which prints nothing, and routes logging through :func:`json_logging`.  Code
that must still tell a caller something (a failure, a saved run) passes its
console to :func:`report_run_failed` or :func:`report_run_saved`, which turn
the message into a JSONL event when the console is headless.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from rich.console import Console
from rich.text import Text

from tool_eval_bench.application.mode_runs import FinalizedRun, ModeRun
from tool_eval_bench.domain.errors import RUN_FAILED
from tool_eval_bench.utils.urls import redact_urls


class HeadlessConsole(Console):
    """The console used under ``--json``: every Rich print is dropped."""

    def __init__(self) -> None:
        super().__init__(quiet=True)


def emit_event(event: dict[str, Any]) -> None:
    """Write one JSONL event to stderr."""
    sys.stderr.write(json.dumps(event) + "\n")
    sys.stderr.flush()


def report_run_failed(console: Console, markup: str) -> None:
    """Print a failure, or emit it as a ``run_failed`` error event under ``--json``.

    The caller still exits; this only decides where the message goes.
    """
    if isinstance(console, HeadlessConsole):
        # One line: the markup is laid out for a terminal.
        emit_run_failed(" ".join(Text.from_markup(markup).plain.split()))
    else:
        console.print(markup)


def emit_run_failed(message: str) -> None:
    """Emit a ``run_failed`` error event; the caller still exits."""
    emit_event({"event": "error", "error": RUN_FAILED, "message": redact_urls(message)})


def report_run_saved(console: Console, run: ModeRun, finalized: FinalizedRun) -> None:
    """Under ``--json``, emit where a run without a result envelope was saved.

    The interactive modes print their own "Report saved" line, so this is a
    no-op for any other console.
    """
    if isinstance(console, HeadlessConsole):
        emit_event(
            {
                "event": "run_saved",
                "run_type": run.run_type,
                "run_id": finalized.run_id,
                "status": run.status,
                "report_path": str(finalized.report_path),
            }
        )


class _JsonLinesLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            text = record.getMessage()
        except Exception:
            # handleError would print a traceback to stderr as bare text.
            text = str(record.msg)
        message = redact_urls(text)
        emit_event(
            {
                "event": "log",
                "level": record.levelname.lower(),
                "logger": record.name,
                "message": message,
            }
        )


@contextmanager
def json_logging(enabled: bool) -> Iterator[None]:
    """Route warnings and errors to stderr as ``log`` events while ``enabled``.

    The CLI configures no logging, so without this a warning reaches stderr
    through ``logging.lastResort`` as bare text.  The handler uses the same
    WARNING threshold, and Python warnings are captured into the same path.
    """
    if not enabled:
        yield
        return
    handler = _JsonLinesLogHandler(level=logging.WARNING)
    root = logging.getLogger()
    root.addHandler(handler)
    logging.captureWarnings(True)
    try:
        yield
    finally:
        logging.captureWarnings(False)
        root.removeHandler(handler)
