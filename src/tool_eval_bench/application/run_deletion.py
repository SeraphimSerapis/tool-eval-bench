"""Delete stored runs together with the Markdown reports they wrote.

A completed run is a SQLite row plus a report under ``<output-dir>/YYYY/MM/``,
so deleting a run removes both. The report path is read back from the
database, which this module does not trust with the filesystem: a file is
removed only when it has the shape the report writers produce,
``.../YYYY/MM/<run_id>[--<label>][<suffix>].md``. Anything else is kept and
reported, so a damaged or hand-edited row cannot delete an unrelated file.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tool_eval_bench.application import run_queries
from tool_eval_bench.storage import db

_YEAR = re.compile(r"\d{4}")
_MONTH = re.compile(r"\d{2}")


@dataclass(frozen=True)
class KeptReport:
    """A report that belonged to a deleted run but was left on disk."""

    run_id: str
    path: str
    reason: str


@dataclass(frozen=True)
class RunDeletion:
    """What :func:`delete_runs` removed and what it left behind."""

    deleted: tuple[str, ...]
    missing: tuple[str, ...]
    reports_removed: tuple[str, ...]
    reports_kept: tuple[KeptReport, ...]


def find_runs(
    *,
    statuses: Collection[str] | None = None,
    before: str | None = None,
    model: str | None = None,
) -> list[dict[str, Any]]:
    """Every stored run matching all given filters, newest first, without scores.

    Each dict has ``run_id``, ``created_at``, ``status``, ``model``,
    ``run_type``, and ``label``. See ``RunRepository.list_summaries`` for the
    filter rules. Returns an empty list when no database exists, without
    creating one.
    """
    if not run_queries.database_exists():
        return []
    with closing(db.RunRepository()) as repo:
        return repo.list_summaries(statuses=statuses, before=before, model=model)


def delete_runs(run_ids: Sequence[str]) -> RunDeletion:
    """Delete runs, their traces and checkpoints, and their Markdown reports.

    The database rows go in one transaction before any file is touched, so a
    report that cannot be removed never leaves a half-deleted run behind. A
    repeated id counts once. Returns an empty result, without creating a
    database, when none exists.
    """
    ids = list(dict.fromkeys(run_ids))
    if not ids:
        return RunDeletion((), (), (), ())
    if not run_queries.database_exists():
        return RunDeletion((), tuple(ids), (), ())
    with closing(db.RunRepository()) as repo:
        rows = repo.delete_runs(ids)

    removed: list[str] = []
    kept: list[KeptReport] = []
    for row in rows:
        if not row.report_path:
            continue
        if row.report_shared:
            kept.append(KeptReport(row.run_id, row.report_path, "another stored run uses it"))
            continue
        path = Path(row.report_path)
        if not path.exists() and not path.is_symlink():
            continue  # already gone; nothing to remove or report
        if not is_run_report(path, row.run_id):
            kept.append(KeptReport(row.run_id, row.report_path, "not a report this tool wrote"))
            continue
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        except OSError as exc:
            kept.append(KeptReport(row.run_id, row.report_path, exc.strerror or str(exc)))
            continue
        removed.append(row.report_path)

    found = {row.run_id for row in rows}
    return RunDeletion(
        deleted=tuple(row.run_id for row in rows),
        missing=tuple(run_id for run_id in ids if run_id not in found),
        reports_removed=tuple(removed),
        reports_kept=tuple(kept),
    )


def is_run_report(path: Path, run_id: str) -> bool:
    """Whether *path* is shaped like the Markdown report the writers name for *run_id*.

    Matches ``report_filename`` in ``storage.reports``: the file name is the
    run id followed by ``.md``, a ``--<label>`` slug, or a ``_<suffix>``, and
    it sits in a ``YYYY/MM`` folder. A directory or a missing file does not
    match.
    """
    name = path.name
    if not run_id or path.suffix != ".md" or not name.startswith(run_id):
        return False
    rest = name[len(run_id) :]
    if rest != ".md" and not rest.startswith(("--", "_")):
        return False
    month, year = path.parent.name, path.parent.parent.name
    if not (_MONTH.fullmatch(month) and _YEAR.fullmatch(year)):
        return False
    return path.is_file()
