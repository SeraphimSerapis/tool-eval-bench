"""Read queries against stored runs.

The delivery layer renders history, diffs, and leaderboards; it does not open
the database to do it.  Every function here owns the repository's lifetime, so
a caller cannot leave a WAL connection behind by returning early.
"""

from __future__ import annotations

from contextlib import closing
from typing import Any

from tool_eval_bench.storage import db


def _repository() -> db.RunRepository:
    """Open a repository, resolving the class at call time.

    Bound per call rather than at import so a test that patches
    ``storage.db.RunRepository`` reaches these queries.
    """
    return db.RunRepository()


def recent_runs(limit: int = 15) -> list[dict[str, Any]]:
    """Return the most recent runs, newest first."""
    with closing(_repository()) as repo:
        return repo.list(limit=limit)


def get_run(run_id: str, *, include_traces: bool = True) -> dict[str, Any] | None:
    """Return one stored run, or ``None`` when the id is unknown."""
    with closing(_repository()) as repo:
        return repo.get(run_id, include_traces=include_traces)


def resolve_run(run_id: str) -> tuple[str, dict[str, Any]] | None:
    """Resolve a run id, accepting ``latest`` for the most recent run.

    Returns the resolved id alongside the run, or ``None`` when nothing
    matches, so a caller can tell "no runs at all" from "no such run" by
    checking which id it asked for.
    """
    with closing(_repository()) as repo:
        if run_id.lower() == "latest":
            run = repo.get_latest()
            return (run["run_id"], run) if run else None
        run = repo.get(run_id)
        return (run_id, run) if run else None


def scenario_results(run_id: str, *, include_traces: bool = False) -> list[dict[str, Any]] | None:
    """Return one run's per-scenario results, or ``None`` when the id is unknown.

    Traces are excluded by default: a diff reads points, status, and duration,
    and rehydrating raw logs for that costs a multi-megabyte read per run.
    """
    with closing(_repository()) as repo:
        return repo.get_scenario_results(run_id, include_traces=include_traces)


def resume_state(run_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    """Return a run and its checkpoints, for deciding whether it can resume."""
    with closing(_repository()) as repo:
        run = repo.get(run_id)
        if run is None:
            return None
        return run, repo.get_checkpoints(run_id)


def persist_run(run_data: dict[str, Any]) -> None:
    """Write a completed run, surfacing mandatory-storage failures to the caller."""
    with _repository() as repo:
        repo.upsert_scenario_run(run_data)
