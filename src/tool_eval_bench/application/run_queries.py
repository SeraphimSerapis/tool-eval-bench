"""Read queries against stored runs.

The delivery layer renders history, diffs, and leaderboards; it does not open
the database to do it.  Every function here owns the repository's lifetime, so
a caller cannot leave a WAL connection behind by returning early.
"""

from __future__ import annotations

from collections.abc import Collection
from contextlib import closing
from pathlib import Path
from typing import Any

from tool_eval_bench.storage import db

# The CLI scopes one invocation's session with these; it may not import storage.
invocation_session = db.invocation_session
current_session_id = db.current_session_id


def _repository() -> db.RunRepository:
    """Open a repository, resolving the class at call time.

    Bound per call rather than at import so a test that patches
    ``storage.db.RunRepository`` reaches these queries.
    """
    return db.RunRepository()


def database_exists() -> bool:
    """Whether the run database exists, checked without creating it.

    Opening a repository creates ``data/benchmarks.sqlite``, so a read-only
    command such as ``card`` checks this first to leave an empty directory
    empty.  The path resolves as it does for the repository, from the current
    working directory at call time.
    """
    return Path(db._default_db_path()).is_file()


def recent_runs(
    limit: int = 15,
    *,
    model: str | None = None,
    run_type: str | None = None,
    status: str | None = None,
    search: str | None = None,
) -> list[dict[str, Any]]:
    """Return the most recent runs, newest first.

    The filters narrow the query before ``limit`` applies, so a search finds an
    old run rather than only matches among the newest ``limit`` rows.
    """
    filters = {
        key: value
        for key, value in (
            ("model", model),
            ("run_type", run_type),
            ("status", status),
            ("search", search),
        )
        if value
    }
    with closing(_repository()) as repo:
        return repo.list(limit=limit, **filters)


def get_run(run_id: str, *, include_traces: bool = True) -> dict[str, Any] | None:
    """Return one stored run, or ``None`` when the id is unknown."""
    with closing(_repository()) as repo:
        return repo.get(run_id, include_traces=include_traces)


def resolve_run(
    run_id: str, *, run_type: str | None = None, status: str | None = None
) -> tuple[str, dict[str, Any]] | None:
    """Resolve a run id, accepting ``latest`` for the most recent run.

    Returns the resolved id alongside the run, or ``None`` when nothing
    matches, so a caller can tell "no runs at all" from "no such run" by
    checking which id it asked for. ``run_type`` and ``status`` narrow what
    ``latest`` may resolve to; an explicit id is returned as stored.
    """
    with closing(_repository()) as repo:
        if run_id.lower() == "latest":
            run = repo.get_latest(run_type=run_type, status=status)
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


def run_progress(
    run_id: str, *, include_traces: bool = False
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    """Return a run row and its checkpoints, for watching a run in progress.

    Traces are off by default because a live view polls this every second and
    needs only statuses and points.
    """
    with closing(_repository()) as repo:
        run = repo.get(run_id, include_traces=include_traces)
        if run is None:
            return None
        return run, repo.get_checkpoints(run_id)


def persist_run(run_data: dict[str, Any]) -> None:
    """Write a completed run, surfacing mandatory-storage failures to the caller."""
    with _repository() as repo:
        repo.upsert_scenario_run(run_data)


def session_runs(session_id: str) -> list[dict[str, Any]]:
    """Return every run one CLI invocation stored, oldest first, without traces.

    Each dict is shaped like ``RunRepository.get()`` output, which includes
    ``session_id``.  An unknown id returns an empty list.
    """
    with closing(_repository()) as repo:
        return repo.list_session(session_id)


def latest_session_id(
    *,
    model: str | None = None,
    run_types: Collection[str] | None = None,
    statuses: Collection[str] | None = None,
) -> str | None:
    """Return the newest session with at least one run matching every filter.

    ``model`` matches the stored ``model`` column exactly or as its trailing
    path segments, so ``Qwen3-32B`` finds ``Qwen/Qwen3-32B``.  ``run_types``
    and ``statuses`` filter on ``run_type`` and ``status``.  ``None`` skips a
    filter and an empty collection matches nothing.  Returns ``None`` when no
    run qualifies, for example in a database written before schema v5.
    """
    with closing(_repository()) as repo:
        return repo.latest_session_id(model=model, run_types=run_types, statuses=statuses)


def resolve_session_prefix(prefix: str) -> str | None:
    """Resolve a full session id, or a unique prefix of at least six characters.

    ``history`` shows the first eight characters, which is enough to type.
    Returns ``None`` when nothing matches or the prefix is too short.

    Raises:
        ValueError: when the prefix matches more than one session.
    """
    with closing(_repository()) as repo:
        return repo.resolve_session_prefix(prefix)
