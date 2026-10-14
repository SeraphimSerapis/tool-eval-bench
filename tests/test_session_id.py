"""Every run one CLI invocation stores shares a session id.

``card`` reads a session's rows back through ``run_queries.session_runs``. The
id is bookkeeping: it must not move a fingerprint or appear in a report.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.conftest import open_repository
from tests.test_e2e import SmokeMockAdapter
from tests.test_scored_perf_scores import OK_CELL
from tool_eval_bench.application import run_queries
from tool_eval_bench.application.leaderboard import cohort_fingerprint as _cohort_fingerprint
from tool_eval_bench.application.service import BenchmarkService
from tool_eval_bench.cli import dispatch
from tool_eval_bench.cli.plugin_lifecycle import finalize_plugin_run
from tool_eval_bench.domain.models import RunContext
from tool_eval_bench.storage.db import _SCHEMA_VERSION
from tool_eval_bench.storage.reports._common import (
    _render_engine_context,
    _render_run_context,
)
from tool_eval_bench.utils import metadata as metadata_module
from tool_eval_bench.utils.fingerprint import comparison_fingerprint, with_config_fingerprint

SESSION = "0123456789abcdef0123456789abcdef"


def _context(**overrides: Any) -> RunContext:
    return RunContext(
        tool_version="1.0",
        git_sha="abc123",
        hostname="h",
        platform_info="p",
        python_version="3",
        model="m",
        backend="vllm",
        base_url="http://gpu.test:8000/v1",
        engine_name="vLLM",
        engine_version="0.9",
        quantization="fp8",
        **overrides,
    )


def _run(
    run_id: str,
    *,
    session_id: str | None,
    model: str = "m",
    run_type: str = "perf",
    status: str = "completed",
) -> dict[str, Any]:
    metadata: dict[str, Any] = {"tool_version": "1.0"}
    if session_id is not None:
        metadata["session_id"] = session_id
    return {
        "run_id": run_id,
        "run_type": run_type,
        "status": status,
        "config": {"model": model},
        "scores": {"samples": 1},
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def _columns_and_indexes(db_path: Path) -> tuple[set[str], set[str], int]:
    with sqlite3.connect(db_path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(scenario_runs)")}
        indexes = {row[1] for row in conn.execute("PRAGMA index_list(scenario_runs)")}
        version = conn.execute("PRAGMA user_version").fetchone()[0]
    conn.close()
    return columns, indexes, version


def test_fresh_database_has_the_session_column_and_index(tmp_path: Path) -> None:
    db_path = tmp_path / "fresh.sqlite"
    open_repository(db_path=str(db_path)).close()

    columns, indexes, version = _columns_and_indexes(db_path)
    assert "session_id" in columns
    assert "idx_scenario_runs_session_id" in indexes
    assert version == _SCHEMA_VERSION == 5


def test_v4_database_migrates_and_keeps_old_rows_untagged(tmp_path: Path) -> None:
    db_path = tmp_path / "v4.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE scenario_runs (run_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, "
            "status TEXT NOT NULL, model TEXT NOT NULL, config_json TEXT NOT NULL, "
            "scores_json TEXT, metadata_json TEXT, "
            "run_type TEXT NOT NULL DEFAULT 'tool_eval', report_path TEXT)"
        )
        conn.execute(
            "CREATE TABLE run_checkpoints (run_id TEXT NOT NULL, scenario_id TEXT NOT NULL, "
            "created_at TEXT NOT NULL, result_json TEXT NOT NULL, "
            "PRIMARY KEY (run_id, scenario_id))"
        )
        conn.execute(
            "CREATE TABLE scenario_traces (run_id TEXT NOT NULL, scenario_id TEXT NOT NULL, "
            "raw_log TEXT NOT NULL, PRIMARY KEY (run_id, scenario_id))"
        )
        # Metadata that happens to carry the key still stays NULL in the column:
        # the migration does not backfill.
        conn.execute(
            "INSERT INTO scenario_runs VALUES ('old', '2026-01-01T00:00:00+00:00', "
            "'completed', 'm', '{}', '{}', '{\"session_id\": \"x\"}', 'perf', NULL)"
        )
        conn.execute("PRAGMA user_version = 4")
    conn.close()

    repo = open_repository(db_path=str(db_path))
    columns, indexes, version = _columns_and_indexes(db_path)
    assert "session_id" in columns
    assert "idx_scenario_runs_session_id" in indexes
    assert version == 5
    assert repo.latest_session_id() is None
    assert repo.get("old") is not None

    repo.upsert_scenario_run(_run("new", session_id=SESSION))
    assert [run["run_id"] for run in repo.list_session(SESSION)] == ["new"]


def test_upsert_tags_from_metadata_and_leaves_untagged_runs_null(tmp_path: Path) -> None:
    db_path = tmp_path / "b.sqlite"
    repo = open_repository(db_path=str(db_path))
    repo.upsert_scenario_run(_run("tagged", session_id=SESSION))
    repo.upsert_scenario_run(_run("untagged", session_id=None))
    repo.upsert_scenario_run({**_run("no-metadata", session_id=None), "metadata": None})

    with sqlite3.connect(db_path) as conn:
        rows = dict(conn.execute("SELECT run_id, session_id FROM scenario_runs"))
    conn.close()
    assert rows == {"tagged": SESSION, "untagged": None, "no-metadata": None}


def test_list_session_returns_the_session_oldest_first_without_traces(tmp_path: Path) -> None:
    repo = open_repository(db_path=str(tmp_path / "b.sqlite"))
    scored = {
        **_run("b-scored", session_id=SESSION),
        "run_type": "tool_eval",
        "scores": {"scenario_results": [{"scenario_id": "TC-01", "raw_log": "trace"}]},
    }
    repo.upsert_scenario_run(_run("a-perf", session_id=SESSION))
    repo.upsert_scenario_run(_run("other", session_id="another-session"))
    repo.upsert_scenario_run(scored)

    runs = repo.list_session(SESSION)

    assert [run["run_id"] for run in runs] == ["a-perf", "b-scored"]
    assert set(runs[0]) == set(repo.get("a-perf")) == set(repo.list()[0])
    assert "session_id" in runs[0]
    assert all(run["session_id"] == SESSION for run in runs)
    assert runs[1]["scores"]["scenario_results"] == [{"scenario_id": "TC-01"}]
    assert repo.list_session("unknown") == []


def test_list_session_keeps_write_order_when_timestamps_tie(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tool_eval_bench.storage import db

    class _Frozen(db.datetime):  # type: ignore[name-defined, misc]
        @classmethod
        def now(cls, tz: Any = None) -> Any:
            return db.datetime(2026, 10, 10, tzinfo=tz)

    monkeypatch.setattr(db, "datetime", _Frozen)
    repo = open_repository(db_path=str(tmp_path / "b.sqlite"))
    for run_id in ("z", "a", "m"):
        repo.upsert_scenario_run(_run(run_id, session_id=SESSION))

    assert [run["run_id"] for run in repo.list_session(SESSION)] == ["z", "a", "m"]


def test_latest_session_id_skips_untagged_rows_and_filters_by_model(tmp_path: Path) -> None:
    repo = open_repository(db_path=str(tmp_path / "b.sqlite"))
    assert repo.latest_session_id() is None

    repo.upsert_scenario_run(_run("1", session_id="s-alpha", model="alpha"))
    repo.upsert_scenario_run(_run("2", session_id="s-beta", model="beta"))
    repo.upsert_scenario_run(_run("3", session_id=None, model="beta"))

    assert repo.latest_session_id() == "s-beta"
    assert repo.latest_session_id(model="alpha") == "s-alpha"
    assert repo.latest_session_id(model="gamma") is None


def test_run_queries_expose_the_session_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    run_queries.persist_run(_run("a", session_id=SESSION))
    run_queries.persist_run(_run("b", session_id="later", model="other"))

    assert [run["run_id"] for run in run_queries.session_runs(SESSION)] == ["a"]
    assert run_queries.latest_session_id() == "later"
    assert run_queries.latest_session_id(model="m") == SESSION
    assert run_queries.latest_session_id(run_types=["perf"], statuses=["completed"]) == "later"
    assert run_queries.resolve_session_prefix(SESSION[:6]) == SESSION


def test_latest_session_id_skips_sessions_without_a_matching_row(tmp_path: Path) -> None:
    repo = open_repository(db_path=str(tmp_path / "b.sqlite"))
    # Oldest first: a complete scored session, then a spec-bench-only session,
    # then a session whose scored run was interrupted, then a concurrent claim.
    repo.upsert_scenario_run(_run("1", session_id="s-done", run_type="tool_eval"))
    repo.upsert_scenario_run(_run("2", session_id="s-done", run_type="perf"))
    repo.upsert_scenario_run(_run("3", session_id="s-spec", run_type="spec-bench"))
    repo.upsert_scenario_run(
        _run("4", session_id="s-cut", run_type="tool_eval", status="interrupted")
    )
    repo.upsert_scenario_run(_run("5", session_id="s-cut", run_type="perf"))
    repo.upsert_scenario_run(
        _run("6", session_id="s-claim", run_type="tool_eval", status="running")
    )

    assert repo.latest_session_id() == "s-claim"
    assert repo.latest_session_id(statuses={"completed"}) == "s-cut"
    assert repo.latest_session_id(run_types=("tool_eval", "gsm8k")) == "s-claim"
    # Both filters must hold on the same row: s-cut has a completed perf row
    # and an interrupted tool_eval row, but no completed tool_eval row.
    assert repo.latest_session_id(run_types=["tool_eval"], statuses=["completed"]) == "s-done"
    assert repo.latest_session_id(run_types=["needle"]) is None
    assert repo.latest_session_id(run_types=[]) is None
    assert repo.latest_session_id(statuses=()) is None
    assert repo.latest_session_id(model="other", statuses=["completed"]) is None


def test_resolve_session_prefix(tmp_path: Path) -> None:
    repo = open_repository(db_path=str(tmp_path / "b.sqlite"))
    repo.upsert_scenario_run(_run("1", session_id="abcdef0011"))
    repo.upsert_scenario_run(_run("2", session_id="abcdef0011"))
    repo.upsert_scenario_run(_run("3", session_id="abcdef9922"))
    repo.upsert_scenario_run(_run("4", session_id="abc"))

    assert repo.resolve_session_prefix("abcdef0011") == "abcdef0011"
    assert repo.resolve_session_prefix("abcdef00") == "abcdef0011"
    # An exact id resolves even when it is shorter than the minimum prefix.
    assert repo.resolve_session_prefix("abc") == "abc"
    assert repo.resolve_session_prefix("abcde") is None
    assert repo.resolve_session_prefix("ffffff") is None
    assert repo.resolve_session_prefix("abcd%_") is None
    with pytest.raises(ValueError, match="ambiguous.*abcdef0011, abcdef9922"):
        repo.resolve_session_prefix("abcdef")


def test_resolve_session_prefix_ignores_case_and_surrounding_whitespace(tmp_path: Path) -> None:
    repo = open_repository(db_path=str(tmp_path / "b.sqlite"))
    repo.upsert_scenario_run(_run("1", session_id="abcdef0011"))

    assert repo.resolve_session_prefix("  ABCDEF00 \n") == "abcdef0011"
    assert repo.resolve_session_prefix("ABCDEF0011") == "abcdef0011"
    assert repo.resolve_session_prefix("   ") is None


def test_latest_session_id_matches_a_bare_model_name_on_a_path_boundary(tmp_path: Path) -> None:
    repo = open_repository(db_path=str(tmp_path / "b.sqlite"))
    repo.upsert_scenario_run(_run("1", session_id="s-org", model="Qwen/Qwen3-32B"))
    repo.upsert_scenario_run(_run("2", session_id="s-path", model="/models/Qwen3-32B"))
    repo.upsert_scenario_run(_run("3", session_id="s-glued", model="MyQwen3-32B"))
    repo.upsert_scenario_run(_run("4", session_id="s-wild", model="org/a_b%c"))

    assert repo.latest_session_id(model="Qwen3-32B") == "s-path"
    assert repo.latest_session_id(model="Qwen/Qwen3-32B") == "s-org"
    assert repo.latest_session_id(model="MyQwen3-32B") == "s-glued"
    # Case-sensitive like the exact match, and LIKE wildcards are plain text.
    assert repo.latest_session_id(model="qwen3-32b") is None
    assert repo.latest_session_id(model="a_b%c") == "s-wild"
    assert repo.latest_session_id(model="aXbYc") is None
    assert repo.latest_session_id(model="%") is None
    assert repo.latest_session_id(model="32B") is None


@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
def test_a_blank_session_id_is_stored_as_null(tmp_path: Path, blank: str) -> None:
    db_path = tmp_path / "b.sqlite"
    repo = open_repository(db_path=str(db_path))
    repo.upsert_scenario_run(_run("blank", session_id=blank))
    repo.upsert_scenario_run(_run("padded", session_id=f"  {SESSION} "))

    with sqlite3.connect(db_path) as conn:
        rows = dict(conn.execute("SELECT run_id, session_id FROM scenario_runs"))
    conn.close()
    assert rows == {"blank": None, "padded": SESSION}
    assert repo.latest_session_id() == SESSION


def test_the_invocation_session_tags_rows_whose_metadata_has_none(tmp_path: Path) -> None:
    repo = open_repository(db_path=str(tmp_path / "b.sqlite"))
    assert run_queries.current_session_id() is None

    with run_queries.invocation_session(SESSION):
        assert run_queries.current_session_id() == SESSION
        repo.upsert_scenario_run({**_run("no-context", session_id=None), "metadata": None})
        repo.upsert_scenario_run(_run("blank", session_id=""))
        repo.upsert_scenario_run(_run("own", session_id="explicit"))
    repo.upsert_scenario_run(_run("after", session_id=None))

    assert run_queries.current_session_id() is None
    assert [run["run_id"] for run in repo.list_session(SESSION)] == ["no-context", "blank"]
    assert repo.get("own")["session_id"] == "explicit"
    assert repo.get("after")["session_id"] is None


def test_database_exists_does_not_create_the_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    assert run_queries.database_exists() is False
    assert not (tmp_path / "data").exists()

    run_queries.persist_run(_run("a", session_id=SESSION))
    assert run_queries.database_exists() is True


def test_a_write_without_a_session_keeps_the_stored_one(tmp_path: Path) -> None:
    repo = open_repository(db_path=str(tmp_path / "b.sqlite"))
    repo.upsert_scenario_run(_run("r", session_id=SESSION, status="interrupted"))
    # A Python API resume rewrites the row with metadata that has no session.
    repo.upsert_scenario_run(_run("r", session_id=None))

    stored = repo.get("r")
    assert stored is not None and stored["status"] == "completed"
    assert stored["session_id"] == SESSION
    assert "session_id" not in stored["metadata"]
    assert [run["run_id"] for run in repo.list_session(SESSION)] == ["r"]
    # A later CLI resume still finds the session through the column.
    assert dispatch._resumed_session_id(stored) == SESSION

    repo.upsert_scenario_run(_run("r", session_id="replaced"))
    assert repo.get("r")["session_id"] == "replaced"


def test_history_shows_a_short_session_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from io import StringIO

    from rich.console import Console

    from tool_eval_bench.cli.history import print_history

    monkeypatch.chdir(tmp_path)
    run_queries.persist_run(_run("tagged-run", session_id=SESSION))
    run_queries.persist_run(_run("untagged-run", session_id=None))
    out = StringIO()

    print_history(Console(file=out, width=200, no_color=True))

    text = out.getvalue()
    header = next(line for line in text.splitlines() if "Run ID" in line)
    rows = {
        cells[1]: cells[2]
        for line in text.splitlines()
        if "-run" in line
        for cells in [[cell.strip() for cell in line.split("│")]]
    }
    assert [cell.strip() for cell in header.split("┃")][2] == "Session"
    assert rows == {"tagged-run": SESSION[:8], "untagged-run": "—"}


# ---------------------------------------------------------------------------
# Fingerprints and reports
# ---------------------------------------------------------------------------


def test_session_id_does_not_move_any_fingerprint() -> None:
    config = {"model": "m", "backend": "vllm", "base_url": "http://gpu.test:8000/v1"}
    without = _context().to_dict()
    with_session = _context(session_id=SESSION).to_dict()
    assert with_session["session_id"] == SESSION

    assert comparison_fingerprint(config, with_session) == comparison_fingerprint(config, without)
    assert with_config_fingerprint(config, with_session) == with_config_fingerprint(config, without)
    assert _cohort_fingerprint(config, with_session) == _cohort_fingerprint(config, without)


def test_scored_run_config_and_id_ignore_the_session_id() -> None:
    from tool_eval_bench.application.run_config import (
        CONFIG_FINGERPRINT_KEY,
        RunSettings,
        build_run_config,
    )
    from tool_eval_bench.evals.scenarios import SCENARIOS

    settings = RunSettings(
        model="m",
        backend="vllm",
        base_url="http://gpu.test:8000/v1",
        temperature=0.0,
        timeout_seconds=120.0,
        max_turns=8,
        seed=None,
        reference_date=None,
        concurrency=1,
        error_rate=0.0,
        alpha=0.7,
        extra_params=None,
        context_pressure_config=None,
        weight_by_difficulty=False,
    )
    scenarios = [s for s in SCENARIOS if s.id == "TC-01"]

    def config(context: RunContext) -> dict[str, Any]:
        return build_run_config(settings, scenarios=scenarios, metadata=context.to_dict())

    without, with_session = config(_context()), config(_context(session_id=SESSION))
    assert without == with_session
    assert without[CONFIG_FINGERPRINT_KEY] == with_session[CONFIG_FINGERPRINT_KEY]


def test_session_id_stays_out_of_the_markdown_context() -> None:
    ctx = _context(session_id=SESSION)
    rendered = "\n".join(_render_run_context(ctx) + _render_engine_context(ctx))

    assert SESSION not in rendered
    assert rendered == "\n".join(
        _render_run_context(replace(ctx, session_id=None))
        + _render_engine_context(replace(ctx, session_id=None))
    )


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def _fake_gsm8k(
    console: Any,
    model: str,
    display_name: str,
    base_url: str,
    api_key: str | None,
    args: Any,
    *,
    output_dir: str | None,
    run_context: RunContext | None,
    **_: Any,
) -> None:
    """Stands in for the GSM8K runner but stores through the real plugin lifecycle."""
    finalize_plugin_run(
        console,
        mode="gsm8k",
        title="GSM8K",
        display_name=display_name,
        result=SimpleNamespace(score=80.0, rating="Good", details={}),
        config={"model": model, "base_url": base_url, "benchmark": "gsm8k"},
        report_metrics=[],
        report_lines=[],
        output_dir=output_dir,
        label=None,
        run_context=run_context,
    )


@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Drive ``dispatch.main`` against a real database with no network."""
    for key in list(os.environ):
        if key.startswith("TOOL_EVAL_"):
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)

    async def collect_run_context(**kwargs: Any) -> RunContext:
        return _context()

    monkeypatch.setattr(dispatch, "_load_dotenv", lambda: None)
    monkeypatch.setattr(metadata_module, "collect_run_context", collect_run_context)
    monkeypatch.setattr(dispatch, "_run_llama_benchy", lambda *a, **k: [OK_CELL])
    monkeypatch.setattr(dispatch, "_run_gsm8k_benchmark", _fake_gsm8k)
    monkeypatch.setattr(
        BenchmarkService, "_adapter_for", lambda *_args, **_kwargs: SmokeMockAdapter()
    )
    return tmp_path


def _main(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *flags: str) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tool-eval-bench",
            "--base-url",
            "http://localhost:9999",
            "--model",
            "m",
            "--backend",
            "vllm",
            "--no-preflight",
            "--no-warmup",
            "--no-probe-engine",
            "--no-live",
            "--output-dir",
            str(tmp_path / "runs"),
            *flags,
        ],
    )
    dispatch.main()


def _stored(tmp_path: Path) -> list[dict[str, Any]]:
    repo = open_repository(db_path=str(tmp_path / "data" / "benchmarks.sqlite"))
    return [
        {**run, "session_id": (run["metadata"] or {}).get("session_id")}
        for run in reversed(repo.list(limit=50))
    ]


def test_runs_from_one_invocation_share_a_session_and_invocations_differ(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _main(monkeypatch, cli_env, "--perf", "--gsm8k", "--skip-tool-eval")
    _main(monkeypatch, cli_env, "--scenarios", "TC-01")
    capsys.readouterr()

    stored = _stored(cli_env)
    assert [run["run_type"] for run in stored] == ["perf", "gsm8k", "tool_eval"]
    first, second = stored[0]["session_id"], stored[2]["session_id"]
    assert first and second and first != second
    assert stored[1]["session_id"] == first

    assert [run["run_type"] for run in run_queries.session_runs(first)] == ["perf", "gsm8k"]
    assert [run["run_type"] for run in run_queries.session_runs(second)] == ["tool_eval"]
    assert run_queries.latest_session_id(model="m") == second
    # Reports are shareable artifacts; the id stays in SQLite.
    for report in (cli_env / "runs").rglob("*.md"):
        text = report.read_text(encoding="utf-8")
        assert first not in text and second not in text


def test_every_trial_of_a_scored_run_joins_the_session(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _main(monkeypatch, cli_env, "--scenarios", "TC-01", "--trials", "2", "--perf")
    capsys.readouterr()

    stored = _stored(cli_env)
    assert [run["run_type"] for run in stored] == ["tool_eval", "tool_eval"]
    assert stored[0]["session_id"] and stored[0]["session_id"] == stored[1]["session_id"]
    # --perf samples ride inside the scored row rather than a row of their own.
    assert "throughput" in stored[0]["scores"]


def test_resume_keeps_the_interrupted_invocations_session(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _main(monkeypatch, cli_env, "--perf", "--skip-tool-eval")
    _main(monkeypatch, cli_env, "--scenarios", "TC-01")
    perf, scored = _stored(cli_env)
    # Model the interruption: the scored run's row is resumable, and the perf
    # row stands in for the work done before it in the same invocation.
    repo = open_repository(db_path=str(cli_env / "data" / "benchmarks.sqlite"))
    original = perf["session_id"]
    repo.upsert_scenario_run(
        {
            **scored,
            "status": "interrupted",
            "metadata": {**scored["metadata"], "session_id": original},
        }
    )
    repo.close()

    _main(monkeypatch, cli_env, "--scenarios", "TC-01", "--resume", scored["run_id"])
    capsys.readouterr()

    resumed = run_queries.get_run(scored["run_id"])
    assert resumed is not None and resumed["status"] == "completed"
    assert resumed["metadata"]["session_id"] == original
    assert [run["run_id"] for run in run_queries.session_runs(original)] == [
        perf["run_id"],
        scored["run_id"],
    ]


def test_resume_of_a_run_without_a_session_starts_a_fresh_one() -> None:
    assert dispatch._resumed_session_id({"metadata": {"session_id": SESSION}}) == SESSION
    assert dispatch._resumed_session_id({"metadata": {}}) is None
    assert dispatch._resumed_session_id({"metadata": None}) is None
    assert dispatch._resumed_session_id({"metadata": {"session_id": ""}}) is None


def _session_column(tmp_path: Path) -> list[tuple[str, str | None]]:
    """``(run_type, session_id)`` per stored run, oldest first, read from the column."""
    with open_repository(db_path=str(tmp_path / "data" / "benchmarks.sqlite")) as repo:
        return [(run["run_type"], run["session_id"]) for run in reversed(repo.list(limit=50))]


def test_runs_without_a_run_context_still_join_the_invocations_session(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def no_context(**_: Any) -> None:
        return None

    monkeypatch.setattr(dispatch, "build_run_context", no_context)

    _main(monkeypatch, cli_env, "--perf", "--gsm8k", "--skip-tool-eval")
    _main(monkeypatch, cli_env, "--scenarios", "TC-01")
    capsys.readouterr()

    (perf_type, first), (gsm8k_type, gsm8k_session), (scored_type, second) = _session_column(
        cli_env
    )
    assert (perf_type, gsm8k_type, scored_type) == ("perf", "gsm8k", "tool_eval")
    assert first and second and first != second
    assert gsm8k_session == first
    # A bare card picks the newest session, which must be the latest invocation.
    assert run_queries.latest_session_id() == second
    assert run_queries.current_session_id() is None


def test_run_saved_event_carries_the_session_id(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _main(monkeypatch, cli_env, "--perf", "--skip-tool-eval", "--json")
    err = capsys.readouterr().err

    (saved,) = [
        event for event in map(json.loads, err.splitlines()) if event["event"] == "run_saved"
    ]
    ((run_type, session_id),) = _session_column(cli_env)
    assert run_type == "perf" and session_id
    assert saved["session_id"] == session_id
