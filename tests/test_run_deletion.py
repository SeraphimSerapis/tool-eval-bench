"""Deleting stored runs: storage, report-file safety, the web endpoint, and the CLI command."""

from __future__ import annotations

import argparse
import asyncio
import io
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from tests.web_support import TEST_TOKEN, web_client
from tool_eval_bench.application.run_deletion import delete_runs, find_runs, is_run_report
from tool_eval_bench.cli import delete as delete_cli
from tool_eval_bench.cli.legacy_parser import make_parser
from tool_eval_bench.cli.parser import parse_cli_args
from tool_eval_bench.storage.db import RunRepository
from tool_eval_bench.web import create_app

OLD = "2026-09-01T08-00-00.000000Z_aaaa1111"
PARTIAL = "2026-10-26T13-30-22.083000Z_bbbb2222"
KEEP = "2026-10-26T14-23-40.033000Z_cccc3333"


def _report(root: Path, run_id: str, *, name: str | None = None) -> Path:
    path = root / "runs" / "2026" / "10" / (name or f"{run_id}.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {run_id}\n", encoding="utf-8")
    return path


def _store(
    run_id: str,
    *,
    status: str = "completed",
    model: str = "Qwen/Qwen3.8-Flash",
    report: Path | None = None,
) -> None:
    with RunRepository() as repo:
        repo.upsert_scenario_run(
            {
                "run_id": run_id,
                "status": status,
                "config": {"model": model},
                "metadata": {"label": "nightly"},
                "scores": {
                    "final_score": 80,
                    "scenario_results": [
                        {"scenario_id": "TC-01", "status": "pass", "points": 2, "raw_log": "trace"}
                    ],
                },
                "report_path": str(report) if report else None,
            }
        )
        repo.checkpoint_scenario_result(run_id, {"scenario_id": "TC-01", "status": "pass"})


def _set_created_at(run_id: str, created_at: str) -> None:
    with RunRepository() as repo, repo._conn as conn:
        conn.execute("UPDATE scenario_runs SET created_at=? WHERE run_id=?", (created_at, run_id))


@pytest.fixture()
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A working directory, so the database and reports resolve under tmp_path."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


class TestDeleteRuns:
    def test_removes_the_row_traces_checkpoints_and_report(self, workdir: Path) -> None:
        report = _report(workdir, PARTIAL)
        kept_report = _report(workdir, KEEP)
        _store(PARTIAL, status="interrupted", report=report)
        _store(KEEP, report=kept_report)

        result = delete_runs([PARTIAL, PARTIAL])

        assert result.deleted == (PARTIAL,)
        assert result.missing == ()
        assert result.reports_removed == (str(report),)
        assert not report.exists()
        assert kept_report.exists()
        with RunRepository() as repo:
            assert repo.get(PARTIAL) is None
            assert repo.get_traces(PARTIAL) == {}
            assert repo.get_checkpoints(PARTIAL) == []
            assert repo.get(KEEP) is not None
            assert repo.get_checkpoints(KEEP) != []

    def test_unknown_ids_are_reported_as_missing(self, workdir: Path) -> None:
        _store(KEEP)
        result = delete_runs(["nope", KEEP])
        assert result.deleted == (KEEP,)
        assert result.missing == ("nope",)

    def test_no_database_is_created(self, workdir: Path) -> None:
        result = delete_runs([KEEP])
        assert result.missing == (KEEP,)
        assert find_runs(statuses=["running"]) == []
        assert not (workdir / "data").exists()

    def test_a_path_that_is_not_a_report_is_kept(self, workdir: Path) -> None:
        unrelated = workdir / "notes.md"
        unrelated.write_text("mine", encoding="utf-8")
        _store(PARTIAL, report=unrelated)

        result = delete_runs([PARTIAL])

        assert result.deleted == (PARTIAL,)
        assert result.reports_removed == ()
        assert [(k.path, k.reason) for k in result.reports_kept] == [
            (str(unrelated), "not a report this tool wrote")
        ]
        assert unrelated.read_text(encoding="utf-8") == "mine"

    def test_a_report_another_run_still_uses_is_kept(self, workdir: Path) -> None:
        report = _report(workdir, PARTIAL)
        _store(PARTIAL, report=report)
        _store(KEEP, report=report)

        result = delete_runs([PARTIAL])

        assert result.reports_kept[0].reason == "another stored run uses it"
        assert report.exists()

    def test_a_report_already_gone_is_not_an_error(self, workdir: Path) -> None:
        report = _report(workdir, PARTIAL)
        _store(PARTIAL, report=report)
        report.unlink()
        result = delete_runs([PARTIAL])
        assert (result.deleted, result.reports_removed, result.reports_kept) == ((PARTIAL,), (), ())


class TestIsRunReport:
    @pytest.mark.parametrize(
        "name", [f"{PARTIAL}.md", f"{PARTIAL}--nightly.md", f"{PARTIAL}--nightly_summary.md"]
    )
    def test_writer_names_match(self, tmp_path: Path, name: str) -> None:
        assert is_run_report(_report(tmp_path, PARTIAL, name=name), PARTIAL)

    @pytest.mark.parametrize(
        "name",
        [
            f"{PARTIAL}x.md",  # a longer run id that shares the prefix
            f"{PARTIAL}.txt",
            f"other-{PARTIAL}.md",
        ],
    )
    def test_other_names_do_not_match(self, tmp_path: Path, name: str) -> None:
        assert not is_run_report(_report(tmp_path, PARTIAL, name=name), PARTIAL)

    def test_a_report_outside_a_year_month_folder_does_not_match(self, tmp_path: Path) -> None:
        path = tmp_path / "elsewhere" / f"{PARTIAL}.md"
        path.parent.mkdir()
        path.write_text("x", encoding="utf-8")
        assert not is_run_report(path, PARTIAL)

    def test_an_empty_run_id_matches_nothing(self, tmp_path: Path) -> None:
        assert not is_run_report(_report(tmp_path, PARTIAL), "")


class TestFindRuns:
    def test_filters_combine(self, workdir: Path) -> None:
        _store(OLD, status="completed")
        _store(PARTIAL, status="interrupted")
        _store(KEEP, status="completed", model="meta-llama/Llama-3.3-70B")
        _set_created_at(OLD, "2026-09-01T08:00:00+00:00")
        _set_created_at(PARTIAL, "2026-10-26T13:30:22+00:00")
        _set_created_at(KEEP, "2026-10-26T14:23:40+00:00")

        assert [r["run_id"] for r in find_runs(statuses=["interrupted"])] == [PARTIAL]
        assert [r["run_id"] for r in find_runs(before="2026-10-26")] == [OLD]
        assert [r["run_id"] for r in find_runs(model="Qwen3.8-Flash")] == [PARTIAL, OLD]
        assert find_runs(statuses=["interrupted"], model="Llama-3.3-70B") == []
        assert find_runs(statuses=[]) == []
        assert find_runs(statuses=["interrupted"])[0]["label"] == "nightly"


class TestWebEndpoint:
    @pytest.mark.asyncio
    async def test_deletes_and_reports_counts(self, workdir: Path) -> None:
        report = _report(workdir, PARTIAL)
        _store(PARTIAL, status="interrupted", report=report)
        _store(KEEP)
        app = create_app(token=TEST_TOKEN)
        async with web_client(app) as client:
            resp = await client.post("/api/runs/delete", json={"run_ids": [PARTIAL, "nope"]})
            assert resp.status_code == 200
            assert resp.json() == {
                "deleted": [PARTIAL],
                "missing": ["nope"],
                "reports_removed": 1,
                "reports_kept": [],
            }
            listed = (await client.get("/api/runs")).json()["runs"]
        assert [r["run_id"] for r in listed] == [KEEP]
        assert not report.exists()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "body", [{}, {"run_ids": []}, {"run_ids": "x"}, {"run_ids": [1]}, {"run_ids": [""]}]
    )
    async def test_rejects_malformed_payloads(self, workdir: Path, body: dict[str, Any]) -> None:
        app = create_app(token=TEST_TOKEN)
        async with web_client(app) as client:
            resp = await client.post("/api/runs/delete", json=body)
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_refuses_a_run_still_executing_here(self, workdir: Path) -> None:
        _store(PARTIAL, status="running")
        app = create_app(token=TEST_TOKEN)
        session = app.state.live_hub.register(PARTIAL, model="m", backend="vllm", total=1)
        session.task = asyncio.create_task(asyncio.sleep(30))
        try:
            async with web_client(app) as client:
                resp = await client.post("/api/runs/delete", json={"run_ids": [PARTIAL]})
        finally:
            session.task.cancel()
        assert resp.status_code == 409
        with RunRepository() as repo:
            assert repo.get(PARTIAL) is not None

    @pytest.mark.asyncio
    async def test_cross_site_and_non_json_requests_delete_nothing(self, workdir: Path) -> None:
        _store(KEEP)
        app = create_app(token=TEST_TOKEN)
        async with web_client(app) as client:
            cross = await client.post(
                "/api/runs/delete",
                json={"run_ids": [KEEP]},
                headers={"Sec-Fetch-Site": "cross-site"},
            )
            plain = await client.post(
                "/api/runs/delete",
                content=f'{{"run_ids": ["{KEEP}"]}}',
                headers={"Content-Type": "text/plain"},
            )
        assert (cross.status_code, plain.status_code) == (403, 415)
        with RunRepository() as repo:
            assert repo.get(KEEP) is not None


def _parse(argv: list[str]) -> argparse.Namespace:
    return parse_cli_args(make_parser, argv)[1]


def _run(argv: list[str], *, tty: bool = False, confirm: bool = False, monkeypatch: Any) -> str:
    monkeypatch.setattr(delete_cli.sys.stdin, "isatty", lambda: tty)
    monkeypatch.setattr(delete_cli, "_confirm", lambda console, prompt: confirm)
    buffer = io.StringIO()
    delete_cli.run_delete_command(_parse(argv), Console(file=buffer, width=200))
    return buffer.getvalue()


class TestCli:
    def test_subcommand_translates_to_the_flat_flags(self) -> None:
        args = _parse(
            ["delete", "--status", "interrupted", "--status", "failed", "--before", "2026-10-01"]
        )
        assert args.delete == []
        assert args.delete_status == ["interrupted", "failed"]
        assert args.delete_before == "2026-10-01"
        assert _parse(["delete", OLD, KEEP, "-y"]).delete == [OLD, KEEP]
        assert _parse(["--delete", OLD, "--yes"]).yes is True

    @pytest.mark.parametrize("before", ["2026-10", "26-10-2026", "2026-10-01T00:00"])
    def test_before_must_be_a_date(self, before: str) -> None:
        with pytest.raises(SystemExit) as exc_info:
            _parse(["delete", "--before", before])
        assert exc_info.value.code == 2

    def test_deletes_with_yes(self, workdir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        report = _report(workdir, PARTIAL)
        _store(PARTIAL, status="interrupted", report=report)
        _store(KEEP)
        text = _run(["delete", "--status", "interrupted", "--yes"], monkeypatch=monkeypatch)
        assert "Deleted 1 run and 1 report" in text
        assert [r["run_id"] for r in find_runs()] == [KEEP]

    def test_confirmation_prompt_decides(
        self, workdir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _store(PARTIAL)
        text = _run(["delete", PARTIAL], tty=True, confirm=False, monkeypatch=monkeypatch)
        assert "Nothing was deleted" in text
        assert find_runs() != []
        _run(["delete", PARTIAL], tty=True, confirm=True, monkeypatch=monkeypatch)
        assert find_runs() == []

    def test_no_terminal_without_yes_is_refused(
        self, workdir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _store(PARTIAL)
        with pytest.raises(SystemExit) as exc_info:
            _run(["delete", PARTIAL], monkeypatch=monkeypatch)
        assert exc_info.value.code == 2
        assert find_runs() != []

    def test_dry_run_lists_and_keeps(self, workdir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _store(PARTIAL, status="running")
        text = _run(["delete", "--status", "running", "--dry-run"], monkeypatch=monkeypatch)
        assert PARTIAL in text
        assert "writes its row back" in text
        assert "1 run would be deleted" in text
        assert find_runs() != []

    def test_an_unknown_id_deletes_nothing(
        self, workdir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _store(PARTIAL)
        with pytest.raises(SystemExit) as exc_info:
            _run(["delete", PARTIAL, "nope", "--yes"], monkeypatch=monkeypatch)
        assert exc_info.value.code == 1
        assert find_runs() != []

    @pytest.mark.parametrize(
        "argv", [["delete"], ["delete", PARTIAL, "--status", "failed"]], ids=["none", "both"]
    )
    def test_selection_must_be_ids_or_filters(
        self, workdir: Path, monkeypatch: pytest.MonkeyPatch, argv: list[str]
    ) -> None:
        with pytest.raises(SystemExit) as exc_info:
            _run(argv, monkeypatch=monkeypatch)
        assert exc_info.value.code == 2

    def test_no_match_is_not_an_error(self, workdir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _store(KEEP)
        text = _run(["delete", "--status", "failed", "--yes"], monkeypatch=monkeypatch)
        assert "No stored runs match" in text
