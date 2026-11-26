"""Checkpoint writes must not stall the event loop.

`checkpoint_scenario_result` is a synchronous INSERT that fsyncs.  Called
directly from an async callback it blocks every scenario in flight, which is
invisible at `--parallel 1` and costly above it.  These tests pin the offload.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from tests.conftest import open_repository
from tool_eval_bench.storage.db import RunRepository


@pytest.fixture
def repo(tmp_path) -> RunRepository:
    repository = open_repository(db_path=str(tmp_path / "bench.sqlite"))
    yield repository
    repository.close()


@pytest.mark.asyncio
async def test_async_checkpoint_round_trips_like_the_sync_one(repo: RunRepository) -> None:
    await repo.acheckpoint_scenario_result("R1", {"scenario_id": "TC-01", "status": "pass"})
    await repo.acheckpoint_scenario_result("R1", {"scenario_id": "TC-02", "status": "fail"})

    stored = repo.get_checkpoints("R1")

    assert [r["scenario_id"] for r in stored] == ["TC-01", "TC-02"]
    assert [r["status"] for r in stored] == ["pass", "fail"]


@pytest.mark.asyncio
async def test_a_slow_write_overlaps_with_other_work(
    repo: RunRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An async task must keep ticking while the offloaded DB write is blocked.

    Uses a deterministic threading.Event handshake: the write thread signals
    entry and blocks until the event-loop ticker observes it and releases it.
    If the write ran on the event loop, the ticker could never run before the
    wait timed out.
    """
    write_entered = threading.Event()
    allow_finish = threading.Event()
    write_released = False

    def slow_write(run_id: str, result: dict) -> None:
        nonlocal write_released
        write_entered.set()
        write_released = allow_finish.wait(timeout=2.0)

    monkeypatch.setattr(repo, "checkpoint_scenario_result", slow_write)
    ticker_ran_while_blocked = False

    async def ticker() -> None:
        nonlocal ticker_ran_while_blocked
        deadline = time.monotonic() + 2.0
        while not write_entered.is_set():
            assert time.monotonic() < deadline, "offloaded write never started"
            await asyncio.sleep(0.001)
        ticker_ran_while_blocked = True
        allow_finish.set()

    await asyncio.gather(
        repo.acheckpoint_scenario_result("R1", {"scenario_id": "TC-01"}),
        ticker(),
    )

    assert ticker_ran_while_blocked is True
    assert write_released is True


@pytest.mark.asyncio
async def test_concurrent_checkpoints_are_all_persisted(repo: RunRepository) -> None:
    """Serialised on one writer thread, so none of them are lost or corrupted."""
    await asyncio.gather(
        *(repo.acheckpoint_scenario_result("R1", {"scenario_id": f"TC-{i:02d}"}) for i in range(25))
    )

    stored = repo.get_checkpoints("R1")

    assert len(stored) == 25
    assert {r["scenario_id"] for r in stored} == {f"TC-{i:02d}" for i in range(25)}


def test_close_shuts_down_the_writer_thread(tmp_path) -> None:
    repository = open_repository(db_path=str(tmp_path / "bench.sqlite"))
    asyncio.run(repository.acheckpoint_scenario_result("R1", {"scenario_id": "TC-01"}))

    assert repository._writer is not None
    repository.close()
    assert repository._writer is None


@pytest.mark.asyncio
async def test_concurrent_async_checkpoints_and_main_thread_writes_do_not_race(
    tmp_path,
) -> None:
    with open_repository(db_path=str(tmp_path / "race.sqlite")) as repo:
        repo.upsert_scenario_run(
            {"run_id": "r1", "status": "running", "config": {"model": "m"}, "scores": {}}
        )

        def hammer_sync() -> None:
            for i in range(25):
                repo.mark_run_status("r1", "running" if i % 2 == 0 else "interrupted")
                _ = repo.get("r1")
                _ = repo.list(limit=5)

        await asyncio.gather(
            asyncio.to_thread(hammer_sync),
            *(
                repo.acheckpoint_scenario_result(
                    "r1", {"scenario_id": f"TC-{i:02d}", "status": "pass", "points": 2}
                )
                for i in range(1, 26)
            ),
        )
        assert len(repo.get_checkpoints("r1")) == 25
