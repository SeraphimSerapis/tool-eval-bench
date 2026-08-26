"""Checkpoint writes must not stall the event loop.

`checkpoint_scenario_result` is a synchronous INSERT that fsyncs.  Called
directly from an async callback it blocks every scenario in flight, which is
invisible at `--parallel 1` and costly above it.  These tests pin the offload.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from tool_eval_bench.storage.db import RunRepository


@pytest.fixture
def repo(tmp_path) -> RunRepository:
    repository = RunRepository(db_path=str(tmp_path / "bench.sqlite"))
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
    """A short task started alongside the write must finish on its own schedule.

    Measured on when the ticker finishes rather than on total wall clock: a
    blocked loop cannot run it until the write returns, so the two outcomes
    differ by the whole write rather than by a fraction of it.
    """
    write_seconds = 0.4
    tick_seconds = 0.05

    def slow_write(run_id: str, result: dict) -> None:
        time.sleep(write_seconds)

    monkeypatch.setattr(repo, "checkpoint_scenario_result", slow_write)
    started = time.monotonic()
    ticker_finished: list[float] = []

    async def ticker() -> None:
        await asyncio.sleep(tick_seconds)
        ticker_finished.append(time.monotonic() - started)

    await asyncio.gather(
        repo.acheckpoint_scenario_result("R1", {"scenario_id": "TC-01"}),
        ticker(),
    )

    assert ticker_finished[0] < write_seconds, (
        f"the write blocked the loop: a {tick_seconds}s task finished after "
        f"{ticker_finished[0]:.3f}s, held up by a {write_seconds}s write"
    )


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
    repository = RunRepository(db_path=str(tmp_path / "bench.sqlite"))
    asyncio.run(repository.acheckpoint_scenario_result("R1", {"scenario_id": "TC-01"}))

    assert repository._writer is not None
    repository.close()
    assert repository._writer is None
