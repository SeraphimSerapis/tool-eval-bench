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
    """A blocking write would serialise against the ticker instead of overlapping."""
    write_seconds = 0.3
    tick_seconds = 0.2

    def slow_write(run_id: str, result: dict) -> None:
        time.sleep(write_seconds)

    monkeypatch.setattr(repo, "checkpoint_scenario_result", slow_write)

    async def ticker() -> None:
        await asyncio.sleep(tick_seconds)

    started = time.monotonic()
    await asyncio.gather(
        repo.acheckpoint_scenario_result("R1", {"scenario_id": "TC-01"}),
        ticker(),
    )
    elapsed = time.monotonic() - started

    # Overlapped, this takes about max(0.3, 0.2). Blocking, it takes their sum.
    assert elapsed < (write_seconds + tick_seconds) * 0.8, (
        f"the write blocked the loop: {elapsed:.3f}s for a {write_seconds}s write "
        f"alongside {tick_seconds}s of other work"
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
