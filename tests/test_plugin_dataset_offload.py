"""A plugin's first-use download must not run on the event loop.

Each dataset loader pages the HuggingFace REST API over a synchronous client.
Called directly from `async def run`, that stalls every task in flight for the
length of the download.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

from tool_eval_bench.plugins.gsm8k import plugin as gsm8k_plugin
from tool_eval_bench.plugins.ifeval import plugin as ifeval_plugin
from tool_eval_bench.plugins.mmlu import plugin as mmlu_plugin

PLUGINS = [
    pytest.param(gsm8k_plugin, "gsm8k", id="gsm8k"),
    pytest.param(mmlu_plugin, "mmlu", id="mmlu"),
    pytest.param(ifeval_plugin, "ifeval", id="ifeval"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("module", "name"), PLUGINS)
async def test_the_loader_runs_off_the_event_loop(
    module: Any, name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    loader_threads: list[int] = []

    def record(*args: object, **kwargs: object) -> list:
        loader_threads.append(threading.get_ident())
        raise _Stop

    monkeypatch.setattr(module, "load_dataset", record)

    plugin = module_plugin(module)
    with pytest.raises(_Stop):
        await plugin.run(_UnusedAdapter(), model="m", base_url="http://localhost:1")

    assert loader_threads, "the loader was never called"
    assert loader_threads[0] != threading.get_ident(), (
        f"{name} loaded its dataset on the event loop's thread"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("module", "name"), PLUGINS)
async def test_a_slow_download_does_not_stall_other_tasks(
    module: Any, name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A concurrent task must run while the dataset loader is blocked.

    The loader sets ``loader_started`` and waits on ``release_loader``. A
    blocked event loop could never run ``ticker`` to release it, so the wait
    would time out rather than completing immediately.
    """
    loader_started = threading.Event()
    release_loader = threading.Event()
    ticker_ran_while_blocked = False
    loading_done = False

    def slow(*args: object, **kwargs: object) -> list:
        loader_started.set()
        if not release_loader.wait(timeout=2.0):
            raise AssertionError(f"{name} blocked the event loop; ticker never ran")
        raise _Stop

    monkeypatch.setattr(module, "load_dataset", slow)
    plugin = module_plugin(module)

    async def loading() -> None:
        nonlocal loading_done
        try:
            with pytest.raises(_Stop):
                await plugin.run(_UnusedAdapter(), model="m", base_url="http://localhost:1")
        finally:
            loading_done = True

    async def ticker() -> None:
        nonlocal ticker_ran_while_blocked
        while not loader_started.is_set() and not loading_done:
            await asyncio.sleep(0)
        if loader_started.is_set():
            ticker_ran_while_blocked = True
            release_loader.set()

    await asyncio.gather(loading(), ticker())

    assert ticker_ran_while_blocked, f"{name} blocked the event loop during dataset download"


class _Stop(Exception):
    """Ends the run once the loader has been reached; nothing past it matters."""


class _UnusedAdapter:
    """The run stops at dataset loading, so no request is ever made."""


def module_plugin(module: Any) -> Any:
    """Instantiate whichever `BenchmarkPlugin` subclass the module defines."""
    from tool_eval_bench.domain.plugin import BenchmarkPlugin

    for value in vars(module).values():
        if isinstance(value, type) and issubclass(value, BenchmarkPlugin):
            if value is not BenchmarkPlugin:
                return value()
    raise AssertionError(f"no plugin class in {module.__name__}")
