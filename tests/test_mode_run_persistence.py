"""Every mode entry point writes its report before it persists the run.

Each case points a mode at an output directory that is a regular file, so the
report cannot be written.  A mode that finalizes through ``finalize_mode_run``
raises before anything reaches SQLite; a mode that persists on its own path, or
persists before writing, fails here.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from tests.test_mode_run_golden import (
    BASE_URL,
    SWEEP_FLAGS,
    _args,
    _context,
    _gsm8k,
    _offline_datasets,  # noqa: F401  (fixture)
    _patch_sweep,
    _spec_samples,
    _target,
    _throughput_samples,
    persisted,  # noqa: F401  (fixture)
)


def _sweep(monkeypatch: pytest.MonkeyPatch, out: Path) -> Callable[[], Any]:
    from tool_eval_bench.cli.dispatch import _run_pressure_sweep_mode

    _patch_sweep(monkeypatch)
    target = _target(_args(out, *SWEEP_FLAGS), _context())
    return lambda: _run_pressure_sweep_mode(target)


def _spec(monkeypatch: pytest.MonkeyPatch, out: Path) -> Callable[[], Any]:
    from tool_eval_bench.cli.dispatch import _run_spec_bench_mode
    from tool_eval_bench.runner import speculative

    samples = _spec_samples("response", 1)

    async def fake_run(*args: Any, on_sample: Callable[..., Any], **kwargs: Any) -> list[Any]:
        for index, sample in enumerate(samples):
            await on_sample(sample, index, len(samples))
        return samples

    monkeypatch.setattr(speculative, "run_spec_bench", fake_run)
    args = _args(out, "--spec-bench", "--depth", "0,4096", "--spec-prompts", "filler,code")
    target = _target(args, _context())
    return lambda: _run_spec_bench_mode(target)


def _throughput(monkeypatch: pytest.MonkeyPatch, out: Path) -> Callable[[], Any]:
    from tool_eval_bench.cli import dispatch

    monkeypatch.setattr(dispatch, "_run_llama_benchy", lambda *a, **k: _throughput_samples(False))
    target = _target(_args(out, "--perf-only"), _context())
    return lambda: dispatch._run_throughput_mode(target)


def _plugin(monkeypatch: pytest.MonkeyPatch, out: Path) -> Callable[[], Any]:
    runner, flags = _gsm8k(monkeypatch)
    args = _args(out, *flags)
    return lambda: runner(
        Console(record=True, width=200),
        "m",
        "Golden Model",
        BASE_URL,
        None,
        args,
        extra_params=None,
        output_dir=str(out),
        run_context=_context(),
    )


@pytest.mark.usefixtures("_offline_datasets")
@pytest.mark.parametrize(
    "mode",
    [_sweep, _spec, _throughput, _plugin],
    ids=["sweep", "spec-bench", "throughput", "plugin"],
)
def test_unwritable_report_persists_nothing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    persisted: list[dict[str, Any]],  # noqa: F811
    mode: Callable[[pytest.MonkeyPatch, Path], Callable[[], Any]],
) -> None:
    out = tmp_path / "not-a-directory"
    out.write_text("", encoding="utf-8")
    run = mode(monkeypatch, out)

    with pytest.raises(OSError):
        run()

    assert persisted == []
