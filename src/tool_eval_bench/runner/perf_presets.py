"""Throughput sweep presets: which llama-benchy grids ``--perf`` runs.

llama-benchy measures the full cross product of pp, tg, depth, and concurrency
in one invocation. A long-context sweep run that way would put four 128K
requests in flight at once and exhaust the KV cache, so the ``long`` preset is
two invocations:

1. a concurrency sweep at depth 0, and
2. a depth sweep at concurrency 1, with llama-benchy's
   ``--enable-prefix-caching`` mode so every depth records both the cold
   prefill and the cached-prefix follow-up an agent loop actually sees.

``short`` is the grid ``--perf`` has always run, and its stored config must
stay byte-identical so existing runs keep their fingerprint.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tool_eval_bench.runner.throughput import ThroughputSample

PERF_PRESETS = ("short", "long")
DEFAULT_PERF_PRESET = "short"

# Chat-template tokens around llama-benchy's system and user messages. The
# served window has to hold them on top of depth + pp + tg, or the deepest
# cell fails with a context-length error instead of being skipped up front.
CHAT_TEMPLATE_HEADROOM = 256


@dataclass(frozen=True)
class PresetDefaults:
    """The values a preset supplies for every workload flag left unset.

    For ``long``, ``depths`` are the depth pass's depths and ``concurrency``
    the depth-0 pass's levels.
    """

    pp: int
    tg: int
    runs: int
    depths: tuple[int, ...]
    concurrency: tuple[int, ...]


PRESET_DEFAULTS: dict[str, PresetDefaults] = {
    "short": PresetDefaults(pp=2048, tg=128, runs=3, depths=(0, 4096, 8192), concurrency=(1, 2, 4)),
    "long": PresetDefaults(
        pp=2048,
        tg=128,
        runs=3,
        depths=(0, 16384, 32768, 65536, 131072),
        concurrency=(1, 2, 4),
    ),
}


@dataclass(frozen=True)
class PerfPass:
    """One llama-benchy invocation: the full ``depths x concurrency`` grid."""

    depths: tuple[int, ...]
    concurrency: tuple[int, ...]
    prefix_caching: bool = False

    def to_config(self) -> dict[str, Any]:
        return {
            "depths": list(self.depths),
            "concurrency": list(self.concurrency),
            "prefix_caching": self.prefix_caching,
        }


@dataclass(frozen=True)
class PerfPlan:
    """A resolved throughput sweep: the workload and the passes that measure it."""

    preset: str
    pp: int
    tg: int
    runs: int
    passes: tuple[PerfPass, ...]
    #: Depth-pass depths left out because ``depth + pp + tg`` does not fit the
    #: window. Never 0: the first pass always measures depth 0.
    dropped_depths: tuple[int, ...] = ()
    #: The served per-request context window, when the run context knows it.
    context_window: int | None = None
    #: Explicit flags changed the preset's pp, tg, depths, or concurrency.
    #: Window capping and ``--benchy-runs`` do not count.
    customized: bool = False

    @property
    def stored_preset(self) -> str:
        """The preset name runs store: ``custom`` once overrides changed the grid."""
        return "custom" if self.customized else self.preset

    @property
    def request_fits(self) -> bool:
        """Whether a depth-0 request, ``pp + tg`` plus template headroom, fits the window."""
        return fits_window(0, pp=self.pp, tg=self.tg, context_window=self.context_window)

    def first_passes(self, count: int) -> PerfPlan:
        """This plan cut to its first *count* passes, for recording what actually ran."""
        return replace(self, passes=self.passes[:count])

    @property
    def depths(self) -> list[int]:
        """Every depth measured. A single pass keeps the flag's list exactly as given."""
        if len(self.passes) == 1:
            return list(self.passes[0].depths)
        return _unique(depth for p in self.passes for depth in p.depths)

    @property
    def concurrency(self) -> list[int]:
        """Every concurrency level measured, in the same form as :attr:`depths`."""
        if len(self.passes) == 1:
            return list(self.passes[0].concurrency)
        return _unique(level for p in self.passes for level in p.concurrency)


def _unique(values: Iterable[int]) -> list[int]:
    return list(dict.fromkeys(values))


def fits_window(depth: int, *, pp: int, tg: int, context_window: int | None) -> bool:
    """Whether one request at *depth* fits the served window; unknown windows fit."""
    if context_window is None:
        return True
    return depth + pp + tg + CHAT_TEMPLATE_HEADROOM <= context_window


def plan_perf(
    preset: str,
    *,
    pp: int | None = None,
    tg: int | None = None,
    runs: int | None = None,
    depths: Sequence[int] | None = None,
    concurrency: Sequence[int] | None = None,
    context_window: int | None = None,
) -> PerfPlan:
    """Resolve *preset* and any explicit flag values into the passes to run.

    An explicit value always wins over the preset's. Under ``long``,
    ``depths`` sets the depth pass and ``concurrency`` the depth-0 pass; the
    depth pass always runs at concurrency 1. Depths that do not fit
    *context_window* are dropped from the depth pass, and so is depth 0 when
    the first pass already measured it at concurrency 1.
    """
    if preset not in PRESET_DEFAULTS:
        raise ValueError(f"Unknown perf preset {preset!r}; choose from {', '.join(PERF_PRESETS)}")
    defaults = PRESET_DEFAULTS[preset]
    pp = defaults.pp if pp is None else pp
    tg = defaults.tg if tg is None else tg
    runs = defaults.runs if runs is None else runs
    depth_list = tuple(defaults.depths if depths is None else depths)
    levels = tuple(defaults.concurrency if concurrency is None else concurrency)
    customized = (pp, tg, depth_list, levels) != (
        defaults.pp,
        defaults.tg,
        defaults.depths,
        defaults.concurrency,
    )

    if preset == "short":
        return PerfPlan(
            preset=preset,
            pp=pp,
            tg=tg,
            runs=runs,
            passes=(PerfPass(depth_list, levels),),
            context_window=context_window,
            customized=customized,
        )

    concurrency_pass = PerfPass((0,), levels)
    # Depth 0 is the first pass's, which always runs it; whether pp + tg fits
    # is reported through ``PerfPlan.request_fits`` rather than by dropping it.
    measures_d0_c1 = 1 in levels
    depth_pass_depths = _unique(
        d
        for d in depth_list
        if (d == 0 and not measures_d0_c1)
        or (d > 0 and fits_window(d, pp=pp, tg=tg, context_window=context_window))
    )
    dropped = tuple(_unique(d for d in depth_list if d > 0 and d not in depth_pass_depths))
    passes: tuple[PerfPass, ...] = (concurrency_pass,)
    if depth_pass_depths:
        passes += (PerfPass(tuple(depth_pass_depths), (1,), prefix_caching=True),)
    return PerfPlan(
        preset=preset,
        pp=pp,
        tg=tg,
        runs=runs,
        passes=passes,
        dropped_depths=dropped,
        context_window=context_window,
        customized=customized,
    )


def failed_cell_count(samples: Iterable[ThroughputSample]) -> int:
    """How many grid cells have no usable sample, the count every store and message uses.

    A cell is one ``pp x tg x depth x concurrency`` point. Under prefix
    caching llama-benchy reports two rows per cell, the context load and the
    inference, and a failure in either marks the cell failed once.
    """
    return len(
        {
            (sample.pp_tokens, sample.tg_tokens, sample.depth, sample.concurrency)
            for sample in samples
            if sample.error
        }
    )


def depth0_speeds(samples: Iterable[ThroughputSample]) -> tuple[float, float] | None:
    """The single-stream depth-0 prefill and decode rates, or None when unmeasured."""
    for sample in samples:
        if (
            sample.depth == 0
            and sample.concurrency == 1
            and not sample.error
            and not sample.is_context_prefill
            and sample.pp_tps > 0
            and sample.tg_tps > 0
        ):
            return sample.pp_tps, sample.tg_tps
    return None


def estimate_pass_seconds(
    perf_pass: PerfPass, *, pp: int, tg: int, runs: int, pp_tps: float, tg_tps: float
) -> float:
    """A rough lower bound on one pass's wall time, from depth-0 speeds.

    Each cell runs ``runs + 1`` times because llama-benchy discards a warm-up
    run per cell. A standard request prefills ``depth + pp`` and decodes
    ``tg``. Under prefix caching a context-load request prefills ``depth`` and
    decodes ``tg``, then the follow-up prefills ``pp`` on the cached prefix and
    decodes ``tg`` again. Every cell is costed as one single-stream request,
    which a concurrent batch can only exceed. Prefill and decode both slow
    down as depth grows, so the real run takes longer than this.
    """
    per_run = 0.0
    for depth in perf_pass.depths:
        prefill = (depth + pp) / pp_tps
        decodes = 2 if perf_pass.prefix_caching and depth > 0 else 1
        per_run += (prefill + decodes * tg / tg_tps) * len(perf_pass.concurrency)
    return per_run * (runs + 1)


def format_duration(seconds: float) -> str:
    """Render an estimate coarsely; the precision would be fake anyway."""
    if seconds < 90:
        return f"{max(1, round(seconds))} s"
    minutes = seconds / 60
    if minutes < 90:
        return f"{round(minutes)} min"
    return f"{minutes / 60:.1f} h"
