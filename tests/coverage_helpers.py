"""Small shared collaborators for subsystem coverage tests."""

from __future__ import annotations

from tool_eval_bench.runner.throughput import ThroughputSample


class MagicAsyncClient:
    """Marker client for functions whose HTTP collaborators are monkeypatched."""


def async_return(value: object):
    async def inner(*args: object, **kwargs: object):
        return value

    return inner


def throughput_sample(**overrides: object) -> ThroughputSample:
    values = {
        "pp_tokens": 100,
        "tg_tokens": 20,
        "depth": 1024,
        "concurrency": 2,
        "ttft_ms": 5.0,
        "total_ms": 100.0,
        "pp_tps": 1000.0,
        "tg_tps": 200.0,
        "requested_pp": 100,
        "requested_depth": 1024,
        "calibration_confidence": "heuristic",
    }
    values.update(overrides)
    return ThroughputSample(**values)
