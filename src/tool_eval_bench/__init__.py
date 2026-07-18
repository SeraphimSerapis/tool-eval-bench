"""tool_eval_bench package."""

__all__ = ["__version__", "run_benchmark"]
__version__ = "2.2.0"


async def run_benchmark(**kwargs):
    """Convenience re-export — see :func:`tool_eval_bench.api.run_benchmark`.

    This is an async function; call it with ``await`` or wrap in
    ``asyncio.run(run_benchmark(...))``.
    """
    from tool_eval_bench.api import run_benchmark as _run

    return await _run(**kwargs)
