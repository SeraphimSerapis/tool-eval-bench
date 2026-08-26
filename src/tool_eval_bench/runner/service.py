"""Compatibility exports for the application-owned benchmark service.

``BenchmarkService`` historically lived in this module. The import remains
supported while composition now belongs to :mod:`tool_eval_bench.application`.
"""

from tool_eval_bench.application.run_config import (
    RunSettings as RunSettings,
)
from tool_eval_bench.application.run_config import (
    build_run_config as build_run_config,
)
from tool_eval_bench.application.service import (
    BenchmarkService,
)
from tool_eval_bench.application.service import (
    _collect_metadata_safe as _collect_metadata_safe,
)

__all__ = ["BenchmarkService"]
