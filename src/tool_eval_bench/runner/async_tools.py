"""Experimental async tool orchestration (WIP).

This module provides an alternative orchestration model where tool calls can
execute asynchronously and the model receives incremental results. This is
a non-breaking WIP — the existing synchronous orchestrator is unchanged.

Activated via ``--experimental-async`` CLI flag (not yet wired).

Design:
  1. Model makes tool calls as usual.
  2. Instead of blocking until all tools complete, tools can return
     "pending" status with a polling handle.
  3. The orchestrator injects intermediate status updates.
  4. The model can continue generating while tools resolve.
  5. Evaluators score both the final result AND the intermediate behavior.

Status: EXPERIMENTAL — API may change. Do not depend on this in production.
"""

from __future__ import annotations


import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Async tool result types
# ---------------------------------------------------------------------------

class AsyncToolStatus(str, Enum):
    """Status of an async tool execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AsyncToolResult:
    """Result from an async tool that may still be in progress."""
    status: AsyncToolStatus
    handle: str  # Opaque handle for polling
    result: Any = None
    error: str | None = None
    progress_percent: float | None = None  # 0.0-1.0
    elapsed_ms: float = 0.0
    intermediate_data: Any = None  # Partial results available before completion


@dataclass
class AsyncToolSpec:
    """Specification for an async tool's behavior in a scenario."""
    tool_name: str
    # How long the tool takes to complete (simulated)
    duration_ms: float = 2000.0
    # Whether the tool supports intermediate results
    supports_streaming: bool = False
    # Number of progress updates before completion
    progress_steps: int = 3
    # Final result (returned on completion)
    final_result: Any = None
    # Intermediate results at each progress step
    intermediate_results: list[Any] = field(default_factory=list)
    # Failure simulation (set to True to simulate tool failure mid-execution)
    simulate_failure: bool = False
    failure_at_percent: float = 0.75  # Fail at 75% progress


# ---------------------------------------------------------------------------
# Async tool executor
# ---------------------------------------------------------------------------

class AsyncToolExecutor:
    """Manages async tool execution with progress tracking.

    This is a mock executor for benchmarking — it simulates async tool
    behavior with configurable timing, progress updates, and failure modes.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, AsyncToolSpec] = {}
        self._started_at: dict[str, float] = {}
        self._handle_counter = 0

    def register_tool(self, spec: AsyncToolSpec) -> None:
        """Register an async tool specification."""
        self._tasks[spec.tool_name] = spec

    def start_tool(self, tool_name: str) -> AsyncToolResult:
        """Start an async tool execution. Returns immediately with a handle."""
        self._handle_counter += 1
        handle = f"async_{tool_name}_{self._handle_counter}"
        self._started_at[handle] = time.monotonic()

        if tool_name not in self._tasks:
            return AsyncToolResult(
                status=AsyncToolStatus.COMPLETED,
                handle=handle,
                result={"error": f"Tool {tool_name} is not registered as async."},
            )

        return AsyncToolResult(
            status=AsyncToolStatus.PENDING,
            handle=handle,
            progress_percent=0.0,
        )

    def poll_tool(self, handle: str) -> AsyncToolResult:
        """Poll an async tool for its current status.

        This simulates time-based progress — in a real system this would
        check actual task status.
        """
        start = self._started_at.get(handle)
        if start is None:
            return AsyncToolResult(
                status=AsyncToolStatus.FAILED,
                handle=handle,
                error="Unknown handle.",
            )

        # Extract tool name from handle
        parts = handle.split("_")
        tool_name = "_".join(parts[1:-1]) if len(parts) > 2 else "unknown"
        spec = self._tasks.get(tool_name)

        if spec is None:
            return AsyncToolResult(
                status=AsyncToolStatus.COMPLETED,
                handle=handle,
                result={"error": f"No spec for tool {tool_name}."},
            )

        elapsed_ms = (time.monotonic() - start) * 1000
        progress = min(1.0, elapsed_ms / spec.duration_ms)

        # Check for simulated failure
        if spec.simulate_failure and progress >= spec.failure_at_percent:
            return AsyncToolResult(
                status=AsyncToolStatus.FAILED,
                handle=handle,
                error=f"Tool {tool_name} failed at {progress:.0%} progress.",
                progress_percent=progress,
                elapsed_ms=elapsed_ms,
            )

        # Not yet complete
        if progress < 1.0:
            # Determine current intermediate result
            intermediate = None
            if spec.supports_streaming and spec.intermediate_results:
                step_idx = min(
                    int(progress * len(spec.intermediate_results)),
                    len(spec.intermediate_results) - 1,
                )
                intermediate = spec.intermediate_results[step_idx]

            return AsyncToolResult(
                status=AsyncToolStatus.RUNNING,
                handle=handle,
                progress_percent=progress,
                elapsed_ms=elapsed_ms,
                intermediate_data=intermediate,
            )

        # Complete
        return AsyncToolResult(
            status=AsyncToolStatus.COMPLETED,
            handle=handle,
            result=spec.final_result,
            progress_percent=1.0,
            elapsed_ms=elapsed_ms,
        )

    def cancel_tool(self, handle: str) -> AsyncToolResult:
        """Cancel a running async tool."""
        if handle in self._started_at:
            del self._started_at[handle]
        return AsyncToolResult(
            status=AsyncToolStatus.CANCELLED,
            handle=handle,
        )


# ---------------------------------------------------------------------------
# Async-aware orchestration helpers
# ---------------------------------------------------------------------------

def format_async_status(result: AsyncToolResult) -> str:
    """Format an async tool result as a human-readable status string.

    This is what gets injected into the conversation as a tool result
    during async orchestration.
    """
    if result.status == AsyncToolStatus.PENDING:
        return (
            f'{{"status": "pending", "handle": "{result.handle}", '
            f'"message": "Task started. Poll with handle to check progress."}}'
        )
    if result.status == AsyncToolStatus.RUNNING:
        parts = {
            "status": "running",
            "handle": result.handle,
            "progress": f"{(result.progress_percent or 0) * 100:.0f}%",
        }
        if result.intermediate_data is not None:
            parts["intermediate_data"] = result.intermediate_data
        import json
        return json.dumps(parts)
    if result.status == AsyncToolStatus.COMPLETED:
        import json
        return json.dumps({"status": "completed", "result": result.result})
    if result.status == AsyncToolStatus.FAILED:
        return (
            f'{{"status": "failed", "handle": "{result.handle}", '
            f'"error": "{result.error}"}}'
        )
    if result.status == AsyncToolStatus.CANCELLED:
        return f'{{"status": "cancelled", "handle": "{result.handle}"}}'
    return '{"status": "unknown"}'


# ---------------------------------------------------------------------------
# Example async scenario definitions (for future use)
# ---------------------------------------------------------------------------

def create_example_async_specs() -> list[AsyncToolSpec]:
    """Create example async tool specifications for testing.

    These demonstrate the kinds of async patterns that could be tested:
    1. A slow file search that returns intermediate results
    2. A code execution that takes time but shows progress
    3. A web search that fails partway through
    """
    return [
        AsyncToolSpec(
            tool_name="search_files",
            duration_ms=3000.0,
            supports_streaming=True,
            progress_steps=3,
            final_result={
                "results": [
                    {"file_id": "file_201", "name": "Project_Plan_2026.docx"},
                    {"file_id": "file_202", "name": "Budget_Analysis_Q1.xlsx"},
                ],
            },
            intermediate_results=[
                {"partial_count": 1, "first_match": "Project_Plan_2026.docx"},
                {"partial_count": 2, "matches": ["Project_Plan_2026.docx", "Budget_Analysis_Q1.xlsx"]},
            ],
        ),
        AsyncToolSpec(
            tool_name="run_code",
            duration_ms=5000.0,
            supports_streaming=True,
            progress_steps=5,
            final_result={
                "output": "Analysis complete. Total: 42,581 records processed.",
                "exit_code": 0,
            },
            intermediate_results=[
                {"stdout": "Loading data...\n"},
                {"stdout": "Loading data...\nProcessing batch 1/3...\n"},
                {"stdout": "Loading data...\nProcessing batch 1/3...\nProcessing batch 2/3...\n"},
                {"stdout": "Loading data...\nProcessing batch 1/3...\nProcessing batch 2/3...\nProcessing batch 3/3...\n"},
            ],
        ),
        AsyncToolSpec(
            tool_name="web_search",
            duration_ms=2000.0,
            supports_streaming=False,
            simulate_failure=True,
            failure_at_percent=0.6,
            final_result=None,  # Never reached due to failure
        ),
    ]
