"""Running screen — live benchmark progress."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import (
    DataTable,
    Static,
)


# Status badge helpers
_STATUS_DISPLAY = {
    "pass": "[green]✅ PASS[/]",
    "partial": "[yellow]⚠️  PARTIAL[/]",
    "fail": "[red]❌ FAIL[/]",
    "running": "[bold cyan]▶ RUNNING[/]",
    "pending": "[dim]· pending[/]",
}


def _render_progress(pct: int, completed: int, total: int, elapsed: str, label: str = "") -> str:
    """Render a text-based progress bar with stats."""
    bar_width = 36
    filled = int(bar_width * pct / 100)
    empty = bar_width - filled

    if pct >= 100:
        bar_color = "green"
        fill_char = "█"
    elif pct > 0:
        bar_color = "cyan"
        fill_char = "█"
    else:
        bar_color = "white"
        fill_char = "█"

    bar = f"[{bar_color}]{fill_char * filled}[/][dim]░{('░' * (empty - 1)) if empty > 1 else ''}[/]" if empty > 0 else f"[{bar_color}]{fill_char * filled}[/]"

    stats = f"[bold]{pct}%[/]  {completed}/{total}  {elapsed}"

    lines = f"  {bar}  {stats}"
    if label:
        lines += f"\n  [dim]{label}[/]"

    return lines


class RunningScreen(Screen):
    """Live benchmark execution with per-scenario progress grid."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=True),
    ]

    CSS = """
    RunningScreen {
        layout: vertical;
    }

    #run-title {
        text-style: bold;
        color: $accent;
        text-align: center;
        padding: 1 0 0 0;
    }

    #run-info {
        color: $text-muted;
        text-align: center;
        padding: 0 0 0 0;
    }

    #progress-area {
        height: auto;
        padding: 0 2 1 2;
    }

    #progress-display {
        height: auto;
        padding: 0 2;
    }

    #scenario-table {
        height: 1fr;
        margin: 0 2;
        scrollbar-size-vertical: 1;
    }

    #run-status {
        height: 1;
        padding: 0 2;
        color: $text-muted;
    }

    /* Disable hover on table rows to avoid color artifacts */
    DataTable > .datatable--cursor {
        background: $accent 20%;
    }

    DataTable:focus > .datatable--cursor {
        background: $accent 30%;
    }

    /* Highlight for the actively running row */
    .datatable--row-highlighted {
        background: $accent 15%;
    }
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        self._start_time = 0.0
        self._completed = 0
        self._total = 0
        self._cancelled = False
        self._active_scenario_id: str | None = None

    def compose(self) -> ComposeResult:
        model = self.config.get("model") or "(auto-detect)"
        yield Static(f"⚡ Running Benchmark — {model}", id="run-title")
        yield Static("", id="run-info")

        with Vertical(id="progress-area"):
            yield Static(
                _render_progress(0, 0, 0, "00:00:00", "Initializing…"),
                id="progress-display",
            )

        yield DataTable(id="scenario-table")
        yield Static("", id="run-status")

    def on_mount(self) -> None:
        """Set up the scenario table and start the benchmark."""
        table = self.query_one("#scenario-table", DataTable)
        table.add_column("ID", width=7)
        table.add_column("Category", width=22)
        table.add_column("Title", width=30)
        table.add_column("Status", width=14)
        table.add_column("Points", width=8)
        table.add_column("Time", width=8)
        table.cursor_type = "none"
        table.zebra_stripes = True

        self.call_after_refresh(self._resize_title_column)
        self._start_time = time.monotonic()
        self._run_benchmark()

    def on_resize(self) -> None:
        """Recompute Title column width when terminal is resized."""
        self._resize_title_column()

    def _resize_title_column(self) -> None:
        """Make the Title column fill remaining horizontal space."""
        table = self.query_one("#scenario-table", DataTable)
        # Table width minus margins (0 2 = 4 total), minus scrollbar (1)
        available = self.app.size.width - 5
        fixed_cols = 7 + 22 + 14 + 8 + 8  # ID + Category + Status + Points + Time
        col_padding = 6 * 2  # 6 columns × 2 cells padding each
        title_width = max(20, available - fixed_cols - col_padding)
        cols = list(table.columns.values())
        if len(cols) > 2:
            cols[2].width = title_width
            cols[2].auto_width = False
        table.refresh()

    @work(thread=True)
    def _run_benchmark(self) -> None:
        """Execute the benchmark in a background thread."""
        config = self.config

        from tool_eval_bench.domain.scenarios import ScenarioDefinition, ScenarioResult
        from tool_eval_bench.evals.scenarios import ALL_SCENARIOS, SCENARIOS
        from tool_eval_bench.runner.service import BenchmarkService

        # Resolve scenarios
        if config.get("categories"):
            cats = set(config["categories"])
            scenarios = [s for s in ALL_SCENARIOS if s.category.value in cats]
        elif config.get("short"):
            scenarios = list(SCENARIOS)
        else:
            scenarios = list(ALL_SCENARIOS)

        self._total = len(scenarios)

        # Populate table with pending rows
        self.app.call_from_thread(self._populate_table, scenarios)

        # Build extra params from sampling flags
        extra_params: dict[str, Any] = {}
        if config.get("no_think"):
            extra_params["chat_template_kwargs"] = {"enable_thinking": False}
        if config.get("top_p") is not None:
            extra_params["top_p"] = config["top_p"]
        if config.get("top_k") is not None:
            extra_params["top_k"] = config["top_k"]
        if config.get("min_p") is not None:
            extra_params["min_p"] = config["min_p"]
        if config.get("repeat_penalty") is not None:
            extra_params["repetition_penalty"] = config["repeat_penalty"]

        # Warm-up
        if not config.get("no_warmup"):
            self.app.call_from_thread(
                self._update_progress_display,
                0, 0, "00:00:00", "Warming up server…",
            )
            try:
                from tool_eval_bench.runner.throughput import warmup

                model = config.get("model") or self._detect_model_sync(config)
                if model:
                    config["model"] = model
                    asyncio.run(warmup(
                        config["base_url"], model,
                        config.get("api_key"), timeout=30.0,
                    ))
            except Exception:
                pass

        # Auto-detect model if needed
        if not config.get("model"):
            self.app.call_from_thread(
                self._update_progress_display,
                0, 0, "00:00:00", "Detecting model…",
            )
            model = self._detect_model_sync(config)
            if model:
                config["model"] = model

        if not config.get("model"):
            self.app.call_from_thread(
                self._update_status,
                "[bold red]Error: Could not detect model. Go back and set it manually.[/]"
            )
            return

        model = config["model"]
        self.app.call_from_thread(
            self._update_title, f"⚡ Running Benchmark — {model}"
        )
        info = (
            f"Server: {config['base_url']}  |  "
            f"Backend: {config['backend']}  |  "
            f"Scenarios: {len(scenarios)}"
        )
        self.app.call_from_thread(self._update_info, info)

        # Run tool-call scenarios
        if config.get("run_tool_call", True):
            service = BenchmarkService()

            # Use synchronous callbacks — these run inside asyncio.run()
            # in the background thread, and we use call_from_thread to
            # push UI updates to the main Textual thread.
            async def on_start(scenario: ScenarioDefinition, idx: int, total: int) -> None:
                self.app.call_from_thread(
                    self._on_scenario_start, scenario.id, scenario.title
                )

            async def on_result(
                scenario: ScenarioDefinition,
                result: ScenarioResult,
                idx: int,
                total: int,
            ) -> None:
                self._completed += 1
                duration = f"{result.duration_seconds:.1f}s" if result.duration_seconds else ""
                points = f"{result.points}/2"
                pct = int(self._completed / self._total * 100)
                elapsed = time.monotonic() - self._start_time
                elapsed_str = _format_elapsed(elapsed)

                self.app.call_from_thread(
                    self._on_scenario_complete,
                    scenario.id,
                    result.status.value,
                    points,
                    duration,
                    pct,
                    elapsed_str,
                )

            try:
                result = asyncio.run(service.run_benchmark(
                    model=model,
                    backend=config.get("backend", "vllm"),
                    base_url=config["base_url"],
                    api_key=config.get("api_key"),
                    scenarios=scenarios,
                    temperature=config.get("temperature", 0.0),
                    timeout_seconds=config.get("timeout", 60.0),
                    max_turns=8,
                    seed=None,
                    concurrency=config.get("parallel", 1),
                    error_rate=config.get("error_rate", 0.0),
                    extra_params=extra_params or None,
                    on_scenario_start=on_start,
                    on_scenario_result=on_result,
                ))

                elapsed = time.monotonic() - self._start_time
                scores = result.get("scores", {})
                final_score = scores.get("final_score", 0)
                rating = scores.get("rating", "")

                self.app.call_from_thread(
                    self._update_progress_display,
                    100, self._total,
                    _format_elapsed(elapsed),
                    f"✓ Complete — {final_score}/100 {rating}",
                )
                self.app.call_from_thread(
                    self._update_status,
                    f"[bold green]Score: {final_score}/100 — {rating}[/]  "
                    f"[dim]Report: {result.get('report_path', '')}[/]"
                )

                # Transition to results
                self.app.call_from_thread(
                    self.app.show_results, config, result  # type: ignore[attr-defined]
                )

            except KeyboardInterrupt:
                self.app.call_from_thread(
                    self._update_status, "[bold red]Cancelled.[/]"
                )
            except Exception as exc:
                self.app.call_from_thread(
                    self._update_status, f"[bold red]Error: {exc}[/]"
                )

    def _detect_model_sync(self, config: dict) -> str | None:
        """Synchronously detect model from the server."""
        import httpx

        base_url = config["base_url"].rstrip("/")
        api_key = config.get("api_key")
        models_endpoint = f"{base_url}/v1/models"
        if base_url.endswith("/v1"):
            models_endpoint = f"{base_url}/models"

        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            async def fetch():
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.get(models_endpoint, headers=headers)
                    if resp.status_code == 404:
                        resp = await client.get(f"{base_url}/models", headers=headers)
                    resp.raise_for_status()
                    return resp.json()

            data = asyncio.run(fetch())
            model_list = data.get("data", [])
            if model_list:
                return model_list[0].get("id", "")
        except Exception:
            pass
        return None

    def _populate_table(self, scenarios) -> None:
        """Add all scenarios as pending rows."""
        from tool_eval_bench.domain.scenarios import CATEGORY_LABELS

        table = self.query_one("#scenario-table", DataTable)
        for s in scenarios:
            cat_label = CATEGORY_LABELS.get(s.category, s.category.value)
            table.add_row(
                s.id, f"{s.category.value} {cat_label}", s.title,
                _STATUS_DISPLAY["pending"], "", "",
                key=s.id,
            )

    def _on_scenario_start(self, scenario_id: str, title: str) -> None:
        """Mark a scenario as running and update progress label."""
        table = self.query_one("#scenario-table", DataTable)

        # Clear highlight from previous running scenario
        if self._active_scenario_id:
            self._set_row_status(self._active_scenario_id, "pending", "", "")

        self._active_scenario_id = scenario_id

        # Update this row to running
        self._set_row_status(scenario_id, "running", "", "")

        # Move cursor to this row to keep it visible
        try:
            row_idx = table.get_row_index(scenario_id)
            table.move_cursor(row=row_idx)
        except Exception:
            pass

        # Update progress display
        elapsed = time.monotonic() - self._start_time
        elapsed_str = _format_elapsed(elapsed)
        pct = int(self._completed / self._total * 100) if self._total else 0
        self._update_progress_display(
            pct, self._completed,
            elapsed_str, f"Running {scenario_id} — {title}",
        )

    def _on_scenario_complete(
        self,
        scenario_id: str,
        status: str,
        points: str,
        duration: str,
        pct: int,
        elapsed_str: str,
    ) -> None:
        """Update a completed scenario row and progress."""
        self._set_row_status(scenario_id, status, points, duration)

        # Update progress display
        self._update_progress_display(pct, self._completed, elapsed_str)

    def _set_row_status(
        self, scenario_id: str, status: str, points: str, duration: str
    ) -> None:
        """Update a scenario row's status, points, and time cells."""
        table = self.query_one("#scenario-table", DataTable)
        display = _STATUS_DISPLAY.get(status, status)
        try:
            col_keys = [c.key for c in table.columns.values()]
            table.update_cell(scenario_id, col_keys[3], display)
            if points:
                table.update_cell(scenario_id, col_keys[4], points)
            if duration:
                table.update_cell(scenario_id, col_keys[5], duration)
        except Exception:
            pass

    def _update_progress_display(
        self, pct: int, completed: int, elapsed: str, label: str = ""
    ) -> None:
        """Update the text-based progress display."""
        display = self.query_one("#progress-display", Static)
        display.update(_render_progress(pct, completed, self._total, elapsed, label))

    def _update_status(self, text: str) -> None:
        self.query_one("#run-status", Static).update(text)

    def _update_title(self, text: str) -> None:
        self.query_one("#run-title", Static).update(text)

    def _update_info(self, text: str) -> None:
        self.query_one("#run-info", Static).update(text)

    def action_cancel(self) -> None:
        """Cancel the current run."""
        self._cancelled = True
        self.app.pop_screen()


def _format_elapsed(seconds: float) -> str:
    """Format elapsed seconds as HH:MM:SS."""
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
