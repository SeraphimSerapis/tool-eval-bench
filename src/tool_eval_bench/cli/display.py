"""Zero-flicker streaming display for benchmark runs.

Pattern: "streaming log + tiny live footer"
- Header is printed once (static).
- Each scenario result is printed as a permanent log line (append-only).
- Only a small progress footer (1-2 lines) is rewritten via Rich Live.
- Summary tables are printed as static output after completion.

This eliminates flicker because only 1-2 lines are ever rewritten.
"""

from __future__ import annotations

import time
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from tool_eval_bench.domain.scenarios import (
    Category,
    ModelScoreSummary,
    ScenarioDefinition,
    ScenarioResult,
    ScenarioStatus,
)
from tool_eval_bench.evals.scenarios import (
    ALL_DISPLAY_DETAILS,
    ALL_SCENARIOS,
)


# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

STATUS_LABELS = {
    ScenarioStatus.PASS: ("✅ PASS", "green"),
    ScenarioStatus.PARTIAL: ("⚠️  PARTIAL", "yellow"),
    ScenarioStatus.FAIL: ("❌ FAIL", "red"),
}

CATEGORY_COLORS = {
    Category.A: "cyan",
    Category.B: "magenta",
    Category.C: "blue",
    Category.D: "yellow",
    Category.E: "red",
    Category.F: "green",
    Category.G: "bright_white",
    Category.H: "bright_cyan",
    Category.I: "bright_magenta",
    Category.J: "bright_yellow",
    Category.K: "bright_red",
    Category.L: "bright_green",
    Category.M: "bright_blue",
    Category.N: "deep_sky_blue1",
    Category.O: "orchid1",
    Category.P: "bold bright_red",
}

RATING_COLORS = {
    "★★★★★ Excellent": "bold green",
    "★★★★ Good": "bold cyan",
    "★★★ Adequate": "bold yellow",
    "★★★ Adequate (safety-capped)": "bold red",
    "★★ Weak": "bold red",
    "★ Poor": "bold red",
}


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

class BenchmarkDisplay:
    """Zero-flicker benchmark display.

    Results appear as permanent log lines printed above a tiny Live footer.
    Only the footer (progress bar + current scenario) is ever rewritten.
    """

    def __init__(
        self,
        model: str,
        backend: str,
        base_url: str,
        scenarios: list[ScenarioDefinition] | None = None,
        run_context: Any | None = None,
    ) -> None:
        from tool_eval_bench.evals.scenarios import SCENARIOS

        self.model = model
        self.backend = backend
        self.base_url = base_url
        self.scenarios = scenarios or SCENARIOS
        self.run_context = run_context
        self.console = Console()

        # State
        self.results: dict[str, ScenarioResult] = {}
        self.active_scenario: str | None = None
        self.started_at = time.time()

        # Live footer (lazy init)
        self._live: Live | None = None

    def start(self) -> None:
        """Print static header, then start the tiny live progress footer."""
        total = len(self.scenarios)

        # Version stamp
        try:
            from tool_eval_bench import __version__
            version_tag = f"  [dim]v{__version__}[/]"
        except ImportError:
            version_tag = ""

        self.console.print()
        self.console.print(
            Panel(
                f"[bold]{self.model}[/]  [dim]via {self.backend} @ {self.base_url}[/]\n"
                f"[dim]{total} scenarios[/]{version_tag}",
                title="[bold]🔧 Tool-Call Benchmark[/]",
                border_style="bright_blue",
            )
        )
        self.console.print()

        # Start Live with just the progress footer — transient so it
        # disappears when we stop(), replaced by the static summary.
        self._live = Live(
            self._build_footer(),
            console=self.console,
            auto_refresh=False,
            transient=True,
        )
        self._live.start()

    def stop(self) -> None:
        if self._live:
            self._live.stop()
            self._live = None

    # -- Event handlers (called by orchestrator) --

    async def on_scenario_start(
        self, scenario: ScenarioDefinition, idx: int, total: int
    ) -> None:
        self.active_scenario = scenario.id
        self._refresh_footer()

    async def on_scenario_result(
        self,
        scenario: ScenarioDefinition,
        result: ScenarioResult,
        idx: int,
        total: int,
    ) -> None:
        self.results[scenario.id] = result
        self.active_scenario = None

        # Print result as a permanent line (appears above the Live footer)
        self.console.print(self._format_result_line(scenario, result))

        # Update the footer
        self._refresh_footer()

    def set_finished(
        self,
        summary: ModelScoreSummary,
        *,
        throughput_samples: list | None = None,
    ) -> None:
        """Stop the live footer and print the static summary."""
        self.stop()
        self.console.print()
        _print_category_scores(self.console, summary)
        self.console.print()
        _print_final_panel(
            self.console, self.model, summary, time.time() - self.started_at,
            throughput_samples=throughput_samples,
            run_context=self.run_context,
        )

    # -- Formatting helpers --

    def _format_result_line(self, scenario: ScenarioDefinition, result: ScenarioResult) -> str:
        """Format a single scenario result as a compact one-line log entry."""
        cat_color = CATEGORY_COLORS.get(scenario.category, "white")
        label, status_color = STATUS_LABELS.get(result.status, ("?", "white"))
        dur = f"{result.duration_seconds:.1f}s"

        # Latency info
        latency_parts: list[str] = []
        if result.ttft_ms is not None:
            latency_parts.append(f"ttft={result.ttft_ms:,.0f}ms")
        if result.turn_count > 1:
            latency_parts.append(f"t{result.turn_count}")
        latency_str = f"  [dim]{' '.join(latency_parts)}[/]" if latency_parts else ""

        # Base line: dot  ID  title  status  points  time  latency
        line = (
            f"  [{cat_color}]●[/] {scenario.id}  {scenario.title:<30s}"
            f"  [{status_color}]{label}[/]"
            f"  [bold]{result.points}[/]/2"
            f"  [dim]{dur:>5s}[/]"
            f"{latency_str}"
        )

        # Append summary for non-pass results
        if result.status != ScenarioStatus.PASS:
            line += f"  [{status_color}]{result.summary}[/]"
        else:
            line += f"  [dim]{result.summary}[/]"

        return line

    def _build_footer(self) -> Text:
        """Build the tiny live progress footer (1-2 lines)."""
        done = len(self.results)
        total = len(self.scenarios)
        elapsed = time.time() - self.started_at

        # Progress bar
        bar_width = 30
        filled = int((done / max(total, 1)) * bar_width)
        bar = f"[green]{'█' * filled}[/][dim]{'░' * (bar_width - filled)}[/]"

        if self.active_scenario:
            sc = next((s for s in self.scenarios if s.id == self.active_scenario), None)
            name = f"{sc.id} {sc.title}" if sc else self.active_scenario
            status = f"[bold cyan]⟳[/] [bold]{name}[/]"
        elif done >= total:
            status = "[bold green]✓ Complete[/]"
        else:
            status = "[dim]Waiting…[/]"

        text = Text.from_markup(
            f"\n  {status}  {bar}  [bold]{done}[/]/{total}  [dim]({elapsed:.0f}s)[/]"
        )
        return text

    def _refresh_footer(self) -> None:
        if self._live:
            self._live.update(self._build_footer(), refresh=True)


# ---------------------------------------------------------------------------
# Static summary output (printed after Live stops)
# ---------------------------------------------------------------------------

def _print_category_scores(console: Console, summary: ModelScoreSummary) -> None:
    """Print category score bars as a static table."""
    table = Table(
        title="[bold]Category Breakdown[/]",
        show_header=True,
        header_style="bold",
        border_style="bright_blue",
        expand=True,
    )
    table.add_column("Category", min_width=22)
    table.add_column("Score", justify="center", width=8)
    table.add_column("Bar", min_width=22)
    table.add_column("Earned", justify="center", width=8)

    for cs in summary.category_scores:
        cat_color = CATEGORY_COLORS.get(cs.category, "white")
        bar_width = 20
        filled = int((cs.percent / 100) * bar_width)
        bar = f"[{cat_color}]{'█' * filled}[/][dim]{'░' * (bar_width - filled)}[/]"
        table.add_row(
            f"[{cat_color}]{cs.label}[/]",
            f"[bold]{cs.percent}%[/]",
            bar,
            f"{cs.earned}/{cs.max_points}",
        )

    console.print(table)


def _print_final_panel(
    console: Console,
    model: str,
    summary: ModelScoreSummary,
    elapsed: float,
    *,
    throughput_samples: list | None = None,
    run_context: Any | None = None,
) -> None:
    """Print the final score panel."""
    score = summary.final_score
    rating = summary.rating
    rating_color = RATING_COLORS.get(rating, "bold white")

    passes = sum(1 for r in summary.scenario_results if r.status == ScenarioStatus.PASS)
    partials = sum(1 for r in summary.scenario_results if r.status == ScenarioStatus.PARTIAL)
    fails = sum(1 for r in summary.scenario_results if r.status == ScenarioStatus.FAIL)

    # Build model info lines from run_context (always included when available)
    model_info_lines = ""
    if run_context is not None:
        rc = run_context
        if rc.engine_name:
            engine_str = rc.engine_name
            if rc.engine_version:
                engine_str += f" {rc.engine_version}"
            model_info_lines += f"\n  [dim]Engine:       {engine_str}[/]"
        if rc.quantization:
            model_info_lines += f"\n  [dim]Quantization: {rc.quantization}[/]"
        if rc.max_model_len:
            model_info_lines += f"\n  [dim]Max context:  {rc.max_model_len:,} tokens[/]"
        if rc.server_model_root and rc.server_model_root != model:
            model_info_lines += f"\n  [dim]Model root:   {rc.server_model_root}[/]"

    content = (
        f"  [bold]Model:[/]  {model}\n"
        f"  [bold]Score:[/]  [{rating_color}]{score} / 100[/]\n"
        f"  [bold]Rating:[/] [{rating_color}]{rating}[/]"
        f"{model_info_lines}\n"
        f"\n"
        f"  [green]✅ {passes} passed[/]   [yellow]⚠️  {partials} partial[/]   [red]❌ {fails} failed[/]\n"
        f"  [bold]Points:[/] {summary.total_points}/{summary.max_points}"
    )

    # Deployability composite (only when latency data is present)
    if summary.deployability is not None and summary.responsiveness is not None:
        med_s = (summary.median_turn_ms or 0) / 1000
        resp_color = "green" if summary.responsiveness >= 75 else (
            "yellow" if summary.responsiveness >= 40 else "red"
        )
        deploy_color = "green" if summary.deployability >= 75 else (
            "yellow" if summary.deployability >= 40 else "red"
        )
        content += (
            f"\n\n  [bold]Quality:[/]        {score}/100"
            f"\n  [bold]Responsiveness:[/] [{resp_color}]{summary.responsiveness}/100[/]"
            f"  [dim](median turn: {med_s:.1f}s)[/]"
            f"\n  [bold]Deployability:[/]  [{deploy_color}]{summary.deployability}/100[/]"
            f"  [dim](α={summary.alpha})[/]"
        )

    # Weakest category (lowest-scoring area) — skip if all at 100%
    if (
        summary.worst_category is not None
        and summary.worst_category_percent is not None
        and summary.worst_category_percent < 100
    ):
        floor_color = "green" if summary.worst_category_percent >= 75 else (
            "yellow" if summary.worst_category_percent >= 40 else "red"
        )
        content += f"\n  [bold]Weakest:[/] [{floor_color}]{summary.worst_category}[/]"

    # Version stamp
    try:
        from tool_eval_bench import __version__
        version_tag = f"  [dim]│  tool-eval-bench v{__version__}[/]"
    except ImportError:
        version_tag = ""

    content += (
        f"\n\n  [dim]Completed in {elapsed:.1f}s[/]{version_tag}"
    )

    # Token usage summary (when server reports usage data)
    if summary.total_tokens > 0:
        content += (
            f"\n\n  [bold]📊 Token Usage:[/]"
            f"\n  [dim]Total:[/] {summary.total_tokens:,} tokens"
        )
        if summary.token_efficiency is not None:
            content += f"  [dim]│  Efficiency: {summary.token_efficiency:.1f} pts/1K tokens[/]"

    # Surface safety-critical failures prominently
    if summary.safety_warnings:
        warning_lines = "\n".join(f"    [bold yellow]⚠ {w}[/]" for w in summary.safety_warnings)
        content += (
            f"\n\n  [bold red]🛡️  SAFETY WARNINGS ({len(summary.safety_warnings)}):[/]\n"
            f"{warning_lines}"
        )

    # Throughput highlights (when --perf is used)
    ok_samples = [s for s in (throughput_samples or []) if not getattr(s, "error", None)]
    if ok_samples:
        # Best single-stream
        single = [s for s in ok_samples if s.concurrency <= 1]
        concurrent = [s for s in ok_samples if s.concurrency > 1]

        content += "\n\n  [bold]⚡ Throughput:[/]"
        if single:
            best_pp = max(single, key=lambda s: s.pp_tps)
            best_tg = max(single, key=lambda s: s.tg_tps)
            content += (
                f"\n  [dim]Single:[/]  {best_pp.pp_tps:,.0f} pp t/s  │  "
                f"{best_tg.tg_tps:,.1f} tg t/s  │  "
                f"TTFT {best_tg.ttft_ms:,.0f}ms"
            )
        if concurrent:
            # Show one line per concurrency level (sorted ascending)
            conc_levels = sorted({s.concurrency for s in concurrent})
            for clevel in conc_levels:
                level_samples = [s for s in concurrent if s.concurrency == clevel]
                best_pp_c = max(level_samples, key=lambda s: s.pp_tps)
                best_tg_c = max(level_samples, key=lambda s: s.tg_tps)
                content += (
                    f"\n  [dim]c{clevel}:[/]      "
                    f"{best_pp_c.pp_tps:,.0f} pp t/s  │  "
                    f"{best_tg_c.tg_tps:,.1f} tg t/s"
                )

    # Scoring methodology (how the numbers are calculated)
    content += (
        "\n\n  [dim]── How this score is calculated ──[/]"
        "\n  [dim]• Each scenario: pass=2pt, partial=1pt, fail=0pt[/]"
        "\n  [dim]• Category %: earned / max per category[/]"
        "\n  [dim]• Final score: (total points / max points) × 100[/]"
    )
    if summary.deployability is not None:
        content += (
            f"\n  [dim]• Deployability: {summary.alpha}×quality + "
            f"{1 - summary.alpha:.1f}×responsiveness[/]"
            f"\n  [dim]• Responsiveness: logistic curve (100 at <1s, ~50 at 3s, 0 at >10s)[/]"
        )

    border = "green" if score >= 75 else ("yellow" if score >= 40 else "red")
    if summary.safety_warnings:
        border = "red"  # always red border when safety issues exist
    console.print(Panel(content, title="[bold]🏆 Benchmark Complete[/]", border_style=border, padding=(1, 2)))


# ---------------------------------------------------------------------------
# Detailed report (used by --no-live path and for expanded diagnostics)
# ---------------------------------------------------------------------------

def print_final_report(
    console: Console,
    model: str,
    summary: ModelScoreSummary,
    elapsed: float,
    *,
    throughput_samples: list | None = None,
    run_context: Any | None = None,
) -> None:
    """Print a detailed static report with per-scenario diagnostics."""
    _print_category_scores(console, summary)
    console.print()

    # Scenario detail table with expected-vs-actual for failures
    detail_table = Table(
        title="[bold]Scenario Details[/]",
        show_header=True,
        header_style="bold",
        border_style="dim",
        expand=True,
    )
    detail_table.add_column("ID", width=6, style="bold")
    detail_table.add_column("Title", min_width=22, no_wrap=True)
    detail_table.add_column("Result", width=12, justify="center", no_wrap=True)
    detail_table.add_column("Pts", width=5, justify="center")
    detail_table.add_column("Time", width=6, justify="right")
    detail_table.add_column("Summary", ratio=1)

    for r in summary.scenario_results:
        label, status_color = STATUS_LABELS.get(r.status, ("?", "white"))
        status_text = f"[bold {status_color}]{label}[/]"
        title = next((s.title for s in ALL_SCENARIOS if s.id == r.scenario_id), "?")
        display_detail = ALL_DISPLAY_DETAILS.get(r.scenario_id)

        if r.status == ScenarioStatus.PASS:
            summary_text = f"[dim]{r.summary}[/]"
        else:
            parts = [f"[{status_color}]{r.summary}[/]"]
            if r.tool_calls_made:
                parts.append(f"[dim]Called: {', '.join(r.tool_calls_made)}[/]")
            else:
                parts.append("[dim]Called: (no tools)[/]")
            expected = r.expected_behavior or (display_detail.success_case if display_detail else "")
            if expected:
                lbl = "Expected" if r.status == ScenarioStatus.FAIL else "For full pass"
                parts.append(f"[dim]{lbl}: {expected}[/]")
            summary_text = "\n".join(parts)

        detail_table.add_row(
            r.scenario_id, title, status_text,
            f"{r.points}/2", f"[dim]{r.duration_seconds:.1f}s[/]",
            summary_text,
        )

    console.print(detail_table)
    console.print()

    _print_final_panel(console, model, summary, elapsed, throughput_samples=throughput_samples, run_context=run_context)
