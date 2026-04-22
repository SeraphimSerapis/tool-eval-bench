"""Main Textual app for tool-eval-bench interactive mode."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header

from tool_eval_bench.tui.screens.configure import ConfigureScreen
from tool_eval_bench.tui.screens.results import ResultsScreen
from tool_eval_bench.tui.screens.running import RunningScreen


class BenchmarkApp(App[None]):
    """Interactive benchmark runner with configuration, live progress, and results."""

    TITLE = "tool-eval-bench"
    SUB_TITLE = "Agentic Tool-Call Benchmark"

    CSS = """
    Screen {
        background: $surface;
    }

    /* Thin scrollbar globally to avoid visual artifacts */
    VerticalScroll, DataTable {
        scrollbar-size-vertical: 1;
    }

    /* Suppress hover background changes on checkboxes and labels */
    _Checkbox:hover {
        background: transparent;
    }

    /* Suppress generic hover tinting on containers */
    Horizontal:hover, Vertical:hover {
        background: transparent;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Footer()

    def on_mount(self) -> None:
        """Start on the configure screen."""
        self.push_screen(ConfigureScreen())

    def start_benchmark(self, config: dict) -> None:
        """Transition from configure → running screen."""
        self.pop_screen()
        self.push_screen(RunningScreen(config))

    def show_results(self, config: dict, results: dict) -> None:
        """Transition from running → results screen."""
        self.pop_screen()
        self.push_screen(ResultsScreen(config, results))

    def back_to_configure(self) -> None:
        """Go back to configure from results."""
        self.pop_screen()
        self.push_screen(ConfigureScreen())


def run_tui() -> None:
    """Entry point for the interactive TUI."""
    app = BenchmarkApp()
    app.run()
