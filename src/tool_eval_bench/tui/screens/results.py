"""Results screen — scores, categories, history, leaderboard."""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button,
    ContentSwitcher,
    DataTable,
    Static,
)


class ResultsScreen(Screen):
    """View benchmark results with tabs for scores, history, and leaderboard."""

    BINDINGS = [
        Binding("escape", "go_back", "Back", show=True),
        Binding("1", "show_tab('scores')", "Scores", show=True),
        Binding("2", "show_tab('categories')", "Categories", show=True),
        Binding("3", "show_tab('history')", "History", show=True),
        Binding("4", "show_tab('leaderboard')", "Leaderboard", show=True),
    ]

    CSS = """
    ResultsScreen {
        layout: vertical;
    }

    #results-title {
        text-style: bold;
        color: $accent;
        text-align: center;
        padding: 1 0 0 0;
    }

    #tab-buttons {
        layout: horizontal;
        height: 3;
        align: center middle;
        padding: 0 0 1 0;
    }

    #tab-buttons Button {
        margin: 0 1;
        min-width: 14;
    }

    #tab-buttons Button.btn-inactive {
        border: tall $text 25%;
        color: $text 80%;
    }

    #results-switcher {
        height: 1fr;
        margin: 0 2;
    }

    .result-panel {
        padding: 1 2;
    }

    #score-hero {
        text-align: center;
        padding: 1 0;
    }

    #score-number {
        text-style: bold;
        text-align: center;
        padding: 1 0 0 0;
    }

    #score-rating {
        text-align: center;
        color: $text-muted;
        padding: 0 0 1 0;
    }

    #result-details {
        height: auto;
        padding: 1 2;
    }

    #action-bar {
        layout: horizontal;
        height: auto;
        align: center middle;
        padding: 1 0;
    }

    #action-bar Button {
        margin: 0 1;
    }

    /* Disable hover on table rows */
    DataTable {
        scrollbar-size-vertical: 1;
    }
    """

    def __init__(
        self,
        config: dict,
        results: dict,
        *,
        initial_tab: str = "scores",
    ) -> None:
        super().__init__()
        self.config = config
        self.results = results
        self.initial_tab = initial_tab

    def compose(self) -> ComposeResult:
        scores = self.results.get("scores", {})
        final_score = scores.get("final_score", "—")
        rating = scores.get("rating", "")

        title = "📊 Benchmark Results"
        if self.config.get("model"):
            title += f" — {self.config['model']}"
        yield Static(title, id="results-title")

        with Horizontal(id="tab-buttons"):
            yield Button("📊 Scores", id="btn-tab-scores", variant="primary")
            yield Button("📋 Categories", id="btn-tab-categories", variant="default")
            yield Button("📜 History", id="btn-tab-history", variant="default")
            yield Button("🏆 Leaderboard", id="btn-tab-leaderboard", variant="default")

        with ContentSwitcher(initial=self.initial_tab, id="results-switcher"):
            # Scores tab
            with VerticalScroll(id="scores"):
                if scores:
                    yield Static(
                        f"[bold]{final_score}[/] / 100",
                        id="score-number",
                    )
                    yield Static(rating, id="score-rating")

                yield DataTable(id="score-table")

            # Categories tab
            with VerticalScroll(id="categories"):
                yield DataTable(id="category-table")

            # History tab
            with VerticalScroll(id="history"):
                yield DataTable(id="history-table")

            # Leaderboard tab
            with VerticalScroll(id="leaderboard"):
                yield DataTable(id="leaderboard-table")

        with Horizontal(id="action-bar"):
            yield Button("🔄 New Run", id="btn-new-run", variant="success")
            yield Button("Quit", id="btn-quit", variant="error")

    def on_mount(self) -> None:
        """Populate tables with data."""
        self._populate_scores()
        self._populate_categories()
        self._populate_history()
        self._populate_leaderboard()

        # Activate the initial tab
        self.action_show_tab(self.initial_tab)

    def _populate_scores(self) -> None:
        """Fill the per-scenario results table."""
        table = self.query_one("#score-table", DataTable)
        table.add_columns("ID", "Title", "Status", "Points", "Summary")
        table.cursor_type = "row"
        table.zebra_stripes = True

        scenario_results = self.results.get("scores", {}).get("scenario_results", [])
        for sr in scenario_results:
            status = sr.get("status", "")
            if status == "pass":
                status_display = "[green]✅ PASS[/]"
            elif status == "partial":
                status_display = "[yellow]⚠️  PARTIAL[/]"
            else:
                status_display = "[red]❌ FAIL[/]"

            table.add_row(
                sr.get("scenario_id", ""),
                sr.get("title", ""),
                status_display,
                str(sr.get("points", 0)),
                sr.get("summary", ""),
            )

    def _populate_categories(self) -> None:
        """Fill the per-category score table."""
        table = self.query_one("#category-table", DataTable)
        table.add_columns("Category", "Score", "Pass", "Partial", "Fail")
        table.cursor_type = "row"
        table.zebra_stripes = True

        category_scores = self.results.get("scores", {}).get("category_scores", [])
        for cs in category_scores:
            pct = cs.get("percent", 0)
            if pct >= 90:
                pct_display = f"[green]{pct:.0f}%[/]"
            elif pct >= 60:
                pct_display = f"[yellow]{pct:.0f}%[/]"
            else:
                pct_display = f"[red]{pct:.0f}%[/]"

            table.add_row(
                f"{cs.get('category', '')} {cs.get('label', '')}",
                pct_display,
                str(cs.get("pass_count", 0)),
                str(cs.get("partial_count", 0)),
                str(cs.get("fail_count", 0)),
            )

    def _populate_history(self) -> None:
        """Load and display run history from storage."""
        table = self.query_one("#history-table", DataTable)
        table.add_columns("Run ID", "Model", "Score", "Scenarios", "Date")
        table.cursor_type = "row"
        table.zebra_stripes = True

        try:
            from tool_eval_bench.storage.db import RunRepository

            repo = RunRepository()
            runs = repo.list_runs(limit=25)
            for run in runs:
                table.add_row(
                    run.get("run_id", "")[:20],
                    run.get("model", ""),
                    str(run.get("final_score", "")),
                    str(run.get("scenario_count", "")),
                    run.get("timestamp", ""),
                )
        except Exception:
            table.add_row("—", "No run history available", "", "", "")

    def _populate_leaderboard(self) -> None:
        """Load and display the model leaderboard."""
        table = self.query_one("#leaderboard-table", DataTable)
        table.add_columns("Rank", "Model", "Score", "Rating", "Runs")
        table.cursor_type = "row"
        table.zebra_stripes = True

        try:
            from tool_eval_bench.storage.db import RunRepository

            repo = RunRepository()
            runs = repo.list_runs(limit=100)

            # Group by model, take best score
            model_best: dict[str, dict] = {}
            model_count: dict[str, int] = {}
            for run in runs:
                model = run.get("model", "unknown")
                score = run.get("final_score", 0)
                model_count[model] = model_count.get(model, 0) + 1
                if model not in model_best or score > model_best[model].get("final_score", 0):
                    model_best[model] = run

            ranked = sorted(model_best.items(), key=lambda x: x[1].get("final_score", 0), reverse=True)
            medals = ["🥇", "🥈", "🥉"]
            for i, (model, run) in enumerate(ranked):
                medal = medals[i] if i < 3 else f" {i + 1}"
                score = run.get("final_score", 0)
                rating = run.get("rating", "")
                count = model_count.get(model, 0)
                table.add_row(medal, model, str(score), rating, str(count))

        except Exception:
            table.add_row("—", "No benchmark data available", "", "", "")

    @on(Button.Pressed, "#btn-tab-scores")
    def on_tab_scores(self) -> None:
        self.action_show_tab("scores")

    @on(Button.Pressed, "#btn-tab-categories")
    def on_tab_categories(self) -> None:
        self.action_show_tab("categories")

    @on(Button.Pressed, "#btn-tab-history")
    def on_tab_history(self) -> None:
        self.action_show_tab("history")

    @on(Button.Pressed, "#btn-tab-leaderboard")
    def on_tab_leaderboard(self) -> None:
        self.action_show_tab("leaderboard")

    def action_show_tab(self, tab: str) -> None:
        """Switch the content to the named tab."""
        switcher = self.query_one("#results-switcher", ContentSwitcher)
        switcher.current = tab

        # Update button variants to show active tab
        tab_map = {
            "scores": "#btn-tab-scores",
            "categories": "#btn-tab-categories",
            "history": "#btn-tab-history",
            "leaderboard": "#btn-tab-leaderboard",
        }
        for tab_name, btn_id in tab_map.items():
            btn = self.query_one(btn_id, Button)
            if tab_name == tab:
                btn.variant = "primary"
                btn.remove_class("btn-inactive")
            else:
                btn.variant = "default"
                btn.add_class("btn-inactive")

    @on(Button.Pressed, "#btn-new-run")
    def on_new_run(self) -> None:
        self.app.back_to_configure()  # type: ignore[attr-defined]

    @on(Button.Pressed, "#btn-quit")
    def on_quit(self) -> None:
        self.app.exit()

    def action_go_back(self) -> None:
        self.app.back_to_configure()  # type: ignore[attr-defined]
