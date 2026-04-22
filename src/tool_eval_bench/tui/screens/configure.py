"""Configure screen — interactive benchmark setup."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button,
    Checkbox,
    Input,
    Label,
    Rule,
    Select,
    Static,
)

if TYPE_CHECKING:
    pass


class _Checkbox(Checkbox):
    """Checkbox with a readable bracket-style indicator."""

    BUTTON_LEFT = "["
    BUTTON_RIGHT = "]"
    BUTTON_INNER = "✓"

# Category labels for the category picker
_CATEGORY_OPTIONS: list[tuple[str, str]] = [
    ("A — Tool Selection", "A"),
    ("B — Parameter Precision", "B"),
    ("C — Multi-Step Chains", "C"),
    ("D — Restraint & Refusal", "D"),
    ("E — Error Recovery", "E"),
    ("F — Localization", "F"),
    ("G — Structured Reasoning", "G"),
    ("H — Instruction Following", "H"),
    ("I — Context & State", "I"),
    ("J — Code Patterns", "J"),
    ("K — Safety & Boundaries", "K"),
    ("L — Toolset Scale", "L"),
    ("M — Autonomous Planning", "M"),
    ("N — Creative Composition", "N"),
    ("O — Structured Output", "O"),
]


class ConfigureScreen(Screen):
    """Interactive configuration form for benchmark runs."""

    BINDINGS = [
        Binding("escape", "app.pop_screen", "Back", show=False),
    ]

    CSS = """
    ConfigureScreen {
        layout: vertical;
    }

    #config-title {
        text-style: bold;
        color: $accent;
        text-align: center;
        padding: 1 0 0 0;
        width: 100%;
    }

    #config-scroll {
        margin: 0 2;
        scrollbar-size-vertical: 1;
    }

    .section-header {
        text-style: bold;
        color: $text;
        padding: 1 0 0 0;
        margin: 0 0 0 0;
    }

    .section-desc {
        color: $text-muted;
        padding: 0 0 0 2;
    }

    .field-row {
        layout: horizontal;
        height: auto;
        padding: 0 0 0 2;
        margin: 0 0;
    }

    .field-row Label {
        width: 18;
        min-width: 12;
        padding: 1 1 0 0;
        text-align: right;
        color: $text;
    }

    .field-row Input {
        width: 1fr;
    }

    .field-row Select {
        width: 1fr;
    }

    .mode-checks {
        layout: horizontal;
        height: auto;
        padding: 0 0 0 2;
    }

    .mode-checks _Checkbox {
        width: 1fr;
        min-width: 20;
        margin: 0 0;
    }

    .category-grid {
        layout: grid;
        grid-size: 3;
        grid-gutter: 0 1;
        height: auto;
        padding: 0 0 0 2;
    }

    .category-grid _Checkbox {
        width: 1fr;
        height: auto;
    }

    /* Responsive: switch to 2-column grid on narrow terminals */
    ConfigureScreen.-narrow .category-grid {
        grid-size: 2;
    }

    ConfigureScreen.-narrow .mode-checks {
        layout: vertical;
        height: auto;
    }

    ConfigureScreen.-narrow .mode-checks _Checkbox {
        width: auto;
    }

    .sampling-row {
        layout: horizontal;
        height: auto;
        padding: 0 0 0 2;
        margin: 0 0;
    }

    .sampling-row Label {
        width: 18;
        min-width: 12;
        padding: 1 1 0 0;
        text-align: right;
        color: $text;
    }

    .sampling-row Input {
        width: 1fr;
    }

    .sampling-grid {
        layout: grid;
        grid-size: 2;
        grid-gutter: 0 2;
        height: auto;
        padding: 0 0 0 2;
    }

    .sampling-grid .field-pair {
        layout: horizontal;
        height: auto;
    }

    .sampling-grid .field-pair Label {
        width: 12;
        min-width: 8;
        padding: 1 1 0 0;
        text-align: right;
        color: $text;
    }

    .sampling-grid .field-pair Input {
        width: 1fr;
    }

    #button-bar {
        layout: horizontal;
        height: auto;
        align: center middle;
        padding: 1 0;
    }

    #button-bar Button {
        margin: 0 1;
    }

    #status-bar {
        height: 1;
        padding: 0 2;
        color: $text-muted;
    }

    #btn-run {
        min-width: 18;
    }

    #btn-detect {
        min-width: 18;
    }

    /* Checkbox indicator styling */
    _Checkbox .toggle--button {
        color: $text 60%;
    }

    _Checkbox.-on .toggle--button {
        color: $success;
        text-style: bold;
    }

    /* Disable hover color changes on all interactive widgets */
    _Checkbox:hover {
        background: transparent;
    }

    Button.-active:hover {
        /* keep default */
    }

    Input:hover {
        border: tall $accent 40%;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("⚡ Benchmark Configuration", id="config-title")

        with VerticalScroll(id="config-scroll"):
            # -- Connection section --
            yield Static("🔌 Connection", classes="section-header")
            yield Static(
                "Server URL and model. Leave model blank for auto-detection.",
                classes="section-desc",
            )

            with Horizontal(classes="field-row"):
                yield Label("Base URL")
                yield Input(
                    value=os.getenv("TOOL_EVAL_BASE_URL", ""),
                    placeholder="http://localhost:8080",
                    id="input-base-url",
                )

            with Horizontal(classes="field-row"):
                yield Label("API Key")
                yield Input(
                    value=os.getenv("TOOL_EVAL_API_KEY", ""),
                    placeholder="(optional)",
                    password=True,
                    id="input-api-key",
                )

            with Horizontal(classes="field-row"):
                yield Label("Model")
                yield Input(
                    value="",
                    placeholder="(auto-detect from server)",
                    id="input-model",
                )
            with Horizontal(classes="field-row"):
                yield Label("")
                yield Button("Detect Models", id="btn-detect", variant="default")

            with Horizontal(classes="field-row"):
                yield Label("Backend")
                yield Select(
                    [("vLLM", "vllm"), ("LiteLLM", "litellm"), ("llama.cpp", "llamacpp")],
                    value="vllm",
                    id="select-backend",
                )

            yield Rule()

            # -- Benchmark mode --
            yield Static("🎯 Benchmark Mode", classes="section-header")
            yield Static(
                "Select which benchmarks to run.",
                classes="section-desc",
            )
            with Horizontal(classes="mode-checks"):
                yield _Checkbox("Tool-Call Scenarios", value=True, id="chk-tool-call")
                yield _Checkbox("Throughput (llama-benchy)", id="chk-perf")
                yield _Checkbox("Spec-Decode", id="chk-spec")

            yield Rule()

            # -- Scenario selection --
            yield Static("📋 Scenario Selection", classes="section-header")
            yield Static(
                "Filter scenarios by category (default: all 69 scenarios).",
                classes="section-desc",
            )
            with Horizontal(classes="field-row"):
                yield Label("Preset")
                yield Select(
                    [
                        ("All 69 scenarios", "all"),
                        ("Core 15 (--short)", "short"),
                        ("Pick categories…", "categories"),
                    ],
                    value="all",
                    id="select-scenario-preset",
                )

            with Vertical(classes="category-grid", id="category-grid"):
                for display_name, value in _CATEGORY_OPTIONS:
                    yield _Checkbox(display_name, value=True, id=f"cat-{value}")

            yield Rule()

            # -- Sampling --
            yield Static("🎲 Sampling", classes="section-header")
            yield Static(
                "Model inference parameters.",
                classes="section-desc",
            )
            with Horizontal(classes="field-row"):
                yield Label("Temperature")
                yield Input(value="0.0", id="input-temperature", type="number")

            with Vertical(classes="sampling-grid"):
                with Horizontal(classes="field-pair"):
                    yield Label("Top-P")
                    yield Input(
                        value="",
                        placeholder="(default)",
                        id="input-top-p",
                        type="number",
                    )
                with Horizontal(classes="field-pair"):
                    yield Label("Top-K")
                    yield Input(
                        value="",
                        placeholder="(default)",
                        id="input-top-k",
                        type="integer",
                    )
                with Horizontal(classes="field-pair"):
                    yield Label("Min-P")
                    yield Input(
                        value="",
                        placeholder="(default)",
                        id="input-min-p",
                        type="number",
                    )
                with Horizontal(classes="field-pair"):
                    yield Label("Repeat Pen.")
                    yield Input(
                        value="",
                        placeholder="(default)",
                        id="input-repeat-penalty",
                        type="number",
                    )

            with Horizontal(classes="field-row"):
                yield Label("")
                yield _Checkbox("Disable thinking (--no-think)", id="chk-no-think")

            yield Rule()

            # -- Run control --
            yield Static("⚙️ Run Control", classes="section-header")
            yield Static(
                "Tuning parameters for the benchmark run.",
                classes="section-desc",
            )
            with Horizontal(classes="field-row"):
                yield Label("Trials")
                yield Input(value="1", id="input-trials", type="integer")

            with Horizontal(classes="field-row"):
                yield Label("Parallel")
                yield Input(value="1", id="input-parallel", type="integer")

            with Horizontal(classes="field-row"):
                yield Label("Timeout (sec)")
                yield Input(value="60", id="input-timeout", type="number")

            with Horizontal(classes="field-row"):
                yield Label("Error Rate")
                yield Input(
                    value="0.0", id="input-error-rate", type="number",
                )

            with Horizontal(classes="field-row"):
                yield Label("")
                yield _Checkbox("Skip warm-up", id="chk-no-warmup")

        # -- Action buttons --
        with Horizontal(id="button-bar"):
            yield Button("🚀 Run Benchmark", id="btn-run", variant="success")
            yield Button("History", id="btn-history", variant="default")
            yield Button("Leaderboard", id="btn-leaderboard", variant="default")

        yield Static("", id="status-bar")

    def on_mount(self) -> None:
        """Load .env and set initial state."""
        from dotenv import load_dotenv

        load_dotenv(override=False)

        # Pre-fill from env if available
        base_url = os.getenv("TOOL_EVAL_BASE_URL", "")
        if not base_url:
            host = os.getenv("TOOL_EVAL_HOST", "")
            port = os.getenv("TOOL_EVAL_PORT", "")
            if host:
                base_url = f"http://{host}:{port}" if port else f"http://{host}"
        if base_url:
            self.query_one("#input-base-url", Input).value = base_url

        api_key = os.getenv("TOOL_EVAL_API_KEY", "")
        if api_key:
            self.query_one("#input-api-key", Input).value = api_key

        model = os.getenv("TOOL_EVAL_MODEL", "")
        if model:
            self.query_one("#input-model", Input).value = model

        # Hide category grid initially
        self.query_one("#category-grid").display = False

        # Apply responsive layout
        self._apply_responsive_layout()

    def on_resize(self) -> None:
        """Adapt layout when terminal is resized."""
        self._apply_responsive_layout()

    def _apply_responsive_layout(self) -> None:
        """Toggle narrow-mode class based on terminal width."""
        if self.app.size.width < 90:
            self.add_class("-narrow")
        else:
            self.remove_class("-narrow")

    @on(Select.Changed, "#select-scenario-preset")
    def on_preset_changed(self, event: Select.Changed) -> None:
        """Show/hide category grid based on preset."""
        grid = self.query_one("#category-grid")
        grid.display = event.value == "categories"

    @on(Button.Pressed, "#btn-detect")
    async def on_detect_models(self) -> None:
        """Auto-detect models from the server."""
        status = self.query_one("#status-bar", Static)
        base_url = self.query_one("#input-base-url", Input).value.strip()
        api_key = self.query_one("#input-api-key", Input).value.strip() or None

        if not base_url:
            status.update("[bold red]⚠ Enter a base URL first[/]")
            return

        status.update(f"[dim]Querying {base_url}/v1/models…[/]")

        try:
            import httpx

            url = base_url.rstrip("/")
            models_endpoint = f"{url}/v1/models"
            if url.endswith("/v1"):
                models_endpoint = f"{url}/models"

            headers: dict[str, str] = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(models_endpoint, headers=headers)
                if resp.status_code == 404:
                    resp = await client.get(f"{url}/models", headers=headers)
                resp.raise_for_status()

            data = resp.json()
            model_list = data.get("data", [])
            models = []
            for m in model_list:
                api_id = m.get("id", "")
                if api_id:
                    root = m.get("root", "")
                    display = root if root and root != api_id else api_id
                    models.append((api_id, display))

            if not models:
                status.update("[bold red]⚠ No models found on server[/]")
                return

            if len(models) == 1:
                api_id, display = models[0]
                self.query_one("#input-model", Input).value = api_id
                label = f"{display} (alias: {api_id})" if display != api_id else api_id
                status.update(f"[bold green]✓[/] Detected: [bold]{label}[/]")
            else:
                # Use first model as default, show all in status
                api_id, display = models[0]
                self.query_one("#input-model", Input).value = api_id
                names = ", ".join(m[1] if m[1] != m[0] else m[0] for m in models)
                status.update(
                    f"[bold green]✓[/] Found {len(models)} models: {names}  "
                    f"[dim](using first: {api_id})[/]"
                )

        except Exception as exc:
            status.update(f"[bold red]✗ {exc}[/]")

    @on(Button.Pressed, "#btn-run")
    def on_run(self) -> None:
        """Collect config and start the benchmark."""
        config = self._collect_config()
        if config is None:
            return
        self.app.start_benchmark(config)  # type: ignore[attr-defined]

    @on(Button.Pressed, "#btn-history")
    def on_history(self) -> None:
        """Show history in results screen."""
        from tool_eval_bench.tui.screens.results import ResultsScreen

        self.app.push_screen(ResultsScreen({}, {}, initial_tab="history"))

    @on(Button.Pressed, "#btn-leaderboard")
    def on_leaderboard(self) -> None:
        """Show leaderboard in results screen."""
        from tool_eval_bench.tui.screens.results import ResultsScreen

        self.app.push_screen(ResultsScreen({}, {}, initial_tab="leaderboard"))

    def _collect_config(self) -> dict | None:
        """Gather all form values into a config dict."""
        status = self.query_one("#status-bar", Static)

        base_url = self.query_one("#input-base-url", Input).value.strip()
        if not base_url:
            status.update("[bold red]⚠ Base URL is required[/]")
            self.query_one("#input-base-url", Input).focus()
            return None

        model = self.query_one("#input-model", Input).value.strip() or None
        api_key = self.query_one("#input-api-key", Input).value.strip() or None
        backend = self.query_one("#select-backend", Select).value

        # Modes
        run_tool_call = self.query_one("#chk-tool-call", _Checkbox).value
        run_perf = self.query_one("#chk-perf", _Checkbox).value
        run_spec = self.query_one("#chk-spec", _Checkbox).value

        if not run_tool_call and not run_perf and not run_spec:
            status.update("[bold red]⚠ Select at least one benchmark mode[/]")
            return None

        # Scenario preset
        preset = self.query_one("#select-scenario-preset", Select).value
        categories = None
        short = False
        if preset == "short":
            short = True
        elif preset == "categories":
            categories = []
            for _, cat_value in _CATEGORY_OPTIONS:
                chk = self.query_one(f"#cat-{cat_value}", _Checkbox)
                if chk.value:
                    categories.append(cat_value)
            if not categories:
                status.update("[bold red]⚠ Select at least one category[/]")
                return None

        # Sampling
        try:
            temperature = float(self.query_one("#input-temperature", Input).value or "0.0")
        except ValueError:
            temperature = 0.0

        # Optional sampling params (empty = not set)
        top_p = _parse_optional_float(self.query_one("#input-top-p", Input).value)
        top_k = _parse_optional_int(self.query_one("#input-top-k", Input).value)
        min_p = _parse_optional_float(self.query_one("#input-min-p", Input).value)
        repeat_penalty = _parse_optional_float(
            self.query_one("#input-repeat-penalty", Input).value
        )

        no_think = self.query_one("#chk-no-think", _Checkbox).value

        # Run control
        try:
            trials = max(1, int(self.query_one("#input-trials", Input).value or "1"))
        except ValueError:
            trials = 1
        try:
            parallel = max(1, int(self.query_one("#input-parallel", Input).value or "1"))
        except ValueError:
            parallel = 1
        try:
            timeout = float(self.query_one("#input-timeout", Input).value or "60")
        except ValueError:
            timeout = 60.0
        try:
            error_rate = float(self.query_one("#input-error-rate", Input).value or "0.0")
        except ValueError:
            error_rate = 0.0

        no_warmup = self.query_one("#chk-no-warmup", _Checkbox).value

        return {
            "base_url": base_url,
            "model": model,
            "api_key": api_key,
            "backend": backend,
            "run_tool_call": run_tool_call,
            "run_perf": run_perf,
            "run_spec": run_spec,
            "short": short,
            "categories": categories,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "repeat_penalty": repeat_penalty,
            "no_think": no_think,
            "trials": trials,
            "parallel": parallel,
            "timeout": timeout,
            "error_rate": error_rate,
            "no_warmup": no_warmup,
        }


def _parse_optional_float(value: str) -> float | None:
    """Parse a float from a string, returning None for empty/invalid."""
    v = value.strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _parse_optional_int(value: str) -> int | None:
    """Parse an int from a string, returning None for empty/invalid."""
    v = value.strip()
    if not v:
        return None
    try:
        return int(v)
    except ValueError:
        return None
