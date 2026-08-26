# Adding a benchmark plugin

A plugin is a benchmark that is not tool-calling: GSM8K, MMLU, and IFEval ship today, each scoring
its own dataset through the same endpoint and reporting into the same run. A plugin owns its
dataset, its prompting, and its scoring. It reuses the adapter, storage, and reporting layers.

## The contract

`domain/plugin.py` defines `BenchmarkPlugin` with two properties and two methods:

| Member | Returns |
|---|---|
| `name` | Short identifier used in the CLI and in storage, such as `"gsm8k"` |
| `description` | One line for `--help` |
| `async run(adapter, *, model, base_url, ...)` | A `BenchmarkResult` |
| `render_report_section(result)` | Markdown lines for the run report |

`BenchmarkResult` normalizes the score to 0–100 alongside a display label, a rating, per-item
results, and metadata. Reporting and comparison read those fields, so a plugin that fills them in
gets history, diffing, and the leaderboard without writing any of it.

`run` receives an `on_progress` callback, `(current, total, item_info)`. Call it after each item;
the shared live display in `cli/plugin_progress.py` turns it into a progress bar and a running
tally. Put `correct`, `is_error`, and the item's prompt into `item_info` and the tally accounting
works unchanged.

## The steps

1. **Write the plugin** under `src/tool_eval_bench/plugins/<name>/`, with `plugin.py` implementing
   the ABC and a `dataset.py` that caches its download under `data/<name>/`.
   `cli/plugin_datasets.py::load_dataset_with_progress` handles the load-or-download flow,
   including resuming a partial download.

2. **Register it** in `plugins/registry.py`, in the dict returned by `_load_builtin_plugins`. The
   import is lazy on purpose: a plugin's dataset dependencies must not load on every CLI start.

3. **Declare its flags** in `cli/legacy_parser.py`, at minimum `--<name>` and `--<name>-only`, and
   mirror them in `schema.py` so the machine-readable flag list stays complete.

4. **Expose it as a subcommand** in `cli/command_registry.py`: add the name to the `plugin`
   command's `choices`, add its flags to `PLUGIN_LEGACY`, and add `--<name>-only` to
   `legacy_flags`. The registry is the authority for both the subcommand form and the legacy flat
   flags, so the two stay in step.

5. **Wire the runner** in `cli/plugin_runners.py`: add a runner and list the name in
   `run_selected_plugins`, which owns the shared selection, invocation, and early-stop lifecycle
   for combined and `--<name>-only` runs. Report writing goes through
   `cli/plugin_lifecycle.py::finalize_plugin_run`.

6. **Regenerate the compatibility snapshots**, which pin the public flag surface:

   ```bash
   .venv/bin/python scripts/update_compat_snapshots.py
   ```

   Review the diff. An unexpected change there means a flag moved that should not have.

Skipping step 4 or 5 is the usual failure: the plugin imports and its tests pass, but
`tool-eval-bench plugin <name>` reports an unknown choice.
