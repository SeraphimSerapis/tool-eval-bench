**Delete stored runs with `delete` and from the web UI.** `tool-eval-bench delete` removes runs by id
or by `--status`, `--before`, and `--model`, lists them, and asks before deleting; `--dry-run`
previews and `--yes` skips the prompt. The web UI gains a **Delete** button on each run and a
**Select** mode on the run list for deleting several at once. Deleting a run removes its database
row, traces, checkpoints, and Markdown report. A report path that does not look like one the tool
wrote is left alone. The argument schema moves to version 10.
