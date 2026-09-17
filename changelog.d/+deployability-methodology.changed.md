**Responsiveness and deployability are documented** — `docs/methodology.md` now defines both
derived scores: what a turn latency measures, which scenarios feed the median, the logistic curve
behind responsiveness, the `alpha` weighting behind deployability, and why the composite exists.
The example values in the `responsiveness_score` docstring were off by up to 15 points and now
match what the function returns. `--alpha` is listed in the CLI reference. No score changes.
