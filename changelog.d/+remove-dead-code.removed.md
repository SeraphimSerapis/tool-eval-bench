Removed three unreferenced functions: `adapters.measurement.bind_measurement_client`,
`runner.llama_benchy.run_llama_benchy_sync`, and `evals.helpers.has_matching_tool_result`. None had
a call site in the package, the tests, or the scripts.
