The llama-benchy command line is no longer logged with credentials embedded in a
URL. `--api-key` values were already redacted, but a base URL of the form
`https://user:password@host` reached the log verbatim, as did an `?api_key=`
query parameter. Host, port, and path are still logged, so the record still
shows which server was benchmarked. The line runs at INFO, which the package
never enables on its own, so it could only leak where the embedding application
turned INFO logging on.
