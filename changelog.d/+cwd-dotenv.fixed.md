**Installed CLI `.env` discovery.** The CLI now looks in the current folder and its parents first,
so `uv tool` installs can read a project's `.env`. If no file is found, the previous module-relative
lookup remains the fallback. Exported environment variables still take precedence.
