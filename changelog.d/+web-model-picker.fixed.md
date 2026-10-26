The web launcher now fills a model dropdown from `/api/probe` instead of a
`<datalist>`. Browsers filter datalist options by the input's current value, so
the auto-selected first model hid every other entry and the list could not be
clicked through. A "Custom model ID…" entry keeps the free-text field for
endpoints that advertise nothing.
