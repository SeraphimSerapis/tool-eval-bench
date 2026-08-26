The three accuracy plugins each carried their own copy of the load-from-cache-or-download flow. That
now lives in `cli/plugin_datasets.py`, parameterised by benchmark name, item noun, and whether an
interrupted download can resume. Console output is unchanged.
