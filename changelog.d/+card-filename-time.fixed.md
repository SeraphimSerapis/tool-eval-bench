**Card filenames no longer collide within a day.** The default card name now carries the UTC time
of the newest run, as `<model>-<YYYYMMDD-HHMM>-<fingerprint>.svg`. A second session on the same day
with the same tool-call configuration, such as a rerun that only adds `--needle`, used to get the
same name and overwrite the first card and its PNG. Rendering one session again still writes the
same file.
