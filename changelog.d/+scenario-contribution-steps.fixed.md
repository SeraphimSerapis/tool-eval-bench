The contributor guide's scenario checklist omitted two required steps: registering the scenario in
its module's `*_DISPLAY_DETAILS` dict, and setting a `difficulty` tier. Both fail silently when
missed, the second by dropping the scenario out of `--weight-by-difficulty` scoring. The guide now
documents all six steps under an `Adding a new scenario` heading.
