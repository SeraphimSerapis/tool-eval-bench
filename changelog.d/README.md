# Changelog fragments

`CHANGELOG.md` is generated. Do not edit it directly.

Every change that users should know about adds a small file to this directory instead. At release
time `towncrier build` collects them into `CHANGELOG.md` and deletes the originals.

This exists for one reason: when every change edits the same lines at the top of `CHANGELOG.md`,
every pair of parallel branches conflicts there. A separate file per change cannot conflict.

## Adding a fragment

```bash
.venv/bin/towncrier create --edit 72.fixed.md
```

The name is `<issue-or-PR-number>.<type>.md`. When a change has no issue or PR number, prefix the
name with `+` and use a short slug instead:

```bash
.venv/bin/towncrier create --edit +gemini-thought-signatures.added.md
```

Writing the file by hand works exactly as well as the `create` command.

## Types

| Type       | Use for                                                          |
| ---------- | ---------------------------------------------------------------- |
| `added`    | New scenarios, CLI flags, backends, plugins, reports              |
| `changed`  | Behavior that already existed and now works differently           |
| `fixed`    | Bugs, scoring corrections, evaluator contract repairs             |
| `removed`  | Deleted flags, scenarios, or supported configurations             |
| `security` | Anything with a security consequence for someone running the tool |

Scoring changes belong in `fixed` or `changed` rather than `added`, since they alter results a
previous run already produced.

## What a fragment contains

One entry, in the style the changelog already uses: a bold subject, then what changed and what it
means for someone running the benchmark. No leading `- `, since towncrier adds the bullet.

```markdown
**TC-49 cancellation scoring** — a model that says the email was *not* sent no longer scores as a
false cancellation claim. The evaluator now checks for negation before matching the sent-confirmation
phrase, so a correct refusal and a hallucinated success stop landing in the same bucket.
```

Continuation lines need no indentation here. Towncrier indents them when it builds.

Describe the observable consequence, not the diff. "Tightened the regex" tells a reader nothing about
whether their scores move.

## Checking your work

```bash
.venv/bin/towncrier build --draft --version 0.0.0
```

That renders every pending fragment exactly as it will appear, and writes nothing.
