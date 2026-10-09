**`git_sha` no longer borrows another repository's commit.** A tool-eval-bench installed into a
virtual environment inside some other Git work tree, such as a project's gitignored `.venv`,
recorded that project's HEAD, plus `-dirty` from its status. Every unrelated commit then moved the
run to a new comparison cohort. `git_sha` is now recorded only when the package sits in this
project's own checkout, and is `None` otherwise, as documented for installed wheels.
