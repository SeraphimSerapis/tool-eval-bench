**Git worktree development safety** — newly created worktrees now link the
primary checkout's `.venv` through a `post-checkout` hook, and Git provenance
subprocesses discard repository-local hook variables before resolving the
package commit. This prevents pre-push tests from treating the active
worktree as a nested test repository, creating stray commits, or changing
shared Git configuration.
