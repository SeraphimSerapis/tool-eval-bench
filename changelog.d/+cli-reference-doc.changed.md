Moved `SKILL.md` to `docs/cli-reference.md`. Its content is user-facing CLI reference, and it was
the only place exit codes and the JSON output shape were documented, so it belongs with the rest of
the docs. The root filename also collided with the agent-skill manifest convention, which expects
YAML frontmatter this file never had.
