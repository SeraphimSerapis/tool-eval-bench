# Security Policy

## Scope

`tool-eval-bench` is a **local benchmarking tool** that connects to
user-specified LLM endpoints. It does not expose any network services itself.

The primary security considerations are:

- **API keys** stored in `.env` files (never committed to git)
- **Prompt injection scenarios** (Category K) — these are intentionally
  adversarial test cases, not vulnerabilities

## Reporting a Vulnerability

If you discover a security issue, please report it responsibly:

1. **Do not** open a public GitHub issue.
2. Email the maintainer or use [GitHub's private vulnerability reporting](https://github.com/SeraphimSerapis/tool-eval-bench/security/advisories/new).
3. Include a description of the issue and steps to reproduce.

You should receive an acknowledgment within 48 hours.

## Supported Versions

| Version | Supported |
|---|---|
| 1.0.x | ✅ |
| < 1.0 | ❌ |
