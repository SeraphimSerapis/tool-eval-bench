# Security Policy

## Scope

`tool-eval-bench` is a **local benchmarking tool** that connects to
user-specified LLM endpoints. It does not expose any network services itself.

The primary security considerations are:

- **API keys** stored in `.env` files (never committed to git)
- **Dataset downloads** — benchmark plugins (GSM8K, MMLU, IFEval) download
  datasets from HuggingFace on first use.  Two download methods are supported:
  - **`datasets` library** (`pip install tool-eval-bench[hf]`): downloads
    directly from the HuggingFace git repo.  Uses `trust_remote_code=False`
    to prevent execution of untrusted code from dataset cards.
  - **REST API fallback**: read-only HTTPS `GET` requests to
    `datasets-server.huggingface.co`.  No authentication tokens are sent.
  Downloaded data is cached locally under `data/` as JSONL files.
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
| 2.x | ✅ |
| < 2.0 | ❌ |
