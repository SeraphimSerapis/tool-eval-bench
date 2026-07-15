"""Compatibility exports for the domain-owned backend adapter contract.

New code should import these ports from :mod:`tool_eval_bench.domain.adapters`.
"""

from tool_eval_bench.domain.adapters import (
    BackendAdapter,
    ChatCompletionResult,
    ProviderToolCall,
)

__all__ = ["BackendAdapter", "ChatCompletionResult", "ProviderToolCall"]
