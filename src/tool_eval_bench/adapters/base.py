"""Backend adapter interface for agentic tool-call benchmarking.

Provides the abstract `BackendAdapter` with the `chat_completion` method
and result types in the OpenAI wire format.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProviderToolCall:
    """A tool call returned by the model in OpenAI wire format."""
    id: str
    name: str
    arguments_str: str

    @property
    def arguments(self) -> dict[str, Any]:
        try:
            parsed = json.loads(self.arguments_str)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}


@dataclass
class ChatCompletionResult:
    """Result of a single chat completion request."""
    content: str
    tool_calls: list[ProviderToolCall] = field(default_factory=list)
    raw_response: dict = field(default_factory=dict)
    elapsed_ms: float = 0.0
    ttft_ms: float | None = None  # Time to first token (set when streaming is used)
    reasoning: str | None = None
    # Token usage (from server's usage response — may be None if server doesn't report)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class BackendAdapter(ABC):
    """Abstract base for backend adapters (vLLM, LiteLLM, llama.cpp, etc.)."""

    @abstractmethod
    async def chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout_seconds: float = 60.0,
        api_key: str | None = None,
        base_url: str = "",
        extra_params: dict[str, Any] | None = None,
        stream: bool = False,
        response_format: dict[str, Any] | None = None,
        parallel_tool_calls: bool | None = True,
    ) -> ChatCompletionResult:
        """Send a full chat completion request with optional tool definitions.

        When stream=True, the adapter should use SSE streaming to measure
        time-to-first-token (TTFT) and set it on the result.
        """
        ...
