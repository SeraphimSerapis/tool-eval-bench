"""Domain ports for chat-completion backends.

The domain owns the request/response contract used by runners and benchmark
plugins. Concrete HTTP adapters implement this port in ``tool_eval_bench.adapters``.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from tool_eval_bench.domain.models import DEFAULT_REQUEST_TIMEOUT_SECONDS, ChatMessage

logger = logging.getLogger(__name__)

# HTTP statuses an adapter should retry, and which — if they survive the retry
# budget — indicate a saturated endpoint rather than a bad request.  A 500 is
# excluded: it normally reflects a request the server cannot process at all.
RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 502, 503, 504})


@dataclass
class ProviderToolCall:
    """A tool call returned by a model in OpenAI wire format."""

    id: str
    name: str
    arguments_str: str
    # Provider-specific fields must survive a multi-turn round trip. Gemini
    # uses this for the thought signature required by Gemini 3 function calls.
    extra_content: dict[str, Any] | None = None

    @property
    def arguments(self) -> dict[str, Any]:
        """Return parsed object arguments, or an empty mapping when invalid."""
        try:
            parsed = json.loads(self.arguments_str)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            logger.debug("Failed to parse tool call arguments: %.80s", self.arguments_str)
            return {}


@dataclass
class ChatCompletionResult:
    """Result of a single chat completion request."""

    content: str
    tool_calls: list[ProviderToolCall] = field(default_factory=list)
    raw_response: dict = field(default_factory=dict)
    elapsed_ms: float = 0.0
    ttft_ms: float | None = None
    reasoning: str | None = None
    # Metadata attached to the assistant message itself, including provider
    # extensions from empty/content-only streaming deltas.
    message_extra_content: dict[str, Any] | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class BackendAdapter(ABC):
    """Port implemented by backend-specific chat-completion adapters."""

    @abstractmethod
    async def chat_completion(
        self,
        *,
        model: str,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = "auto",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        api_key: str | None = None,
        base_url: str = "",
        extra_params: dict[str, Any] | None = None,
        stream: bool = False,
        response_format: dict[str, Any] | None = None,
        parallel_tool_calls: bool | None = True,
    ) -> ChatCompletionResult:
        """Send a full chat-completion request."""
        ...
