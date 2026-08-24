from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from biocoder.state import TokenUsage


class ModelOutput(BaseModel):
    content: Any = ""
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    model: str
    finish_reason: str | None = None
    raw: Any = Field(default=None, exclude=True)


class ModelProvider(ABC):
    """Provider-neutral model interface used by runtime and evaluators."""

    @property
    @abstractmethod
    def langchain_chat_model(self) -> Any:
        """Compatibility bridge for the existing LangGraph runtime."""

    @abstractmethod
    async def generate(self, messages: list[Any], **kwargs: Any) -> ModelOutput: ...

    @abstractmethod
    async def tool_call(
        self,
        messages: list[Any],
        tools: list[Any],
        **kwargs: Any,
    ) -> ModelOutput: ...

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]: ...
