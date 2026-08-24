from __future__ import annotations

import asyncio
from typing import Any

from langchain_openai import ChatOpenAI

from app.config import Settings
from app.services.llm import create_chat_model, create_embeddings
from inference.model_provider import ModelOutput, ModelProvider
from observability.cost import token_usage_from_message


class OpenAICompatibleProvider(ModelProvider):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._chat_model: ChatOpenAI = create_chat_model(settings)

    @property
    def langchain_chat_model(self) -> ChatOpenAI:
        return self._chat_model

    def _output(self, response: Any) -> ModelOutput:
        metadata = getattr(response, "response_metadata", {}) or {}
        return ModelOutput(
            content=getattr(response, "content", response),
            tool_calls=list(getattr(response, "tool_calls", []) or []),
            token_usage=token_usage_from_message(response),
            model=self.settings.llm_model,
            finish_reason=metadata.get("finish_reason"),
            raw=response,
        )

    async def generate(self, messages: list[Any], **kwargs: Any) -> ModelOutput:
        response = await self._chat_model.ainvoke(messages, **kwargs)
        return self._output(response)

    async def tool_call(
        self,
        messages: list[Any],
        tools: list[Any],
        **kwargs: Any,
    ) -> ModelOutput:
        response = await self._chat_model.bind_tools(tools).ainvoke(messages, **kwargs)
        return self._output(response)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = create_embeddings(self.settings)
        return await asyncio.to_thread(embeddings.embed_documents, texts)
