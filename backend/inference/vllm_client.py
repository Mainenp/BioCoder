from __future__ import annotations

import asyncio
from typing import Any

import httpx
from langchain_openai import ChatOpenAI

from app.config import Settings
from app.services.llm import LocalHashEmbeddings
from inference.model_provider import ModelOutput, ModelProvider
from observability.cost import token_usage_from_message


class VLLMClient(ModelProvider):
    """Thin OpenAI-compatible vLLM client; serving remains outside the Agent process."""

    def __init__(self, settings: Settings) -> None:
        if not settings.vllm_base_url:
            raise ValueError("VLLM_BASE_URL is required when MODEL_PROVIDER=vllm")
        self.settings = settings
        self.base_url = settings.vllm_base_url.rstrip("/")
        self.model_name = settings.vllm_model or settings.llm_model
        self.api_key = settings.vllm_api_key or "not-required"
        self._chat_model = ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=0.1,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.request_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )

    @property
    def langchain_chat_model(self) -> ChatOpenAI:
        return self._chat_model

    def _output(self, response: Any) -> ModelOutput:
        metadata = getattr(response, "response_metadata", {}) or {}
        return ModelOutput(
            content=getattr(response, "content", response),
            tool_calls=list(getattr(response, "tool_calls", []) or []),
            token_usage=token_usage_from_message(response),
            model=self.model_name,
            finish_reason=metadata.get("finish_reason"),
            raw=response,
        )

    async def generate(self, messages: list[Any], **kwargs: Any) -> ModelOutput:
        return self._output(await self._chat_model.ainvoke(messages, **kwargs))

    async def tool_call(
        self,
        messages: list[Any],
        tools: list[Any],
        **kwargs: Any,
    ) -> ModelOutput:
        return self._output(await self._chat_model.bind_tools(tools).ainvoke(messages, **kwargs))

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.settings.vllm_embedding_model:
            embeddings = LocalHashEmbeddings()
            return await asyncio.to_thread(embeddings.embed_documents, texts)
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
            response = await client.post(
                f"{self.base_url}/embeddings",
                headers=headers,
                json={"model": self.settings.vllm_embedding_model, "input": texts},
            )
            response.raise_for_status()
        rows = sorted(response.json().get("data", []), key=lambda row: row.get("index", 0))
        return [list(map(float, row["embedding"])) for row in rows]
