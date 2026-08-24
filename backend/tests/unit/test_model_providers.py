import asyncio
from typing import Any

import pytest

from app.config import Settings
from app.services.llm import create_chat_model, create_model_provider
from inference.model_provider import ModelOutput, ModelProvider
from inference.vllm_client import VLLMClient


class FakeProvider(ModelProvider):
    @property
    def langchain_chat_model(self) -> str:
        return "fake"

    async def generate(self, messages: list[Any], **kwargs: Any) -> ModelOutput:
        return ModelOutput(content="answer", model="fake")

    async def tool_call(
        self,
        messages: list[Any],
        tools: list[Any],
        **kwargs: Any,
    ) -> ModelOutput:
        return ModelOutput(content="", tool_calls=[{"name": "tool"}], model="fake")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text))] for text in texts]


def test_provider_contract_supports_generation_tool_call_and_embedding() -> None:
    provider = FakeProvider()

    assert asyncio.run(provider.generate([])).content == "answer"
    assert asyncio.run(provider.tool_call([], [])).tool_calls == [{"name": "tool"}]
    assert asyncio.run(provider.embed(["abc"])) == [[3.0]]


def test_vllm_provider_is_config_driven_and_uses_local_embedding_fallback() -> None:
    settings = Settings(
        model_provider="vllm",
        vllm_base_url="http://localhost:8001/v1/",
        vllm_model="local-biocoder",
        vllm_embedding_model="",
    )
    provider = create_model_provider(settings)

    assert isinstance(provider, VLLMClient)
    assert provider.base_url == "http://localhost:8001/v1"
    assert provider.model_name == "local-biocoder"
    vectors = asyncio.run(provider.embed(["EGFR", "PARP"]))
    assert len(vectors) == 2
    assert len(vectors[0]) == 768


def test_openai_compatible_loopback_endpoint_does_not_require_an_api_key() -> None:
    settings = Settings(
        openai_api_key="",
        openai_base_url="http://127.0.0.1:8080/v1",
        llm_model="local-model",
    )

    model = create_chat_model(settings)

    assert model.openai_api_key is not None
    assert model.openai_api_key.get_secret_value() == "not-required"


def test_remote_openai_compatible_endpoint_still_requires_an_api_key() -> None:
    settings = Settings(
        openai_api_key="",
        openai_base_url="https://models.example.com/v1",
        llm_model="remote-model",
    )

    with pytest.raises(RuntimeError, match="non-local"):
        create_chat_model(settings)
