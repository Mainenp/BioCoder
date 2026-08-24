from __future__ import annotations

import hashlib
import math
import re
from urllib.parse import urlparse

from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.config import Settings

LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


def is_loopback_base_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and parsed.hostname in LOOPBACK_HOSTS


def chat_api_key(settings: Settings) -> str:
    if settings.openai_api_key:
        return settings.openai_api_key
    if is_loopback_base_url(settings.openai_base_url):
        return "not-required"
    raise RuntimeError(
        "OPENAI_API_KEY is required for non-local model endpoints. "
        "Configure it in .env before starting the Agent."
    )


def create_chat_model(settings: Settings, *, temperature: float = 0.1) -> ChatOpenAI:
    api_key = chat_api_key(settings)
    options = {}
    if "deepseek.com" in settings.openai_base_url.lower():
        options["extra_body"] = {"thinking": {"type": settings.llm_thinking_mode}}
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=api_key,
        base_url=settings.openai_base_url,
        temperature=temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.request_timeout_seconds,
        max_retries=settings.llm_max_retries,
        **options,
    )


def create_model_provider(settings: Settings):
    """Build the provider adapter without coupling the Agent graph to a serving backend."""
    provider = settings.model_provider.strip().lower()
    if provider in {"openai", "api", "openai_compatible"}:
        from inference.openai_provider import OpenAICompatibleProvider

        return OpenAICompatibleProvider(settings)
    if provider == "vllm":
        from inference.vllm_client import VLLMClient

        return VLLMClient(settings)
    raise RuntimeError("MODEL_PROVIDER must be 'openai' or 'vllm'.")


class LocalHashEmbeddings(Embeddings):
    """Deterministic local embeddings suitable for small Chinese/English knowledge bases."""

    def __init__(self, dimensions: int = 768) -> None:
        self.dimensions = dimensions

    def _embed(self, text: str) -> list[float]:
        normalized = re.sub(r"\s+", " ", text.lower()).strip()
        compact = normalized.replace(" ", "")
        features = re.findall(r"[a-z0-9_\-]+", normalized)
        for size in (1, 2, 3):
            features.extend(compact[index : index + size] for index in range(len(compact) - size + 1))

        vector = [0.0] * self.dimensions
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            slot = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[slot] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def create_embeddings(settings: Settings) -> Embeddings:
    provider = settings.embedding_provider.lower()
    if provider == "auto":
        provider = "openai" if settings.embedding_base_url or "api.openai.com" in settings.openai_base_url else "local"
    if provider == "local":
        return LocalHashEmbeddings()
    if provider != "openai":
        raise RuntimeError("EMBEDDING_PROVIDER must be 'auto', 'local', or 'openai'.")

    api_key = settings.embedding_api_key or settings.openai_api_key
    if not api_key:
        raise RuntimeError("EMBEDDING_API_KEY is required when EMBEDDING_PROVIDER=openai.")
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=api_key,
        base_url=settings.embedding_base_url or settings.openai_base_url,
        request_timeout=settings.request_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )
