from app.config import Settings
from app.services.llm import LocalHashEmbeddings, create_embeddings


def test_auto_provider_uses_local_embeddings_for_deepseek() -> None:
    settings = Settings(
        openai_api_key="test-key",
        openai_base_url="https://api.deepseek.com",
        embedding_provider="auto",
    )
    embeddings = create_embeddings(settings)
    assert isinstance(embeddings, LocalHashEmbeddings)


def test_local_embeddings_are_deterministic_and_normalized() -> None:
    embeddings = LocalHashEmbeddings(dimensions=64)
    first = embeddings.embed_query("EGFR C797S 耐药机制")
    second = embeddings.embed_query("EGFR C797S 耐药机制")
    assert first == second
    assert len(first) == 64
    assert abs(sum(value * value for value in first) - 1.0) < 1e-9
