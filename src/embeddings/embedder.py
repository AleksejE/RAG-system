"""Embedding model shared by indexing and retrieval.

The model is loaded once and cached, because loading bge-m3 takes noticeable time.
"""

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from src.config.settings import settings


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the embedding model (only on the first call).

    Embeddings are normalized, so cosine distance in Qdrant works correctly.

    Returns:
        HuggingFaceEmbeddings: Model configured from settings.
    """
    print(f"Загрузка эмбеддингов {settings.embedding_model}...")
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": settings.embedding_device},
        encode_kwargs={"normalize_embeddings": True},
    )
