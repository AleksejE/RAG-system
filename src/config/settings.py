"""Project settings loaded from environment variables and the ``.env`` file.

All other modules import the ready-made ``settings`` object instead of
reading environment variables or hardcoding values.
"""

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed configuration of the RAG system.

    Field names match the variables in ``.env.example`` (case-insensitive).
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM (Groq) - используется модулем RAG, для индексации не нужен
    groq_api_key: SecretStr | None = None
    llm_model: str = "openai/gpt-oss-120b"
    llm_temperature: float = 0.1

    # Embeddings
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"
    embedding_dim: int = 1024

    # Vector database
    qdrant_url: str = "http://localhost:6333"
    # If set, Qdrant runs in local (embedded) mode in this folder, without Docker
    qdrant_path: str | None = None
    collection_name: str = "constitution_rf"

    # Data source
    source_url: str = "https://constitution.garant.ru/english/"

    # RAG parameters
    chunk_size: int = 600
    chunk_overlap: int = 100
    page_size_chars: int = 1500
    top_k_results: int = 5
    llm_timeout_sec: float = 60.0
    llm_max_retries: int = 3


settings = Settings()
