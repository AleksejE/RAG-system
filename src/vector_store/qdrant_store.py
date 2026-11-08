"""Vector database module for Qdrant.

Creates the collection, writes chunks into it and connects to it for retrieval.
"""

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from src.config.settings import settings
from src.embeddings.embedder import get_embeddings


def get_client() -> QdrantClient:
    """Create a Qdrant client.

    If ``QDRANT_PATH`` is set, Qdrant runs in local (embedded) mode and stores
    data in that folder, so Docker is not needed. Otherwise the client connects
    to the server at ``QDRANT_URL``.

    Returns:
        QdrantClient: Qdrant client.
    """
    if settings.qdrant_path:
        return QdrantClient(path=settings.qdrant_path)
    return QdrantClient(url=settings.qdrant_url)


def recreate_collection(client: QdrantClient) -> None:
    """Delete the collection if it exists and create an empty one.

    Args:
        client (QdrantClient): Qdrant client.
    """
    if client.collection_exists(settings.collection_name):
        client.delete_collection(settings.collection_name)

    client.create_collection(
        collection_name=settings.collection_name,
        vectors_config=VectorParams(size=settings.embedding_dim, distance=Distance.COSINE),
    )


def index_documents(documents: list[Document]) -> None:
    """Recreate the collection and load chunks into it.

    Args:
        documents (list[Document]): Chunks with metadata.
    """
    print("Запись в локальный Qdrant...")
    client = get_client()
    recreate_collection(client)

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=settings.collection_name,
        embedding=get_embeddings(),
    )
    vector_store.add_documents(documents)
    print(f"Успешно загружено {len(documents)} векторов в коллекцию '{settings.collection_name}'!")


def load_vector_store() -> QdrantVectorStore:
    """Connect to the existing collection.

    Returns:
        QdrantVectorStore: Vector store for similarity search.
    """
    return QdrantVectorStore(
        client=get_client(),
        collection_name=settings.collection_name,
        embedding=get_embeddings(),
    )
