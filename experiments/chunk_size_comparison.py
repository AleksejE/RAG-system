"""Comparison of chunk sizes and search types on the control question set.

For every chunk size the Constitution is split again and indexed into a
temporary in-memory Qdrant collection (the main collection is not touched).
Then similarity and MMR search are evaluated with Hit@k and MRR.

Example:
    uv run python -m experiments.chunk_size_comparison
"""

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from analytics.evaluate_retrieval import score_retriever
from src.chunking.splitter import extract_article_number
from src.embeddings.embedder import get_embeddings
from src.parsing.constitution_parser import fetch_and_preprocess

CHUNK_SIZES = [(600, 100), (1000, 150), (1500, 200)]
SEARCH_TYPES = ["similarity", "mmr"]
TOP_K = 5


def chunk_blocks(blocks: list[tuple[str, str]], size: int, overlap: int) -> list[Document]:
    """Split blocks into chunks of the given size (same format as the main splitter).

    Args:
        blocks (list[tuple[str, str]]): Blocks of ``(header, text)``.
        size (int): Chunk size in characters.
        overlap (int): Overlap between neighbouring chunks.

    Returns:
        list[Document]: Chunks with ``Header`` and ``article`` metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap, separators=["\n\n", "\n", ". ", " "]
    )
    return [
        Document(
            page_content=f"{header}\n{chunk}",
            metadata={"Header": header, "article": extract_article_number(header)},
        )
        for header, text in blocks
        for chunk in splitter.split_text(text)
    ]


def main() -> None:
    """Evaluate every combination of chunk size and search type and print a table."""
    blocks = fetch_and_preprocess()
    rows = []

    for size, overlap in CHUNK_SIZES:
        documents = chunk_blocks(blocks, size, overlap)
        print(f"\nchunk_size={size}: {len(documents)} чанков, считаем эмбеддинги...")
        store = QdrantVectorStore.from_documents(
            documents, get_embeddings(), location=":memory:", collection_name=f"exp_{size}"
        )
        for search_type in SEARCH_TYPES:
            retriever = store.as_retriever(search_type=search_type, search_kwargs={"k": TOP_K})
            hit, mrr = score_retriever(retriever, verbose=False)
            rows.append((size, overlap, len(documents), search_type, hit, mrr))
            print(f"  {search_type:<10} Hit@{TOP_K}={hit:.2f}  MRR={mrr:.2f}")

    print("\n| chunk_size | overlap | чанков | поиск | Hit@5 | MRR |\n|---|---|---|---|---|---|")
    for size, overlap, count, search_type, hit, mrr in rows:
        print(f"| {size} | {overlap} | {count} | {search_type} | {hit:.2f} | {mrr:.2f} |")


if __name__ == "__main__":
    main()
