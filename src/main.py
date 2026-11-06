"""Command-line entry point of the data part of the RAG system.

Provides two commands:
    index:  download the Constitution, split it into chunks and load them into Qdrant.
    search: find the chunks closest to a query (checks the vector DB without an LLM).

Example:
    uv run python -m src.main index
    uv run python -m src.main search "term of office of the President"
"""

import argparse

from src.chunking.splitter import create_chunked_documents
from src.config.settings import settings
from src.parsing.constitution_parser import fetch_and_preprocess
from src.vector_store.qdrant_store import index_documents, load_vector_store


def run_indexing() -> None:
    """Download, chunk and index the Constitution into Qdrant."""
    blocks = fetch_and_preprocess()
    documents = create_chunked_documents(blocks)

    if documents:
        print("\nПример первого чанка:")
        print(f"Metadata: {documents[0].metadata}")
        print(f"Content: {documents[0].page_content[:100]}...\n")

    index_documents(documents)


def run_search(query: str) -> None:
    """Print the top-k chunks most similar to the query with their scores.

    Args:
        query (str): Search text.
    """
    vector_store = load_vector_store()
    results = vector_store.similarity_search_with_score(query, k=settings.top_k_results)

    for rank, (doc, score) in enumerate(results, start=1):
        header = doc.metadata.get("Header", "Unknown")
        page = doc.metadata.get("page_number", "Unknown")
        print(f"\n#{rank}  score={score:.3f}  [{header} | Page {page}]")
        print(doc.page_content[:300])


def main() -> None:
    """Parse command-line arguments and run the selected command."""
    parser = argparse.ArgumentParser(description="Данные и векторная БД для RAG по Конституции РФ")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("index", help="скачать Конституцию и построить индекс в Qdrant")
    search_parser = subparsers.add_parser("search", help="найти похожие фрагменты в Qdrant")
    search_parser.add_argument("query", help="текст запроса")

    args = parser.parse_args()
    if args.command == "index":
        run_indexing()
    else:
        run_search(args.query)


if __name__ == "__main__":
    main()
