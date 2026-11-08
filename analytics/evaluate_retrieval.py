"""Retrieval quality evaluation on the control question set.

For each question from ``eval_questions.json`` the retriever returns top-k
chunks, and we check whether a chunk of the expected article is among them.
The LLM is not called, so a Groq key is not needed.

Metrics:
    Hit@k: share of questions where an expected article is in the top-k.
    MRR:   mean reciprocal rank of the first expected article (0 if not found).

Example:
    uv run python -m analytics.evaluate_retrieval
    uv run python -m analytics.evaluate_retrieval --k 3
"""

import argparse
import json
from pathlib import Path
from typing import Any

from langchain_core.retrievers import BaseRetriever

QUESTIONS_PATH = Path(__file__).parent / "eval_questions.json"


def load_questions() -> list[dict[str, Any]]:
    """Load the control questions.

    Returns:
        list[dict[str, Any]]: Items with ``question`` and ``expected_articles``.
    """
    return json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))


def first_hit_rank(retrieved_articles: list[str | None], expected: list[str]) -> int | None:
    """Find the rank of the first retrieved chunk that belongs to an expected article.

    Args:
        retrieved_articles (list[str | None]): Article numbers of retrieved chunks, in order.
        expected (list[str]): Article numbers that contain the answer.

    Returns:
        int | None: 1-based rank, or None if no expected article was retrieved.
    """
    for rank, article in enumerate(retrieved_articles, start=1):
        if article in expected:
            return rank
    return None


def score_retriever(retriever: BaseRetriever, verbose: bool = True) -> tuple[float, float]:
    """Run a retriever on all control questions and compute metrics.

    Args:
        retriever (BaseRetriever): Retriever to evaluate.
        verbose (bool): Print the result for every question.

    Returns:
        tuple[float, float]: ``(hit_at_k, mrr)``.
    """
    questions = load_questions()
    hits = 0
    reciprocal_ranks = 0.0

    for item in questions:
        docs = retriever.invoke(item["question"])
        retrieved = [doc.metadata.get("article") for doc in docs]
        rank = first_hit_rank(retrieved, item["expected_articles"])

        if rank is not None:
            hits += 1
            reciprocal_ranks += 1 / rank
        if verbose:
            mark = f"rank {rank}" if rank else "MISS"
            print(f"[{mark:>6}] {item['question']}")
            print(f"         expected={item['expected_articles']}  got={retrieved}")

    total = len(questions)
    return hits / total, reciprocal_ranks / total


def main() -> None:
    """Parse command-line arguments, evaluate the main retriever and print metrics."""
    from src.config.settings import settings
    from src.rag.chain import get_retriever

    parser = argparse.ArgumentParser(description="Оценка качества поиска по контрольным вопросам")
    parser.add_argument(
        "--k", type=int, default=settings.top_k_results, help="сколько чанков искать"
    )
    k = parser.parse_args().k

    hit, mrr = score_retriever(get_retriever(k))
    print(f"\nk={k}  questions={len(load_questions())}  Hit@k={hit:.2f}  MRR={mrr:.2f}")


if __name__ == "__main__":
    main()
