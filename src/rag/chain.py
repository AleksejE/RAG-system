"""RAG chain: retrieval from Qdrant and answer generation with Groq.

Pipeline::

    question -> retriever (Qdrant, top-k) -> format_docs -> prompt -> ChatGroq -> answer

Example:
    >>> chain = build_rag_chain()
    >>> result = chain.invoke({"question": "What is the term of office of the President?"})
    >>> result["answer"], result["sources"]
"""

from operator import itemgetter
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_groq import ChatGroq

from src.config.settings import settings
from src.rag.prompts import RAG_PROMPT
from src.vector_store.qdrant_store import load_vector_store


def get_llm() -> ChatGroq:
    """Create the Groq chat model configured from settings.

    Returns:
        ChatGroq: Chat model with timeout and automatic retries
            (including rate-limit errors).

    Raises:
        ValueError: If ``GROQ_API_KEY`` is not set in ``.env``.
    """
    if settings.groq_api_key is None or not settings.groq_api_key.get_secret_value():
        raise ValueError("Не задан GROQ_API_KEY: добавьте ключ в файл .env")

    return ChatGroq(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=settings.groq_api_key,
        timeout=settings.llm_timeout_sec,
        max_retries=settings.llm_max_retries,
    )


def get_retriever(k: int | None = None) -> VectorStoreRetriever:
    """Create a retriever over the Qdrant collection.

    Args:
        k (int | None): Number of chunks to retrieve. Defaults to ``TOP_K_RESULTS``.

    Returns:
        VectorStoreRetriever: Similarity-search retriever.
    """
    return load_vector_store().as_retriever(
        search_type="similarity",
        search_kwargs={"k": k or settings.top_k_results},
    )


def format_docs(docs: list[Document]) -> str:
    """Join retrieved chunks into one context string for the prompt.

    Args:
        docs (list[Document]): Retrieved chunks. Their text already starts
            with the article header.

    Returns:
        str: Numbered fragments separated by blank lines.
    """
    return "\n\n".join(
        f"[{i}] (page ~{doc.metadata.get('page_number', '?')})\n{doc.page_content}"
        for i, doc in enumerate(docs, start=1)
    )


def extract_sources(docs: list[Document]) -> list[dict[str, Any]]:
    """Collect unique sources of the retrieved chunks, keeping retrieval order.

    Args:
        docs (list[Document]): Retrieved chunks.

    Returns:
        list[dict[str, Any]]: Items with ``header``, ``article`` and ``page_number``.
    """
    sources: list[dict[str, Any]] = []
    seen: set[str] = set()
    for doc in docs:
        header = doc.metadata.get("Header", "Unknown")
        if header in seen:
            continue
        seen.add(header)
        sources.append(
            {
                "header": header,
                "article": doc.metadata.get("article"),
                "page_number": doc.metadata.get("page_number"),
            }
        )
    return sources


def _to_output(state: dict[str, Any]) -> dict[str, Any]:
    """Keep only the public fields of the chain state.

    Args:
        state (dict[str, Any]): Chain state with ``answer`` and ``docs``.

    Returns:
        dict[str, Any]: ``answer``, ``sources`` and ``docs``.
    """
    return {
        "answer": state["answer"],
        "sources": extract_sources(state["docs"]),
        "docs": state["docs"],
    }


def build_rag_chain(k: int | None = None) -> Runnable:
    """Build the RAG chain.

    Input is a dict with ``question`` (str) and optional ``chat_history``
    (list of messages, filled in by the memory module).

    Args:
        k (int | None): Number of chunks to retrieve. Defaults to ``TOP_K_RESULTS``.

    Returns:
        Runnable: Chain whose ``invoke`` returns a dict with ``answer`` (str),
            ``sources`` (list[dict]) and ``docs`` (list[Document]).
    """
    retriever = get_retriever(k)
    answer_chain = RAG_PROMPT | get_llm() | StrOutputParser()

    retrieve = RunnablePassthrough.assign(docs=itemgetter("question") | retriever)
    add_context = RunnablePassthrough.assign(context=lambda x: format_docs(x["docs"]))
    generate = RunnablePassthrough.assign(answer=answer_chain)

    return retrieve | add_context | generate | RunnableLambda(_to_output)


def ask(question: str, k: int | None = None) -> dict[str, Any]:
    """Answer a single question without chat history.

    Args:
        question (str): User question (Russian or English).
        k (int | None): Number of chunks to retrieve. Defaults to ``TOP_K_RESULTS``.

    Returns:
        dict[str, Any]: ``answer``, ``sources`` and ``docs`` as in ``build_rag_chain``.
    """
    return build_rag_chain(k).invoke({"question": question})
