"""Splitting of header-text blocks into chunks with metadata.

Each chunk starts with the header of its block (so the header is part of the
embedding) and keeps ``Header``, ``article`` and an estimated ``page_number``
in metadata, so that answers can refer to the source.
"""

import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.settings import settings

ARTICLE_PATTERN = re.compile(r"^Article\s+(\d+(?:\.\d+)?)", re.IGNORECASE)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
    separators=["\n\n", "\n", ". ", " "],
)


def extract_article_number(header: str) -> str | None:
    """Extract the article number from a block header.

    Args:
        header (str): Block header, e.g. ``"Article 80"`` or ``"Article 67.1"``.

    Returns:
        str | None: Article number as a string (``"80"``, ``"67.1"``) or None
            if the header is not an article (preamble, section, chapter).
    """
    match = ARTICLE_PATTERN.match(header.strip())
    return match.group(1) if match else None


def create_chunked_documents(raw_blocks: list[tuple[str, str]]) -> list[Document]:
    """Split blocks into chunks and attach ``Header``, ``article`` and ``page_number``.

    The header is prepended to the chunk text, so that queries like
    "What does Article 80 say?" match the right chunk.

    The page number is an estimate: it is calculated from the total length of
    the preceding text, assuming ``PAGE_SIZE_CHARS`` characters per page.

    Args:
        raw_blocks (list[tuple[str, str]]): Blocks of ``(header, text)`` from the parser.

    Returns:
        list[Document]: Chunks with metadata, ready for indexing.
    """
    print("Чанкинг и генерация метаданных {Header, article, page_number}...")
    documents: list[Document] = []
    total_chars = 0

    for header, block_text in raw_blocks:
        article = extract_article_number(header)
        for chunk in text_splitter.split_text(block_text):
            page_number = total_chars // settings.page_size_chars + 1
            total_chars += len(chunk)
            documents.append(
                Document(
                    page_content=f"{header}\n{chunk}",
                    metadata={"Header": header, "article": article, "page_number": page_number},
                )
            )

    print(f"Всего создано чанков: {len(documents)}")
    return documents
