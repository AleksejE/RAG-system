"""Splitting of header-text blocks into chunks with metadata.

Each chunk keeps the header of its block and an estimated page number,
so that answers can refer to the source.
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.settings import settings

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
    separators=["\n\n", "\n", ". ", " "],
)


def create_chunked_documents(raw_blocks: list[tuple[str, str]]) -> list[Document]:
    """Split blocks into chunks and attach ``Header`` and ``page_number`` metadata.

    The page number is an estimate: it is calculated from the total length of
    the preceding text, assuming ``PAGE_SIZE_CHARS`` characters per page.

    Args:
        raw_blocks (list[tuple[str, str]]): Blocks of ``(header, text)`` from the parser.

    Returns:
        list[Document]: Chunks with metadata, ready for indexing.
    """
    print("Чанкинг и генерация метаданных {Header, page_number}...")
    documents: list[Document] = []
    total_chars = 0

    for header, block_text in raw_blocks:
        for chunk in text_splitter.split_text(block_text):
            page_number = total_chars // settings.page_size_chars + 1
            total_chars += len(chunk)
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={"Header": header, "page_number": page_number},
                )
            )

    print(f"Всего создано чанков: {len(documents)}")
    return documents
