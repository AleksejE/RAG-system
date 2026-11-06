"""Downloading and parsing of the Constitution text from constitution.garant.ru.

The page is split into blocks of ``(header, text)``, where a header is a
section, chapter or article title and text is the paragraphs that follow it.
"""

import requests
from bs4 import BeautifulSoup, Tag

from src.config.settings import settings

HEADERS = {"User-Agent": "Mozilla/5.0"}
HEADER_TAGS = ("h2", "h3", "h4")
HEADER_PREFIXES = ("SECTION", "Chapter", "Article")
REQUEST_TIMEOUT_SEC = 30


def fetch_html(url: str) -> str:
    """Download the HTML page.

    Args:
        url (str): Page address.

    Returns:
        str: Page HTML decoded as UTF-8.

    Raises:
        requests.HTTPError: If the server responds with an error status.
    """
    response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT_SEC)
    response.raise_for_status()
    response.encoding = "utf-8"
    return response.text


def extract_content(html: str) -> Tag:
    """Find the main text container and remove service elements from it.

    Args:
        html (str): Page HTML.

    Returns:
        Tag: ``div.content`` element without scripts, styles and the page title.

    Raises:
        ValueError: If the page has no ``div.content`` element (site layout changed).
    """
    soup = BeautifulSoup(html, "html.parser")
    content = soup.find("div", class_="content")
    if content is None:
        raise ValueError("Не найден блок div.content: возможно, изменилась вёрстка сайта")

    for tag in content.find_all(["style", "script"]):
        tag.decompose()
    if content.find("h1"):
        content.find("h1").decompose()
    return content


def is_header(element: Tag, text: str) -> bool:
    """Check whether an element is a section, chapter or article title.

    Args:
        element (Tag): HTML element.
        text (str): Element text without surrounding whitespace.

    Returns:
        bool: True if the element starts a new block.
    """
    return element.name in HEADER_TAGS or text.startswith(HEADER_PREFIXES)


def split_into_blocks(content: Tag) -> list[tuple[str, str]]:
    """Group paragraphs under the nearest preceding header.

    Args:
        content (Tag): Main text container.

    Returns:
        list[tuple[str, str]]: Blocks of ``(header, text)``. Text before the
            first header gets the header ``"Preamble"``.
    """
    blocks: list[tuple[str, str]] = []
    current_header = "Preamble"
    current_text: list[str] = []

    for element in content.find_all([*HEADER_TAGS, "p"]):
        text = element.get_text(strip=True)
        if not text:
            continue

        if is_header(element, text):
            if current_text:
                blocks.append((current_header, "\n".join(current_text)))
                current_text = []
            current_header = text
        else:
            current_text.append(text)

    if current_text:
        blocks.append((current_header, "\n".join(current_text)))
    return blocks


def fetch_and_preprocess(url: str = settings.source_url) -> list[tuple[str, str]]:
    """Download the Constitution page and split it into header-text blocks.

    Args:
        url (str): Page address. Defaults to ``SOURCE_URL`` from settings.

    Returns:
        list[tuple[str, str]]: Blocks of ``(header, text)``.
    """
    print("Парсинг Конституции РФ...")
    content = extract_content(fetch_html(url))
    return split_into_blocks(content)
