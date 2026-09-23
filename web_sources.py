"""
Web-source lookup for plagiarism checks: search + page-text scraping.
"""
import logging
import os

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger("web_sources")

SCRAPE_TIMEOUT_SECONDS = 10
SCRAPE_USER_AGENT = (
    "Mozilla/5.0 (compatible; SimilarityService/1.0; "
    "+https://example.com/bot)"
)

SKIP_TAGS = {"nav", "footer", "script", "style", "header", "aside", "form", "noscript"}
CONTENT_TAGS = ["p", "h1", "h2", "h3", "h4", "h5", "h6"]


class ScrapeError(Exception):
    """Raised when a page can't be fetched or has no usable text."""


DEFAULT_CHUNK_TARGET_WORDS = 200

# target_word_count is normally the submission's own word count, so a
# short submission (a paragraph or two) checked against a page made of
# many short paragraphs (forum threads, wiki lists) would otherwise chunk
# at a fine granularity: only 1-2 paragraphs needed to hit the target
# means each new chunk drops just one paragraph and adds one, so the
# one-paragraph overlap step barely advances start — producing close to
# one chunk per paragraph. Flooring the target keeps chunks coarse enough
# that this can't happen.
MIN_CHUNK_TARGET_WORDS = 100

# Backstop regardless of the floor above — bounds the worst case (a very
# long page) to a fixed number of model.encode() calls per web hit.
MAX_CHUNKS_PER_PAGE = 30


def chunk_page_text(page_text: str, target_word_count: int = None) -> list[str]:
    """
    Split scraped page text into passage-sized chunks so a short
    submission can be compared against comparably-sized segments of a
    page instead of the whole page (which dilutes Jaccard similarity).

    scrape_page_text() joins each <p>/heading tag's text with "\n", so a
    newline is a paragraph boundary here. Adjacent paragraphs are grouped
    until a chunk reaches roughly target_word_count words; consecutive
    chunks overlap by one paragraph so a match spanning a chunk boundary
    isn't missed.
    """
    if not target_word_count or target_word_count <= 0:
        target_word_count = DEFAULT_CHUNK_TARGET_WORDS
    target_word_count = max(target_word_count, MIN_CHUNK_TARGET_WORDS)

    paragraphs = [p for p in page_text.split("\n") if p.strip()]
    if not paragraphs:
        return []

    total_words = sum(len(p.split()) for p in paragraphs)
    if total_words <= target_word_count:
        return [page_text.strip()]

    chunks = []
    start = 0
    n = len(paragraphs)
    while start < n and len(chunks) < MAX_CHUNKS_PER_PAGE:
        word_count = 0
        end = start
        while end < n and word_count < target_word_count:
            word_count += len(paragraphs[end].split())
            end += 1
        chunks.append(" ".join(paragraphs[start:end]))
        if end >= n:
            break
        # Overlap by one paragraph — but only when this chunk grouped more
        # than one paragraph. If a single paragraph alone already met the
        # target, overlapping on it would set start back to itself and
        # loop forever, so advance past it instead.
        start = end - 1 if end - start > 1 else end

    return chunks


def search_web(query: str, num_results: int = 5) -> list[dict]:
    api_key = os.getenv("SERP_API_KEY")
    response = requests.get(
        "https://serpapi.com/search",
        params={
            "q": query,
            "api_key": api_key,
            "num": num_results,
            "engine": "google",
        },
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()

    results = []
    for item in data.get("organic_results", [])[:num_results]:
        results.append({
            "title": item.get("title", ""),
            "url": item.get("link", ""),
        })
    return results


def scrape_page_text(url: str) -> str:
    """
    Fetch a URL and extract main body text from <p> and heading tags,
    skipping nav/footer/script/style/etc. Returns "" on any failure
    that shouldn't crash the caller; raises ScrapeError for cases the
    caller may want to catch and log specifically.
    """
    try:
        response = requests.get(
            url,
            timeout=SCRAPE_TIMEOUT_SECONDS,
            headers={"User-Agent": SCRAPE_USER_AGENT},
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise ScrapeError(f"failed to fetch {url}: {exc}") from exc

    content_type = response.headers.get("Content-Type", "")
    if "html" not in content_type:
        raise ScrapeError(f"non-HTML content-type for {url}: {content_type!r}")

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup.find_all(SKIP_TAGS):
        tag.decompose()
    for tag in soup.find_all(attrs={"role": "navigation"}):
        tag.decompose()

    # Prefer a real content container when the page has one; falling back
    # to the whole (nav-stripped) document pulls in menu/sidebar boilerplate.
    scope = soup.find("article") or soup.find("main") or soup

    chunks = []
    for tag in scope.find_all(CONTENT_TAGS):
        text = tag.get_text(separator=" ", strip=True)
        if text:
            chunks.append(text)

    page_text = "\n".join(chunks)

    if not page_text.strip():
        raise ScrapeError(f"no extractable text found on {url}")

    return page_text
