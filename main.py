import logging
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from web_sources import search_web, scrape_page_text, chunk_page_text, ScrapeError

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("similarity_service")

app = FastAPI()

# Load the SBERT model once at startup (not per-request — this is slow to load)
model = SentenceTransformer('all-MiniLM-L6-v2')

WEB_SEARCH_MAX_RESULTS = 5
WEB_SEARCH_NUM_QUERY_SENTENCES = 3

# ---------- Request/Response models ----------

class Submission(BaseModel):
    id: int
    text: str

class CheckRequest(BaseModel):
    new_submission: Submission
    existing_submissions: List[Submission]
    check_web: bool = True

class ScorePair(BaseModel):
    source_type: str  # "submission" or "web"
    compared_submission_id: Optional[int] = None
    source_url: Optional[str] = None
    source_title: Optional[str] = None
    lexical_score: float
    semantic_score: float
    combined_score: float
    matched_shingles: List[str]
    matched_passage: Optional[str] = None  # best-matching chunk, web sources only

class CheckResponse(BaseModel):
    results: List[ScorePair]


# ---------- Preprocessing ----------

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "of",
    "and", "or", "to", "for", "with", "as", "by", "this", "that", "it",
    "be", "been", "has", "have", "had", "not", "but", "from", "which"
}

def preprocess(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)  # strip punctuation
    words = text.split()
    words = [w for w in words if w not in STOPWORDS]
    return words


# ---------- Lexical similarity: shingling + Jaccard ----------

def get_shingles(words: List[str], n: int = 5) -> set:
    # An empty word list (blank text, or text that's entirely stopwords)
    # must produce an empty set. Without this, `{" ".join([])}` below
    # returns {''} — non-empty, so jaccard_similarity's "not set_a" guard
    # doesn't catch it, and two blank/stopword-only submissions come back
    # with a spurious 100% lexical match instead of 0%.
    if not words:
        return set()
    if len(words) < n:
        return {" ".join(words)}
    return {" ".join(words[i:i+n]) for i in range(len(words) - n + 1)}

def jaccard_similarity(set_a: set, set_b: set) -> float:
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


# ---------- Semantic similarity: SBERT + cosine similarity ----------

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))


# ---------- Combined score ----------

LEXICAL_WEIGHT = 0.4
SEMANTIC_WEIGHT = 0.6

def combined_score(lexical: float, semantic: float) -> float:
    return (LEXICAL_WEIGHT * lexical) + (SEMANTIC_WEIGHT * semantic)


# ---------- Web source query building ----------

def build_web_query(text: str, num_sentences: int = WEB_SEARCH_NUM_QUERY_SENTENCES) -> str:
    """Pick the longest (most distinctive) sentences to use as a search query."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return text.strip()
    longest = sorted(sentences, key=len, reverse=True)[:num_sentences]
    return " ".join(longest)


def score_against_text(new_shingles, new_embedding, other_text: str):
    other_words = preprocess(other_text)
    other_shingles = get_shingles(other_words)
    other_embedding = model.encode(other_text)

    lexical = jaccard_similarity(new_shingles, other_shingles)
    semantic = cosine_similarity(new_embedding, other_embedding)
    combined = combined_score(lexical, semantic)
    matched = list(new_shingles & other_shingles)

    return lexical, semantic, combined, matched


# ---------- Endpoint ----------

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/check-submission", response_model=CheckResponse)
def check_submission(request: CheckRequest):
    new_words = preprocess(request.new_submission.text)
    new_shingles = get_shingles(new_words)
    new_embedding = model.encode(request.new_submission.text)
    submission_word_count = len(request.new_submission.text.split())

    results = []

    for existing in request.existing_submissions:
        lexical, semantic, combined, matched = score_against_text(
            new_shingles, new_embedding, existing.text
        )
        results.append(ScorePair(
            source_type="submission",
            compared_submission_id=existing.id,
            lexical_score=round(lexical, 4),
            semantic_score=round(semantic, 4),
            combined_score=round(combined, 4),
            matched_shingles=matched[:20]  # cap so response isn't huge
        ))

    # A blank/whitespace-only submission has nothing worth searching for —
    # build_web_query() would otherwise fall back to sending an empty
    # query to SerpApi, burning a paid search call for no useful signal.
    if request.check_web and request.new_submission.text.strip():
        query = build_web_query(request.new_submission.text)
        web_hits = []
        # SerpApi has been observed to intermittently stall past its 10s
        # timeout regardless of query length; one retry absorbs that
        # without masking a genuine outage.
        for attempt in (1, 2):
            try:
                web_hits = search_web(query, num_results=WEB_SEARCH_MAX_RESULTS)
                break
            except Exception as exc:
                logger.warning(
                    "web search attempt %d/2 failed for query %r: %s",
                    attempt, query, exc
                )

        for hit in web_hits:
            url = hit.get("url")
            if not url:
                continue
            try:
                page_text = scrape_page_text(url)
            except ScrapeError as exc:
                logger.warning("skipping web result %s: %s", url, exc)
                continue
            except Exception as exc:
                logger.warning("unexpected error scraping %s: %s", url, exc)
                continue

            chunks = chunk_page_text(page_text, target_word_count=submission_word_count)
            if not chunks:
                logger.warning("no chunks produced for %s, skipping", url)
                continue

            best = None  # (lexical, semantic, combined, matched, chunk_text)
            for chunk_text in chunks:
                lexical, semantic, combined, matched = score_against_text(
                    new_shingles, new_embedding, chunk_text
                )
                if best is None or combined > best[2]:
                    best = (lexical, semantic, combined, matched, chunk_text)

            lexical, semantic, combined, matched, best_chunk = best
            results.append(ScorePair(
                source_type="web",
                source_url=url,
                source_title=hit.get("title"),
                lexical_score=round(lexical, 4),
                semantic_score=round(semantic, 4),
                combined_score=round(combined, 4),
                matched_shingles=matched[:20],
                matched_passage=best_chunk
            ))

    # sort by combined score, highest first
    results.sort(key=lambda r: r.combined_score, reverse=True)

    return CheckResponse(results=results)