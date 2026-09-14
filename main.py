from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import re
import numpy as np
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Load the SBERT model once at startup (not per-request — this is slow to load)
model = SentenceTransformer('all-MiniLM-L6-v2')

# ---------- Request/Response models ----------

class Submission(BaseModel):
    id: int
    text: str

class CheckRequest(BaseModel):
    new_submission: Submission
    existing_submissions: List[Submission]

class ScorePair(BaseModel):
    compared_submission_id: int
    lexical_score: float
    semantic_score: float
    combined_score: float
    matched_shingles: List[str]

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


# ---------- Endpoint ----------

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/check-submission", response_model=CheckResponse)
def check_submission(request: CheckRequest):
    new_words = preprocess(request.new_submission.text)
    new_shingles = get_shingles(new_words)
    new_embedding = model.encode(request.new_submission.text)

    results = []

    for existing in request.existing_submissions:
        existing_words = preprocess(existing.text)
        existing_shingles = get_shingles(existing_words)
        existing_embedding = model.encode(existing.text)

        lexical = jaccard_similarity(new_shingles, existing_shingles)
        semantic = cosine_similarity(new_embedding, existing_embedding)
        combined = combined_score(lexical, semantic)

        matched = list(new_shingles & existing_shingles)

        results.append(ScorePair(
            compared_submission_id=existing.id,
            lexical_score=round(lexical, 4),
            semantic_score=round(semantic, 4),
            combined_score=round(combined, 4),
            matched_shingles=matched[:20]  # cap so response isn't huge
        ))

    # sort by combined score, highest first
    results.sort(key=lambda r: r.combined_score, reverse=True)

    return CheckResponse(results=results)