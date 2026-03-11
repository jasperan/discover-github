"""
Embedding generation for semantic search and deduplication.

Uses a lightweight sentence-transformers model to generate embeddings
for README content, enabling vector similarity search in Oracle DB.
Falls back to a simple TF-IDF approach if sentence-transformers is unavailable.
"""

import hashlib
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy-loaded model
_model = None
_model_type = None  # "sentence-transformers" or "tfidf"


def _load_model():
    """Load embedding model, trying sentence-transformers first."""
    global _model, _model_type

    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        _model_type = "sentence-transformers"
        logger.info("Loaded sentence-transformers model: all-MiniLM-L6-v2")
    except ImportError:
        logger.warning(
            "sentence-transformers not installed. "
            "Install with: pip install sentence-transformers. "
            "Falling back to TF-IDF embeddings (lower quality)."
        )
        _model = "tfidf"
        _model_type = "tfidf"


def generate_embedding(text: str, max_chars: int = 2000) -> list[float]:
    """
    Generate a 384-dimensional embedding from text content.

    Args:
        text: README or description text to embed.
        max_chars: Truncate input to this many characters (model token limit).

    Returns:
        List of 384 floats representing the text embedding.
    """
    global _model, _model_type
    if _model is None:
        _load_model()

    # Preprocess
    clean_text = _preprocess_for_embedding(text)
    if not clean_text:
        return [0.0] * 384

    # Truncate for token limits
    clean_text = clean_text[:max_chars]

    if _model_type == "sentence-transformers":
        embedding = _model.encode(clean_text, normalize_embeddings=True)
        return embedding.tolist()
    else:
        return _tfidf_fallback(clean_text)


def _preprocess_for_embedding(text: str) -> str:
    """Clean text for embedding: remove markdown, URLs, code blocks."""
    # Remove code blocks
    text = re.sub(r"```[\s\S]*?```", " ", text)
    # Remove inline code
    text = re.sub(r"`[^`]+`", " ", text)
    # Remove URLs
    text = re.sub(r"https?://\S+", " ", text)
    # Remove markdown images
    text = re.sub(r"!\[.*?\]\(.*?\)", " ", text)
    # Remove markdown links but keep text
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove markdown headers
    text = re.sub(r"#{1,6}\s*", "", text)
    # Remove badges/shields
    text = re.sub(r"\[!\[.*?\]\(.*?\)\]\(.*?\)", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tfidf_fallback(text: str) -> list[float]:
    """
    Generate a deterministic pseudo-embedding using hashing.
    Not as good as sentence-transformers but works without GPU/heavy deps.
    Produces a 384-dimensional vector using feature hashing.
    """
    words = text.lower().split()
    embedding = [0.0] * 384

    for word in words:
        # Feature hashing: hash word to get index and sign
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        idx = h % 384
        sign = 1.0 if (h // 384) % 2 == 0 else -1.0
        embedding[idx] += sign

    # L2 normalize
    norm = sum(x * x for x in embedding) ** 0.5
    if norm > 0:
        embedding = [x / norm for x in embedding]

    return embedding
