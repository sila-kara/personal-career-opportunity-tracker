"""Optional semantic matching with sentence-transformers embeddings."""

import importlib.util

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from matcher import build_profile_text


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def sentence_transformers_available() -> bool:
    """Return True when the optional sentence-transformers package is installed."""
    return importlib.util.find_spec("sentence_transformers") is not None


def require_sentence_transformers():
    """Import sentence-transformers or raise a helpful dependency error."""
    if not sentence_transformers_available():
        raise ImportError(
            "sentence-transformers is not installed. Install it with: "
            "pip install sentence-transformers"
        )

    from sentence_transformers import SentenceTransformer

    return SentenceTransformer


def calculate_embedding_similarity_scores(
    profile_text: str,
    job_texts: pd.Series,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> np.ndarray:
    """Calculate cosine similarity using sentence-transformer embeddings."""
    SentenceTransformer = require_sentence_transformers()
    model = SentenceTransformer(model_name)

    documents = [profile_text] + job_texts.fillna("").tolist()
    embeddings = model.encode(documents, convert_to_numpy=True, show_progress_bar=False)

    profile_embedding = embeddings[0:1]
    job_embeddings = embeddings[1:]

    return cosine_similarity(profile_embedding, job_embeddings).flatten()


def add_semantic_scores(
    profile: dict,
    jobs: pd.DataFrame,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> pd.DataFrame:
    """Add semantic embedding similarity scores to ranked jobs."""
    jobs_with_scores = jobs.copy()
    profile_text = build_profile_text(profile)
    semantic_scores = calculate_embedding_similarity_scores(
        profile_text,
        jobs_with_scores["clean_text"],
        model_name=model_name,
    )

    jobs_with_scores["semantic_similarity_score"] = (semantic_scores * 100).round(2)
    jobs_with_scores = jobs_with_scores.sort_values(
        "semantic_similarity_score", ascending=False
    )

    return jobs_with_scores.reset_index(drop=True)
