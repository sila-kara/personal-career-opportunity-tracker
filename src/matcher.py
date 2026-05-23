"""Job matching logic using TF-IDF, cosine similarity, and simple rules."""

import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import clean_text


def _flatten_profile_value(value: Any) -> list[str]:
    """Turn nested profile values into a flat list of readable strings."""
    if isinstance(value, dict):
        flattened = []
        for nested_value in value.values():
            flattened.extend(_flatten_profile_value(nested_value))
        return flattened

    if isinstance(value, list):
        flattened = []
        for item in value:
            flattened.extend(_flatten_profile_value(item))
        return flattened

    return [str(value)]


def get_profile_list(profile: dict, key: str) -> list[str]:
    """Safely get a list field from the profile."""
    value = profile.get(key, [])
    if isinstance(value, list):
        return [str(item) for item in value]
    if value:
        return [str(value)]
    return []


def build_profile_text(profile: dict) -> str:
    """Create one text document that represents the user's career interests."""
    useful_keys = [
        "education",
        "target_roles",
        "skills",
        "preferred_industries",
        "preferred_locations",
        "job_type_preference",
        "keywords_like",
    ]

    profile_parts = []
    for key in useful_keys:
        profile_parts.extend(_flatten_profile_value(profile.get(key, [])))

    return clean_text(" ".join(profile_parts))


def find_matching_terms(text: str, terms: list[str]) -> list[str]:
    """Return profile terms that appear in the cleaned job text."""
    cleaned_text = clean_text(text)
    matches = []

    for term in terms:
        cleaned_term = clean_text(term)
        if cleaned_term and _contains_term(cleaned_text, cleaned_term):
            matches.append(term)

    return matches


def _contains_term(cleaned_text: str, cleaned_term: str) -> bool:
    """Match complete words/phrases so short terms like AI do not match campaign."""
    pattern = rf"(?<![a-z0-9+#]){re.escape(cleaned_term)}(?![a-z0-9+#])"
    return bool(re.search(pattern, cleaned_text))


def calculate_similarity_scores(profile_text: str, job_texts: pd.Series) -> np.ndarray:
    """Calculate TF-IDF cosine similarity between the profile and each job."""
    documents = [profile_text] + job_texts.tolist()

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(documents)

    profile_vector = tfidf_matrix[0:1]
    job_vectors = tfidf_matrix[1:]
    similarities = cosine_similarity(profile_vector, job_vectors).flatten()

    return similarities


def score_jobs(profile: dict, jobs: pd.DataFrame) -> pd.DataFrame:
    """Rank jobs using an explainable hybrid scoring formula.

    Final score formula:
    - 60% TF-IDF cosine similarity between the profile and job text
    - up to 15% bonus for preferred keywords
    - up to 10% bonus for preferred locations
    - up to 25% penalty for non-preferred required locations
    - up to 10% bonus for matching target roles in the title
    - up to 5% bonus for preferred job type
    - up to 25% penalty for non-preferred job type
    - up to 20% penalty for avoid keywords

    The result is clipped between 0 and 100 for readability.
    """
    jobs = jobs.copy()
    profile_text = build_profile_text(profile)
    jobs["similarity_score"] = calculate_similarity_scores(profile_text, jobs["clean_text"])

    like_keywords = get_profile_list(profile, "keywords_like")
    avoid_keywords = get_profile_list(profile, "keywords_avoid")
    preferred_locations = get_profile_list(profile, "preferred_locations")
    target_roles = get_profile_list(profile, "target_roles")
    job_type_preferences = get_profile_list(profile, "job_type_preference")

    scored_rows = []

    for _, row in jobs.iterrows():
        full_text = row["combined_text"]
        title_text = str(row["title"])
        location_text = str(row["location"])
        job_type_text = str(row["job_type"])

        matched_keywords = find_matching_terms(full_text, like_keywords)
        avoid_matches = find_matching_terms(full_text, avoid_keywords)
        location_matches = find_matching_terms(location_text, preferred_locations)
        role_matches = find_matching_terms(title_text, target_roles)
        job_type_matches = find_matching_terms(job_type_text, job_type_preferences)

        keyword_bonus = min(len(matched_keywords) / max(len(like_keywords), 1), 1.0) * 0.15
        avoid_penalty = min(len(avoid_matches) / max(len(avoid_keywords), 1), 1.0) * 0.20
        location_bonus = 0.10 if location_matches else 0.0
        # A non-preferred required location is a serious blocker because the
        # candidate cannot realistically attend an on-site role there.
        location_penalty = 0.25 if not location_matches else 0.0
        role_bonus = 0.10 if role_matches else 0.0
        job_type_bonus = 0.05 if job_type_matches else 0.0
        job_type_penalty = 0.25 if not job_type_matches else 0.0

        raw_score = (
            row["similarity_score"] * 0.60
            + keyword_bonus
            + location_bonus
            + role_bonus
            + job_type_bonus
            - location_penalty
            - job_type_penalty
            - avoid_penalty
        )
        final_score = float(np.clip(raw_score * 100, 0, 100))

        scored_row = row.to_dict()
        scored_row.update(
            {
                "keyword_bonus": round(keyword_bonus * 100, 2),
                "avoid_penalty": round(avoid_penalty * 100, 2),
                "location_bonus": round(location_bonus * 100, 2),
                "location_penalty": round(location_penalty * 100, 2),
                "role_bonus": round(role_bonus * 100, 2),
                "job_type_bonus": round(job_type_bonus * 100, 2),
                "job_type_penalty": round(job_type_penalty * 100, 2),
                "match_score": round(final_score, 2),
                "matched_keywords": ", ".join(matched_keywords),
                "avoid_keywords_found": ", ".join(avoid_matches),
            }
        )
        scored_rows.append(scored_row)

    ranked_jobs = pd.DataFrame(scored_rows)
    ranked_jobs["similarity_score"] = (ranked_jobs["similarity_score"] * 100).round(2)
    ranked_jobs = ranked_jobs.sort_values("match_score", ascending=False)
    return ranked_jobs.reset_index(drop=True)
