"""Job matching logic using TF-IDF, cosine similarity, and simple rules."""

import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import clean_text


FEEDBACK_ADJUSTMENTS = {
    "liked": 0.10,
    "maybe": 0.03,
    "rejected": -0.30,
}

TECHNICAL_SKILL_TERMS = [
    "Python",
    "C++",
    "Java",
    "JavaScript",
    "SQL",
    "Dart",
    "Flutter",
    "pandas",
    "numpy",
    "scikit-learn",
    "TensorFlow",
    "PyTorch",
    "machine learning",
    "deep learning",
    "natural language processing",
    "NLP",
    "data analysis",
    "data visualization",
    "business analytics",
    "Power BI",
    "Tableau",
    "Excel",
    "Gurobi",
    "Arena",
    "optimization",
    "simulation",
    "algorithms",
    "data structures",
    "software development",
    "mobile application",
    "cybersecurity",
    "Git",
    "Docker",
    "AWS",
    "cloud",
    "API",
    "ETL",
    "dashboard",
]

SKILL_ALIASES = {
    "algorithms and data structures": ["algorithms", "data structures"],
    "artificial intelligence": ["AI"],
    "machine learning": ["ML"],
    "microsoft excel": ["Excel"],
    "mobile application development": ["mobile application"],
    "natural language processing": ["NLP"],
}


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


def expand_skill_terms(skills: list[str]) -> list[str]:
    """Add common aliases so skill matching catches equivalent phrases."""
    expanded_terms = []
    seen_terms = set()

    for skill in skills:
        terms = [skill] + SKILL_ALIASES.get(clean_text(skill), [])
        for term in terms:
            cleaned_term = clean_text(term)
            if cleaned_term and cleaned_term not in seen_terms:
                expanded_terms.append(term)
                seen_terms.add(cleaned_term)

    return expanded_terms


def analyze_skill_gap(text: str, profile_skills: list[str]) -> dict:
    """Compare job skill signals with the candidate's profile skills."""
    candidate_skill_terms = expand_skill_terms(profile_skills)
    candidate_skill_keys = {clean_text(skill) for skill in candidate_skill_terms}
    skill_catalog = expand_skill_terms(TECHNICAL_SKILL_TERMS + profile_skills)

    profile_skills_found = find_profile_skill_matches(text, profile_skills)
    job_skills_found = find_matching_terms(text, skill_catalog)
    missing_skills = [
        skill for skill in job_skills_found if clean_text(skill) not in candidate_skill_keys
    ]

    denominator = len(profile_skills_found) + len(missing_skills)
    skill_match_rate = len(profile_skills_found) / denominator if denominator else 0.0

    return {
        "profile_skills_found": profile_skills_found,
        "job_skills_found": job_skills_found,
        "missing_skills": missing_skills,
        "skill_match_rate": skill_match_rate,
    }


def find_profile_skill_matches(text: str, profile_skills: list[str]) -> list[str]:
    """Find profile skills without double-counting their aliases."""
    matches = []
    seen_terms = set()

    for skill in profile_skills:
        skill_terms = [skill] + SKILL_ALIASES.get(clean_text(skill), [])
        matched_terms = find_matching_terms(text, skill_terms)
        if not matched_terms:
            continue

        preferred_match = skill if skill in matched_terms else matched_terms[0]
        cleaned_match = clean_text(preferred_match)
        if cleaned_match not in seen_terms:
            matches.append(preferred_match)
            seen_terms.add(cleaned_match)

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


def get_feedback_adjustment(feedback_value: str) -> float:
    """Convert user feedback into a simple score adjustment.

    This is the first step toward personalization from feedback. Later, these
    labels can become training data for a relevance classifier.
    """
    normalized_feedback = clean_text(feedback_value)
    return FEEDBACK_ADJUSTMENTS.get(normalized_feedback, 0.0)


def build_match_reason(
    matched_keywords: list[str],
    avoid_matches: list[str],
    location_matches: list[str],
    role_matches: list[str],
    job_type_matches: list[str],
    feedback_value: str,
    profile_skills_found: list[str] | None = None,
    missing_skills: list[str] | None = None,
) -> str:
    """Build a short human-readable explanation for a job score."""
    reasons = []
    profile_skills_found = profile_skills_found or []
    missing_skills = missing_skills or []

    if matched_keywords:
        top_keywords = ", ".join(matched_keywords[:4])
        reasons.append(f"Matches preferred keywords: {top_keywords}")

    if profile_skills_found:
        top_skills = ", ".join(profile_skills_found[:4])
        reasons.append(f"Profile skills found: {top_skills}")

    if missing_skills:
        top_missing = ", ".join(missing_skills[:4])
        reasons.append(f"Possible skill gaps: {top_missing}")

    if location_matches:
        reasons.append(f"Location fits preference: {location_matches[0]}")
    else:
        reasons.append("Location is not in preferred locations")

    if role_matches:
        reasons.append(f"Title matches target role: {role_matches[0]}")

    if job_type_matches:
        reasons.append(f"Job type fits preference: {job_type_matches[0]}")
    else:
        reasons.append("Job type is not preferred")

    normalized_feedback = clean_text(feedback_value)
    if normalized_feedback:
        reasons.append(f"User feedback: {normalized_feedback}")

    if avoid_matches:
        top_avoid_terms = ", ".join(avoid_matches[:3])
        reasons.append(f"Contains avoid keywords: {top_avoid_terms}")

    return "; ".join(reasons)


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
    - +10% for liked, +3% for maybe, or -30% for rejected feedback

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
    profile_skills = get_profile_list(profile, "skills")

    scored_rows = []

    for _, row in jobs.iterrows():
        full_text = row["combined_text"]
        title_text = str(row["title"])
        location_text = str(row["location"])
        job_type_text = str(row["job_type"])
        feedback_value = str(row.get("user_feedback", ""))

        matched_keywords = find_matching_terms(full_text, like_keywords)
        avoid_matches = find_matching_terms(full_text, avoid_keywords)
        location_matches = find_matching_terms(location_text, preferred_locations)
        role_matches = find_matching_terms(title_text, target_roles)
        job_type_matches = find_matching_terms(job_type_text, job_type_preferences)
        skill_gap = analyze_skill_gap(full_text, profile_skills)
        match_reason = build_match_reason(
            matched_keywords=matched_keywords,
            avoid_matches=avoid_matches,
            location_matches=location_matches,
            role_matches=role_matches,
            job_type_matches=job_type_matches,
            feedback_value=feedback_value,
            profile_skills_found=skill_gap["profile_skills_found"],
            missing_skills=skill_gap["missing_skills"],
        )

        keyword_bonus = min(len(matched_keywords) / max(len(like_keywords), 1), 1.0) * 0.15
        avoid_penalty = min(len(avoid_matches) / max(len(avoid_keywords), 1), 1.0) * 0.20
        location_bonus = 0.10 if location_matches else 0.0
        # A non-preferred required location is a serious blocker because the
        # candidate cannot realistically attend an on-site role there.
        location_penalty = 0.25 if not location_matches else 0.0
        role_bonus = 0.10 if role_matches else 0.0
        job_type_bonus = 0.05 if job_type_matches else 0.0
        job_type_penalty = 0.25 if not job_type_matches else 0.0
        feedback_adjustment = get_feedback_adjustment(feedback_value)

        raw_score = (
            row["similarity_score"] * 0.60
            + keyword_bonus
            + location_bonus
            + role_bonus
            + job_type_bonus
            + feedback_adjustment
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
                "feedback_adjustment": round(feedback_adjustment * 100, 2),
                "match_score": round(final_score, 2),
                "match_reason": match_reason,
                "profile_skills_found": ", ".join(skill_gap["profile_skills_found"]),
                "job_skills_found": ", ".join(skill_gap["job_skills_found"]),
                "missing_skills": ", ".join(skill_gap["missing_skills"]),
                "skill_match_rate": round(skill_gap["skill_match_rate"] * 100, 2),
                "matched_keywords": ", ".join(matched_keywords),
                "avoid_keywords_found": ", ".join(avoid_matches),
            }
        )
        scored_rows.append(scored_row)

    ranked_jobs = pd.DataFrame(scored_rows)
    ranked_jobs["similarity_score"] = (ranked_jobs["similarity_score"] * 100).round(2)
    ranked_jobs = ranked_jobs.sort_values("match_score", ascending=False)
    return ranked_jobs.reset_index(drop=True)
