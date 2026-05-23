"""Hybrid scoring that combines explainable rules with feedback learning."""

import pandas as pd

from feedback_model import (
    predict_relevance,
    prepare_training_data,
    train_feedback_classifier,
)


MATCH_SCORE_WEIGHT = 0.70
ML_SCORE_WEIGHT = 0.30


def calculate_hybrid_score(
    match_score: float,
    predicted_relevance_score: float,
    match_score_weight: float = MATCH_SCORE_WEIGHT,
) -> float:
    """Combine rule/NLP match score with feedback-model relevance score."""
    ml_score_weight = 1 - match_score_weight
    hybrid_score = (
        match_score * match_score_weight
        + predicted_relevance_score * ml_score_weight
    )
    return round(float(hybrid_score), 2)


def calculate_hybrid_score_v3(
    match_score: float,
    predicted_relevance_score: float,
    semantic_similarity_score: float | None = None,
) -> float:
    """Combine match, feedback, and optional semantic similarity scores."""
    if predicted_relevance_score == "":
        return round(float(match_score), 2)

    if semantic_similarity_score is None or semantic_similarity_score == "":
        return calculate_hybrid_score(match_score, predicted_relevance_score)

    hybrid_score = (
        match_score * 0.55
        + predicted_relevance_score * 0.25
        + float(semantic_similarity_score) * 0.20
    )
    return round(float(hybrid_score), 2)


def add_hybrid_scores(jobs: pd.DataFrame) -> pd.DataFrame:
    """Add feedback-model predictions and hybrid_score_v2 when possible.

    If there is not enough feedback to train the model, the function keeps the
    pipeline running and falls back to match_score for hybrid_score_v2.
    """
    jobs_with_scores = jobs.copy()

    try:
        training_data = prepare_training_data(jobs_with_scores)
        classifier = train_feedback_classifier(training_data)
        jobs_with_scores = predict_relevance(jobs_with_scores, classifier)
        jobs_with_scores["hybrid_score_v2"] = jobs_with_scores.apply(
            lambda row: calculate_hybrid_score(
                row["match_score"],
                row["predicted_relevance_score"],
            ),
            axis=1,
        )
        jobs_with_scores["hybrid_score_note"] = (
            "70% match_score + 30% feedback_model_score"
        )
    except ValueError as error:
        jobs_with_scores["predicted_relevance_score"] = ""
        jobs_with_scores["hybrid_score_v2"] = jobs_with_scores["match_score"]
        jobs_with_scores["hybrid_score_note"] = f"Fallback to match_score: {error}"

    jobs_with_scores["hybrid_score_v3"] = jobs_with_scores.apply(
        lambda row: calculate_hybrid_score_v3(
            row["match_score"],
            row["predicted_relevance_score"],
            row.get("semantic_similarity_score", None),
        ),
        axis=1,
    )
    has_semantic_scores = "semantic_similarity_score" in jobs_with_scores.columns
    jobs_with_scores["hybrid_score_v3_note"] = (
        "55% match_score + 25% feedback_model_score + 20% semantic_similarity_score"
        if has_semantic_scores
        else "Fallback to hybrid_score_v2 because semantic scores are unavailable"
    )

    jobs_with_scores = jobs_with_scores.sort_values(
        "hybrid_score_v3", ascending=False
    )
    return jobs_with_scores.reset_index(drop=True)
