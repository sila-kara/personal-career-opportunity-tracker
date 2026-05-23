"""Train a small feedback-based relevance classifier.

This module is intentionally optional. The main ranking system works without it,
but this gives the project a first step toward learning from user feedback.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


RELEVANT_FEEDBACK = {"liked", "maybe"}
NOT_RELEVANT_FEEDBACK = {"rejected"}


def feedback_to_label(feedback_value: object) -> int | None:
    """Map feedback text to a binary relevance label."""
    normalized_feedback = str(feedback_value).strip().lower()

    if normalized_feedback in RELEVANT_FEEDBACK:
        return 1
    if normalized_feedback in NOT_RELEVANT_FEEDBACK:
        return 0
    return None


def prepare_training_data(jobs: pd.DataFrame) -> pd.DataFrame:
    """Keep only jobs with feedback labels that can train the classifier."""
    training_data = jobs.copy()
    training_data["relevance_label"] = training_data["user_feedback"].apply(
        feedback_to_label
    )
    training_data = training_data.dropna(subset=["relevance_label"])
    training_data["relevance_label"] = training_data["relevance_label"].astype(int)

    return training_data


def validate_training_data(training_data: pd.DataFrame) -> None:
    """Make sure there is enough feedback to train a binary classifier."""
    label_counts = training_data["relevance_label"].value_counts()

    if len(training_data) < 4:
        raise ValueError("At least 4 feedback-labeled jobs are needed for training.")

    if set(label_counts.index) != {0, 1}:
        raise ValueError("Training data must include both relevant and rejected jobs.")


def build_classifier() -> Pipeline:
    """Create a simple TF-IDF + logistic regression text classifier."""
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def train_feedback_classifier(training_data: pd.DataFrame) -> Pipeline:
    """Train a classifier from feedback-labeled job postings."""
    validate_training_data(training_data)

    classifier = build_classifier()
    classifier.fit(training_data["clean_text"], training_data["relevance_label"])

    return classifier


def predict_relevance(jobs: pd.DataFrame, classifier: Pipeline) -> pd.DataFrame:
    """Add predicted relevance probabilities to the full jobs table."""
    jobs_with_predictions = jobs.copy()
    probabilities = classifier.predict_proba(jobs_with_predictions["clean_text"])[:, 1]

    jobs_with_predictions["predicted_relevance_score"] = (probabilities * 100).round(2)
    jobs_with_predictions = jobs_with_predictions.sort_values(
        "predicted_relevance_score", ascending=False
    )

    return jobs_with_predictions.reset_index(drop=True)
