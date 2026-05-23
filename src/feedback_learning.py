"""Active-learning style helpers for collecting better user feedback."""

import pandas as pd


REVIEW_QUEUE_COLUMNS = [
    "feedback_priority_score",
    "predicted_relevance_score",
    "hybrid_score_v2",
    "match_score",
    "title",
    "company",
    "location",
    "job_type",
    "cluster_label",
    "match_reason",
    "link",
]


def calculate_feedback_priority(predicted_relevance_score: float) -> float:
    """Score how useful a new feedback label would be.

    Scores near 50 are more uncertain, so they receive higher priority.
    """
    uncertainty = 100 - abs(float(predicted_relevance_score) - 50) * 2
    return round(max(0.0, uncertainty), 2)


def build_feedback_review_queue(jobs: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Suggest unlabeled jobs that would be useful to review next."""
    if "predicted_relevance_score" not in jobs.columns:
        return pd.DataFrame(columns=REVIEW_QUEUE_COLUMNS)

    candidates = jobs.copy()
    if "user_feedback" not in candidates.columns:
        candidates["user_feedback"] = ""
    candidates["user_feedback"] = candidates["user_feedback"].fillna("")
    candidates = candidates[candidates["user_feedback"].astype(str).str.strip() == ""]

    if candidates.empty:
        return pd.DataFrame(columns=REVIEW_QUEUE_COLUMNS)

    candidates["feedback_priority_score"] = candidates["predicted_relevance_score"].apply(
        calculate_feedback_priority
    )
    candidates = candidates.sort_values(
        ["feedback_priority_score", "hybrid_score_v2"],
        ascending=[False, False],
    )

    columns = [col for col in REVIEW_QUEUE_COLUMNS if col in candidates.columns]
    return candidates.head(top_n)[columns].reset_index(drop=True)


def export_feedback_review_queue(
    jobs: pd.DataFrame,
    output_path,
    top_n: int = 10,
) -> pd.DataFrame:
    """Create and save a feedback review queue."""
    queue = build_feedback_review_queue(jobs, top_n=top_n)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    queue.to_csv(output_path, index=False)
    return queue
