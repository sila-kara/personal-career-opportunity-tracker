"""Data helpers for the Streamlit dashboard."""

from pathlib import Path

import pandas as pd


def load_dashboard_data(csv_path: Path) -> pd.DataFrame:
    """Load ranked opportunities for dashboard display."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dashboard data not found: {csv_path}. Run python src/main.py first."
        )

    return pd.read_csv(csv_path).fillna("")


def filter_dashboard_data(
    jobs: pd.DataFrame,
    cluster_label: str = "All",
    job_type: str = "All",
    min_score: float = 0,
) -> pd.DataFrame:
    """Filter dashboard rows by cluster, job type, and minimum hybrid score."""
    filtered_jobs = jobs.copy()

    if cluster_label != "All" and "cluster_label" in filtered_jobs.columns:
        filtered_jobs = filtered_jobs[filtered_jobs["cluster_label"] == cluster_label]

    if job_type != "All" and "job_type" in filtered_jobs.columns:
        filtered_jobs = filtered_jobs[filtered_jobs["job_type"] == job_type]

    score_column = "hybrid_score_v2" if "hybrid_score_v2" in filtered_jobs.columns else "match_score"
    filtered_jobs = filtered_jobs[filtered_jobs[score_column] >= min_score]

    return filtered_jobs.reset_index(drop=True)
