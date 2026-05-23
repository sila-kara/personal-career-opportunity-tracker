"""Streamlit dashboard for the Personal Career Opportunity Tracker."""

import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "src"))

from config import GOOGLE_SHEETS_READY_PATH
from dashboard_data import filter_dashboard_data, load_dashboard_data


DISPLAY_COLUMNS = [
    "hybrid_score_v2",
    "match_score",
    "predicted_relevance_score",
    "title",
    "company",
    "location",
    "job_type",
    "cluster_label",
    "user_feedback",
    "profile_skills_found",
    "missing_skills",
    "match_reason",
    "link",
]


def main() -> None:
    """Render the dashboard."""
    st.set_page_config(
        page_title="Career Opportunity Tracker",
        layout="wide",
    )

    st.title("Personal Career Opportunity Tracker")
    st.caption("NLP recommendation, feedback learning, skill gaps, and clustering.")

    try:
        jobs = load_dashboard_data(GOOGLE_SHEETS_READY_PATH)
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()

    score_column = "hybrid_score_v2" if "hybrid_score_v2" in jobs.columns else "match_score"

    clusters = ["All"] + sorted(jobs.get("cluster_label", []).dropna().unique().tolist())
    job_types = ["All"] + sorted(jobs.get("job_type", []).dropna().unique().tolist())

    with st.sidebar:
        st.header("Filters")
        selected_cluster = st.selectbox("Cluster", clusters)
        selected_job_type = st.selectbox("Job type", job_types)
        min_score = st.slider("Minimum score", 0, 100, 0)

    filtered_jobs = filter_dashboard_data(
        jobs,
        cluster_label=selected_cluster,
        job_type=selected_job_type,
        min_score=min_score,
    )

    metric_columns = st.columns(4)
    metric_columns[0].metric("Total jobs", len(jobs))
    metric_columns[1].metric("Visible jobs", len(filtered_jobs))
    metric_columns[2].metric("Best score", f"{jobs[score_column].max():.2f}")
    metric_columns[3].metric("Clusters", jobs["cluster_label"].nunique())

    st.subheader("Ranked Opportunities")
    columns = [col for col in DISPLAY_COLUMNS if col in filtered_jobs.columns]
    st.dataframe(filtered_jobs[columns], use_container_width=True, hide_index=True)

    st.subheader("Cluster Summary")
    cluster_summary = (
        jobs.groupby("cluster_label")
        .agg(
            jobs=("title", "count"),
            average_score=(score_column, "mean"),
        )
        .reset_index()
        .sort_values("average_score", ascending=False)
    )
    cluster_summary["average_score"] = cluster_summary["average_score"].round(2)
    st.dataframe(cluster_summary, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
