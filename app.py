"""Streamlit dashboard for the Personal Career Opportunity Tracker."""

import sys
from datetime import date
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "src"))

from config import (
    FEEDBACK_MODEL_REPORT_PATH,
    FEEDBACK_PATH,
    GOOGLE_SHEETS_READY_PATH,
    SEMANTIC_MATCHES_PATH,
)
from add_job import add_job_to_csv
from dashboard_data import (
    filter_dashboard_data,
    load_dashboard_data,
    load_text_report,
    parse_metric_from_report,
)
from feedback_store import upsert_feedback_entry


DISPLAY_COLUMNS = [
    "hybrid_score_v3",
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

FEEDBACK_OPTIONS = ["", "liked", "maybe", "rejected"]


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

    score_column = "hybrid_score_v3" if "hybrid_score_v3" in jobs.columns else "hybrid_score_v2" if "hybrid_score_v2" in jobs.columns else "match_score"

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

    st.subheader("Update Feedback")
    if filtered_jobs.empty:
        st.info("No jobs match the current filters.")
    else:
        job_options = {
            f"{row.title} | {row.company} | {row.location}": row
            for row in filtered_jobs.itertuples(index=False)
        }
        selected_job_label = st.selectbox("Select a job", list(job_options.keys()))
        selected_job = job_options[selected_job_label]

        current_feedback = getattr(selected_job, "user_feedback", "")
        current_notes = getattr(selected_job, "notes", "")
        default_feedback_index = (
            FEEDBACK_OPTIONS.index(current_feedback)
            if current_feedback in FEEDBACK_OPTIONS
            else 0
        )

        feedback_value = st.selectbox(
            "Feedback",
            FEEDBACK_OPTIONS,
            index=default_feedback_index,
        )
        notes = st.text_area("Notes", value=current_notes)

        if st.button("Save feedback"):
            upsert_feedback_entry(
                link=selected_job.link,
                user_feedback=feedback_value,
                notes=notes,
                feedback_path=FEEDBACK_PATH,
            )
            st.success("Feedback saved to data/feedback.csv.")
            st.info("Run python src/main.py to recompute scores with the new feedback.")

    st.subheader("Add New Job")
    with st.form("add_job_form", clear_on_submit=True):
        form_columns = st.columns(2)
        title = form_columns[0].text_input("Title")
        company = form_columns[1].text_input("Company")
        location = form_columns[0].text_input("Location")
        job_type = form_columns[1].selectbox(
            "Job type",
            ["Internship", "Long-term Internship", "Part-time", "Full-time", "Other"],
        )
        source = form_columns[0].text_input("Source", value="Manual Entry")
        date_found = form_columns[1].date_input("Date found", value=date.today())
        link = st.text_input("Link")
        description = st.text_area("Description")
        submitted = st.form_submit_button("Add job")

        if submitted:
            job_record = {
                "title": title,
                "company": company,
                "location": location,
                "job_type": job_type,
                "description": description,
                "link": link,
                "source": source,
                "date_found": date_found.isoformat(),
            }

            try:
                updated_jobs = add_job_to_csv(job_record)
                st.success(f"Job added. Active dataset now has {len(updated_jobs)} jobs.")
                st.info("Run python src/main.py to score the new job.")
            except ValueError as error:
                st.error(str(error))

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

    st.subheader("Feedback Model Evaluation")
    report = load_text_report(FEEDBACK_MODEL_REPORT_PATH)
    if not report:
        st.info("Run python src/train_feedback_model.py to generate the evaluation report.")
    else:
        evaluation_columns = st.columns(4)
        evaluation_columns[0].metric("Accuracy", parse_metric_from_report(report, "Accuracy"))
        evaluation_columns[1].metric("Precision", parse_metric_from_report(report, "Precision"))
        evaluation_columns[2].metric("Recall", parse_metric_from_report(report, "Recall"))
        evaluation_columns[3].metric("F1 Score", parse_metric_from_report(report, "F1 Score"))
        st.text(report)

    st.subheader("Semantic Matching Comparison")
    if not SEMANTIC_MATCHES_PATH.exists():
        st.info(
            "Optional semantic scores are not available yet. "
            "Run python src/run_semantic_matching.py after installing sentence-transformers."
        )
    else:
        semantic_jobs = load_dashboard_data(SEMANTIC_MATCHES_PATH)
        comparison_columns = [
            col
            for col in [
                "semantic_similarity_score",
                "match_score",
                "title",
                "company",
                "location",
                "job_type",
                "link",
            ]
            if col in semantic_jobs.columns
        ]
        st.dataframe(
            semantic_jobs[comparison_columns].head(10),
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
