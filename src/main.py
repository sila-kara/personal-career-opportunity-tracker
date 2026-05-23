"""Run the Personal Career Opportunity Tracker MVP."""

from config import (
    FEEDBACK_PATH,
    FEEDBACK_REVIEW_QUEUE_PATH,
    GOOGLE_SHEETS_READY_PATH,
    JOBS_PATH,
    OUTPUT_CSV_PATH,
    OUTPUT_MD_PATH,
    PROFILE_PATH,
)
from clustering import add_job_clusters
from data_loader import load_feedback, load_jobs, load_profile, merge_feedback
from exporter import export_google_sheets_ready_csv, export_to_csv, export_to_markdown
from feedback_learning import export_feedback_review_queue
from hybrid_scorer import add_hybrid_scores
from matcher import score_jobs
from preprocessing import add_clean_text_columns


def main() -> None:
    """Load data, score jobs, and export ranked results."""
    profile = load_profile(PROFILE_PATH)
    jobs = load_jobs(JOBS_PATH)
    feedback = load_feedback(FEEDBACK_PATH)

    jobs = add_clean_text_columns(jobs)
    jobs = merge_feedback(jobs, feedback)
    ranked_jobs = score_jobs(profile, jobs)
    ranked_jobs = add_hybrid_scores(ranked_jobs)
    ranked_jobs = add_job_clusters(ranked_jobs)

    export_to_csv(ranked_jobs, OUTPUT_CSV_PATH)
    export_to_markdown(ranked_jobs, OUTPUT_MD_PATH)
    export_google_sheets_ready_csv(ranked_jobs, GOOGLE_SHEETS_READY_PATH)
    feedback_queue = export_feedback_review_queue(
        ranked_jobs,
        FEEDBACK_REVIEW_QUEUE_PATH,
    )

    print("Career opportunity matching complete.")
    print(f"Jobs processed: {len(ranked_jobs)}")
    print(f"CSV output: {OUTPUT_CSV_PATH}")
    print(f"Markdown output: {OUTPUT_MD_PATH}")
    print(f"Google Sheets-ready output: {GOOGLE_SHEETS_READY_PATH}")
    print(f"Feedback review queue: {FEEDBACK_REVIEW_QUEUE_PATH}")
    print(f"Feedback review suggestions: {len(feedback_queue)}")
    print("\nTop matches:")

    preview_columns = [
        "hybrid_score_v3",
        "hybrid_score_v2",
        "match_score",
        "title",
        "company",
        "location",
        "cluster_label",
    ]
    print(ranked_jobs[preview_columns].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
