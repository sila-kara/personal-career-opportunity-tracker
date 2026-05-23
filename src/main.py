"""Run the Personal Career Opportunity Tracker MVP."""

from config import FEEDBACK_PATH, JOBS_PATH, OUTPUT_CSV_PATH, OUTPUT_MD_PATH, PROFILE_PATH
from data_loader import load_feedback, load_jobs, load_profile, merge_feedback
from exporter import export_to_csv, export_to_markdown
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

    export_to_csv(ranked_jobs, OUTPUT_CSV_PATH)
    export_to_markdown(ranked_jobs, OUTPUT_MD_PATH)

    print("Career opportunity matching complete.")
    print(f"Jobs processed: {len(ranked_jobs)}")
    print(f"CSV output: {OUTPUT_CSV_PATH}")
    print(f"Markdown output: {OUTPUT_MD_PATH}")
    print("\nTop matches:")

    preview_columns = ["match_score", "title", "company", "location"]
    print(ranked_jobs[preview_columns].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
