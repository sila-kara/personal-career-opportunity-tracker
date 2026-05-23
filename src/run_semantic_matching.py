"""Run optional sentence-transformers semantic matching."""

from config import FEEDBACK_PATH, JOBS_PATH, PROFILE_PATH, SEMANTIC_MATCHES_PATH
from data_loader import load_feedback, load_jobs, load_profile, merge_feedback
from matcher import score_jobs
from preprocessing import add_clean_text_columns
from semantic_matcher import add_semantic_scores


SEMANTIC_OUTPUT_COLUMNS = [
    "semantic_similarity_score",
    "match_score",
    "title",
    "company",
    "location",
    "job_type",
    "match_reason",
    "link",
]


def main() -> None:
    """Run semantic matching and export semantic scores."""
    profile = load_profile(PROFILE_PATH)
    jobs = load_jobs(JOBS_PATH)
    feedback = load_feedback(FEEDBACK_PATH)

    jobs = add_clean_text_columns(jobs)
    jobs = merge_feedback(jobs, feedback)
    scored_jobs = score_jobs(profile, jobs)

    try:
        semantic_jobs = add_semantic_scores(profile, scored_jobs)
    except ImportError as error:
        print(error)
        print("Skipping semantic matching. The main TF-IDF pipeline still works.")
        return

    columns = [col for col in SEMANTIC_OUTPUT_COLUMNS if col in semantic_jobs.columns]
    SEMANTIC_MATCHES_PATH.parent.mkdir(parents=True, exist_ok=True)
    semantic_jobs[columns].to_csv(SEMANTIC_MATCHES_PATH, index=False)

    print("Semantic matching complete.")
    print(f"Semantic matches output: {SEMANTIC_MATCHES_PATH}")
    print("\nTop semantic matches:")
    print(semantic_jobs[columns].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
