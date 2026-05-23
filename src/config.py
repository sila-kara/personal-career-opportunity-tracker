"""Project paths and shared settings."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROFILE_PATH = PROJECT_ROOT / "profile.yaml"
SAMPLE_JOBS_PATH = PROJECT_ROOT / "data" / "sample_jobs.csv"
JOBS_PATH = PROJECT_ROOT / "data" / "jobs.csv"
FEEDBACK_PATH = PROJECT_ROOT / "data" / "feedback.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_CSV_PATH = OUTPUT_DIR / "matched_jobs.csv"
OUTPUT_MD_PATH = OUTPUT_DIR / "matched_jobs.md"
GOOGLE_SHEETS_READY_PATH = OUTPUT_DIR / "google_sheets_ready.csv"
FEEDBACK_PREDICTIONS_PATH = OUTPUT_DIR / "feedback_model_predictions.csv"
FEEDBACK_MODEL_REPORT_PATH = OUTPUT_DIR / "feedback_model_report.txt"
SEMANTIC_MATCHES_PATH = OUTPUT_DIR / "semantic_matches.csv"
FEEDBACK_REVIEW_QUEUE_PATH = OUTPUT_DIR / "feedback_review_queue.csv"

REQUIRED_JOB_COLUMNS = [
    "title",
    "company",
    "location",
    "job_type",
    "description",
    "link",
    "source",
    "date_found",
]

REQUIRED_FEEDBACK_COLUMNS = [
    "link",
    "user_feedback",
    "notes",
]
