"""Project paths and shared settings."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROFILE_PATH = PROJECT_ROOT / "profile.yaml"
JOBS_PATH = PROJECT_ROOT / "data" / "sample_jobs.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_CSV_PATH = OUTPUT_DIR / "matched_jobs.csv"
OUTPUT_MD_PATH = OUTPUT_DIR / "matched_jobs.md"

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
