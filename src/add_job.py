"""Add a new job posting to the local jobs CSV."""

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from config import JOBS_PATH, REQUIRED_JOB_COLUMNS


ALLOWED_JOB_TYPES = {
    "Internship",
    "Long-term Internship",
    "Part-time",
    "Full-time",
    "Other",
}


def build_job_record(args: argparse.Namespace) -> dict:
    """Create a job record in the same structure as data/sample_jobs.csv."""
    return {
        "title": args.title.strip(),
        "company": args.company.strip(),
        "location": args.location.strip(),
        "job_type": args.job_type.strip(),
        "description": args.description.strip(),
        "link": args.link.strip(),
        "source": args.source.strip(),
        "date_found": args.date_found.strip(),
    }


def validate_job_record(job_record: dict) -> None:
    """Check that all required fields are present before saving."""
    missing_values = [
        column for column in REQUIRED_JOB_COLUMNS if not str(job_record.get(column, "")).strip()
    ]

    if missing_values:
        raise ValueError("Missing required values: " + ", ".join(missing_values))

    if job_record["job_type"] not in ALLOWED_JOB_TYPES:
        valid_types = ", ".join(sorted(ALLOWED_JOB_TYPES))
        raise ValueError(f"Invalid job_type '{job_record['job_type']}'. Use: {valid_types}.")

    if not str(job_record["link"]).startswith(("http://", "https://")):
        raise ValueError("Job link must start with http:// or https://.")

    try:
        date.fromisoformat(str(job_record["date_found"]))
    except ValueError as error:
        raise ValueError("date_found must use YYYY-MM-DD format.") from error


def add_job_to_csv(job_record: dict, jobs_path: Path = JOBS_PATH) -> pd.DataFrame:
    """Append a job posting to the jobs CSV and return the updated DataFrame."""
    validate_job_record(job_record)

    if jobs_path.exists():
        jobs = pd.read_csv(jobs_path).fillna("")
    else:
        jobs = pd.DataFrame(columns=REQUIRED_JOB_COLUMNS)

    missing_columns = [col for col in REQUIRED_JOB_COLUMNS if col not in jobs.columns]
    if missing_columns:
        raise ValueError(
            "Jobs CSV is missing required columns: " + ", ".join(missing_columns)
        )

    if job_record["link"] in set(jobs["link"].astype(str)):
        raise ValueError("A job with this link already exists in the dataset.")

    updated_jobs = pd.concat(
        [jobs, pd.DataFrame([job_record], columns=REQUIRED_JOB_COLUMNS)],
        ignore_index=True,
    )
    updated_jobs.to_csv(jobs_path, index=False)

    return updated_jobs


def parse_args() -> argparse.Namespace:
    """Parse command-line inputs for a new job posting."""
    parser = argparse.ArgumentParser(description="Add a job posting to data/sample_jobs.csv.")
    parser.add_argument("--title", required=True, help="Job title.")
    parser.add_argument("--company", required=True, help="Company name.")
    parser.add_argument("--location", required=True, help="Job location.")
    parser.add_argument("--job-type", required=True, help="Internship, Part-time, Full-time, etc.")
    parser.add_argument("--description", required=True, help="Job description text.")
    parser.add_argument("--link", required=True, help="Unique job posting link.")
    parser.add_argument("--source", default="Manual Entry", help="Where the job was found.")
    parser.add_argument(
        "--date-found",
        default=date.today().isoformat(),
        help="Date the job was found in YYYY-MM-DD format.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the add-job workflow."""
    args = parse_args()
    job_record = build_job_record(args)
    updated_jobs = add_job_to_csv(job_record)

    print("Job added successfully.")
    print(f"Dataset path: {JOBS_PATH}")
    print(f"Total jobs: {len(updated_jobs)}")
    print("Run python src/main.py to re-score the updated dataset.")


if __name__ == "__main__":
    main()
