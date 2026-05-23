"""Functions for loading the personal profile and job dataset."""

from pathlib import Path

import pandas as pd
import yaml

from config import REQUIRED_FEEDBACK_COLUMNS, REQUIRED_JOB_COLUMNS


def load_profile(profile_path: Path) -> dict:
    """Load the user's career profile from a YAML file."""
    if not profile_path.exists():
        raise FileNotFoundError(f"Profile file not found: {profile_path}")

    with profile_path.open("r", encoding="utf-8") as file:
        profile = yaml.safe_load(file)

    if not isinstance(profile, dict):
        raise ValueError("Profile file must contain a YAML dictionary.")

    return profile


def load_jobs(jobs_path: Path) -> pd.DataFrame:
    """Load job postings from CSV and check that expected columns exist."""
    if not jobs_path.exists():
        raise FileNotFoundError(f"Jobs file not found: {jobs_path}")

    jobs = pd.read_csv(jobs_path)
    missing_columns = [col for col in REQUIRED_JOB_COLUMNS if col not in jobs.columns]

    if missing_columns:
        raise ValueError(
            "Jobs CSV is missing required columns: " + ", ".join(missing_columns)
        )

    return jobs.fillna("")


def load_feedback(feedback_path: Path) -> pd.DataFrame:
    """Load optional user feedback from CSV.

    Feedback is kept separate from generated outputs so it is not lost when the
    ranking pipeline runs again.
    """
    if not feedback_path.exists():
        return pd.DataFrame(columns=REQUIRED_FEEDBACK_COLUMNS)

    feedback = pd.read_csv(feedback_path)
    missing_columns = [
        col for col in REQUIRED_FEEDBACK_COLUMNS if col not in feedback.columns
    ]

    if missing_columns:
        raise ValueError(
            "Feedback CSV is missing required columns: " + ", ".join(missing_columns)
        )

    return feedback[REQUIRED_FEEDBACK_COLUMNS].fillna("")


def merge_feedback(jobs: pd.DataFrame, feedback: pd.DataFrame) -> pd.DataFrame:
    """Attach user feedback to jobs using the job link as a stable key."""
    if feedback.empty:
        jobs = jobs.copy()
        jobs["user_feedback"] = ""
        jobs["notes"] = ""
        return jobs

    feedback = feedback.drop_duplicates(subset=["link"], keep="last")
    jobs_with_feedback = jobs.merge(feedback, on="link", how="left")
    jobs_with_feedback[["user_feedback", "notes"]] = jobs_with_feedback[
        ["user_feedback", "notes"]
    ].fillna("")

    return jobs_with_feedback
