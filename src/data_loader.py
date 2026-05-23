"""Functions for loading the personal profile and job dataset."""

from pathlib import Path

import pandas as pd
import yaml

from config import REQUIRED_JOB_COLUMNS


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
