"""Reset the active jobs dataset from the sample dataset."""

import shutil

from config import JOBS_PATH, SAMPLE_JOBS_PATH


def reset_jobs_from_sample() -> None:
    """Copy data/sample_jobs.csv into data/jobs.csv."""
    if not SAMPLE_JOBS_PATH.exists():
        raise FileNotFoundError(f"Sample jobs file not found: {SAMPLE_JOBS_PATH}")

    JOBS_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SAMPLE_JOBS_PATH, JOBS_PATH)


def main() -> None:
    """Run the reset workflow."""
    reset_jobs_from_sample()
    print("Jobs dataset reset from sample data.")
    print(f"Sample: {SAMPLE_JOBS_PATH}")
    print(f"Active dataset: {JOBS_PATH}")


if __name__ == "__main__":
    main()
