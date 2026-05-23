"""Simple text cleaning helpers for job matching."""

import re
from html import unescape

import pandas as pd


def clean_text(text: str) -> str:
    """Lowercase text and remove noisy punctuation/spacing.

    This is intentionally simple for the MVP. Later, you could add stemming,
    lemmatization, stopword removal, or domain-specific phrase handling.
    """
    text = unescape(str(text)).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9+#.\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def combine_job_text(row: pd.Series) -> str:
    """Combine the most useful fields into one searchable text field."""
    return " ".join(
        [
            str(row.get("title", "")),
            str(row.get("company", "")),
            str(row.get("location", "")),
            str(row.get("job_type", "")),
            str(row.get("description", "")),
        ]
    )


def add_clean_text_columns(jobs: pd.DataFrame) -> pd.DataFrame:
    """Add raw combined text and cleaned combined text columns."""
    jobs = jobs.copy()
    jobs["combined_text"] = jobs.apply(combine_job_text, axis=1)
    jobs["clean_text"] = jobs["combined_text"].apply(clean_text)
    return jobs
