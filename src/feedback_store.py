"""Read and update persistent user feedback."""

from pathlib import Path

import pandas as pd

from config import FEEDBACK_PATH, REQUIRED_FEEDBACK_COLUMNS


ALLOWED_FEEDBACK_VALUES = {"", "liked", "maybe", "rejected"}


def normalize_feedback_value(value: object) -> str:
    """Normalize feedback labels before saving."""
    return str(value).strip().lower()


def validate_feedback_value(value: str) -> None:
    """Validate a feedback label."""
    if value not in ALLOWED_FEEDBACK_VALUES:
        valid_values = ", ".join(sorted(ALLOWED_FEEDBACK_VALUES - {""}))
        raise ValueError(f"Invalid feedback value '{value}'. Use: {valid_values}.")


def load_feedback_table(feedback_path: Path = FEEDBACK_PATH) -> pd.DataFrame:
    """Load feedback CSV or return an empty feedback table."""
    if not feedback_path.exists():
        return pd.DataFrame(columns=REQUIRED_FEEDBACK_COLUMNS)

    feedback = pd.read_csv(feedback_path).fillna("")
    missing_columns = [
        column for column in REQUIRED_FEEDBACK_COLUMNS if column not in feedback.columns
    ]

    if missing_columns:
        raise ValueError(
            "Feedback CSV is missing required columns: " + ", ".join(missing_columns)
        )

    return feedback[REQUIRED_FEEDBACK_COLUMNS]


def upsert_feedback_entry(
    link: str,
    user_feedback: str,
    notes: str = "",
    feedback_path: Path = FEEDBACK_PATH,
) -> pd.DataFrame:
    """Insert or update feedback for a job link."""
    normalized_feedback = normalize_feedback_value(user_feedback)
    validate_feedback_value(normalized_feedback)

    link = str(link).strip()
    if not link:
        raise ValueError("Job link is required to save feedback.")

    feedback = load_feedback_table(feedback_path)
    feedback = feedback[feedback["link"].astype(str).str.strip() != link]

    new_row = pd.DataFrame(
        [
            {
                "link": link,
                "user_feedback": normalized_feedback,
                "notes": str(notes).strip(),
            }
        ],
        columns=REQUIRED_FEEDBACK_COLUMNS,
    )
    updated_feedback = pd.concat([feedback, new_row], ignore_index=True)

    feedback_path.parent.mkdir(parents=True, exist_ok=True)
    updated_feedback.to_csv(feedback_path, index=False)

    return updated_feedback
