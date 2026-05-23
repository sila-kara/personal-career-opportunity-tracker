"""Update data/feedback.csv from an edited Google Sheets-style CSV export."""

import argparse
from pathlib import Path

import pandas as pd

from config import FEEDBACK_PATH, GOOGLE_SHEETS_READY_PATH, REQUIRED_FEEDBACK_COLUMNS


ALLOWED_FEEDBACK_VALUES = {"", "liked", "maybe", "rejected"}


def normalize_feedback(value: object) -> str:
    """Normalize user feedback labels to simple lowercase values."""
    return str(value).strip().lower()


def load_sheets_export(input_path: Path) -> pd.DataFrame:
    """Load a CSV exported from Google Sheets and validate required columns."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    sheets_data = pd.read_csv(input_path).fillna("")
    missing_columns = [
        col for col in REQUIRED_FEEDBACK_COLUMNS if col not in sheets_data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Sheets CSV is missing required columns: " + ", ".join(missing_columns)
        )

    return sheets_data


def build_feedback_table(sheets_data: pd.DataFrame) -> pd.DataFrame:
    """Extract and validate feedback columns from the Sheets export."""
    feedback = sheets_data[REQUIRED_FEEDBACK_COLUMNS].copy()
    feedback["user_feedback"] = feedback["user_feedback"].apply(normalize_feedback)
    feedback["notes"] = feedback["notes"].astype(str).str.strip()
    feedback["link"] = feedback["link"].astype(str).str.strip()

    invalid_feedback = sorted(
        set(feedback["user_feedback"]) - ALLOWED_FEEDBACK_VALUES
    )
    if invalid_feedback:
        valid_values = ", ".join(sorted(ALLOWED_FEEDBACK_VALUES - {""}))
        raise ValueError(
            "Invalid feedback values found: "
            + ", ".join(invalid_feedback)
            + f". Use only: {valid_values}."
        )

    has_feedback_or_notes = (feedback["user_feedback"] != "") | (feedback["notes"] != "")
    feedback = feedback[has_feedback_or_notes]
    feedback = feedback[feedback["link"] != ""]
    feedback = feedback.drop_duplicates(subset=["link"], keep="last")

    return feedback


def save_feedback(feedback: pd.DataFrame, output_path: Path) -> None:
    """Save the cleaned feedback table."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feedback.to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Update data/feedback.csv from an edited Google Sheets CSV."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default=str(GOOGLE_SHEETS_READY_PATH),
        help="Path to the edited Google Sheets CSV export.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the feedback update workflow."""
    args = parse_args()
    input_path = Path(args.input_path)

    sheets_data = load_sheets_export(input_path)
    feedback = build_feedback_table(sheets_data)
    save_feedback(feedback, FEEDBACK_PATH)

    print("Feedback update complete.")
    print(f"Input file: {input_path}")
    print(f"Feedback rows saved: {len(feedback)}")
    print(f"Feedback output: {FEEDBACK_PATH}")


if __name__ == "__main__":
    main()
