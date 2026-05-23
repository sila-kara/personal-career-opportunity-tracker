"""Tests for persistent feedback updates."""

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from feedback_store import load_feedback_table, upsert_feedback_entry


class TestFeedbackStore(unittest.TestCase):
    def test_upsert_feedback_entry_creates_feedback_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            feedback_path = Path(temp_dir) / "feedback.csv"

            result = upsert_feedback_entry(
                link="https://example.com/job",
                user_feedback="Liked",
                notes="Strong role",
                feedback_path=feedback_path,
            )

            self.assertTrue(feedback_path.exists())
            self.assertEqual(result.iloc[0]["user_feedback"], "liked")
            self.assertEqual(result.iloc[0]["notes"], "Strong role")

    def test_upsert_feedback_entry_updates_existing_link(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            feedback_path = Path(temp_dir) / "feedback.csv"
            upsert_feedback_entry(
                "https://example.com/job",
                "maybe",
                "Check later",
                feedback_path,
            )
            result = upsert_feedback_entry(
                "https://example.com/job",
                "rejected",
                "Not suitable",
                feedback_path,
            )

            self.assertEqual(len(result), 1)
            self.assertEqual(result.iloc[0]["user_feedback"], "rejected")

    def test_upsert_feedback_entry_rejects_invalid_feedback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            feedback_path = Path(temp_dir) / "feedback.csv"

            with self.assertRaises(ValueError):
                upsert_feedback_entry(
                    "https://example.com/job",
                    "great",
                    "",
                    feedback_path,
                )

    def test_load_feedback_table_returns_empty_table_when_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            feedback_path = Path(temp_dir) / "missing.csv"

            result = load_feedback_table(feedback_path)

            self.assertEqual(len(result), 0)
            self.assertEqual(result.columns.tolist(), ["link", "user_feedback", "notes"])


if __name__ == "__main__":
    unittest.main()
