"""Tests for updating feedback from a Sheets-style CSV."""

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from update_feedback import build_feedback_table


class TestUpdateFeedback(unittest.TestCase):
    def test_build_feedback_table_keeps_only_rows_with_feedback_or_notes(self):
        sheets_data = pd.DataFrame(
            [
                {
                    "link": "https://example.com/liked",
                    "user_feedback": "Liked",
                    "notes": "Strong fit",
                },
                {
                    "link": "https://example.com/empty",
                    "user_feedback": "",
                    "notes": "",
                },
                {
                    "link": "https://example.com/note-only",
                    "user_feedback": "",
                    "notes": "Check later",
                },
            ]
        )

        result = build_feedback_table(sheets_data)

        self.assertEqual(len(result), 2)
        self.assertIn("liked", result["user_feedback"].tolist())
        self.assertIn("https://example.com/note-only", result["link"].tolist())

    def test_build_feedback_table_rejects_invalid_feedback_values(self):
        sheets_data = pd.DataFrame(
            [
                {
                    "link": "https://example.com/bad",
                    "user_feedback": "great",
                    "notes": "",
                }
            ]
        )

        with self.assertRaises(ValueError):
            build_feedback_table(sheets_data)


if __name__ == "__main__":
    unittest.main()
