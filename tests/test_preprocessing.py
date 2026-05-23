"""Tests for text preprocessing helpers."""

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from preprocessing import add_clean_text_columns, clean_text, combine_job_text


class TestPreprocessing(unittest.TestCase):
    def test_clean_text_lowercases_and_removes_html(self):
        text = "<b>Python &amp; SQL Internship!</b>"

        result = clean_text(text)

        self.assertEqual(result, "python sql internship")

    def test_combine_job_text_uses_core_job_fields(self):
        row = pd.Series(
            {
                "title": "Data Science Intern",
                "company": "Example AI",
                "location": "Remote",
                "job_type": "Internship",
                "description": "Use Python and SQL.",
            }
        )

        result = combine_job_text(row)

        self.assertIn("Data Science Intern", result)
        self.assertIn("Example AI", result)
        self.assertIn("Use Python and SQL.", result)

    def test_add_clean_text_columns_keeps_original_dataframe_shape(self):
        jobs = pd.DataFrame(
            [
                {
                    "title": "ML Intern",
                    "company": "Data Co",
                    "location": "Remote",
                    "job_type": "Internship",
                    "description": "Build machine learning models.",
                }
            ]
        )

        result = add_clean_text_columns(jobs)

        self.assertIn("combined_text", result.columns)
        self.assertIn("clean_text", result.columns)
        self.assertEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main()
