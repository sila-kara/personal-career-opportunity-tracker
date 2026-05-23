"""Tests for resetting active jobs from sample data."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from reset_jobs_from_sample import reset_jobs_from_sample


class TestResetJobsFromSample(unittest.TestCase):
    def test_reset_jobs_from_sample_copies_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            sample_path = temp_path / "sample_jobs.csv"
            jobs_path = temp_path / "jobs.csv"
            sample_path.write_text("title,company\nData Intern,Example\n", encoding="utf-8")

            with patch("reset_jobs_from_sample.SAMPLE_JOBS_PATH", sample_path):
                with patch("reset_jobs_from_sample.JOBS_PATH", jobs_path):
                    reset_jobs_from_sample()

            self.assertEqual(
                jobs_path.read_text(encoding="utf-8"),
                "title,company\nData Intern,Example\n",
            )


if __name__ == "__main__":
    unittest.main()
