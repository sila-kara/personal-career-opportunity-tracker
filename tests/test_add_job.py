"""Tests for adding jobs to the local dataset."""

import argparse
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from add_job import add_job_to_csv, build_job_record
from config import REQUIRED_JOB_COLUMNS


class TestAddJob(unittest.TestCase):
    def test_build_job_record_strips_values(self):
        args = argparse.Namespace(
            title=" Data Intern ",
            company=" Example Co ",
            location=" Remote ",
            job_type=" Internship ",
            description=" Use Python. ",
            link=" https://example.com/job ",
            source=" Manual Entry ",
            date_found=" 2026-05-23 ",
        )

        result = build_job_record(args)

        self.assertEqual(result["title"], "Data Intern")
        self.assertEqual(result["link"], "https://example.com/job")

    def test_add_job_to_csv_appends_record(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            jobs_path = Path(temp_dir) / "jobs.csv"
            pd.DataFrame(columns=REQUIRED_JOB_COLUMNS).to_csv(jobs_path, index=False)
            job_record = {
                "title": "Data Intern",
                "company": "Example Co",
                "location": "Remote",
                "job_type": "Internship",
                "description": "Use Python and SQL.",
                "link": "https://example.com/job",
                "source": "Test",
                "date_found": "2026-05-23",
            }

            result = add_job_to_csv(job_record, jobs_path)

            self.assertEqual(len(result), 1)
            self.assertEqual(result.iloc[0]["title"], "Data Intern")

    def test_add_job_to_csv_rejects_duplicate_link(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            jobs_path = Path(temp_dir) / "jobs.csv"
            job_record = {
                "title": "Data Intern",
                "company": "Example Co",
                "location": "Remote",
                "job_type": "Internship",
                "description": "Use Python and SQL.",
                "link": "https://example.com/job",
                "source": "Test",
                "date_found": "2026-05-23",
            }
            pd.DataFrame([job_record], columns=REQUIRED_JOB_COLUMNS).to_csv(
                jobs_path, index=False
            )

            with self.assertRaises(ValueError):
                add_job_to_csv(job_record, jobs_path)


if __name__ == "__main__":
    unittest.main()
