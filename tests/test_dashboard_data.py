"""Tests for Streamlit dashboard data helpers."""

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from dashboard_data import filter_dashboard_data, load_dashboard_data


class TestDashboardData(unittest.TestCase):
    def test_load_dashboard_data_reads_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "dashboard.csv"
            pd.DataFrame([{"title": "Data Intern"}]).to_csv(csv_path, index=False)

            result = load_dashboard_data(csv_path)

            self.assertEqual(result.iloc[0]["title"], "Data Intern")

    def test_filter_dashboard_data_filters_cluster_type_and_score(self):
        jobs = pd.DataFrame(
            [
                {
                    "title": "Data Intern",
                    "cluster_label": "Data & Analytics",
                    "job_type": "Internship",
                    "hybrid_score_v2": 80,
                },
                {
                    "title": "Sales Intern",
                    "cluster_label": "Sales & Marketing",
                    "job_type": "Internship",
                    "hybrid_score_v2": 20,
                },
            ]
        )

        result = filter_dashboard_data(
            jobs,
            cluster_label="Data & Analytics",
            job_type="Internship",
            min_score=50,
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["title"], "Data Intern")


if __name__ == "__main__":
    unittest.main()
