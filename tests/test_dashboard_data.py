"""Tests for Streamlit dashboard data helpers."""

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from dashboard_data import (
    filter_dashboard_data,
    load_dashboard_data,
    load_text_report,
    parse_metric_from_report,
)


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

    def test_load_text_report_returns_empty_string_when_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "missing.txt"

            self.assertEqual(load_text_report(report_path), "")

    def test_parse_metric_from_report_extracts_metric_value(self):
        report = "- Accuracy: 0.75\n- Precision: 0.80\n"

        self.assertEqual(parse_metric_from_report(report, "Accuracy"), "0.75")
        self.assertEqual(parse_metric_from_report(report, "Recall"), "N/A")


if __name__ == "__main__":
    unittest.main()
