"""Tests for unsupervised job clustering."""

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from clustering import (
    add_job_clusters,
    build_cluster_label,
    choose_cluster_count,
    infer_cluster_label,
)


class TestClustering(unittest.TestCase):
    def test_choose_cluster_count_handles_small_datasets(self):
        self.assertEqual(choose_cluster_count(0), 1)
        self.assertEqual(choose_cluster_count(1), 1)
        self.assertEqual(choose_cluster_count(3, requested_clusters=5), 3)
        self.assertEqual(choose_cluster_count(10, requested_clusters=5), 5)

    def test_build_cluster_label_uses_top_terms(self):
        result = build_cluster_label(["python", "data science", "internship"])

        self.assertEqual(result, "Python / Data Science / Internship")

    def test_infer_cluster_label_uses_category_signals(self):
        cluster_text = "python sql data analytics dashboard reporting"

        result = infer_cluster_label(cluster_text, ["python", "sql"])

        self.assertEqual(result, "Data & Analytics")

    def test_add_job_clusters_adds_cluster_columns(self):
        jobs = pd.DataFrame(
            [
                {"clean_text": "python sql data science internship"},
                {"clean_text": "machine learning python model evaluation"},
                {"clean_text": "flutter dart mobile application"},
                {"clean_text": "java backend sql software development"},
            ]
        )

        result = add_job_clusters(jobs, n_clusters=2)

        self.assertIn("job_cluster", result.columns)
        self.assertIn("cluster_label", result.columns)
        self.assertEqual(len(result), 4)
        self.assertEqual(result["job_cluster"].nunique(), 2)


if __name__ == "__main__":
    unittest.main()
