"""Tests for active-learning feedback review queue helpers."""

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from feedback_learning import (
    build_feedback_review_queue,
    calculate_feedback_priority,
    export_feedback_review_queue,
)


class TestFeedbackLearning(unittest.TestCase):
    def test_calculate_feedback_priority_prefers_uncertain_scores(self):
        self.assertEqual(calculate_feedback_priority(50), 100)
        self.assertEqual(calculate_feedback_priority(90), 20)
        self.assertEqual(calculate_feedback_priority(10), 20)

    def test_build_feedback_review_queue_keeps_unlabeled_uncertain_jobs(self):
        jobs = pd.DataFrame(
            [
                {
                    "title": "Uncertain Data Role",
                    "predicted_relevance_score": 51,
                    "hybrid_score_v2": 40,
                    "match_score": 35,
                    "user_feedback": "",
                },
                {
                    "title": "Already Liked",
                    "predicted_relevance_score": 50,
                    "hybrid_score_v2": 90,
                    "match_score": 80,
                    "user_feedback": "liked",
                },
                {
                    "title": "Confident Bad Role",
                    "predicted_relevance_score": 5,
                    "hybrid_score_v2": 10,
                    "match_score": 10,
                    "user_feedback": "",
                },
            ]
        )

        result = build_feedback_review_queue(jobs, top_n=2)

        self.assertEqual(result.iloc[0]["title"], "Uncertain Data Role")
        self.assertNotIn("Already Liked", result["title"].tolist())

    def test_export_feedback_review_queue_writes_csv(self):
        jobs = pd.DataFrame(
            [
                {
                    "title": "Review Me",
                    "predicted_relevance_score": 50,
                    "hybrid_score_v2": 40,
                    "match_score": 35,
                    "user_feedback": "",
                }
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "queue.csv"
            result = export_feedback_review_queue(jobs, output_path)

            self.assertTrue(output_path.exists())
            self.assertEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main()
