"""Tests for hybrid score v2."""

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from hybrid_scorer import (
    add_hybrid_scores,
    calculate_hybrid_score,
    calculate_hybrid_score_v3,
)


class TestHybridScorer(unittest.TestCase):
    def test_calculate_hybrid_score_combines_two_scores(self):
        result = calculate_hybrid_score(80, 50)

        self.assertEqual(result, 71.0)

    def test_calculate_hybrid_score_v3_uses_semantic_score_when_available(self):
        result = calculate_hybrid_score_v3(80, 50, 90)

        self.assertEqual(result, 74.5)

    def test_calculate_hybrid_score_v3_falls_back_without_semantic_score(self):
        result = calculate_hybrid_score_v3(80, 50)

        self.assertEqual(result, 71.0)

    def test_add_hybrid_scores_uses_feedback_model_when_possible(self):
        jobs = pd.DataFrame(
            [
                {
                    "match_score": 80,
                    "clean_text": "python sql data science internship",
                    "user_feedback": "liked",
                },
                {
                    "match_score": 70,
                    "clean_text": "flutter dart mobile application part-time",
                    "user_feedback": "liked",
                },
                {
                    "match_score": 20,
                    "clean_text": "sales outreach crm calls",
                    "user_feedback": "rejected",
                },
                {
                    "match_score": 10,
                    "clean_text": "full-time graduate senior role",
                    "user_feedback": "rejected",
                },
            ]
        )

        result = add_hybrid_scores(jobs)

        self.assertIn("predicted_relevance_score", result.columns)
        self.assertIn("hybrid_score_v2", result.columns)
        self.assertIn("hybrid_score_v3", result.columns)
        self.assertIn("hybrid_score_note", result.columns)
        self.assertTrue((result["hybrid_score_v2"] >= 0).all())
        self.assertTrue((result["hybrid_score_v2"] <= 100).all())

    def test_add_hybrid_scores_falls_back_when_feedback_is_insufficient(self):
        jobs = pd.DataFrame(
            [
                {
                    "match_score": 50,
                    "clean_text": "python internship",
                    "user_feedback": "",
                }
            ]
        )

        result = add_hybrid_scores(jobs)

        self.assertEqual(result.iloc[0]["hybrid_score_v2"], 50)
        self.assertEqual(result.iloc[0]["hybrid_score_v3"], 50)
        self.assertEqual(result.iloc[0]["predicted_relevance_score"], "")
        self.assertIn("Fallback to match_score", result.iloc[0]["hybrid_score_note"])


if __name__ == "__main__":
    unittest.main()
