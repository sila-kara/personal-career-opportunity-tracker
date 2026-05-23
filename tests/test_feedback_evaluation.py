"""Tests for feedback model evaluation reports."""

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from feedback_evaluation import evaluate_with_leave_one_out, format_evaluation_report


class TestFeedbackEvaluation(unittest.TestCase):
    def test_evaluate_with_leave_one_out_returns_metrics(self):
        training_data = pd.DataFrame(
            [
                {
                    "clean_text": "python sql data science internship",
                    "relevance_label": 1,
                },
                {
                    "clean_text": "flutter dart mobile application",
                    "relevance_label": 1,
                },
                {
                    "clean_text": "sales outreach crm calls",
                    "relevance_label": 0,
                },
                {
                    "clean_text": "full-time graduate senior role",
                    "relevance_label": 0,
                },
            ]
        )

        result = evaluate_with_leave_one_out(training_data)

        self.assertEqual(result["examples"], 4)
        self.assertGreater(result["evaluated_folds"], 0)
        self.assertIn("accuracy", result)
        self.assertEqual(result["confusion_matrix"].shape, (2, 2))

    def test_format_evaluation_report_contains_key_sections(self):
        metrics = {
            "examples": 4,
            "evaluated_folds": 4,
            "relevant_examples": 2,
            "rejected_examples": 2,
            "accuracy": 0.75,
            "precision": 0.8,
            "recall": 0.7,
            "f1": 0.74,
            "confusion_matrix": [[1, 1], [0, 2]],
        }

        report = format_evaluation_report(metrics)

        self.assertIn("Feedback Model Evaluation", report)
        self.assertIn("Leave-One-Out CV", report)
        self.assertIn("Accuracy: 0.75", report)
        self.assertIn("Confusion Matrix", report)


if __name__ == "__main__":
    unittest.main()
