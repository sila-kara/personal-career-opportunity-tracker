"""Tests for the optional feedback-based classifier."""

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from feedback_model import (
    feedback_to_label,
    predict_relevance,
    prepare_training_data,
    train_feedback_classifier,
    validate_training_data,
)


class TestFeedbackModel(unittest.TestCase):
    def test_feedback_to_label_maps_supported_values(self):
        self.assertEqual(feedback_to_label("liked"), 1)
        self.assertEqual(feedback_to_label("maybe"), 1)
        self.assertEqual(feedback_to_label("rejected"), 0)
        self.assertIsNone(feedback_to_label(""))

    def test_prepare_training_data_keeps_labeled_rows(self):
        jobs = pd.DataFrame(
            [
                {"user_feedback": "liked", "clean_text": "python internship"},
                {"user_feedback": "", "clean_text": "unknown role"},
                {"user_feedback": "rejected", "clean_text": "sales full-time"},
            ]
        )

        result = prepare_training_data(jobs)

        self.assertEqual(len(result), 2)
        self.assertEqual(result["relevance_label"].tolist(), [1, 0])

    def test_validate_training_data_requires_two_classes(self):
        training_data = pd.DataFrame(
            [
                {"relevance_label": 1},
                {"relevance_label": 1},
                {"relevance_label": 1},
                {"relevance_label": 1},
            ]
        )

        with self.assertRaises(ValueError):
            validate_training_data(training_data)

    def test_train_feedback_classifier_predicts_relevance_scores(self):
        training_data = pd.DataFrame(
            [
                {
                    "clean_text": "python sql data science internship",
                    "relevance_label": 1,
                },
                {
                    "clean_text": "flutter dart mobile application part-time",
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
        jobs = pd.DataFrame(
            [
                {"clean_text": "python data science internship"},
                {"clean_text": "sales outreach calls"},
            ]
        )

        classifier = train_feedback_classifier(training_data)
        predictions = predict_relevance(jobs, classifier)

        self.assertIn("predicted_relevance_score", predictions.columns)
        self.assertEqual(len(predictions), 2)
        self.assertGreaterEqual(predictions["predicted_relevance_score"].min(), 0)
        self.assertLessEqual(predictions["predicted_relevance_score"].max(), 100)


if __name__ == "__main__":
    unittest.main()
