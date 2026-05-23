"""Evaluation helpers for the feedback-based relevance classifier."""

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import LeaveOneOut

from feedback_model import build_classifier, validate_training_data


def evaluate_with_leave_one_out(training_data: pd.DataFrame) -> dict:
    """Evaluate the feedback classifier with Leave-One-Out cross-validation."""
    validate_training_data(training_data)

    labels = training_data["relevance_label"].tolist()
    predictions = []
    loo = LeaveOneOut()

    for train_index, test_index in loo.split(training_data):
        train_fold = training_data.iloc[train_index]
        test_fold = training_data.iloc[test_index]

        if train_fold["relevance_label"].nunique() < 2:
            continue

        classifier = build_classifier()
        classifier.fit(train_fold["clean_text"], train_fold["relevance_label"])
        prediction = classifier.predict(test_fold["clean_text"])[0]
        predictions.append(
            {
                "actual": int(test_fold["relevance_label"].iloc[0]),
                "predicted": int(prediction),
            }
        )

    if not predictions:
        raise ValueError("Not enough class variety to run cross-validation.")

    actual = [row["actual"] for row in predictions]
    predicted = [row["predicted"] for row in predictions]
    matrix = confusion_matrix(actual, predicted, labels=[0, 1])

    return {
        "examples": len(training_data),
        "evaluated_folds": len(predictions),
        "relevant_examples": int(sum(labels)),
        "rejected_examples": int(len(labels) - sum(labels)),
        "accuracy": accuracy_score(actual, predicted),
        "precision": precision_score(actual, predicted, zero_division=0),
        "recall": recall_score(actual, predicted, zero_division=0),
        "f1": f1_score(actual, predicted, zero_division=0),
        "confusion_matrix": matrix,
    }


def format_evaluation_report(metrics: dict) -> str:
    """Format model evaluation metrics as a readable text report."""
    matrix = metrics["confusion_matrix"]

    lines = [
        "Feedback Model Evaluation",
        "=========================",
        "",
        "Note: This evaluation is experimental because the feedback dataset is small.",
        "",
        f"Training examples: {metrics['examples']}",
        f"Evaluated folds: {metrics['evaluated_folds']}",
        f"Relevant examples: {metrics['relevant_examples']}",
        f"Rejected examples: {metrics['rejected_examples']}",
        "",
        "Cross-validation strategy: Leave-One-Out CV",
        "",
        "Metrics:",
        f"- Accuracy: {metrics['accuracy']:.2f}",
        f"- Precision: {metrics['precision']:.2f}",
        f"- Recall: {metrics['recall']:.2f}",
        f"- F1 Score: {metrics['f1']:.2f}",
        "",
        "Confusion Matrix (labels: rejected=0, relevant=1):",
        f"[[TN={matrix[0][0]}, FP={matrix[0][1]}],",
        f" [FN={matrix[1][0]}, TP={matrix[1][1]}]]",
        "",
    ]

    return "\n".join(lines)


def save_evaluation_report(report: str, output_path) -> None:
    """Save the evaluation report to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
