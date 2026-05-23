"""Train the optional feedback model and export relevance predictions."""

from config import (
    FEEDBACK_MODEL_REPORT_PATH,
    FEEDBACK_PATH,
    FEEDBACK_PREDICTIONS_PATH,
    JOBS_PATH,
    PROFILE_PATH,
)
from data_loader import load_feedback, load_jobs, load_profile, merge_feedback
from feedback_evaluation import (
    evaluate_with_leave_one_out,
    format_evaluation_report,
    save_evaluation_report,
)
from feedback_model import (
    predict_relevance,
    prepare_training_data,
    train_feedback_classifier,
)
from matcher import score_jobs
from preprocessing import add_clean_text_columns


PREDICTION_COLUMNS = [
    "predicted_relevance_score",
    "match_score",
    "user_feedback",
    "match_reason",
    "title",
    "company",
    "location",
    "job_type",
    "matched_keywords",
    "link",
]


def main() -> None:
    """Train the feedback classifier and export predictions."""
    profile = load_profile(PROFILE_PATH)
    jobs = load_jobs(JOBS_PATH)
    feedback = load_feedback(FEEDBACK_PATH)
    jobs = add_clean_text_columns(jobs)
    jobs = merge_feedback(jobs, feedback)

    scored_jobs = score_jobs(profile, jobs)
    training_data = prepare_training_data(scored_jobs)
    classifier = train_feedback_classifier(training_data)
    predictions = predict_relevance(scored_jobs, classifier)
    metrics = evaluate_with_leave_one_out(training_data)
    report = format_evaluation_report(metrics)

    columns = [col for col in PREDICTION_COLUMNS if col in predictions.columns]
    FEEDBACK_PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions[columns].to_csv(FEEDBACK_PREDICTIONS_PATH, index=False)
    save_evaluation_report(report, FEEDBACK_MODEL_REPORT_PATH)

    print("Feedback model training complete.")
    print(f"Feedback-labeled jobs used: {len(training_data)}")
    print(f"Predictions output: {FEEDBACK_PREDICTIONS_PATH}")
    print(f"Evaluation report: {FEEDBACK_MODEL_REPORT_PATH}")
    print("\nTop predicted relevant jobs:")
    preview_columns = [
        "predicted_relevance_score",
        "title",
        "company",
        "user_feedback",
    ]
    print(predictions[preview_columns].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
