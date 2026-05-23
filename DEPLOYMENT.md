# Deployment Notes

This project can be deployed as a Streamlit app after checking data privacy.

## Before Deploying

- Review `profile.yaml` and remove private personal details if needed.
- Review `data/jobs.csv` and remove private links or confidential notes.
- Review `data/feedback.csv` and remove personal notes if needed.
- Never commit real API keys, Google credentials, tokens, or secrets.

## Run Locally

```bash
pip install -r requirements.txt
python src/main.py
streamlit run app.py
```

## Streamlit Community Cloud

Suggested settings:

```text
Main file path: app.py
Python version: 3.11 or 3.12
```

The app expects generated output files such as:

```text
outputs/google_sheets_ready.csv
outputs/feedback_model_report.txt
```

If outputs are missing, run:

```bash
python src/main.py
python src/train_feedback_model.py
```

## Optional Semantic Matching

Semantic matching uses `sentence-transformers`, which can make deployment heavier.
It is intentionally optional.

To enable it:

```bash
pip install sentence-transformers
python src/run_semantic_matching.py
```
