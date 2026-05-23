# Personal Career Opportunity Tracker

A Python-based NLP and recommendation system that ranks internship and early-career job opportunities against a personal career profile.

This project started as a simple MVP and now includes content-based matching, rule-based personalization, feedback learning, skill gap analysis, unsupervised clustering, and a hybrid recommendation score.

## Project Motivation

Finding relevant internships can be noisy and repetitive. Job descriptions often contain similar language, but not every technically relevant role is realistic for a student. Location, job type, graduation requirements, preferred skills, and personal feedback all matter.

This project helps answer:

- Which opportunities best match my profile?
- Why did a role receive a high or low score?
- Which skills from my profile appear in the job description?
- Which missing skills should I notice?
- Can my feedback improve future recommendations?

## Why This Is an ML/NLP Project

This is not just a scraping or automation project. The core system uses NLP and machine learning techniques:

- TF-IDF vectorization for text representation
- Cosine similarity for profile-to-job matching
- Rule-based recommendation scoring for real-world constraints
- Logistic Regression classifier trained from user feedback
- KMeans clustering to group similar job postings
- Skill gap analysis using technical keyword extraction
- Hybrid scoring that combines explainable matching with learned feedback signals

## Current Features

- Stores a personal career profile in `profile.yaml`
- Reads active job postings from `data/jobs.csv`
- Cleans and preprocesses job/profile text
- Calculates TF-IDF cosine similarity
- Adds bonuses for preferred keywords, locations, roles, and job types
- Adds penalties for unsuitable locations, job types, and avoid keywords
- Tracks user feedback through `data/feedback.csv`
- Produces a feedback-trained relevance score
- Combines rule-based and ML scores into `hybrid_score_v2`
- Produces a model evaluation report
- Performs skill gap analysis
- Clusters job postings with KMeans
- Suggests high-value jobs to review next with an active-learning style feedback queue
- Provides an optional sentence-transformers semantic matching script
- Includes a Streamlit dashboard
- Exports ranked results to CSV, Markdown, and Google Sheets-ready CSV
- Includes unit tests for the main pipeline components

## Project Structure

```text
personal-career-opportunity-tracker/
  README.md
  requirements.txt
  profile.yaml
  data/
    jobs.csv
    sample_jobs.csv
    feedback.csv
  src/
    main.py
    config.py
    data_loader.py
    preprocessing.py
    matcher.py
    hybrid_scorer.py
    feedback_model.py
    feedback_evaluation.py
    feedback_learning.py
    train_feedback_model.py
    semantic_matcher.py
    run_semantic_matching.py
    clustering.py
    exporter.py
    update_feedback.py
    add_job.py
    reset_jobs_from_sample.py
    dashboard_data.py
  outputs/
    matched_jobs.csv
    matched_jobs.md
    google_sheets_ready.csv
    feedback_model_predictions.csv
    feedback_model_report.txt
    feedback_review_queue.csv
    semantic_matches.csv
  tests/
    test_add_job.py
    test_clustering.py
    test_feedback_model.py
    test_hybrid_scorer.py
    test_matcher.py
    test_preprocessing.py
    test_update_feedback.py
  notebooks/
    exploration.ipynb
```

## Input Data

The active job dataset is stored in:

```text
data/jobs.csv
```

The sample dataset is stored separately:

```text
data/sample_jobs.csv
```

Required columns:

- `title`
- `company`
- `location`
- `job_type`
- `description`
- `link`
- `source`
- `date_found`

The personal profile is stored in:

```text
profile.yaml
```

The profile includes education, target roles, skills, preferred locations, preferred job types, liked keywords, and avoid keywords.

User feedback is stored in:

```text
data/feedback.csv
```

Supported feedback labels:

```text
liked
maybe
rejected
```

## Methodology

1. Load the personal profile from YAML.
2. Load job postings from CSV.
3. Load optional user feedback.
4. Combine job fields into one text field.
5. Clean and normalize text.
6. Use TF-IDF to vectorize profile and job descriptions.
7. Calculate cosine similarity between profile and each job.
8. Add rule-based bonuses and penalties.
9. Add feedback adjustment.
10. Perform skill gap analysis.
11. Train an optional feedback classifier.
12. Produce `hybrid_score_v2`.
13. Cluster jobs using KMeans.
14. Export ranked results.

## Scoring Logic

The first score, `match_score`, is explainable and rule-based around the NLP similarity score.

```text
match_score =
  60% TF-IDF cosine similarity
  + up to 15% preferred keyword bonus
  + up to 10% preferred location bonus
  - up to 25% non-preferred required location penalty
  + up to 10% target role/title bonus
  + up to 5% preferred job type bonus
  - up to 25% non-preferred job type penalty
  - up to 20% avoid keyword penalty
  + feedback adjustment
```

Feedback adjustment:

```text
liked    +10
maybe    +3
rejected -30
```

The second score, `hybrid_score_v2`, combines the explainable score with a feedback-trained classifier:

```text
hybrid_score_v2 =
  70% match_score
  + 30% predicted_relevance_score
```

If there is not enough feedback to train the classifier, the system falls back to `match_score`.

## Skill Gap Analysis

For each job, the system extracts:

- `profile_skills_found`: skills from the profile that appear in the job
- `job_skills_found`: technical skill signals found in the job
- `missing_skills`: skills found in the job but not in the profile
- `skill_match_rate`: percentage-style signal showing how well profile skills cover detected job skills

Example:

```text
Profile skills found: Python, pandas, machine learning
Possible skill gaps: scikit-learn, natural language processing
```

## Clustering

The project uses TF-IDF + KMeans to group similar job postings without labels.

Output columns:

```text
job_cluster
cluster_label
```

Example cluster labels:

```text
Data & Analytics
AI & Machine Learning
Software & Mobile
Optimization & Simulation
Sales & Marketing
```

## Outputs

Running the main pipeline creates:

```text
outputs/matched_jobs.csv
outputs/matched_jobs.md
outputs/google_sheets_ready.csv
```

Running the feedback model script creates:

```text
outputs/feedback_model_predictions.csv
outputs/feedback_model_report.txt
```

The main pipeline also creates an active-learning style feedback queue:

```text
outputs/feedback_review_queue.csv
```

If the optional sentence-transformers workflow is installed and run, it creates:

```text
outputs/semantic_matches.csv
```

The Google Sheets-ready file keeps feedback columns near the front so it can be imported, edited, and exported back into the project.

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the main ranking pipeline:

```bash
python src/main.py
```

Run tests:

```bash
python -m unittest discover -s tests
```

Train the optional feedback model and export predictions:

```bash
python src/train_feedback_model.py
```

Run the optional semantic matching workflow:

```bash
pip install sentence-transformers
python src/run_semantic_matching.py
```

Launch the Streamlit dashboard:

```bash
streamlit run app.py
```

## Google Sheets Feedback Workflow

1. Run the main pipeline:

```bash
python src/main.py
```

2. Import this file into Google Sheets:

```text
outputs/google_sheets_ready.csv
```

3. Edit `user_feedback` and `notes` in Google Sheets.

4. Export the sheet back to CSV.

5. Update the local feedback file:

```bash
python src/update_feedback.py path/to/edited_google_sheet.csv
```

6. Re-run the ranking pipeline:

```bash
python src/main.py
```

## Add a New Job Manually

Use `src/add_job.py` to add a posting without editing the CSV by hand:

```bash
python src/add_job.py \
  --title "Data Science Intern" \
  --company "Example Company" \
  --location "Remote" \
  --job-type "Internship" \
  --description "Use Python, SQL, and pandas for analytics projects." \
  --link "https://example.com/jobs/data-science-intern" \
  --source "Manual Entry"
```

Then re-run:

```bash
python src/main.py
```

Reset the active jobs dataset from the sample dataset:

```bash
python src/reset_jobs_from_sample.py
```

## Sample Output

Example terminal preview:

```text
Career opportunity matching complete.
Jobs processed: 25

Top matches:
 hybrid_score_v2  match_score                           title              company         location         cluster_label
           57.49        53.42     Software Development Intern         FinCore Tech Hybrid - Kocaeli     Software & Mobile
           47.61        39.32    Data Science Working Student          Quantiva AI           Remote AI & Machine Learning
           44.11        36.34         Machine Learning Intern DataBridge Analytics           Remote AI & Machine Learning
```

## Testing

The project uses Python's built-in `unittest` framework.

Current test coverage includes:

- Text preprocessing
- Job matching logic
- Feedback scoring
- Skill gap analysis
- Feedback CSV updates
- Manual job entry
- Feedback classifier
- Feedback model evaluation
- Feedback review queue
- KMeans clustering
- Hybrid score calculation
- Dashboard data filtering

Run all tests:

```bash
python -m unittest discover -s tests
```

## Future Improvements

- Add optional Google Sheets API integration
- Add real job sources that allow scraping or provide RSS feeds
- Add more feedback data and compare classifier performance over time
- Add visual charts for score distributions and cluster summaries

## Deployment

The Streamlit dashboard can be launched locally with:

```bash
streamlit run app.py
```

Before public deployment, review `profile.yaml`, `data/jobs.csv`, and `data/feedback.csv` for private information. See `DEPLOYMENT.md` for deployment notes.

## Notes

The project is intentionally modular and beginner-friendly:

- `data_loader.py`: reads profile, jobs, and feedback
- `preprocessing.py`: cleans text
- `matcher.py`: computes similarity, rule-based score, explanations, and skill gaps
- `feedback_model.py`: trains a simple feedback classifier
- `feedback_evaluation.py`: evaluates the feedback classifier
- `feedback_learning.py`: suggests useful jobs to label next
- `hybrid_scorer.py`: combines explainable and learned scores
- `semantic_matcher.py`: optional sentence-transformers semantic similarity
- `clustering.py`: groups similar jobs using unsupervised learning
- `exporter.py`: saves outputs
- `main.py`: connects the full pipeline

This makes the system easy to understand, test, and extend.
