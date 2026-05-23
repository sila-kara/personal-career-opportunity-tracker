# Personal Career Opportunity Tracker

A beginner-friendly NLP / recommendation project that ranks internship and job postings against a personal career profile.

The goal is to help a student or early-career candidate quickly identify the most relevant opportunities from a structured dataset. The first version uses a local CSV file instead of web scraping, so the machine learning pipeline is easy to run, explain, and improve.

## Why This Is an ML/NLP Project

This project is not just a job scraping script. It uses natural language processing to compare the meaning of a personal profile with job descriptions.

The MVP uses:

- Text preprocessing to clean profile and job posting text
- TF-IDF vectorization to convert text into numerical features
- Cosine similarity to compare the profile vector with each job vector
- A hybrid recommendation score that combines NLP similarity with explainable career preferences

This makes the project suitable for a machine learning portfolio because it demonstrates data loading, feature extraction, similarity-based recommendation, scoring logic, and ranked output generation.

## Project Structure

```text
personal-career-opportunity-tracker/
  README.md
  requirements.txt
  profile.yaml
  data/
    sample_jobs.csv
  src/
    main.py
    data_loader.py
    preprocessing.py
    matcher.py
    exporter.py
    config.py
  outputs/
    matched_jobs.csv
    matched_jobs.md
  notebooks/
    exploration.ipynb
```

## Dataset Structure

The MVP reads job postings from `data/sample_jobs.csv`.

Required columns:

- `title`
- `company`
- `location`
- `job_type`
- `description`
- `link`
- `source`
- `date_found`

You can replace the sample rows with your own collected postings as long as the column names stay the same.

## Personal Profile

Your preferences are stored in `profile.yaml`.

The profile includes:

- Education
- Target roles
- Skills
- Preferred industries
- Preferred locations
- Internship or full-time preference
- Keywords you like
- Keywords you want to avoid

Update this file when your goals change. The scoring pipeline will automatically use the new profile the next time you run the project.

## Methodology

1. Load the personal profile from YAML.
2. Load job postings from CSV.
3. Combine useful job fields into one text field.
4. Clean the text by lowercasing, removing noisy characters, and normalizing spaces.
5. Convert the profile and job descriptions into TF-IDF vectors.
6. Calculate cosine similarity between the profile and each job.
7. Add rule-based bonuses and penalties.
8. Rank jobs by final match score.
9. Export results to CSV and Markdown.

## Scoring Logic

The final score is a hybrid of ML/NLP similarity and clear rule-based preferences.

```text
final_score =
  60% TF-IDF cosine similarity
  + up to 15% preferred keyword bonus
  + up to 10% preferred location bonus
  + up to 10% target role/title bonus
  + up to 5% job type preference bonus
  - up to 20% avoid keyword penalty
```

The score is converted to a 0 to 100 scale and clipped so it never goes below 0 or above 100.

Why this formula works for an MVP:

- TF-IDF similarity captures overall text relevance.
- Keyword bonuses reward specific interests like `NLP`, `Python`, or `machine learning`.
- Avoid penalties reduce matches that contain unwanted terms.
- Location, role, and job type bonuses make the ranking more personal and practical.

## How To Run

From the project folder:

```bash
pip install -r requirements.txt
python src/main.py
```

The script creates:

- `outputs/matched_jobs.csv`
- `outputs/matched_jobs.md`

## Sample Output

Example terminal preview:

```text
Career opportunity matching complete.
Jobs processed: 10

Top matches:
 match_score                   title              company          location
       55.80 Machine Learning Intern DataBridge Analytics            Remote
       39.82     Data Science Intern    BrightRetail Labs          Istanbul
       34.68       AI Product Intern         EduFuture AI            Remote
```

The CSV output contains the full ranked dataset, including similarity score, bonuses, penalties, matched keywords, and avoid keywords found.

## Development Phases

### Phase 1: Project Structure and Sample Dataset

Created folders, `profile.yaml`, `requirements.txt`, `data/sample_jobs.csv`, and a starter notebook.

Test:

```bash
ls
```

### Phase 2: Profile and Job Loading

Implemented `src/data_loader.py` to load the YAML profile and CSV jobs with basic validation.

Test:

```bash
python src/main.py
```

### Phase 3: Text Preprocessing

Implemented `src/preprocessing.py` to combine job fields and clean text for NLP.

Test:

```bash
python src/main.py
```

### Phase 4: TF-IDF and Cosine Similarity

Implemented TF-IDF vectorization and cosine similarity in `src/matcher.py`.

Test:

```bash
python src/main.py
```

### Phase 5: Bonuses and Penalties

Added preferred keyword, avoid keyword, location, role, and job type scoring.

Test:

```bash
python src/main.py
```

### Phase 6: Export Ranked Results

Implemented CSV and Markdown exports in `src/exporter.py`.

Test:

```bash
python src/main.py
```

### Phase 7: README

Documented the project motivation, methodology, scoring formula, setup instructions, and future improvements.

## Future Improvements

Good next upgrades:

- Add an optional scraper module for sites that allow scraping or provide RSS feeds.
- Export an Excel file with formatting using `openpyxl`.
- Add sentence-transformers embeddings for stronger semantic matching.
- Cluster job postings into groups like data science, backend, product, and business.
- Add a feedback file where you mark jobs as relevant or not relevant.
- Train a simple classifier from your feedback.
- Build a Streamlit dashboard after the command-line MVP is stable.
- Add charts in the notebook to explore score distributions and keyword matches.

## Notes

This project intentionally starts simple. The code is split into small files so each part has one clear responsibility:

- `data_loader.py`: reads input files
- `preprocessing.py`: cleans text
- `matcher.py`: scores and ranks jobs
- `exporter.py`: saves outputs
- `main.py`: connects the full pipeline

That structure makes the project easier to understand now and easier to upgrade later.
