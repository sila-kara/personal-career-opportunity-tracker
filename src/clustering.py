"""Unsupervised clustering for grouping similar job postings."""

import os

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

DEFAULT_CLUSTER_COUNT = 5
TOP_TERMS_PER_CLUSTER = 4

CLUSTER_CATEGORY_KEYWORDS = {
    "Data & Analytics": [
        "data",
        "analytics",
        "analysis",
        "sql",
        "pandas",
        "dashboard",
        "business intelligence",
        "etl",
        "reporting",
    ],
    "AI & Machine Learning": [
        "ai",
        "machine learning",
        "model",
        "nlp",
        "classification",
        "embeddings",
        "evaluation",
    ],
    "Software & Mobile": [
        "software",
        "development",
        "java",
        "backend",
        "flutter",
        "dart",
        "mobile",
        "api",
    ],
    "Optimization & Simulation": [
        "optimization",
        "simulation",
        "gurobi",
        "arena",
        "operations research",
        "production planning",
    ],
    "Cybersecurity": [
        "cybersecurity",
        "security",
        "alerts",
        "incident",
        "network",
    ],
    "Business & Product": [
        "product",
        "stakeholder",
        "user research",
        "sprint",
        "business",
    ],
    "Sales & Marketing": [
        "sales",
        "marketing",
        "social media",
        "crm",
        "outreach",
        "customer calls",
    ],
}


def choose_cluster_count(num_jobs: int, requested_clusters: int = DEFAULT_CLUSTER_COUNT) -> int:
    """Choose a safe number of clusters for the current dataset size."""
    if num_jobs <= 1:
        return 1
    return min(requested_clusters, num_jobs)


def build_cluster_label(top_terms: list[str]) -> str:
    """Convert top TF-IDF terms into a readable cluster label."""
    if not top_terms:
        return "General opportunities"

    return " / ".join(term.title() for term in top_terms[:3])


def infer_cluster_label(cluster_text: str, top_terms: list[str]) -> str:
    """Create a readable label for a cluster using category keyword signals."""
    category_scores = {}
    normalized_text = cluster_text.lower()

    for category, keywords in CLUSTER_CATEGORY_KEYWORDS.items():
        category_scores[category] = sum(
            normalized_text.count(keyword) for keyword in keywords
        )

    best_category = max(category_scores, key=category_scores.get)
    if category_scores[best_category] > 0:
        return best_category

    return build_cluster_label(top_terms)


def add_job_clusters(
    jobs: pd.DataFrame,
    text_column: str = "clean_text",
    n_clusters: int = DEFAULT_CLUSTER_COUNT,
) -> pd.DataFrame:
    """Add KMeans cluster IDs and readable labels to job postings.

    This is an unsupervised ML component: the model groups jobs by text
    similarity without needing labels such as liked/rejected.
    """
    clustered_jobs = jobs.copy()

    if clustered_jobs.empty:
        clustered_jobs["job_cluster"] = []
        clustered_jobs["cluster_label"] = []
        return clustered_jobs

    cluster_count = choose_cluster_count(len(clustered_jobs), n_clusters)

    if cluster_count == 1:
        clustered_jobs["job_cluster"] = 0
        clustered_jobs["cluster_label"] = "General opportunities"
        return clustered_jobs

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    tfidf_matrix = vectorizer.fit_transform(clustered_jobs[text_column].fillna(""))

    model = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
    cluster_ids = model.fit_predict(tfidf_matrix)

    feature_names = vectorizer.get_feature_names_out()
    cluster_labels = {}

    for cluster_id, center in enumerate(model.cluster_centers_):
        top_indices = center.argsort()[::-1][:TOP_TERMS_PER_CLUSTER]
        top_terms = [feature_names[index] for index in top_indices]
        cluster_text = " ".join(
            clustered_jobs.loc[cluster_ids == cluster_id, text_column].fillna("")
        )
        cluster_labels[cluster_id] = infer_cluster_label(cluster_text, top_terms)

    clustered_jobs["job_cluster"] = cluster_ids
    clustered_jobs["cluster_label"] = clustered_jobs["job_cluster"].map(cluster_labels)

    return clustered_jobs
