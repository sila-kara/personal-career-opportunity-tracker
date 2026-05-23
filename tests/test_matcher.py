"""Tests for matching and scoring logic."""

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from matcher import (
    build_match_reason,
    find_matching_terms,
    get_feedback_adjustment,
    score_jobs,
)
from preprocessing import add_clean_text_columns


class TestMatcher(unittest.TestCase):
    def test_find_matching_terms_matches_complete_terms_only(self):
        text = "This internship focuses on campaign reporting."

        result = find_matching_terms(text, ["AI", "internship"])

        self.assertEqual(result, ["internship"])

    def test_feedback_adjustments_are_mapped_correctly(self):
        self.assertEqual(get_feedback_adjustment("liked"), 0.10)
        self.assertEqual(get_feedback_adjustment("Maybe"), 0.03)
        self.assertEqual(get_feedback_adjustment("rejected"), -0.30)
        self.assertEqual(get_feedback_adjustment(""), 0.0)

    def test_build_match_reason_explains_positive_and_negative_signals(self):
        result = build_match_reason(
            matched_keywords=["Python", "SQL", "data science"],
            avoid_matches=["full-time"],
            location_matches=["Remote"],
            role_matches=["Data Science Intern"],
            job_type_matches=["Internship"],
            feedback_value="liked",
        )

        self.assertIn("Matches preferred keywords: Python, SQL, data science", result)
        self.assertIn("Location fits preference: Remote", result)
        self.assertIn("Title matches target role: Data Science Intern", result)
        self.assertIn("Job type fits preference: Internship", result)
        self.assertIn("User feedback: liked", result)
        self.assertIn("Contains avoid keywords: full-time", result)

    def test_score_jobs_applies_location_job_type_and_feedback_effects(self):
        profile = {
            "education": {"degree": "Computer Engineering student"},
            "target_roles": ["Data Science Intern"],
            "skills": ["Python", "SQL"],
            "preferred_industries": ["analytics"],
            "preferred_locations": ["Remote"],
            "job_type_preference": ["Internship"],
            "keywords_like": ["Python", "SQL", "data science", "internship"],
            "keywords_avoid": ["full-time"],
        }
        jobs = pd.DataFrame(
            [
                {
                    "title": "Data Science Intern",
                    "company": "Remote Analytics",
                    "location": "Remote",
                    "job_type": "Internship",
                    "description": "Use Python and SQL for data science.",
                    "link": "https://example.com/remote",
                    "source": "Test",
                    "date_found": "2026-05-23",
                    "user_feedback": "liked",
                    "notes": "",
                },
                {
                    "title": "Data Science Intern",
                    "company": "Far Analytics",
                    "location": "Berlin",
                    "job_type": "Full-time",
                    "description": "Use Python and SQL for data science.",
                    "link": "https://example.com/berlin",
                    "source": "Test",
                    "date_found": "2026-05-23",
                    "user_feedback": "rejected",
                    "notes": "",
                },
            ]
        )
        jobs = add_clean_text_columns(jobs)

        result = score_jobs(profile, jobs)
        remote_row = result[result["link"] == "https://example.com/remote"].iloc[0]
        berlin_row = result[result["link"] == "https://example.com/berlin"].iloc[0]

        self.assertEqual(remote_row["location_penalty"], 0.0)
        self.assertEqual(remote_row["job_type_penalty"], 0.0)
        self.assertEqual(remote_row["feedback_adjustment"], 10.0)
        self.assertEqual(berlin_row["location_penalty"], 25.0)
        self.assertEqual(berlin_row["job_type_penalty"], 25.0)
        self.assertEqual(berlin_row["feedback_adjustment"], -30.0)
        self.assertIn("Location is not in preferred locations", berlin_row["match_reason"])
        self.assertIn("Job type is not preferred", berlin_row["match_reason"])
        self.assertGreater(remote_row["match_score"], berlin_row["match_score"])


if __name__ == "__main__":
    unittest.main()
