"""Tests for optional semantic matching helpers."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from semantic_matcher import require_sentence_transformers, sentence_transformers_available


class TestSemanticMatcher(unittest.TestCase):
    def test_sentence_transformers_available_returns_bool(self):
        self.assertIsInstance(sentence_transformers_available(), bool)

    def test_require_sentence_transformers_raises_helpful_error_when_missing(self):
        with patch("semantic_matcher.sentence_transformers_available", return_value=False):
            with self.assertRaisesRegex(ImportError, "pip install sentence-transformers"):
                require_sentence_transformers()


if __name__ == "__main__":
    unittest.main()
