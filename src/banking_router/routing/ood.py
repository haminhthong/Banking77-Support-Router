"""Out-of-Distribution (OOD) and Out-of-Scope Query Guard."""

from __future__ import annotations

import re
from typing import Any, Sequence
import numpy as np

STOPWORDS = frozenset({
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s",
    "t", "can", "will", "just", "don", "should", "now", "d", "ll", "m",
    "o", "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn",
    "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn", "shan",
    "shouldn", "wasn", "weren", "won", "wouldn", "please", "help"
})


class OODGuard:
    """Detects queries that lie outside the supported 77 banking intents.

    Employs a multi-signal gate combining:
    1. Input syntactic & length validity (too short, pure gibberish/symbols).
    2. Domain content-word overlap against banking vocabulary (filtering stopwords).
    3. Extreme predictive dispersion / uncertainty (confidence < 0.22 in 77-class model).
    """

    def __init__(
        self,
        vocabulary: set[str] | None = None,
        min_chars: int = 4,
        min_tokens: int = 2,
        lexical_similarity_threshold: float = 0.30,
        low_confidence_ood_threshold: float = 0.22,
    ) -> None:
        self.vocabulary = vocabulary or set()
        self.min_chars = min_chars
        self.min_tokens = min_tokens
        self.lexical_similarity_threshold = lexical_similarity_threshold
        self.low_confidence_ood_threshold = low_confidence_ood_threshold

    def set_vocabulary(self, vocabulary: Sequence[str] | set[str]) -> None:
        """Set or update the banking domain vocabulary (filtered of stopwords)."""
        self.vocabulary = {w.lower() for w in vocabulary if w.lower() not in STOPWORDS}

    def detect(
        self,
        text: str,
        confidence: float,
        margin: float | None = None,
    ) -> tuple[bool, list[str]]:
        """Detect whether a query is out-of-scope or uninterpretable.

        Returns:
            tuple of (is_ood: bool, reason_codes: list[str]).
        """
        clean_text = text.strip()
        tokens = re.findall(r"\b\w+\b", clean_text.lower())

        # 1. Syntactic / Length Quality Checks
        if len(clean_text) < self.min_chars:
            return True, ["OOD_TOO_SHORT"]

        if len(tokens) < self.min_tokens:
            if len(tokens) == 1 and tokens[0] not in self.vocabulary:
                return True, ["OUT_OF_SCOPE_QUERY"]
            if len(tokens) == 0:
                return True, ["OOD_NO_TOKENS"]

        # Repeated characters / gibberish detection
        if re.search(r"(.)\1{4,}", clean_text.lower()):
            return True, ["OOD_GIBBERISH_REPETITION"]

        # Pure numeric or symbol check
        if re.fullmatch(r"[\d\W_]+", clean_text):
            return True, ["OOD_NON_TEXTUAL"]

        # 2. Extreme predictive dispersion in 77 classes
        # Uniform probability is 1/77 ~= 0.013. Confidence < 0.22 indicates severe query ambiguity/OOD.
        if confidence < self.low_confidence_ood_threshold:
            return True, ["OUT_OF_SCOPE_QUERY"]

        # 3. Domain Content Keyword Footprint
        content_tokens = [t for t in tokens if t not in STOPWORDS]
        if self.vocabulary:
            if not content_tokens:
                if confidence < 0.60:
                    return True, ["OUT_OF_SCOPE_QUERY"]
            else:
                known_content = [t for t in content_tokens if t in self.vocabulary]
                content_ratio = len(known_content) / len(content_tokens)

                if content_ratio == 0.0:
                    return True, ["OUT_OF_SCOPE_QUERY"]

                if content_ratio < self.lexical_similarity_threshold and confidence < 0.65:
                    return True, ["OUT_OF_SCOPE_QUERY"]

        return False, []
