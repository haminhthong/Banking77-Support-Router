"""Kiểm tra chất lượng input và phạm vi hỗ trợ của Banking77."""

from __future__ import annotations

import re
from typing import Any, Sequence

STOPWORDS = frozenset({
    "i", "me", "my", "we", "our", "you", "your", "he", "his", "she", "her",
    "it", "they", "them", "this", "that", "what", "which", "who", "why", "how",
    "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does",
    "did", "a", "an", "the", "and", "but", "or", "because", "as", "of", "at", "by",
    "for", "with", "about", "to", "from", "in", "out", "on", "off", "over", "under",
    "when", "where", "all", "any", "both", "each", "more", "most", "some", "no", "not",
    "only", "same", "so", "than", "too", "very", "can", "will", "just", "should", "now",
    "please", "help",
})


class ScopeGuard:
    """Kết hợp heuristic input, scope classifier và vocabulary của intent model."""

    def __init__(
        self,
        vocabulary: set[str] | None = None,
        min_chars: int = 4,
        min_tokens: int = 2,
        lexical_similarity_threshold: float = 0.08,
        low_confidence_threshold: float = 0.22,
        scope_classifier: Any | None = None,
        unsupported_threshold: float = 0.80,
    ) -> None:
        self.vocabulary = vocabulary or set()
        self.min_chars = min_chars
        self.min_tokens = min_tokens
        self.lexical_similarity_threshold = lexical_similarity_threshold
        self.low_confidence_threshold = low_confidence_threshold
        self.scope_classifier = scope_classifier
        self.unsupported_threshold = unsupported_threshold

    def set_vocabulary(self, vocabulary: Sequence[str] | set[str]) -> None:
        self.vocabulary = {
            str(word).lower()
            for word in vocabulary
            if str(word).lower() not in STOPWORDS
        }

    def set_scope_classifier(self, classifier: Any | None) -> None:
        self.scope_classifier = classifier

    def detect(self, text: str, confidence: float, margin: float | None = None) -> tuple[bool, list[str]]:
        del margin
        clean_text = text.strip()
        tokens = re.findall(r"\b\w+\b", clean_text.lower())

        if len(clean_text) < self.min_chars:
            return True, ["SCOPE_TOO_SHORT"]
        if len(tokens) < self.min_tokens:
            if not tokens:
                return True, ["SCOPE_NO_TOKENS"]
            if tokens[0] not in self.vocabulary:
                return True, ["SCOPE_LOW_INFORMATION"]
        if re.search(r"(.)\1{4,}", clean_text.lower()):
            return True, ["SCOPE_GIBBERISH"]
        if re.fullmatch(r"[\d\W_]+", clean_text):
            return True, ["SCOPE_NON_TEXTUAL"]

        if self.scope_classifier is not None:
            try:
                unsupported = float(self.scope_classifier.unsupported_probability(clean_text))
            except AttributeError:
                probabilities = self.scope_classifier.predict_proba([clean_text])[0]
                classes = list(self.scope_classifier.classes_)
                unsupported = float(probabilities[classes.index("UNSUPPORTED")])
            if unsupported >= self.unsupported_threshold:
                return True, ["OUT_OF_SCOPE_QUERY"]

        low_confidence = confidence < self.low_confidence_threshold
        content_tokens = [token for token in tokens if token not in STOPWORDS]
        if self.vocabulary:
            if not content_tokens and confidence < 0.60:
                return True, ["OUT_OF_SCOPE_QUERY"]
            known_content = [token for token in content_tokens if token in self.vocabulary]
            content_ratio = len(known_content) / len(content_tokens) if content_tokens else 0.0
            if content_ratio == 0.0 and low_confidence:
                return True, ["OUT_OF_SCOPE_QUERY"]
            if content_ratio < self.lexical_similarity_threshold and low_confidence:
                return True, ["OUT_OF_SCOPE_QUERY"]

        return False, []
