"""Data module exports for Banking77 Support Router."""

from .contracts import (
    BANKING77_77_CLASSES,
    DEFAULT_HIGH_RISK_INTENTS,
    INTENT_TO_DOMAIN,
    get_domain_for_intent,
)
from .loader import (
    load_official_test,
    normalize_dataset,
    read_raw_dataset,
)
from .normalization import (
    compute_file_sha256,
    normalize_text_for_audit,
    normalize_whitespace,
)
from .audit import (
    audit_conflicting_labels,
    audit_dataset,
    audit_normalized_duplicates,
    clean_dataset,
)
from .split import (
    load_training_splits,
    summarize_split_quality,
)

__all__ = [
    "BANKING77_77_CLASSES",
    "DEFAULT_HIGH_RISK_INTENTS",
    "INTENT_TO_DOMAIN",
    "get_domain_for_intent",
    "read_raw_dataset",
    "normalize_dataset",
    "load_official_test",
    "normalize_whitespace",
    "normalize_text_for_audit",
    "compute_file_sha256",
    "audit_conflicting_labels",
    "audit_normalized_duplicates",
    "audit_dataset",
    "clean_dataset",
    "load_training_splits",
    "summarize_split_quality",
]
