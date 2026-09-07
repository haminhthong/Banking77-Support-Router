"""Operational taxonomy and queue routing definitions."""

from __future__ import annotations

from typing import Any
from ..config import get_taxonomy_config
from ..data.contracts import (
    BANKING77_77_CLASSES,
    DEFAULT_HIGH_RISK_INTENTS,
    INTENT_TO_DOMAIN,
    get_domain_for_intent,
)


class TaxonomyResolver:
    """Resolves intents to domains, queues, and operational priorities."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        if not cfg:
            try:
                cfg = get_taxonomy_config()
            except Exception:
                cfg = {}

        self.intents_cfg: dict[str, dict[str, Any]] = cfg.get("intents", {})
        self.system_queues: dict[str, str] = cfg.get(
            "system_queues",
            {
                "priority_fraud_security": "fraud_security_queue",
                "general_human_review": "general_human_review_queue",
                "ood_review": "general_human_review_queue",
            },
        )
        self.high_risk_intents: frozenset[str] = frozenset(
            cfg.get("high_risk_intents", list(DEFAULT_HIGH_RISK_INTENTS))
        )

    def get_domain(self, intent: str) -> str:
        """Get high-level domain for an intent."""
        if intent in self.intents_cfg:
            return str(self.intents_cfg[intent].get("domain", get_domain_for_intent(intent)))
        return get_domain_for_intent(intent)

    def get_queue(self, intent: str) -> str:
        """Resolve business queue for a given intent."""
        if intent in self.intents_cfg:
            return str(self.intents_cfg[intent].get("queue", "general_operations_queue"))
        # Fallback queue based on domain
        domain = self.get_domain(intent)
        return f"{domain}_queue"

    def get_priority(self, intent: str) -> str:
        """Resolve operational priority for a given intent ('normal', 'high', 'critical')."""
        if intent in self.intents_cfg:
            return str(self.intents_cfg[intent].get("priority", "normal"))
        if intent in self.high_risk_intents:
            return "critical"
        return "normal"
