"""Operational taxonomy and queue routing definitions."""

from __future__ import annotations

from typing import Any
from ..config import get_taxonomy_config
from ..data.contracts import (
    BANKING77_77_CLASSES,
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
        # ``risk_tier`` is authoritative.  The legacy ``high_risk_intents``
        # field is accepted for reading old bundles but is deliberately not
        # used to determine runtime safety.
        self.critical_intents: frozenset[str] = frozenset(
            intent
            for intent, values in self.intents_cfg.items()
            if str(values.get("risk_tier", "")).lower() == "critical"
            or str(values.get("priority", "")).lower() == "critical"
            or bool(values.get("is_high_risk", False))
        )
        self.high_risk_intents = self.critical_intents

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

    def get_risk_tier(self, intent: str) -> str:
        """Resolve risk independently from the operational queue."""
        if intent in self.intents_cfg:
            values = self.intents_cfg[intent]
            return str(values.get("risk_tier", values.get("priority", "normal"))).lower()
        return "critical" if intent in self.critical_intents else "normal"

    def get_critical_intents(self) -> frozenset[str]:
        """Return critical intents derived from taxonomy metadata."""
        return self.critical_intents
