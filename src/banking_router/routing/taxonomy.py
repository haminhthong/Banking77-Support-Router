"""Đọc taxonomy và chiếu intent sang queue/risk dùng chung cho train và serve."""

from __future__ import annotations

from typing import Any

from ..config import get_taxonomy_config
from ..data.contracts import get_domain_for_intent


class TaxonomyResolver:
    """Nguồn sự thật duy nhất cho domain, queue, priority và security risk.

    ``priority`` mô tả mức ưu tiên vận hành. Security risk chỉ được suy ra từ
    ``risk.priority_escalation``; không dùng priority để biến mọi ticket khẩn
    thành ticket gian lận.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        if not cfg:
            try:
                cfg = get_taxonomy_config()
            except Exception:
                cfg = {}

        self.config = cfg
        self.intents_cfg: dict[str, dict[str, Any]] = cfg.get("intents", {})
        self.system_queues: dict[str, str] = cfg.get(
            "system_queues",
            {
                "priority_fraud_security": "fraud_security_queue",
                "general_human_review": "general_human_review_queue",
                "ood_review": "general_human_review_queue",
            },
        )
        self.critical_intents = frozenset(
            intent
            for intent in self.intents_cfg
            if self.is_priority_escalation(intent)
        )
        # Alias để client cũ không bị gãy; runtime mới dùng metadata risk.
        self.high_risk_intents = self.critical_intents

    def _values(self, intent: str) -> dict[str, Any]:
        return self.intents_cfg.get(intent, {})

    def get_domain(self, intent: str) -> str:
        """Lấy domain nghiệp vụ của intent."""
        return str(self._values(intent).get("domain", get_domain_for_intent(intent)))

    def get_queue(self, intent: str) -> str:
        """Lấy queue đích của intent."""
        values = self._values(intent)
        if values.get("queue"):
            return str(values["queue"])
        return f"{self.get_domain(intent)}_queue"

    def get_priority(self, intent: str) -> str:
        """Lấy priority vận hành, không đồng nhất với security risk."""
        values = self._values(intent)
        return str(values.get("operational_priority", values.get("priority", "normal"))).lower()

    def get_risk(self, intent: str) -> dict[str, Any]:
        """Lấy metadata risk độc lập với priority vận hành."""
        risk = self._values(intent).get("risk", {})
        if isinstance(risk, dict) and risk:
            return risk
        # Bundle cũ chỉ được suy ra security từ is_high_risk, không từ priority.
        if bool(self._values(intent).get("is_high_risk", False)):
            return {
                "category": "security",
                "tier": "critical",
                "priority_escalation": True,
                "escalation_queue": self.system_queues.get(
                    "priority_fraud_security", "fraud_security_queue"
                ),
            }
        return {"category": "none", "tier": "normal", "priority_escalation": False}

    def get_risk_category(self, intent: str) -> str:
        return str(self.get_risk(intent).get("category", "none")).lower()

    def get_risk_tier(self, intent: str) -> str:
        return str(self.get_risk(intent).get("tier", "normal")).lower()

    def is_priority_escalation(self, intent: str) -> bool:
        return bool(self.get_risk(intent).get("priority_escalation", False))

    def get_escalation_queue(self, risk_category: str | None = None) -> str:
        """Chọn queue escalation theo nhóm risk, mặc định là fraud/security."""
        groups = self.config.get("risk_groups", {})
        if risk_category and isinstance(groups, dict):
            group = groups.get(risk_category, {})
            if isinstance(group, dict) and group.get("escalation_queue"):
                return str(group["escalation_queue"])
        return self.system_queues.get("priority_fraud_security", "fraud_security_queue")

    def get_critical_intents(self) -> frozenset[str]:
        """Trả về các intent có cờ escalation security rõ ràng."""
        return self.critical_intents
