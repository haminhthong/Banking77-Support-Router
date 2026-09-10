"""Ánh xạ duy nhất từ intent Banking77 sang domain, queue và priority."""

from __future__ import annotations

from typing import Any

from ..config import get_taxonomy_config
from ..data.contracts import get_domain_for_intent


class TaxonomyResolver:
    """Đọc taxonomy dùng chung cho training, evaluation và serving."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or get_taxonomy_config()
        self.config = cfg
        self.intents_cfg: dict[str, dict[str, Any]] = cfg.get("intents", {})
        self.system_queues: dict[str, str] = cfg.get(
            "system_queues",
            {
                "priority_sensitive_case": "sensitive_case_review_queue",
                "general_human_review": "general_human_review_queue",
                "scope_review": "general_human_review_queue",
            },
        )
        self.sensitive_intents = frozenset(
            intent for intent in self.intents_cfg if self.requires_priority_review(intent)
        )

    def _values(self, intent: str) -> dict[str, Any]:
        return self.intents_cfg.get(intent, {})

    def get_domain(self, intent: str) -> str:
        """Lấy domain nghiệp vụ của intent."""
        return str(self._values(intent).get("domain", get_domain_for_intent(intent)))

    def get_queue(self, intent: str) -> str:
        """Lấy queue nghiệp vụ của intent."""
        values = self._values(intent)
        return str(values.get("queue", f"{self.get_domain(intent)}_queue"))

    def get_priority(self, intent: str) -> str:
        """Lấy mức ưu tiên vận hành của intent."""
        return str(self._values(intent).get("priority", "normal")).lower()

    def get_sensitive_case(self, intent: str) -> dict[str, Any]:
        """Lấy metadata của nhóm intent cần xem xét ưu tiên."""
        values = self._values(intent)
        sensitive_case = values.get("sensitive_case", {})
        if isinstance(sensitive_case, dict):
            return sensitive_case
        return {}

    def get_sensitive_category(self, intent: str) -> str:
        return str(self.get_sensitive_case(intent).get("category", "none")).lower()

    def requires_priority_review(self, intent: str) -> bool:
        return bool(self.get_sensitive_case(intent).get("priority_review", False))

    def get_escalation_queue(self, category: str | None = None) -> str:
        """Chọn queue cho ca nhạy cảm theo nhóm taxonomy."""
        groups = self.config.get("sensitive_groups", {})
        if category and isinstance(groups, dict):
            group = groups.get(category, {})
            if isinstance(group, dict) and group.get("review_queue"):
                return str(group["review_queue"])
        return self.system_queues.get(
            "priority_sensitive_case", "sensitive_case_review_queue"
        )

    def get_scope_review_queue(self) -> str:
        """Queue chung cho ticket ngoài phạm vi hoặc thiếu thông tin."""
        return self.system_queues.get("scope_review", "general_human_review_queue")

    def get_sensitive_intents(self) -> frozenset[str]:
        """Trả về các intent được taxonomy đánh dấu cần priority review."""
        return self.sensitive_intents
