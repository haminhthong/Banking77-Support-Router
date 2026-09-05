"""Mô-đun định nghĩa chính sách phân luồng (Routing Policy Engine).

Tách biệt logic nghiệp vụ vận hành khỏi mô hình học máy:
- High-Risk Priority Escalation: các ý định liên quan tới gian lận, mất thẻ, kẹt thẻ luôn được ưu tiên
  chuyển cho nhân viên xử lý khẩn cấp (kể cả khi confidence thấp hoặc xuất hiện trong Top-K).
- Selective Decision Gate (Abstain Semantics): khi độ tin cậy dưới ngưỡng hoặc độ mập mờ cao (margin hẹp,
  entropy cao), hệ thống chủ động từ chối tự động hóa (Abstain), chuyển human review thường.
- Safe Auto-Routing: chỉ tự động phân luồng khi thỏa mãn toàn bộ tiêu chí an toàn & tự tin.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Danh sách mặc định các intent rủi ro cao cần ưu tiên cho nhân viên kiểm duyệt khẩn cấp
DEFAULT_HIGH_RISK_INTENTS: frozenset[str] = frozenset(
    {
        "cash_withdrawal_not_recognised",
        "card_swallowed",
        "compromised_card",
        "lost_or_stolen_card",
        "lost_or_stolen_phone",
    }
)


@dataclass(frozen=True)
class RoutingDecision:
    """Cấu trúc dữ liệu đại diện cho quyết định phân luồng vận hành cuối cùng."""

    intent: str | None
    domain: str | None
    route: str
    decision: str  # "auto_route" | "priority_escalation" | "abstain"
    abstained: bool
    requires_human_review: bool
    review_reason: str | None
    is_unknown: bool = False  # Giữ tương thích ngược với API contract cũ


@dataclass(frozen=True)
class RoutingPolicy:
    """Bộ quy tắc chuyển đổi xác suất dự đoán thành quyết định vận hành có kiểm soát rủi ro."""

    threshold: float
    high_risk_intents: frozenset[str] = field(
        default_factory=lambda: DEFAULT_HIGH_RISK_INTENTS
    )
    high_risk_trigger: float = 0.20
    min_margin: float | None = None
    max_entropy: float | None = None

    def __post_init__(self) -> None:
        """Kiểm tra tính hợp lệ của ngưỡng tin cậy."""
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("Ngưỡng threshold phải nằm trong khoảng [0.0, 1.0]")
        if not 0.0 <= self.high_risk_trigger <= 1.0:
            raise ValueError("Ngưỡng high_risk_trigger phải nằm trong khoảng [0.0, 1.0]")
        if self.min_margin is not None and not 0.0 <= self.min_margin <= 1.0:
            raise ValueError("min_margin phải nằm trong khoảng [0.0, 1.0]")
        if self.max_entropy is not None and self.max_entropy < 0.0:
            raise ValueError("max_entropy không được là số âm")

    def decide(
        self,
        top_intent: str,
        confidence: float,
        top_domain: str | None = None,
        top_k_candidates: list[tuple[str, float]] | None = None,
        margin: float | None = None,
        entropy: float | None = None,
    ) -> RoutingDecision:
        """Đưa ra quyết định phân luồng dựa trên mức độ rủi ro và độ tin cậy.

        Quy tắc ưu tiên (Precedence):
        1. RỦI RO CAO (High-Risk First):
           - Nếu top_intent thuộc nhóm rủi ro cao -> Lập tức leo thang priority_human_review
             (ngay cả khi confidence < threshold).
           - Nếu bất kỳ intent rủi ro cao nào nằm trong Top-K với xác suất >= high_risk_trigger ->
             Leo thang priority_human_review.
        2. TỪ CHỐI AN TOÀN (Uncertainty / Selective Gate):
           - Nếu confidence < threshold -> Từ chối tự động hóa (Abstain) với lý do LOW_CONFIDENCE.
           - Nếu margin < min_margin -> Abstain với lý do AMBIGUOUS_MARGIN.
           - Nếu entropy > max_entropy -> Abstain với lý do HIGH_ENTROPY.
        3. TỰ ĐỘNG HÓA AN TOÀN (Safe Auto-Route):
           - Đủ tin cậy và không rủi ro -> Tự động đưa vào intent queue.
        """
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Độ tin cậy confidence phải nằm trong khoảng [0.0, 1.0]")

        # 1. Kiểm tra ưu tiên Rủi Ro Cao
        if top_intent in self.high_risk_intents:
            return RoutingDecision(
                intent=top_intent,
                domain=top_domain,
                route="priority_human_review",
                decision="priority_escalation",
                abstained=False,
                requires_human_review=True,
                review_reason="HIGH_RISK_INTENT",
                is_unknown=False,
            )

        if top_k_candidates:
            for cand_intent, cand_prob in top_k_candidates:
                if (
                    cand_intent in self.high_risk_intents
                    and cand_prob >= self.high_risk_trigger
                ):
                    return RoutingDecision(
                        intent=cand_intent,
                        domain=top_domain,
                        route="priority_human_review",
                        decision="priority_escalation",
                        abstained=False,
                        requires_human_review=True,
                        review_reason="HIGH_RISK_CANDIDATE",
                        is_unknown=False,
                    )

        # 2. Cổng kiểm soát độ mập mờ / không chắc chắn (Uncertainty Gate)
        if confidence < self.threshold:
            return RoutingDecision(
                intent=None,
                domain=None,
                route="human",
                decision="abstain",
                abstained=True,
                requires_human_review=True,
                review_reason="LOW_CONFIDENCE",
                is_unknown=True,
            )

        if self.min_margin is not None and margin is not None and margin < self.min_margin:
            return RoutingDecision(
                intent=None,
                domain=None,
                route="human",
                decision="abstain",
                abstained=True,
                requires_human_review=True,
                review_reason="AMBIGUOUS_MARGIN",
                is_unknown=True,
            )

        if (
            self.max_entropy is not None
            and entropy is not None
            and entropy > self.max_entropy
        ):
            return RoutingDecision(
                intent=None,
                domain=None,
                route="human",
                decision="abstain",
                abstained=True,
                requires_human_review=True,
                review_reason="HIGH_ENTROPY",
                is_unknown=True,
            )

        # 3. Tự động hóa an toàn
        return RoutingDecision(
            intent=top_intent,
            domain=top_domain,
            route=top_intent,
            decision="auto_route",
            abstained=False,
            requires_human_review=False,
            review_reason=None,
            is_unknown=False,
        )
