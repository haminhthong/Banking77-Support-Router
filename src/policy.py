"""Mô-đun định nghĩa chính sách phân luồng (Routing Policy) độc lập với mô hình.

Tách biệt logic nghiệp vụ vận hành khỏi mô hình học máy để dễ dàng kiểm thử,
thay đổi quy tắc và bảo trì hệ thống mà không cần huấn luyện lại mô hình.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Danh sách mặc định các intent rủi ro cao cần ưu tiên cho nhân viên kiểm duyệt
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
    """Cấu trúc dữ liệu đại diện cho quyết định phân luồng cuối cùng."""

    intent: str | None
    domain: str | None
    route: str
    is_unknown: bool
    requires_human_review: bool
    review_reason: str | None


@dataclass(frozen=True)
class RoutingPolicy:
    """Bộ quy tắc chuyển đổi kết quả dự đoán của mô hình thành quyết định vận hành có kiểm soát."""

    threshold: float
    high_risk_intents: frozenset[str] = field(
        default_factory=lambda: DEFAULT_HIGH_RISK_INTENTS
    )

    def __post_init__(self) -> None:
        """Kiểm tra tính hợp lệ của ngưỡng tin cậy."""
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("Ngưỡng threshold phải nằm trong khoảng [0.0, 1.0]")

    def decide(
        self, top_intent: str, confidence: float, top_domain: str | None = None
    ) -> RoutingDecision:
        """Đưa ra quyết định phân luồng dựa trên intent, xác suất tin cậy và miền nghiệp vụ.

        Args:
            top_intent (str): Ý định có xác suất cao nhất từ mô hình.
            confidence (float): Xác suất tin cậy đã được hiệu chỉnh (0.0 đến 1.0).
            top_domain (str | None): Miền nghiệp vụ tương ứng với ý định.

        Returns:
            RoutingDecision: Quyết định phân luồng vận hành chi tiết.
        """
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Độ tin cậy confidence phải nằm trong khoảng [0.0, 1.0]")

        # 1. Trường hợp độ tin cậy dưới ngưỡng -> Từ chối tự động hóa, chuyển cho nhân viên
        if confidence < self.threshold:
            return RoutingDecision(
                intent=None,
                domain=None,
                route="human",
                is_unknown=True,
                requires_human_review=True,
                review_reason="LOW_CONFIDENCE",
            )

        # 2. Trường hợp ý định thuộc nhóm rủi ro cao -> Gán nhãn nhưng yêu cầu nhân viên duyệt ưu tiên
        if top_intent in self.high_risk_intents:
            return RoutingDecision(
                intent=top_intent,
                domain=top_domain,
                route="priority_human_review",
                is_unknown=False,
                requires_human_review=True,
                review_reason="HIGH_RISK_INTENT",
            )

        # 3. Trường hợp an toàn -> Tự động xử lý theo luồng intent
        return RoutingDecision(
            intent=top_intent,
            domain=top_domain,
            route=top_intent,
            is_unknown=False,
            requires_human_review=False,
            review_reason=None,
        )
