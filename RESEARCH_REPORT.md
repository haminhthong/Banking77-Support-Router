# Nghiên Cứu & Báo Cáo Cải Tiến: Calibrated Selective Routing trên BANKING77

## 1. Nghiên Cứu Nền Tảng (Foundational Research)

- **Casanueva et al. (2020)**, *Efficient Intent Detection with Dual Sentence Encoders* ([ACL Anthology](https://aclanthology.org/2020.nlp4convai-1.5/)): Giới thiệu tập dữ liệu BANKING77 với 77 ý định chi tiết, độ phân mảnh cao và nhiều cặp ý định có ngữ nghĩa rất gần nhau.
- **Geifman & El-Yaniv (2019)**, *SelectiveNet: A Deep Neural Network with an Integrated Reject Option* ([PMLR](https://proceedings.mlr.press/v97/geifman19a.html)): Cơ chế phân loại có chọn lọc (Selective Classification) với hàm mục tiêu đánh đổi giữa độ phủ (Coverage) và rủi ro tự động hóa (Selective Risk).
- **Niculescu-Mizil & Caruana (2005)**, *Predicting Good Probabilities With Supervised Learning* ([ICML](https://dl.acm.org/doi/10.1145/1102351.1102430)): Tầm quan trọng của việc hiệu chỉnh xác suất (Probability Calibration) bằng Platt Scaling và Isotonic Regression trên tập dữ liệu độc lập.

---

## 2. Thiết Kế Pipeline & Giao Thức Phân Chia Dữ Liệu (4-Split Protocol)

Nhằm đảm bảo tính khách quan và ngăn ngừa triệt để rò rỉ thông tin (Data Leakage), hệ thống thiết lập giao thức 4 vai trò dữ liệu hoàn toàn tách biệt:

```text
BANKING77 (13,082 mẫu)
├── Train (6,999 mẫu - 70% source train): Huấn luyện TF-IDF unigram+bigram & Logistic Regression.
├── Calibration (1,500 mẫu - 15% source train): Hiệu chỉnh xác suất bằng Platt Scaling (Sigmoid Calibrator).
├── Threshold Validation (1,500 mẫu - 15% source train): Quét chọn ngưỡng từ chối tối ưu thỏa mãn Coverage >= 80%.
└── Official Test (3,079 mẫu - độc lập): Giữ nguyên vẹn, chỉ dùng cho bước báo cáo hiệu năng cuối cùng.
```

Tất cả 6 cặp phân tách (`train ↔ cal`, `train ↔ threshold_val`, `train ↔ test`, `cal ↔ threshold_val`, `cal ↔ test`, `threshold_val ↔ test`) được kiểm định tự động qua `summarize_split_quality()` để đảm bảo tính phân tách (Pairwise Disjointness).

---

## 3. Kết Quả Thực Nghiệm Canonical Trên Tập Test Chính Thức (3,079 Mẫu)

Dưới đây là bộ chỉ số chuẩn xác duy nhất kết xuất từ `reports/test_metrics.json` và `reports/validation_metrics.json`:

### 3.1. Phân Loại Ý Định & Phóng Chiếu Miền Nghiệp Vụ (Intent & Taxonomy Projection)
- **Test Accuracy**: **86.72%** (2,670 / 3,079)
- **Test Macro-F1**: **86.66%**
- **Top-3 Accuracy**: **95.94%** (nhãn thực tế nằm trong 3 gợi ý hàng đầu)
- **Domain Taxonomy Projection Accuracy**: **93.73%** (quy đổi 77 ý định chi tiết sang 10 miền nghiệp vụ cấp cao)

### 3.2. Đánh Giá Hiệu Chỉnh Xác Suất (Calibration Layer)
- **Validation Log-Loss**: Giảm từ **0.8722** (mô hình gốc) xuống **0.7125** (sau Platt Scaling).
- **Test Log-Loss**: **0.6674**
- **Test Multi-class Brier Score**: **0.2610**
- **Test ECE (Expected Calibration Error)**: **0.2176**
  > *Nhận định kỹ thuật*: Mặc dù Platt Scaling tối ưu hóa đáng kể log-loss, ECE trên test vẫn còn ở mức 0.2176 do đặc thù 77 lớp của BANKING77. Xác suất đầu ra được định vị là *calibrated scores* phục vụ ngưỡng quyết định hơn là xác suất hoàn hảo tuyệt đối.

### 3.3. Phân Loại Có Chọn Lọc (Selective Routing & Risk-Coverage Curve)
- **Reject Threshold**: **0.4500** (chọn trên Threshold Validation với ràng buộc Coverage $\ge 80\%$).
- **Selective Coverage**: **81.07%** (2,496 ticket được mô hình tự động xử lý).
- **Selective Risk**: **6.01%** (chỉ 150 ticket bị phân loại sai trong 2,496 ca được tự động hóa).
- **Accepted Accuracy**: **93.99%** (tăng mạnh 7.27% so với mức 86.72% khi không có cơ chế từ chối).
- **AURC (Area Under the Risk-Coverage Curve)**: **0.0283**.
- **Coverage @ 5% Risk**: **77.04%**.
- **Coverage @ 3% Risk**: **67.55%**.

### 3.4. Kiểm Soát Rủi Ro Cao (High-Risk Escalation Safety)
- Tổng số mẫu rủi ro cao thực tế trong tập test: 200 mẫu (`compromised_card`, `lost_or_stolen_card`, `card_swallowed`, `lost_or_stolen_phone`, `cash_withdrawal_not_recognised`).
- **High-Risk Escalation Recall**: **92.50%** (185 / 200 ca nhạy cảm được đưa vào hàng đợi ưu tiên `priority_human_review`).
- **High-Risk Escalation Precision**: **86.45%** (185 / 214 ca trong hàng đợi ưu tiên là rủi ro thực sự).

---

## 4. Phân Tích Lỗi Chuyên Sâu (Top Confusion Pairs)

Dữ liệu BANKING77 chứa nhiều cặp ý định phân định rất mỏng (fine-grained distinctions). Báo cáo từ `reports/confusion_pairs.json` chỉ ra các cụm lỗi tiêu biểu:

1. **`verify_my_identity` vs `why_verify_identity`** (8 lỗi, margin TB: 0.0764):
   - *Ví dụ*: *"What can I use to verify my identify?"*, *"What do you need for my identity check?"*
   - *Bản chất*: Một bên hỏi về quy trình thực hiện, một bên hỏi về lý do yêu cầu xác minh danh tính.
2. **`request_refund` vs `Refund_not_showing_up`** (6 lỗi, margin TB: 0.0421):
   - *Ví dụ*: *"I need to do a refund"*, *"How do I apply for a refund?"*
   - *Bản chất*: Yêu cầu hoàn tiền mới bị nhầm với việc tra cứu một khoản hoàn tiền chưa tới tài khoản.
3. **`wrong_exchange_rate_for_cash_withdrawal` vs `card_payment_wrong_exchange_rate`** (6 lỗi, margin TB: 0.1071):
   - *Ví dụ*: *"Why is the exchange rate wrong for my international withdrawal?"*
   - *Bản chất*: Lỗi tỷ giá khi rút tiền mặt ngoại tệ qua ATM bị nhầm với lỗi tỷ giá khi thanh toán quẹt thẻ tại POS.
4. **`virtual_card_not_working` vs `getting_virtual_card`** (5 lỗi, margin TB: 0.2239):
   - *Ví dụ*: *"Why was my virtual card declined when attempting to setup automatic billing?"*
5. **`top_up_failed` vs `top_up_reverted`** (5 lỗi, margin TB: 0.1048):
   - *Ví dụ*: *"Why did the app deny my top up?"*, *"Why was my top-up unsuccessful?"*

---

## 5. Hướng Phát Triển Tiếp Theo (P2 Roadmap)

1. **Out-of-Scope (OOS) Benchmark Thật**: Xây dựng tập kiểm thử OOS chuyên biệt gồm: các câu hỏi ngoài ngành ngân hàng, chit-chat/lời chào, và các sản phẩm ngân hàng không hỗ trợ (ví dụ: vay thế chấp) để đo lường tỷ lệ chấp nhận sai (False Acceptance Rate - FAR).
2. **Modern Embedding Baseline (MiniLM / DistilBERT)**: Benchmark mô hình Transformer trên cùng một đường ống (Calibration $\rightarrow$ Selective Gate $\rightarrow$ Risk Policy) để so sánh hiệu năng F1, Selective Risk và độ trễ phục vụ CPU (Latency).
3. **Domain-Specific / Tiered Thresholds**: Nghiên cứu thiết lập ngưỡng từ chối theo từng miền nghiệp vụ thay vì một ngưỡng toàn cục (Global Threshold).
