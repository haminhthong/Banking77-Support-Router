# 🚀 Banking77 Support Triage Platform — Calibrated Intent Routing with Selective Automation & Risk-Aware Human Escalation

Dịch vụ phân luồng ticket hỗ trợ khách hàng ngân hàng (77 ý định) kết hợp **TF-IDF Classification**, **Probability Calibration** (Platt Scaling), **Selective Prediction Gate** (Từ chối dự đoán khi mập mờ/rủi ro), **Exact 77-Intent Domain Taxonomy Projection**, **High-Risk Escalation Policy** và **FastAPI Serving** sẵn sàng cho môi trường vận hành thực tế.

---

## 🎯 Bài Toán Nghiệp Vụ & Tư Duy AI Engineering

Trong dịch vụ tài chính - ngân hàng, việc phân loại sai nhãn (**False Routing**) của một yêu cầu hỗ trợ (ví dụ: chuyển nhầm một sự cố *thẻ bị xâm nhập/lừa đảo* sang hàng đợi *thắc mắc tỷ giá thông thường*) gây ra thiệt hại nghiêm trọng và chi phí xử lý cao hơn nhiều so với việc chuyển ticket cho nhân viên kiểm duyệt.

Hệ thống được thiết kế theo tư duy **Decision Support & Risk-Aware Triage**: không ép mô hình phải dự đoán 100% các ticket, mà tách rõ ranh giới giữa những ca tự động hóa an toàn, những ca cần chuyên viên an ninh ưu tiên xử lý khẩn cấp, và những ca mập mờ cần nhân viên hỗ trợ thông thường.

### 📐 Canonical System Architecture

```text
                        OFFLINE ML PIPELINE

BANKING77
Train Source (10,003) + Official Test (3,080)
        ↓
1. DATA QUALITY & CONTRACT AUDIT
   ├── Schema validation
   ├── Empty-text & whitespace normalization
   ├── Conflicting-label audit (same text, different intents)
   └── Pairwise disjoint audit across all splits
        ↓
2. DEVELOPMENT SPLIT (4 Distinct Roles)
   ┌──────────────┬───────────────┬─────────────────────┐
   │              │               │                     │
 Train (70%)   Calibration (15%) Threshold Val (15%)  Official Test (3,079)
 (6,999 rows)   (1,500 rows)     (1,500 rows)           │
   │              │               │                     │
TF-IDF +       Platt Scaling     Selective Reject       │
Logistic Reg   (Sigmoid Calib)   Policy Selection       │
   │              │               │                     │
   └──────────────┴───────┬───────┘                     │
                          ↓                             │
                    Frozen Router                       │
                          ↓                             │
                     FINAL TEST ◄───────────────────────┘
                          ↓
        Versioned Artifacts & Canonical Reports
        ├── models/router.joblib
        ├── models/config.json & model_manifest.json
        └── reports/validation_metrics.json & test_metrics.json


                        ONLINE ROUTING PIPELINE

Customer Query
      ↓
Input Validation & PII Redaction Layer
      ↓
Text Feature Pipeline (TF-IDF unigram + bigram)
      ↓
77-Intent Classifier & Calibrated Probabilities
      ↓
Top-K Candidate Intents + Margin + Entropy
      ↓
Risk & Uncertainty Policy Gate
 ┌─────────────────────────────────────────────────────────────┐
 │ 1. High-Risk Check: Top-1 in High-Risk OR                   │
 │    any High-Risk in Top-K (prob >= high_risk_trigger)?      │
 ├─────────────────────────────────────────────────────────────┤
 │ 2. Uncertainty Check: Confidence < Threshold (0.45) OR      │
 │    Margin < min_margin OR Entropy > max_entropy?            │
 └──────────────────────────────┬──────────────────────────────┘
                                │
        ┌───────────────────────┼────────────────────────┐
        ▼                       ▼                        ▼
[Priority Escalation]       [Abstain]              [Auto Route]
        │                       │                        │
Priority Human Review     Human Review          Intent Queue / Auto
(Fraud / Stolen Card)    (Low Confidence)        (High Confidence)
        └───────────────────────┼────────────────────────┘
                                ▼
               PII-Safe Telemetry & Monitoring
```

---

## ✨ Các Điểm Sáng Kỹ Thuật (Engineering Highlights)

1. **Protocol 4 Split Độc Lập — Tránh Tối Đa Leakage**:
   - **Train (70%)**: Huấn luyện TF-IDF vectorizer và bộ phân loại Logistic Regression.
   - **Calibration (15%)**: Hiệu chỉnh xác suất bằng Platt Scaling (Sigmoid Calibrator), hoàn toàn tách biệt khỏi tập train.
   - **Threshold Validation (15%)**: Quét tìm ngưỡng từ chối tối ưu ($0.20 \rightarrow 0.95$) thỏa mãn ràng buộc nghiệp vụ $Coverage \ge 80\%$.
   - **Official Test (3,079 mẫu)**: Giữ nguyên vẹn làm mốc benchmark cuối cùng, không tham gia vào bất kỳ bước tuning nào.
   - Toàn bộ 6 cặp split đều được audit tự động độ trùng lặp văn bản (Pairwise Disjoint Audit).

2. **Cơ Chế Phân Luồng Chọn Lọc (Selective Prediction & Abstain Semantics)**:
   - Khi độ tin cậy dưới ngưỡng threshold ($0.45$), hệ thống chủ động **từ chối tự động hóa (Abstain)** với hợp đồng: `{"decision": "abstain", "abstained": true, "reason": "LOW_CONFIDENCE"}`.
   - Việc từ chối 18.93% các ca khó giúp độ chính xác của các ca được tự động xử lý (**Accepted Accuracy**) tăng vọt từ **86.72% lên 93.99%**, giảm tỷ lệ lỗi tự động (**Selective Risk**) xuống chỉ còn **6.01%**.

3. **High-Risk Escalation Policy (Ưu Tiên Rủi Ro Tuyệt Đối)**:
   - Các ý định nhạy cảm liên quan tới an ninh/thất thoát tài sản (`compromised_card`, `lost_or_stolen_card`, `card_swallowed`, `lost_or_stolen_phone`, `cash_withdrawal_not_recognised`) luôn được ưu tiên:
     - Nếu Top-1 là intent rủi ro: lập tức chuyển `priority_human_review` với lý do `HIGH_RISK_INTENT`, **không bao giờ** bị hạ cấp xuống review thường vì low confidence.
     - Nếu intent rủi ro xuất hiện trong Top-K với xác suất $\ge 0.20$: kích hoạt cảnh báo sớm `HIGH_RISK_CANDIDATE`.
   - Kết quả: Đạt **92.50% High-Risk Escalation Recall** và **86.45% Precision** trên tập Test độc lập.

4. **Exact 77-Class Domain Taxonomy Projection**:
   - Thay thế toàn bộ substring rules bằng từ điển taxonomy tường minh ánh xạ chính xác 77 ý định vào 10 miền nghiệp vụ (`card_services`, `transfers_payments`, `account_security`, `topup_recharge`, `transactions_refunds`, `atm_cash`, `fees_rates`, `account_management`, `app_features`, `international_services`).
   - Khắc phục triệt để lỗi phân loại nhầm các nhãn `top_up_*`. CI test tự động kiểm tra tính đầy đủ: `set(INTENT_TO_DOMAIN.keys()) == set(BANKING77_77_CLASSES)`.
   - Đạt độ chính xác ánh xạ cấp miền (**Domain Taxonomy Projection Accuracy**) là **93.73%**.

5. **REST API Production-Ready & PII-Safe Telemetry**:
   - **Vectorized Batch Inference**: Endpoint `/predict/batch` vector hóa toàn bộ danh sách truy vấn trong một lần gọi ma trận duy nhất, tăng tốc độ xử lý nhiều lần so với xử lý tuần tự.
   - **Độ chính xác dấu phẩy động**: Sử dụng raw probabilities cho quyết định của policy; chỉ làm tròn số khi format JSON response.
   - **PII Redaction**: Tự động che giấu số thẻ ngân hàng (13-19 số), email và số điện thoại trước khi ghi log sự kiện.

---

## 📊 Kết Quả Thực Nghiệm Canonical (Single Source of Truth)

Toàn bộ các chỉ số dưới đây được kết xuất tự động từ một lần chạy duy nhất trên tập Test chính thức độc lập (**3,079 mẫu**):

| Tầng Đánh Giá (Evaluation Layer) | Chỉ Số (Metric) | Giá Trị Canonical | Diễn Giải Kỹ Thuật & Nghiệp Vụ |
|---|---|---:|---|
| **Intent Classifier** | **Test Accuracy** | **86.72%** | Tỷ lệ dự đoán chính xác trên toàn bộ 77 intent |
| | **Test Macro-F1** | **86.66%** | F1 trung bình giữa 77 nhãn (tính đồng đều) |
| | **Top-3 Accuracy** | **95.94%** | Tỷ lệ nhãn thực tế nằm trong 3 gợi ý hàng đầu |
| **Taxonomy Projection** | **Domain Accuracy** | **93.73%** | Độ chính xác ánh xạ vào 10 miền nghiệp vụ (Coarse Routing) |
| **Calibration** | **Calibrated Test ECE** | **0.2176** | Expected Calibration Error sau Platt Scaling |
| | **Test Log-Loss** | **0.6674** | Độ mất mát hàm entropy chéo đa lớp |
| | **Test Brier Score** | **0.2610** | Trung bình bình phương sai số xác suất |
| **Selective Routing** | **Reject Threshold** | **0.4500** | Ngưỡng xác suất tối ưu (chọn trên Validation) |
| | **Selective Coverage** | **81.07%** | Tỷ lệ ticket đủ điều kiện tự động xử lý (2,496 / 3,079) |
| | **Selective Risk** | **6.01%** | Tỷ lệ lỗi trong số các ticket được tự động xử lý |
| | **Accepted Accuracy** | **93.99%** | Độ chính xác thực tế của luồng tự động hóa |
| **Risk-Coverage Curve** | **AURC** | **0.0283** | Diện tích dưới đường cong Risk-Coverage (càng nhỏ càng tốt) |
| | **Coverage @ 5% Risk** | **77.04%** | Độ phủ tối đa khi chặn ngưỡng lỗi tự động $\le 5\%$ |
| | **Coverage @ 3% Risk** | **67.55%** | Độ phủ tối đa khi chặn ngưỡng lỗi tự động $\le 3\%$ |
| **High-Risk Safety** | **Escalation Recall** | **92.50%** | Tỷ lệ phát hiện và leo thang thành công các ca nhạy cảm |
| | **Escalation Precision**| **86.45%** | Tỷ lệ ca được leo thang thực sự là rủi ro an ninh |

> 📌 **Đánh giá về Calibration & ECE**: Platt Scaling giúp giảm đáng kể log-loss trên validation từ 0.8722 xuống 0.7125. Tuy nhiên ECE trên tập test vẫn còn ở mức 0.2176. Do đó trong hệ thống, xác suất được coi là *calibrated scores phục vụ routing policy* chứ chưa phải xác suất hoàn hảo tuyệt đối.

---

## 📁 Cấu Trúc Mã Nguồn (Repository Structure)

```text
Banking77-Support-Router/
├── data/
│   └── raw/                       # Dữ liệu gốc train.csv, test.csv (kèm SHA256 checksum)
├── models/
│   ├── config.json                # Metadata runtime, threshold, domain taxonomy
│   ├── model_manifest.json        # Manifest đầy đủ: checksum, split sizes, parameters
│   └── router.joblib              # Pipeline TF-IDF + Platt Scaled Classifier
├── reports/
│   ├── confusion_pairs.json       # Top-20 cặp intent hay nhầm lẫn kèm ví dụ
│   ├── high_risk_metrics.json     # Báo cáo chi tiết về recall/precision ca rủi ro cao
│   ├── risk_coverage_curve.json   # 100 điểm đường cong Risk-Coverage & AURC
│   ├── test_metrics.json          # Bộ chỉ số Canonical trên tập Test độc lập
│   └── validation_metrics.json    # Báo cáo Data Quality Contract & Validation metrics
├── scripts/
│   └── download_data.py           # Script tải dữ liệu kèm kiểm tra SHA-256
├── src/
│   ├── __init__.py
│   ├── api.py                     # FastAPI REST API (/health, /predict, /predict/batch)
│   ├── data.py                    # Exact 77-class taxonomy, 4 splits, quality contracts
│   ├── evaluate.py                # Đánh giá đa tầng: Intent, Calibration, Selective, High-Risk
│   ├── policy.py                  # Routing Policy Engine: Abstain & High-Risk Precedence
│   ├── train.py                   # 4-split training, Platt scaling, threshold selection
│   └── utils.py                   # ECE, Shannon Entropy, PII Redaction, logging
├── tests/
│   ├── test_data_contract.py      # Kiểm tra 77-class taxonomy, overlap 4 split, label conflicts
│   └── test_smoke.py              # Kiểm tra policy abstain, high-risk precedence, batch API
├── Dockerfile                     # Containerization cho triển khai Production
├── Makefile                       # Lệnh tắt điều khiển dự án
├── pytest.ini                     # Cấu hình Pytest
├── README.md                      # Tài liệu dự án chuẩn mực
└── requirements.txt               # Thư viện phụ thuộc
```

---

## ⚙️ Hướng Dẫn Tái Lập Dự Án (Quickstart)

### 1. Cài đặt môi trường

```bash
# Khởi tạo môi trường ảo Python
python -m venv .venv

# Kích hoạt môi trường (Windows PowerShell)
.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate

# Cài đặt thư viện phụ thuộc
pip install -r requirements.txt
```

### 2. Tải dữ liệu & Huấn luyện Pipeline Canonical

```bash
# Tải tập dữ liệu BANKING77 và kiểm tra SHA-256
python scripts/download_data.py

# Huấn luyện mô hình, hiệu chỉnh xác suất và chọn ngưỡng trên Threshold Validation
python -m src.train

# Đánh giá toàn diện trên tập Test độc lập (sinh toàn bộ reports JSON)
python -m src.evaluate
```

### 3. Chạy Kiểm Thử Tự Động (Unit Tests & Data Contracts)

```bash
python -m pytest -v
```

### 4. Khởi chạy REST API Service

```bash
python -m uvicorn src.api:app --reload --port 8000
```

---

## 🔌 Hướng Dẫn Sử Dụng API & Contract Mới

### Endpoint: `POST /predict`

**Request:**
```json
{
  "text": "I think someone stole my card and took money at an ATM",
  "top_k": 3
}
```

**Response (Ưu tiên Rủi Ro Cao - Priority Escalation):**
```json
{
  "decision": "priority_escalation",
  "abstained": false,
  "is_unknown": false,
  "intent": "compromised_card",
  "domain": "account_security",
  "top_intent": "compromised_card",
  "confidence": 0.8124,
  "margin": 0.6512,
  "entropy": 0.9421,
  "alternatives": [
    {"intent": "compromised_card", "domain": "account_security", "confidence": 0.8124},
    {"intent": "lost_or_stolen_card", "domain": "account_security", "confidence": 0.1612},
    {"intent": "card_payment_not_recognised", "domain": "transactions_refunds", "confidence": 0.0125}
  ],
  "route": "priority_human_review",
  "requires_human_review": true,
  "review_reason": "HIGH_RISK_INTENT",
  "model_version": "banking77-tfidf-calibrated-lr-v2",
  "policy_version": "risk-aware-policy-v2"
}
```

**Response (Độ Tin Cậy Dưới Ngưỡng - Abstain Semantics):**
```json
{
  "decision": "abstain",
  "abstained": true,
  "is_unknown": true,
  "intent": null,
  "domain": null,
  "top_intent": "pending_cash_withdrawal",
  "confidence": 0.3812,
  "margin": 0.0921,
  "entropy": 2.4512,
  "alternatives": [
    {"intent": "pending_cash_withdrawal", "domain": "atm_cash", "confidence": 0.3812},
    {"intent": "declined_cash_withdrawal", "domain": "atm_cash", "confidence": 0.2891}
  ],
  "route": "human",
  "requires_human_review": true,
  "review_reason": "LOW_CONFIDENCE",
  "model_version": "banking77-tfidf-calibrated-lr-v2",
  "policy_version": "risk-aware-policy-v2"
}
```

---

## 📝 Mẫu Trình Bày Chuẩn Hóa Trong CV AI / ML Engineer

### Tiếng Việt:
- **Xây dựng Banking77 Support Triage Platform (77 intents)** với cơ chế **Selective Classification** và **Probability Calibration (Platt Scaling)**, đạt **Selective Coverage 81.07%** với **Selective Risk chỉ 6.01%** (độ chính xác luồng tự động đạt **93.99%**) trên tập Test độc lập (3,079 mẫu).
- **Thiết kế giao thức 4 split nghiêm ngặt**: Tách biệt hoàn toàn Train (70%), Calibration (15%), Threshold Validation (15%) và Official Test, ngăn chặn 100% rò rỉ dữ liệu và tránh tình trạng tối ưu hóa ngưỡng trên tập kiểm thử.
- **Xây dựng High-Risk Escalation Policy**: Tự động nhận diện và leo thang ưu tiên các ca nhạy cảm (thẻ bị xâm nhập, mất thẻ, nuốt thẻ) với **Escalation Recall đạt 92.50%** và **Precision 86.45%**, ngăn ngừa rủi ro gian lận.
- **Triển khai Production REST Service (FastAPI)**: Hỗ trợ Vectorized Batch Inference, tích hợp PII Redaction bảo mật dữ liệu khách hàng, đo lường độ mập mờ qua Margin/Entropy và thiết lập hệ thống Data Quality Contract CI/CD.

### Tiếng Anh:
- **Architected the Banking77 Support Triage Platform (77 classes)** integrating **Platt Scaling probability calibration** and **selective prediction**, achieving **81.07% coverage** at a low **6.01% selective risk** (**93.99% accepted accuracy**) on an independent test set.
- **Implemented a Strict 4-Split Pipeline**: Decoupled Train (70%), Calibration (15%), Threshold Validation (15%), and Official Test to eliminate data leakage and prevent threshold overfitting.
- **Engineered Risk-Aware Human Escalation**: Built an automated priority routing policy for high-risk intents (compromised card, stolen card, card swallowed), achieving **92.50% escalation recall** and **86.45% precision**.
- **Deployed Production FastAPI REST Microservices**: Optimized batch inference via single-pass vectorization, added PII redaction for customer privacy, and integrated CI-enforced data quality and taxonomy contracts.
