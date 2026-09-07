# 🛡️ Risk-Aware Banking Support Triage System — Calibrated Intent Intelligence, Multi-Tiered Safety Gates & Operational Decision Routing

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116+-009688.svg)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.5%20%E2%86%92%201.9-F7931E.svg)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/tests-32%20passed%20(100%25)-brightgreen.svg)]()
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

Hệ thống phân luồng và ra quyết định hỗ trợ khách hàng ngân hàng tự động (**Risk-Aware Banking Support Triage System**). Kết hợp **TF-IDF + Char N-gram Feature Union**, **Platt Probability Calibration**, **Out-of-Distribution (OOD) Detection**, **Decoupled Security Risk Scanner**, **Operational Queue Routing**, và **FastAPI Serving** với vòng phản hồi chuyên viên (**Human-in-the-Loop Feedback Loop**).

---

## 🎯 1. Định Vị Dự Án & Bài Toán Nghiệp Vụ Thực Tế

Trong môi trường tài chính - ngân hàng, bài toán không đơn giản là *"phân loại văn bản vào 77 nhãn"*. Một mô hình ML với độ chính xác 87% vẫn có thể gây thảm họa nếu tự động phân luồng sai một sự cố thẻ bị xâm nhập gian lận (`compromised_card`) vào hàng đợi thắc mắc thường, hoặc cố ép một câu hỏi ngoài phạm vi (`"Làm sao vay mua nhà 30 năm?"`) vào một trong 77 nhãn thẻ.

Hệ thống được thiết kế theo tư duy **Operational Decision System** hoàn chỉnh:

```text
Customer Support Ticket
        │
        ▼
Request Validation & PII Privacy Shield
        │
        ▼
Intent Understanding (Calibrated Probabilities)
        │
        ▼
Uncertainty / Margin / Entropy Profiling
        │
        ▼
Out-of-Distribution (OOD) Guard
        │
        ▼
Security Risk Assessment (All Risk Classes Scanned)
        │
        ▼
Operational Routing Policy Engine
        │
   ┌────┴───────────────────────────┬────────────────────────────────┐
   ▼                                ▼                                ▼
AUTO ROUTE                     HUMAN REVIEW                  PRIORITY REVIEW
Normal high-confidence         Ambiguous queries             Fraud / stolen card /
operations queue               or Out-of-Scope (OOD)         security incidents
   │                                │                                │
   └────────────────────────────────┼────────────────────────────────┘
                                    ▼
                 Operational Queues + Structured Telemetry
                                    │
                                    ▼
                    Human Reviewer Feedback Loop
```

ML model chỉ là một thành phần cảm biến (signal generator); toàn bộ giá trị nằm ở tầng **Safety Gates, Decision Policy và Queue Resolution**.

---

## 📐 2. Kiến Trúc Canonical System

### A. Offline Machine Learning & Policy Pipeline

```text
BANKING77 RAW DATA (Train 10,003 rows + Official Test 3,080 rows)
        │
        ▼
1. DATA QUALITY & INTEGRITY CONTRACT
   ├── Schema validation (['text', 'category'] -> ['text', 'intent'])
   ├── Whitespace & NFKC normalization
   ├── Conflicting-label audit (BEFORE deduplication)
   └── Deduplication (only after audit)
        │
        ▼
2. DUAL BENCHMARK SPLIT STRATEGY
   ├── [A] Official Published Split: Preserves published test benchmark (3,080 rows).
   └── [B] Strict Decontaminated Split: Purges normalized test duplicates from train.
        │
        ▼
3. 4-WAY DEVELOPMENT ROLE PARTITION
   ┌────────────────────┬────────────────────┬────────────────────────┐
   │                    │                    │                        │
Train (70%)        Calibration (15%)    Threshold Val (15%)     Official Test (3,080)
(6,999 rows)        (1,500 rows)         (1,500 rows)             (Untouched Benchmark)
   │                    │                    │                        │
Word + Char         Platt Scaling        Joint Grid Search        Immutable Evaluation
TF-IDF Pipeline     (Sigmoid Calib)      ├── Reject Threshold     ├── Classification
   │                    │                └── High-Risk Trigger    ├── Calibration
   └────────────────────┴──────────┬─────────┘                    ├── Selective Risk
                                   ▼                              ├── Operational Safety
                         FROZEN MODEL BUNDLE                      └── OOD Benchmark
                         ├── router.joblib
                         ├── model_manifest.json (SHA-256 verified)
                         ├── config.json
                         ├── taxonomy.json
                         └── policy.json
```

### B. Online Request Triage Pipeline

```text
                            CUSTOMER TICKET
                                   │
                                   ▼
                         1. REQUEST VALIDATION
                         Length & Encoding Checks
                                   │
                                   ▼
                           2. PRIVACY LAYER
                         PII Masking (Card/Phone/Email/Account)
                                   │
                                   ▼
                           3. INTENT MODEL
                    Word (1-2) + Char (3-5) TF-IDF + LR
                                   │
                                   ▼
                         77 CALIBRATED SCORES
                                   │
             ┌─────────────────────┼─────────────────────┐
             │                     │                     │
             ▼                     ▼                     ▼
          Top-1               Uncertainty             Security
        Prediction              Signals             Risk Scanner
     (Intent & Domain)     Confidence / Margin    ALWAYS scans ALL
                              and Entropy           risk classes
             │                     │                     │
             └─────────────────────┼─────────────────────┘
                                   ▼
                        4. OOD / QUALITY GATE
                     Dispersion + Content Keyword Footprint
                                   │
                                   ▼
                        5. POLICY ENGINE
                     Priority Precedence Order:
                     
                     1. SECURITY THREAT DETECTED?
                        └─► YES: PRIORITY HUMAN REVIEW
                     2. OUT-OF-DISTRIBUTION (OOD)?
                        └─► YES: GENERAL HUMAN REVIEW
                     3. UNCERTAINTY GATE TRIGGERED?
                        └─► YES: GENERAL HUMAN REVIEW
                     4. SAFE & CONFIDENT?
                        └─► YES: SAFE AUTO-ROUTE
                                   │
                     ┌─────────────┼─────────────┐
                     ▼             ▼             ▼
              PRIORITY REVIEW HUMAN REVIEW   AUTO ROUTE
                     │             │             │
                     └─────────────┼─────────────┘
                                   ▼
                         6. QUEUE RESOLUTION
                  Intent != Queue (Operational Queues)
                                   │
                                   ▼
                         7. STRUCTURED TELEMETRY
                     JSONL Event Log (PII-Safe)
                                   │
                                   ▼
                         8. HUMAN FEEDBACK LOOP
                     Reviewer Intent Corrections -> Retraining Pool
```

---

## 🔬 3. Kết Quả Thực Nghiệm Toàn Diện (Canonical Benchmarks)

Các chỉ số được đo lường độc lập trên tập **Official Test Benchmark (3,080 mẫu nguyên vẹn)**:

| Tầng Đánh Giá | Chỉ Số (Metric) | Official Benchmark | Strict Decontaminated | Ý Nghĩa Kỹ Thuật & Vận Hành |
|---|---|---:|---:|---|
| **Classification** | **Accuracy** | **89.38%** | **89.42%** | Tỷ lệ dự đoán chính xác trên 77 intent |
| | **Macro-F1** | **89.35%** | **89.38%** | Tính đồng đều giữa các nhóm nhãn |
| | **Top-3 Accuracy** | **96.92%** | **96.98%** | Nhãn đúng nằm trong 3 gợi ý hàng đầu |
| | **Domain Accuracy**| **95.49%** | **95.42%** | Ánh xạ vào 10 miền nghiệp vụ (Coarse Routing) |
| **Calibration** | **Calibrated ECE** | **0.2045** | **0.2111** | Expected Calibration Error sau Platt Scaling |
| | **Log-Loss** | **0.5646** | **0.5707** | Cross-entropy loss sau hiệu chỉnh xác suất |
| | **Brier Score** | **0.2212** | **0.2241** | Trung bình bình phương sai số xác suất |
| **Model Selective**| **Reject Threshold** | **0.4800** | **0.4600** | Ngưỡng xác suất tối ưu chọn trên Validation |
| | **Acceptance Coverage** | **82.82%** | **84.90%** | Tỷ lệ ticket có $Confidence \ge Threshold$ |
| | **Accepted Accuracy** | **95.73%** | **95.60%** | Độ chính xác chỉ tính trên các ca accepted |
| | **Selective Risk** | **4.27%** | **4.40%** | Tỷ lệ lỗi trong các ca accepted (ngưỡng SLO $\le 5\%$) |
| **Operational Policy** | **Auto-Route Coverage** | **77.01%** | **78.77%** | Tỷ lệ thực tế được chuyển thẳng vào hàng đợi tự động |
| *(Full System)* | **Auto-Route Accuracy** | **95.70%** | **95.75%** | **Độ chính xác thực tế của luồng tự động hóa** |
| | **Auto-Route Error** | **4.30%** | **4.25%** | Tỷ lệ lỗi thực tế của toàn bộ hệ thống |
| | **Human Review Rate** | **16.36%** | **14.19%** | Tỷ lệ ca mập mờ / OOD chuyển kiểm duyệt thường |
| | **Priority Escalation Rate** | **6.62%** | **7.05%** | Tỷ lệ ca an ninh chuyển hàng đợi khẩn cấp |
| **High-Risk Safety**| **High-Risk Recall** | **94.00%** | **97.00%** | Tỷ lệ phát hiện và leo thang thành công ca nhạy cảm |
| | **High-Risk Precision**| **92.16%** | **89.40%** | Độ tin cậy của các ca được leo thang khẩn cấp |
| **Curve & Limits** | **AURC** | **0.0190** | **0.0206** | Area Under Risk-Coverage Curve (càng nhỏ càng tốt) |
| | **Coverage @ 5% Risk** | **86.59%** | **86.56%** | Độ phủ tối đa khi chặn ngưỡng lỗi $\le 5\%$ |
| | **Coverage @ 3% Risk** | **75.65%** | **76.82%** | Độ phủ tối đa khi chặn ngưỡng lỗi $\le 3\%$ |
| **OOD Benchmark** | **OOD Recall** | **82.00%** | **82.00%** | Tỷ lệ nhận diện các truy vấn ngoài phạm vi |
| *(Dedicated 50-item)*| **False Acceptance Rate** | **8.00%** | **8.00%** | Tỷ lệ truy vấn OOD bị auto-route nhầm |
| | **Safe Containment Rate** | **92.00%** | **92.00%** | Tỷ lệ câu hỏi OOD được giữ an toàn khỏi auto-route |

---

## 🌟 4. Các Điểm Nâng Cấp Kỹ Thuật (Engineering Highlights)

### 1. Data Quality Contract: Audit trước Deduplication
- **Lỗi trước đây**: Hàm clean gọi `drop_duplicates(subset=["text"])` trước khi audit nhãn xung đột, khiến hai câu cùng text nhưng khác nhãn bị xoá mất 1 dòng, che giấu lỗi data quality.
- **Thiết kế mới**:
  `Raw CSV` $\rightarrow$ `Schema Validation` $\rightarrow$ `NFKC Normalization` $\rightarrow$ `Audit Conflicting Labels` $\rightarrow$ `Audit Duplicates` $\rightarrow$ `Fail on Violation` $\rightarrow$ `Clean Dataset`.

### 2. Dual Benchmarks & Immutable Official Test
- Tập dữ liệu gốc Banking77 từ PolyAI tồn tại **11 câu trùng lặp ngữ nghĩa** giữa tập train và tập test.
- Hệ thống hỗ trợ 2 benchmark rõ ràng:
  - **Official Benchmark**: Giữ nguyên phân chia công bố để đối sánh học thuật.
  - **Strict Decontaminated Benchmark**: Loại bỏ các mẫu dev trùng lặp với test bằng normalized fingerprint, đảm bảo **0% leakage chéo** giữa dev và test.
- `load_official_test()` là **Report-Only & Immutable**: Tuyệt đối không drop duplicates hay lọc bỏ mẫu, bảo toàn nguyên vẹn 3,080 hàng công bố.

### 3. Tách Biệt Hoàn Toàn Display Top-K và Safety Scanner
- **Lỗi trước đây**: Safety scan duyệt qua danh sách `top_k_candidates`. Người dùng truyền `top_k=1` thì policy chỉ kiểm tra Top-1, truyền `top_k=5` thì policy quét Top-5 $\Rightarrow$ Hành vi an ninh thay đổi theo tham số hiển thị!
- **Thiết kế mới**: `RiskAssessor` **luôn luôn quét toàn bộ các lớp rủi ro cao trên toàn bộ 77 xác suất**, độc lập tuyệt đối với tham số `top_k` của người dùng.

### 4. Tách Rời Prediction, Risk Assessment, và Routing Decision
Tách thành 3 hợp đồng dữ liệu độc lập tuân thủ Single Responsibility:
- `IntentPrediction`: Intent dự đoán của model, Domain tương ứng, Confidence, Margin, Entropy, và danh sách Alternatives.
- `RiskAssessment`: Cờ cảnh báo rủi ro cao, intent rủi ro phát hiện, điểm rủi ro, cờ OOD.
- `RoutingDecision`: Hành động (`auto_route`, `human_review`, `priority_human_review`), `queue_id`, `priority`, cờ review, reason codes.
- Khi có cảnh báo `HIGH_RISK_CANDIDATE`, hệ thống ưu tiên leo thang nhưng **không thay đổi hoặc gán sai intent/domain dự đoán ban đầu của mô hình**.

### 5. Intent $\ne$ Queue (Operational Queue Resolution)
Intent chi tiết được ánh xạ sang các hàng đợi vận hành chuyên biệt:
- `card_arrival`, `card_delivery_estimate` $\rightarrow$ `card_operations`
- `compromised_card`, `lost_or_stolen_card` $\rightarrow$ `fraud_security_queue` (Priority: `critical`)
- `top_up_failed` $\rightarrow$ `topup_support_queue` (Priority: `high`)
- Abstain / Out-of-Distribution $\rightarrow$ `general_human_review_queue`

### 6. Tối Ưu Hóa Ngưỡng Đồng Thời Theo Safety SLO
Không dùng ngưỡng hard-code 0.45 và trigger 0.20 cố định. Bộ tối ưu hóa duyệt grid search trên tập Threshold Validation độc lập:
$$\max \text{Auto-Route Coverage} \quad \text{s.t.} \quad \text{Error} \le 5\% \quad \text{and} \quad \text{High-Risk Recall} \ge 95\%$$

### 7. Out-of-Distribution (OOD) Guard & Benchmark Set
- Bộ lọc OOD Guard đa tầng phát hiện các câu hỏi phi ngân hàng (crypto, bảo hiểm, bất động sản, chit-chat, gibberish) dựa trên:
  - Kiểm tra cú pháp, độ dài cực ngắn, chuỗi ký tự lặp.
  - Phân tích độ phủ từ khóa nội dung (Content-word Footprint, lọc bỏ stopwords).
  - Độ phân tán xác suất cực hạn (xác suất $< 0.22$ trong bài toán 77 nhãn).
- Bộ dữ liệu đánh giá riêng biệt `data/evaluation/ood.jsonl` (50 ca thực tế) kiểm chứng tỷ lệ an toàn đạt **92.00%**.

### 8. Production ModelBundle Contract & Readiness Probe
Gói đóng gói mô hình bao gồm đầy đủ:
- `router.joblib`: Binary weights.
- `model_manifest.json`: Băm SHA-256 của artifact, checksums dữ liệu nguồn, runtime phiên bản (`scikit-learn 1.9.0`, `numpy 2.4.6`, `joblib 1.5.3`).
- `config.json`, `taxonomy.json`, `policy.json`.
- Endpoint `GET /health/ready` thực sự nạp weights, kiểm tra checksum SHA-256, kiểm định 77 lớp nhãn và tính toàn vẹn của bảng taxonomy trước khi nhận traffic.

### 9. Structured Telemetry & Human Feedback Loop
- Log sự kiện có tầng lọc dữ liệu nhạy cảm PII (`redact_pii` che số thẻ 13-19 số, email, số điện thoại, số tài khoản).
- Endpoint `POST /v1/feedback` cho phép ghi nhận các ca chuyên viên hiệu chỉnh nhãn đúng vào `reports/feedback_events.jsonl`, tạo nguồn dữ liệu cho chu kỳ tái huấn luyện trong tương lai.

---

## 📂 5. Cấu Trúc Mã Nguồn

```text
Banking77-Support-Router/
├── configs/
│   ├── model.yaml                     # Cấu hình siêu tham số TF-IDF và Logistic Regression
│   ├── routing_policy.yaml            # Cấu hình SLOs tối ưu hóa và ngưỡng runtime
│   └── taxonomy.yaml                  # Ánh xạ 77 intent, 10 domain và operational queues
├── data/
│   ├── raw/                           # Dữ liệu gốc train.csv, test.csv
│   └── evaluation/
│       └── ood.jsonl                  # Dataset đánh giá Out-of-Distribution độc lập
├── models/
│   ├── router.joblib                  # Frozen Model weights
│   ├── model_manifest.json            # Manifest kiểm định băm SHA-256 & runtime
│   ├── config.json                    # Cấu hình runtime & domain mapping
│   ├── taxonomy.json                  # Snapshot taxonomy đã nạp
│   └── policy.json                    # Snapshot policy đã tối ưu
├── reports/
│   ├── test_metrics.json              # Kết quả đánh giá trên Official Test Benchmark
│   ├── strict_decontaminated_test_metrics.json  # Kết quả đánh giá trên Strict Benchmark
│   ├── risk_coverage_curve.json       # Tọa độ đường cong Risk-Coverage & AURC
│   ├── high_risk_metrics.json         # Báo cáo an ninh rủi ro cao
│   ├── ood_metrics.json               # Báo cáo phát hiện Out-of-Scope
│   └── confusion_pairs.json           # Top 20 cặp nhãn dễ nhầm lẫn nhất
├── src/
│   ├── banking_router/
│   │   ├── config.py                  # Trình nạp YAML cấu hình
│   │   ├── utils.py                   # Tiện ích logging, ECE, entropy, save_json
│   │   ├── data/                      # Tầng hợp đồng dữ liệu & chuẩn hóa
│   │   │   ├── contracts.py           # 77 nhãn chuẩn & domain mapping
│   │   │   ├── loader.py              # Đọc CSV & nạp official test bất biến
│   │   │   ├── normalization.py       # NFKC text normalization & SHA-256
│   │   │   ├── audit.py               # Audit conflicts & duplicates trước dedup
│   │   │   └── split.py               # Phân chia 4 vai trò & dual benchmark
│   │   ├── modeling/                  # Tầng huấn luyện & hiệu chỉnh xác suất
│   │   │   ├── pipeline.py            # FeatureUnion (Word 1-2 + Char 3-5)
│   │   │   ├── calibration.py         # Platt Scaling Calibrator
│   │   │   ├── artifact.py            # ModelBundle contract & SHA-256 check
│   │   │   └── training.py            # Huấn luyện & tối ưu ngưỡng đồng thời
│   │   ├── routing/                   # Bộ máy phân luồng & quyết định
│   │   │   ├── schemas.py             # Decoupled Prediction, Risk, Decision
│   │   │   ├── taxonomy.py            # Taxonomy & Queue Resolver
│   │   │   ├── risk.py                # Security Risk Scanner (độc lập top-k)
│   │   │   ├── ood.py                 # Out-of-Distribution Query Guard
│   │   │   ├── policy.py              # Routing Policy Engine đa tầng
│   │   │   └── service.py             # RoutingService orchestrator
│   │   ├── evaluation/                # Tầng đánh giá đa tầng
│   │   │   ├── classification.py      # Accuracy, Macro-F1, Top-3, Domain Acc
│   │   │   ├── calibration.py         # ECE, Log-loss, Brier Score
│   │   │   ├── selective.py           # Coverage, Selective Risk, AURC
│   │   │   ├── safety.py              # Operational Safety & Confusion Pairs
│   │   │   ├── ood_eval.py            # Đánh giá OOD Recall & False Acceptance
│   │   │   └── evaluator.py           # Bộ đánh giá benchmark tổng hợp
│   │   ├── telemetry/                 # Giám sát & An toàn thông tin
│   │   │   ├── privacy.py             # PII Masking regex
│   │   │   └── events.py              # Ghi nhận sự kiện & Feedback Loop
│   │   └── api/                       # HTTP REST Adapter
│   │       ├── schemas.py             # Pydantic schemas v1/route & feedback
│   │       └── app.py                 # FastAPI endpoints (live, ready, route)
│   ├── data.py                        # Facade tương thích ngược
│   ├── policy.py                      # Facade tương thích ngược
│   ├── train.py                       # CLI Runner huấn luyện
│   ├── evaluate.py                    # CLI Runner đánh giá
│   ├── api.py                         # Facade tương thích ngược
│   └── utils.py                       # Facade tương thích ngược
├── tests/
│   ├── unit/
│   │   ├── test_data_invariants.py    # Kiểm định hợp đồng dữ liệu & benchmark
│   │   └── test_policy_invariants.py  # Kiểm định tính độc lập của safety scan
│   ├── integration/
│   │   └── test_service_and_api.py    # Kiểm định RoutingService, Bundle, API
│   ├── test_data_contract.py          # Unit tests dữ liệu chuẩn hóa
│   └── test_smoke.py                  # Unit tests chính sách & API smoke
├── scripts/
│   ├── download_data.py               # Tải dữ liệu mẫu BANKING77
│   └── manual_api_test.py             # Kịch bản kiểm thử API thủ công
├── Dockerfile                         # Container image với Readiness check
└── requirements.txt                   # Phụ thuộc môi trường chính xác
```

---

## 🚀 6. Hướng Dẫn Sử Dụng Nhanh (Quickstart)

### 1. Cài Đặt Môi Trường
```bash
# Tạo môi trường ảo
python -m venv .venv
source .venv/bin/activate  # Trên Windows: .venv\Scripts\activate

# Cài đặt thư viện phụ thuộc
pip install -r requirements.txt
```

### 2. Huấn Luyện & Tối Ưu Hóa Chính Sách
```bash
# Huấn luyện trên chuẩn Official Published Split
python -m src.train --benchmark official

# HOẶC huấn luyện trên chuẩn Strict Decontaminated Split (loại bỏ trùng lặp test)
python -m src.train --benchmark strict_decontaminated
```

### 3. Đánh Giá Toàn Diện Benchmark
```bash
python -m src.evaluate
```

### 4. Chạy Toàn Bộ Test Suite
```bash
python -m pytest tests/ -v
```

### 5. Khởi Động REST API Service
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

Kiểm tra trạng thái sẵn sàng:
```bash
curl -X GET http://127.0.0.1:8000/health/ready
```

### 6. Phân Luồng Ticket Khách Hàng (Canonical `/v1/route`)
```bash
curl -X POST "http://127.0.0.1:8000/v1/route" \
     -H "Content-Type: application/json" \
     -d '{"text": "I lost my phone and card, help me immediately!", "top_k": 3}'
```

**JSON Response mẫu (Decoupled Output):**
```json
{
  "request_id": "req_8d66cb0260ce",
  "prediction": {
    "intent": "lost_or_stolen_phone",
    "domain": "account_security",
    "confidence": 0.8133,
    "margin": 0.6215,
    "entropy": 1.1245,
    "alternatives": [
      {
        "intent": "lost_or_stolen_phone",
        "domain": "account_security",
        "confidence": 0.8133
      },
      {
        "intent": "lost_or_stolen_card",
        "domain": "account_security",
        "confidence": 0.1241
      },
      {
        "intent": "compromised_card",
        "domain": "account_security",
        "confidence": 0.0312
      }
    ]
  },
  "risk": {
    "high_risk_detected": true,
    "high_risk_intent": "lost_or_stolen_phone",
    "high_risk_score": 0.8133,
    "ood_detected": false
  },
  "decision": {
    "action": "priority_human_review",
    "queue": "fraud_security_queue",
    "priority": "critical",
    "requires_human_review": true,
    "reason_codes": [
      "HIGH_RISK_INTENT"
    ]
  },
  "metadata": {
    "model_version": "banking77-support-triage-v3",
    "policy_version": "risk-aware-triage-v3"
  }
}
```

### 7. Gửi Phản Hồi Chuyên Viên (Feedback Loop)
```bash
curl -X POST "http://127.0.0.1:8000/v1/feedback" \
     -H "Content-Type: application/json" \
     -d '{
       "request_id": "req_8d66cb0260ce",
       "reviewed_intent": "lost_or_stolen_card",
       "reviewer_id": "senior_agent_07",
       "notes": "Customer confirmed card was stolen along with phone"
     }'
```

---

## 🐳 7. Đóng Gói Docker Container

```bash
# 1. Đảm bảo model artifact đã được tạo trước khi build
python -m src.train

# 2. Build Docker container image
docker build -t banking77-support-router:v3 .

# 3. Chạy container với port 8000
docker run -d -p 8000:8000 --name banking-router banking77-support-router:v3

# 4. Kiểm tra sức khỏe container (Readiness Probe)
curl http://127.0.0.1:8000/health/ready
```

---

## 📜 8. License

Phát hành theo giấy phép [MIT License](LICENSE). Bộ dữ liệu gốc thuộc bản quyền của [PolyAI Banking77](https://github.com/PolyAI-LDN/task-specific-datasets).
