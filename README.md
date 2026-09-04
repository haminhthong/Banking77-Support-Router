# 🚀 02 — AI Customer Support Router (BANKING77)

Dịch vụ AI phân loại tự động yêu cầu hỗ trợ khách hàng ngân hàng (77 ý định) tích hợp **Selective Classification** (Từ chối dự đoán rủi ro), **Probability Calibration** (Hiệu chỉnh xác suất ECE), **High-Risk Escalation Policy** và **Domain Grouping** (Gom nhóm miền nghiệp vụ).

---

## 🎯 Bài Toán Nghiệp Vụ & Tư Duy AI Engineering

Trong ngành Tài chính - Ngân hàng, việc **phân loại sai nhãn (False Routing)** của một ticket hỗ trợ (ví dụ: nhầm ca *thẻ bị mất* thành *thắc mắc tỷ giá*) gây ra rủi ro nghiệp vụ và chi phí vận hành cao hơn nhiều so với việc chuyển ticket đó cho nhân viên xử lý.

```text
[Khách hàng nhập câu hỏi] 
       │
       ▼
[Chuẩn hóa văn bản & TF-IDF (1-2 gram)] 
       │
       ▼
[Mô hình Logistic Regression + Platt Scaling Calibration]
       │
       ├──► Dự đoán Top Intent & Calibrated Confidence
       ├──► Phân loại Miền Nghiệp Vụ (Domain - 10 nhóm lớn)
       │
       ▼
[Bộ Quy Tắc Vận Hành - Routing Policy Engine]
       │
       ├── ❶ Confidence < Ngưỡng (0.485) ──► Route: "human" (Low Confidence / Unknown)
       ├── ❷ Intent thuộc Nhóm Rủi Ro  ──► Route: "priority_human_review" (Fraud/Lost Card)
       └── ❸ An toàn & Độ tin cậy cao  ──► Route: [Intent Queue tự động]
```

### 💡 Điểm cốt lõi về tư duy AI Engineering:
- **Không chạy theo Accuracy thuần túy**: Thay vì ép mô hình đưa ra dự đoán cho 100% ticket, hệ thống tự động từ chối tự động hóa (Reject Option) các ca có độ tin cậy thấp.
- **Tối ưu hóa Selective Risk**: Ngưỡng từ chối được chọn tự động trên tập Validation bằng thuật toán quét ngưỡng với ràng buộc độ phủ tối thiểu $\ge 80\%$. Kết quả thực tế trên tập Test độc lập giúp tự động hóa **78.17%** lượng ticket với tỷ lệ lỗi (Selective Risk) chỉ **4.32%**.

---

## ✨ Các Điểm Sáng Kỹ Thuật (Technical Highlights)

1. **Probability Calibration (Hiệu chỉnh xác suất ECE)**:
   - Logistic Regression mặc định đưa ra xác suất thô chưa được hiệu chỉnh. Hệ thống áp dụng **Platt Scaling (Sigmoid Calibrator)** fit trên tập Validation để đưa xác suất dự đoán về đúng tần suất xuất hiện thực tế, giúp giảm chỉ số **Expected Calibration Error (ECE)** từ `0.2505` xuống `0.2171`.

2. **Domain Grouping (Gom nhóm Miền Nghiệp Vụ Cấp Cao)**:
   - Tự động gom 77 ý định chi tiết thành 10 miền nghiệp vụ chính (`card_services`, `atm_cash`, `account_security`, `transfers_payments`, `fees_rates`, `account_management`, `topup_recharge`, `transactions_refunds`, `app_features`, `international_services`).
   - Đạt độ chính xác cấp miền (**Domain-level Accuracy**) lên tới **93.99%**, hỗ trợ phân luồng cấp cao cho các phòng ban vận hành.

3. **High-Risk Escalation Policy (Kiểm soát rủi ro nhạy cảm)**:
   - Logic vận hành tách biệt hoàn toàn khỏi mô hình AI (`src/policy.py`). Các ý định nhạy cảm như `compromised_card`, `lost_or_stolen_card`, `card_swallowed` luôn được gắn cờ `requires_human_review=True` và chuyển sang luồng ưu tiên `priority_human_review`.

4. **REST API Production-Ready (FastAPI)**:
   - Lazy-load model artifact một lần duy nhất vào bộ nhớ.
   - Hỗ trợ endpoint đơn `/predict` và endpoint xử lý hàng loạt `/predict/batch`.
   - Chuẩn hóa Schema Pydantic, báo cáo chi tiết top-$k$ alternatives giúp nhân viên tư vấn dễ dàng tra cứu.

---

## 📊 Kết Quả Thực Nghiệm Trên Tập Test Độc Lập

Pipeline v2 tách dữ liệu thành bốn vai trò độc lập: train để fit TF-IDF/classifier, calibration để fit Platt scaling, threshold-validation để chọn reject threshold và official test chỉ để báo cáo cuối. Kết quả mới nhất đạt Accuracy 0,8672, Macro-F1 0,8666, coverage 0,8107 và selective risk 0,0601. Calibration làm log-loss validation giảm từ 0,8722 xuống 0,7125; ECE vẫn còn 0,2176 và được ghi nhận là limitation cần tiếp tục cải thiện.

Tập Test chính thức (**3,079 mẫu**) tuyệt đối không tham gia vào quá trình huấn luyện hay tinh chỉnh ngưỡng (tuning).

| Chỉ số Đánh giá (Metric) | Giá trị | Ý nghĩa Nghiệp vụ |
|---|---:|---|
| **Test Accuracy** | **87.20%** | Độ chính xác tổng thể trên 77 nhãn ý định |
| **Test Macro-F1** | **87.20%** | Độ cân bằng hiệu năng giữa các lớp |
| **Domain-level Accuracy** | **93.99%** | Độ chính xác phân loại vào 10 phòng ban lớn |
| **Calibrated Test ECE** | **0.2120** | Chỉ số đo lường độ tin cậy xác suất dự đoán |
| **Selective Coverage** | **78.17%** | Tỷ lệ ticket được mô hình tự động xử lý |
| **Selective Risk** | **4.32%** | Tỷ lệ lỗi trong số các ticket được tự động xử lý |

> 📌 **Nhận xét**: Với Selective Risk chỉ **4.32%**, hệ thống đảm bảo 95.68% các ticket được xử lý tự động là hoàn toàn chính xác, đáp ứng tiêu chuẩn khắt khe trong vận hành ngân hàng.

---

## 📁 Cấu Trúc Mã Nguồn (Project Structure)

```text
02_banking77_support_router/
├── configs/                # File cấu hình môi trường
├── data/
│   └── raw/                # Dữ liệu gốc train.csv, test.csv (BANKING77)
├── models/
│   ├── config.json         # Metadata mô hình, ngưỡng threshold, domain map
│   └── router.joblib       # Pipeline TF-IDF + Calibrated Classifier
├── reports/
│   ├── test_metrics.json   # Chỉ số đánh giá chính thức trên tập Test
│   └── validation_metrics.json # Data quality & validation metrics
├── scripts/
│   └── download_data.py    # Script tải tự động tập dữ liệu BANKING77
├── src/
│   ├── __init__.py
│   ├── api.py              # FastAPI Web Service (/health, /predict, /predict/batch)
│   ├── data.py             # Data loader, làm sạch, stratified split, domain mapping
│   ├── evaluate.py         # Đánh giá Accuracy, F1, ECE, Selective Risk trên Test
│   ├── policy.py           # Bộ quy tắc phân luồng độc lập (Routing Policy)
│   ├── train.py            # Huấn luyện TF-IDF + Logistic Regression, Platt Scaling, chọn threshold
│   └── utils.py            # Hàm bổ trợ: logging, set seed, calculate_ece, save_json
├── tests/
│   ├── test_data_contract.py # Kiểm thử hợp đồng dữ liệu & thuật toán chọn threshold
│   └── test_smoke.py       # Unit test kiểm tra Policy, ECE, API FastAPI
├── Dockerfile              # Docker containerization cho Production
├── Makefile                # Câu lệnh tắt quản lý dự án
├── pytest.ini              # Cấu hình kiểm thử Pytest
├── README.md               # Tài liệu hướng dẫn dự án chi tiết
└── requirements.txt        # Các thư viện phụ thuộc
```

---

## ⚙️ Hướng Dẫn Tái Lập Dự Án (Quickstart)

### 1. Cài đặt môi trường

```bash
# Khởi tạo môi trường ảo Python
python -m venv .venv

# Kích hoạt môi trường (Windows PowerShell)
.venv\Scripts\Activate.ps1
# Lựa chọn Linux/macOS: source .venv/bin/activate

# Cài đặt thư viện phụ thuộc
pip install -r requirements.txt
```

### 2. Tải dữ liệu & Huấn luyện mô hình

```bash
# Tải tập dữ liệu BANKING77
python scripts/download_data.py

# Huấn luyện mô hình, hiệu chỉnh xác suất ECE và tự động chọn ngưỡng trên Validation
python -m src.train

# Đánh giá hiệu năng chi tiết trên tập Test độc lập
python -m src.evaluate
```

### 3. Kiểm thử tự động (Unit Tests)

```bash
python -m pytest -v
```

### 4. Khởi chạy REST API Service

```bash
python -m uvicorn src.api:app --reload --port 8000
```

---

## 🔌 Hướng Dẫn Sử Dụng API

### Endpoint: `POST /predict`

**Request Body:**
```json
{
  "text": "Why has my cash withdrawal been declined?",
  "top_k": 3
}
```

**Response Body:**
```json
{
  "intent": null,
  "domain": null,
  "top_intent": "pending_cash_withdrawal",
  "confidence": 0.3798,
  "alternatives": [
    {
      "intent": "pending_cash_withdrawal",
      "domain": "atm_cash",
      "confidence": 0.3798
    },
    {
      "intent": "declined_cash_withdrawal",
      "domain": "atm_cash",
      "confidence": 0.2442
    }
  ],
  "is_unknown": true,
  "route": "human",
  "requires_human_review": true,
  "review_reason": "LOW_CONFIDENCE",
  "model_version": "banking77-tfidf-calibrated-lr-v1"
}
```

### Endpoint: `POST /predict/batch`

**Request Body:**
```json
{
  "queries": [
    {"text": "Where is my card?"},
    {"text": "I lost my phone and card, help!"}
  ]
}
```

---

## 📝 Mẫu Trình Bày Trong CV AI / ML Engineer

Dưới đây là các câu mẫu (bullet points) chuẩn hóa để bạn đưa trực tiếp vào CV:

### Tiếng Việt:
- **Xây dựng AI Customer Support Router cho ngành Ngân hàng (77 intents)** đạt **Selective Coverage 78.2%** với **Selective Risk 4.3%** trên tập Test độc lập (3,079 mẫu).
- **Tối ưu hóa Selective Classification**: Thiết kế thuật toán tự động chọn ngưỡng từ chối (Reject Threshold) trên tập Validation với ràng buộc độ phủ $\ge 80\%$, chuyển các ca rủi ro cao hoặc mập mờ cho nhân viên kiểm duyệt.
- **Tăng cường độ tin cậy mô hình**: Áp dụng **Platt Scaling Calibration** giúp giảm **Expected Calibration Error (ECE)** từ 0.2505 xuống 0.2171; thiết kế **Domain Grouping** gom nhóm ý định đạt **93.99% Domain Accuracy**.
- **Triển khai REST API Production-Ready**: Đóng gói dịch vụ với FastAPI hỗ trợ xử lý hàng loạt batch prediction, tích hợp unit tests bằng Pytest và Docker containerization.

### Tiếng Anh:
- **Engineered an AI Customer Support Router for Banking Intent Classification (77 classes)**, achieving **78.2% Selective Coverage** with a low **4.3% Selective Risk** on an independent test set.
- **Implemented Selective Prediction & Risk Management**: Developed an automated confidence thresholding algorithm constrained by a minimum 80% coverage on validation data, escalating low-confidence and high-risk queries to human agents.
- **Enhanced Model Reliability & Domain Hierarchies**: Applied **Platt Scaling Probability Calibration** (reducing ECE from 0.2505 to 0.2171) and designed a 10-domain hierarchical mapping achieving **93.99% domain-level accuracy**.
- **Deployed Production REST Services**: Built low-latency FastAPI endpoints supporting single & batch predictions, complete with unit test suites (Pytest) and Docker deployment readiness.
