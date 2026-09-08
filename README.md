# Banking77 Support Router

Hệ thống phân loại ticket Banking77 thành intent và queue vận hành, có ba lớp bảo vệ: scope/input quality, security risk và policy abstain. Model ML chỉ tạo xác suất; quyết định cuối cùng do policy engine đưa ra.

## Luồng runtime

```text
request
  -> chuẩn hóa PII ngữ nghĩa
  -> intent model (word + char TF-IDF, Logistic Regression, calibration)
  -> cộng xác suất theo queue
  -> scope/input guard
  -> risk assessor quét toàn bộ 77 lớp
  -> RoutingPolicy: priority review / human review / auto route
  -> telemetry đã redact PII + ticket/review SQLite
```

Thứ tự policy là cố định:

1. Security risk đạt ngưỡng: chuyển `priority_human_review` vào queue escalation.
2. Ngoài scope hoặc input lỗi: chuyển `human_review`.
3. Queue confidence, queue margin hoặc entropy không đạt: chuyển `human_review`.
4. Chỉ ticket còn lại mới được `auto_route`.

`priority: critical` là priority vận hành, không đồng nghĩa với fraud/security. Security escalation chỉ đến từ `risk.priority_escalation: true` trong `configs/taxonomy.yaml`.

## Release đang chạy

`models/production.json` đang trỏ tới `releases/banking-router-v5`. Đây là release đã được promote qua `release_gate.json`.

`releases/banking-router-v4` chỉ được giữ làm rollback artifact; không được dùng khi chạy production pointer.

Đánh giá được sinh từ official test 3.080 mẫu và locked scope split 10 mẫu:

| Chỉ số | Giá trị |
|---|---:|
| Test accuracy | 0.9016 |
| Test macro-F1 | 0.9019 |
| Auto-route coverage | 0.6981 |
| Auto-route wrong-queue rate | 0.0051 |
| Security escalation recall | 0.9821 |
| Locked OOS auto-route rate | 0.0000 |
| Locked OOS containment | 1.0000 |

Nguồn: [reports/v5-evaluation/test_metrics.json](reports/v5-evaluation/test_metrics.json), [reports/v5-evaluation/ood_metrics.json](reports/v5-evaluation/ood_metrics.json), [models/releases/banking-router-v5/release_gate.json](models/releases/banking-router-v5/release_gate.json).

## Cấu trúc chính

```text
configs/
  model.yaml                 # cấu hình model và calibration
  taxonomy.yaml              # 77 intent, domain, queue, risk metadata
  routing_policy.yaml        # threshold, scope và release SLO
data/
  raw/                       # train.csv, test.csv
  evaluation/ood.jsonl       # mẫu ngoài phạm vi; tách train/dev/locked deterministic
models/
  production.json            # pointer release duy nhất
  releases/<release>/        # model + manifest + policy + scope model + gate
reports/<release>-evaluation/
src/banking_router/
  data/                      # loader, normalization, split và scope split
  modeling/                  # pipeline, calibration, artifact, scope, release
  routing/                   # taxonomy, queue projector, risk, scope, policy, service
  evaluation/                # classification, calibration, queue safety, OOS
  api/                       # FastAPI và schema tương thích
  storage/                   # SQLite ticket/review lifecycle
scripts/promote_release.py   # gate rồi mới cập nhật production pointer
```

## Chạy local

```powershell
python -m pip install -r requirements.txt
python -m src.train
python -m src.evaluate
uvicorn src.banking_router.api.app:app --reload
```

Nếu cần train candidate riêng, không ghi đè production pointer:

```powershell
python -c "from src.banking_router.modeling.training import train_and_optimize; train_and_optimize(models_dir='models/releases/banking-router-v6', reports_dir='reports/v6-training')"
```

Sau khi chạy evaluation cho candidate, chỉ promote qua gate:

```powershell
python scripts/promote_release.py --release banking-router-v6 --reports reports/v6-evaluation --models models
```

Lệnh sẽ từ chối và giữ nguyên pointer nếu một trong các SLO sau không đạt: wrong queue rate tối đa 5%, security recall tối thiểu 95%, OOS auto-route tối đa 5%, coverage tối thiểu 65%.

## API tối thiểu

```powershell
curl -X POST http://127.0.0.1:8000/v1/route `
  -H "Content-Type: application/json" `
  -d '{"text":"My card was stolen", "top_k":3}'
```

Response canonical gồm `prediction`, `queue_prediction`, `scope`, `risk`, `decision` và `versions`. Các field phẳng cũ vẫn được giữ để client legacy không bị gãy. Endpoint review là `POST /v1/tickets/{request_id}/review`.

## Kiểm tra

```powershell
python -m pytest -q
```

Official test là benchmark bất biến, không dùng để tune threshold. Scope model được train trên scope-train; scope-dev dùng cho threshold; locked scope chỉ dùng để báo cáo/gate.
