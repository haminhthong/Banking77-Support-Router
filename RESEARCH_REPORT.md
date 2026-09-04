# Báo cáo cải tiến dựa trên nghiên cứu

## Nghiên cứu nền tảng

- Casanueva et al. (2020), *Efficient Intent Detection with Dual Sentence Encoders* (giới thiệu BANKING77): https://aclanthology.org/2020.nlp4convai-1.5/
- Geifman & El-Yaniv (2019), *SelectiveNet: A Deep Neural Network with an Integrated Reject Option*: https://proceedings.mlr.press/v97/geifman19a.html

## Pipeline trước và sau

Trước: làm sạch trùng → stratified split → TF-IDF + Logistic Regression → ngưỡng unknown hard-code 0.45.

Sau: chuẩn hóa chuỗi/rỗng trước khử trùng → giữ test chính thức độc lập → train/validation phân tầng → tự chọn ngưỡng reject trên validation theo risk–coverage, với coverage tối thiểu 80% → lưu threshold và metric selective để API dùng nhất quán.

Reject option phản ánh selective classification: hệ thống chuyển ca ít tin cậy cho người thay vì buộc dự đoán. Việc chọn ngưỡng hoàn toàn trên validation tránh tuning bằng test.

## Đánh giá

- Chất lượng phân lớp: accuracy, macro-F1 và per-class F1 trên test.
- Từ chối: coverage, selective risk, risk–coverage curve; báo cáo riêng accuracy của ca được chấp nhận.
- Vận hành: tỷ lệ chuyển người, p95 latency và phân phối confidence theo thời gian.

Bước tiếp theo: thêm tập out-of-scope thật, calibration (temperature/isotonic) trên validation, so sánh TF-IDF với sentence encoder; chỉ đổi model khi macro-F1 và risk–coverage cùng tốt hơn.

## Kết quả chạy thực tế

Trên test chính thức 3.079 mẫu: **accuracy = 0,8867**, **macro-F1 = 0,8871**. Validation chọn threshold 0,35 với coverage 0,8007 và selective risk 0,0525. Model, config và JSON metric đã được tạo trong `models/` và `reports/`.
