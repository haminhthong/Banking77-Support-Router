FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Cài dependency runtime; image API không chứa công cụ train/test.
COPY requirements.txt .
RUN python -m pip install -r requirements.txt \
    && addgroup --system app \
    && adduser --system --ingroup app app

# Image chỉ chứa mã suy luận và artifact chuẩn duy nhất.
COPY --chown=app:app src ./src
COPY --chown=app:app artifacts ./artifacts

# SQLite review cần thư mục có quyền ghi khi chạy bằng user app.
RUN mkdir -p /app/reports \
    && chown -R app:app /app/reports

# Dừng build nếu artifact không load hoặc không đủ 77 intent.
RUN python -c "from src.banking_router.config import ARTIFACTS_DIR; from src.banking_router.modeling.artifact import load_artifacts; artifacts = load_artifacts(ARTIFACTS_DIR); assert len(artifacts.intent_model.classes_) == 77"

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=3)"

CMD ["uvicorn", "src.banking_router.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
