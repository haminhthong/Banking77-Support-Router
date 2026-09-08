FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Cài dependencies
COPY requirements.txt .
RUN python -m pip install -r requirements.txt \
    && addgroup --system app \
    && adduser --system --ingroup app app

# Copy source và release artifact đã được kiểm tra vào image
COPY --chown=app:app . .

USER app

# Healthcheck xác nhận model, SHA-256 và taxonomy 77 lớp
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=3)"

CMD ["uvicorn", "src.banking_router.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
