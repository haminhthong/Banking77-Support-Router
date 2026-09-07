FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN python -m pip install -r requirements.txt \
    && addgroup --system app \
    && adduser --system --ingroup app app

# Copy application source code and models
# NOTE: Artifact Delivery Strategy:
# Ensure models/router.joblib is generated beforehand via `python -m src.train`
# or downloaded from a release artifact registry prior to building the production container.
COPY --chown=app:app . .

USER app

# Readiness healthcheck validating model loading, manifest SHA-256, and 77-class taxonomy
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=3)"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
