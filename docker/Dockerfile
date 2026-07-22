FROM python:3.12-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.12-slim

RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

COPY --from=builder /install /usr/local
COPY app ./app
COPY alembic ./alembic
COPY alembic.ini .
COPY scripts ./scripts

USER appuser
EXPOSE 8000

# Uses Python's own stdlib rather than installing curl into the slim image
# just for this one check.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/healthz')"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
