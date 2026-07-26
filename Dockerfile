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

# Default for local/docker-compose use. Render (and Railway) inject their
# own $PORT automatically at container start - both the healthcheck and the
# CMD below read $PORT at runtime rather than baking a port in at build time.
ENV PORT=8000
EXPOSE $PORT

# Uses Python's own stdlib rather than installing curl into the slim image
# just for this one check.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\", \"8000\")}/api/v1/healthz')"]

# Apply any pending schema migrations before serving, so a deploy that ships
# a new migration (e.g. new tables) never leaves the API querying a table that
# doesn't exist yet. Safe for the single-instance Render service; if this ever
# scales to multiple replicas, move migrations to a dedicated release step.
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
