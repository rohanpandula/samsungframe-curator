# CURATOR CPU image — headless server via `curator --headless start`.
# Build:       docker build -f packaging/docker/Dockerfile -t curator:cpu .
# Run:         docker compose -f packaging/docker/docker-compose.yml up
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CURATOR_DATA_ROOT=/data

WORKDIR /app

RUN addgroup --system curator && adduser --system --ingroup curator curator

COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv \
    && uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev

RUN mkdir -p /data && chown -R curator:curator /app /data
USER curator

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=3)"

ENTRYPOINT ["curator", "--headless", "start"]
