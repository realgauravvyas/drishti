# Smriti -- single-stage image. The application is small; the weights are not,
# so they are fetched into a mounted volume on first run rather than baked in.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    SMRITI_DATA_DIR=/data \
    SMRITI_HOST=0.0.0.0 \
    SMRITI_PORT=8000

# opencv-python-headless still links against glib. libgl is deliberately absent:
# the headless wheel does not need it, and it is 100 MB of X11 we never use.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libglib2.0-0 curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY smriti/ ./smriti/
COPY web/ ./web/
COPY scripts/ ./scripts/
COPY cli.py run.py ./

# Run as a non-root user; /data is a volume so it must be writable by that user.
RUN useradd --create-home --uid 10001 smriti \
 && mkdir -p /data && chown -R smriti:smriti /data /app
USER smriti
VOLUME ["/data"]

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD curl -fsS http://localhost:8000/api/health || exit 1

CMD ["python", "run.py"]
