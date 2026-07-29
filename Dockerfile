# Dockerfile — one image, used for both the gateway and the bootstrap job.
#
# Same image on purpose: bootstrap.py imports adplatform.db.SCHEMA and
# adplatform.settings, so it needs the application code and the same asyncpg /
# clickhouse-connect / aiokafka versions the gateway runs. A separate slimmer
# bootstrap image would be two dependency sets to keep in sync.

FROM python:3.12-slim AS base

# libgomp1 is xgboost's OpenMP runtime. Without it `import xgboost` fails at
# runtime with a linker error, which is a confusing way to discover a missing
# 200kB library.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Requirements first so a code edit does not invalidate the dependency layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY adplatform/ ./adplatform/
COPY scripts/ ./scripts/
COPY sql/ ./sql/

# models/ is a bind mount in compose; this just guarantees the path exists so
# ctr_model.load() finds an absent directory rather than a missing parent.
RUN mkdir -p /app/models/current

# Non-root. The gateway needs no write access to anything except /tmp.
RUN useradd --create-home --uid 10001 app && chown -R app:app /app
USER app

EXPOSE 8000

CMD ["uvicorn", "adplatform.gateway:app", "--host=0.0.0.0", "--port=8000"]
