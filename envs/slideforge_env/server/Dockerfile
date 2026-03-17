FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY envs/slideforge_env/server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    playwright install --with-deps chromium && \
    rm /tmp/requirements.txt

COPY envs/ /app/envs/
COPY rewards/ /app/rewards/

ENV SLIDEFORGE_MAX_STEPS=50
ENV SLIDEFORGE_RENDER_TIMEOUT=30

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "envs.slideforge_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
