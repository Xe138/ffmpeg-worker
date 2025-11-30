FROM python:3.14-slim

# Install FFmpeg and curl (for uv installation)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install Python dependencies
COPY pyproject.toml .
RUN uv pip install --system --no-cache .

# Copy application
COPY app/ ./app/

# Create data directory
RUN mkdir -p /data

ENV DATA_PATH=/data
ENV FFMPEG_TIMEOUT=3600
ENV HOST=0.0.0.0
ENV PORT=8000

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host $HOST --port $PORT"]
