# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FFmpeg Worker is a dockerized REST API service that executes arbitrary FFmpeg commands. Jobs are submitted via API, processed sequentially by a background worker, and clients poll for status/progress.

## Commands

```bash
# Run with Docker (recommended)
docker-compose up -d --build

# Install dependencies locally (requires uv: https://docs.astral.sh/uv/)
uv sync

# Run tests (excludes integration test)
uv run pytest tests/ -v --ignore=tests/test_integration.py

# Run single test
uv run pytest tests/test_api.py::test_create_job -v

# Run integration test (requires Docker running)
uv run pytest tests/test_integration.py -v -s

# Run locally (requires ffmpeg installed)
uv run uvicorn app.main:app --reload
```

## Architecture

**Request Flow:**
1. `POST /jobs` → creates Job, stores in JobStore, enqueues job_id in JobQueue
2. Background `worker_loop` dequeues job_id, runs FFmpeg, updates job status/progress
3. `GET /jobs/{id}` → polls job status and real-time progress

**Key Components:**
- `app/main.py` - FastAPI app with lifespan manager that starts/stops worker
- `app/queue.py` - JobQueue (asyncio.Queue wrapper), run_ffmpeg(), worker_loop()
- `app/ffmpeg.py` - Command parsing, path resolution, progress parsing, ffprobe duration
- `app/models.py` - Pydantic models (Job, JobStatus, Progress, CreateJobRequest, JobResponse)
- `app/store.py` - In-memory JobStore (dict-based)

**File Path Handling:**
- API accepts relative paths (e.g., `input/video.mp4`)
- `resolve_paths()` converts to absolute using DATA_PATH environment variable
- Output files returned as relative paths

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| DATA_PATH | /data | Mount point for media files |
| FFMPEG_TIMEOUT | 3600 | Max job duration in seconds |
| HOST | 0.0.0.0 | Server bind address |
| PORT | 8000 | Server port |
