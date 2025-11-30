# FFmpeg Worker Design

## Overview

A dockerized worker capable of running arbitrary FFmpeg commands, exposing API endpoints to create jobs and check job status. Files are provided and returned via a mounted volume using relative paths.

## Technology Stack

- **Language:** Python 3.14
- **Framework:** FastAPI
- **Task Queue:** In-memory (asyncio.Queue)
- **Container:** Docker with FFmpeg installed

## Architecture

### Components

- **FastAPI application** - Exposes REST endpoints for job management
- **Job Queue** - In-memory queue (asyncio.Queue) holding pending jobs
- **Worker Loop** - Background async task that processes jobs one at a time
- **Job Store** - In-memory dict mapping job IDs to job state

### File Handling

- A single volume mounted at `/data` (configurable via env var)
- API requests specify relative paths like `input/video.mp4`
- Worker resolves to absolute paths: `/data/input/video.mp4`
- Output files written to the same volume, returned as relative paths

### Project Structure

```
ffmpeg-worker/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI app, endpoints
│   ├── models.py        # Pydantic models
│   ├── queue.py         # Job queue and worker loop
│   └── ffmpeg.py        # FFmpeg execution and progress parsing
├── Dockerfile
├── requirements.txt
└── docker-compose.yml
```

## API Endpoints

### POST /jobs - Create a new job

```json
Request:
{
  "command": "-i input/source.mp4 -c:v libx264 -crf 23 output/result.mp4"
}

Response (201 Created):
{
  "id": "job_abc123",
  "status": "queued",
  "created_at": "2025-11-30T10:30:00Z"
}
```

### GET /jobs/{id} - Get job status

```json
Response:
{
  "id": "job_abc123",
  "status": "running",
  "command": "-i input/source.mp4 ...",
  "created_at": "2025-11-30T10:30:00Z",
  "started_at": "2025-11-30T10:30:05Z",
  "completed_at": null,
  "progress": {
    "frame": 1234,
    "fps": 30.2,
    "time": "00:01:23.45",
    "bitrate": "1250kbits/s",
    "percent": 45.2
  },
  "output_files": [],
  "error": null
}
```

### GET /jobs - List all jobs

Optional query parameter: `?status=running`

### GET /health - Health check

Returns 200 OK for container orchestration.

## Job States

```
queued → running → completed
                 ↘ failed
```

## FFmpeg Execution

### Command Execution

- Parse the raw command string to extract input/output file paths
- Resolve relative paths against the `/data` mount point
- Run FFmpeg as async subprocess via `asyncio.create_subprocess_exec`
- Add `-progress pipe:1` flag to get machine-readable progress output

### Progress Parsing

FFmpeg with `-progress pipe:1` outputs key=value pairs:

```
frame=1234
fps=30.24
total_size=5678900
out_time_ms=83450000
bitrate=1250.5kbits/s
progress=continue
```

For percent complete, probe the input file with `ffprobe -show_entries format=duration` before starting.

### Output File Detection

- Parse command for output path (last non-flag argument)
- Verify file exists after completion
- Return as relative path in response

### Error Handling

- Non-zero exit code → job marked `failed`, stderr captured in `error` field
- Missing input file → immediate failure with clear message
- FFmpeg timeout (configurable, default 1 hour) → kill process, mark failed

## Configuration

Environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `DATA_PATH` | Mount point for media files | `/data` |
| `FFMPEG_TIMEOUT` | Max job duration in seconds | `3600` |
| `HOST` | Server bind address | `0.0.0.0` |
| `PORT` | Server port | `8000` |

## Memory Management

- Jobs stay in memory indefinitely (simple approach)
- Optional `DELETE /jobs/{id}` endpoint for cleanup
- Future: auto-purge completed/failed jobs older than N hours
