# FFmpeg Worker Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a dockerized FFmpeg worker with REST API for job submission and status polling.

**Architecture:** FastAPI app with in-memory job queue, single background worker processing jobs sequentially, real-time FFmpeg progress parsing via `-progress pipe:1`.

**Tech Stack:** Python 3.14, FastAPI, Pydantic, asyncio, ffmpeg/ffprobe CLI

---

## Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `app/__init__.py`
- Create: `app/main.py`

**Step 1: Create requirements.txt**

```txt
fastapi==0.115.0
uvicorn[standard]==0.32.0
pydantic==2.9.0
```

**Step 2: Create app package**

Create `app/__init__.py` (empty file).

**Step 3: Create minimal FastAPI app**

Create `app/main.py`:

```python
from fastapi import FastAPI

app = FastAPI(title="FFmpeg Worker", version="1.0.0")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
```

**Step 4: Test manually**

Run: `cd /home/bballou/ffmpeg-worker && uvicorn app.main:app --reload`

In another terminal: `curl http://localhost:8000/health`

Expected: `{"status":"ok"}`

Stop the server (Ctrl+C).

**Step 5: Commit**

```bash
git add requirements.txt app/
git commit -m "feat: initial FastAPI setup with health endpoint"
```

---

## Task 2: Job Models

**Files:**
- Create: `app/models.py`
- Create: `tests/__init__.py`
- Create: `tests/test_models.py`

**Step 1: Create tests package**

Create `tests/__init__.py` (empty file).

**Step 2: Write the failing test**

Create `tests/test_models.py`:

```python
from datetime import datetime

from app.models import Job, JobStatus, CreateJobRequest, JobResponse


def test_job_creation():
    job = Job(command="-i input.mp4 output.mp4")

    assert job.id.startswith("job_")
    assert len(job.id) == 20  # job_ + 16 hex chars
    assert job.status == JobStatus.QUEUED
    assert job.command == "-i input.mp4 output.mp4"
    assert isinstance(job.created_at, datetime)
    assert job.started_at is None
    assert job.completed_at is None
    assert job.progress is None
    assert job.output_files == []
    assert job.error is None


def test_create_job_request():
    request = CreateJobRequest(command="-i test.mp4 out.mp4")
    assert request.command == "-i test.mp4 out.mp4"


def test_job_response_from_job():
    job = Job(command="-i input.mp4 output.mp4")
    response = JobResponse.model_validate(job.model_dump())

    assert response.id == job.id
    assert response.status == JobStatus.QUEUED
```

**Step 3: Run test to verify it fails**

Run: `cd /home/bballou/ffmpeg-worker && python -m pytest tests/test_models.py -v`

Expected: FAIL with ModuleNotFoundError (app.models doesn't exist)

**Step 4: Write the implementation**

Create `app/models.py`:

```python
import secrets
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Progress(BaseModel):
    frame: int = 0
    fps: float = 0.0
    time: str = "00:00:00.00"
    bitrate: str = "0kbits/s"
    percent: float | None = None


class Job(BaseModel):
    id: str = Field(default_factory=lambda: f"job_{secrets.token_hex(8)}")
    status: JobStatus = JobStatus.QUEUED
    command: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    progress: Progress | None = None
    output_files: list[str] = Field(default_factory=list)
    error: str | None = None


class CreateJobRequest(BaseModel):
    command: str


class JobResponse(BaseModel):
    id: str
    status: JobStatus
    command: str
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    progress: Progress | None = None
    output_files: list[str] = Field(default_factory=list)
    error: str | None = None
```

**Step 5: Run test to verify it passes**

Run: `cd /home/bballou/ffmpeg-worker && python -m pytest tests/test_models.py -v`

Expected: PASS (3 tests)

**Step 6: Commit**

```bash
git add app/models.py tests/
git commit -m "feat: add job models with Pydantic"
```

---

## Task 3: Job Store

**Files:**
- Create: `app/store.py`
- Create: `tests/test_store.py`

**Step 1: Write the failing test**

Create `tests/test_store.py`:

```python
import pytest

from app.models import Job, JobStatus
from app.store import JobStore


def test_add_and_get_job():
    store = JobStore()
    job = Job(command="-i input.mp4 output.mp4")

    store.add(job)
    retrieved = store.get(job.id)

    assert retrieved is not None
    assert retrieved.id == job.id
    assert retrieved.command == job.command


def test_get_nonexistent_job():
    store = JobStore()

    result = store.get("nonexistent")

    assert result is None


def test_list_all_jobs():
    store = JobStore()
    job1 = Job(command="-i a.mp4 b.mp4")
    job2 = Job(command="-i c.mp4 d.mp4")

    store.add(job1)
    store.add(job2)
    jobs = store.list_all()

    assert len(jobs) == 2
    assert job1 in jobs
    assert job2 in jobs


def test_list_jobs_by_status():
    store = JobStore()
    job1 = Job(command="-i a.mp4 b.mp4")
    job2 = Job(command="-i c.mp4 d.mp4")
    job2.status = JobStatus.RUNNING

    store.add(job1)
    store.add(job2)

    queued = store.list_by_status(JobStatus.QUEUED)
    running = store.list_by_status(JobStatus.RUNNING)

    assert len(queued) == 1
    assert len(running) == 1
    assert queued[0].id == job1.id
    assert running[0].id == job2.id
```

**Step 2: Run test to verify it fails**

Run: `cd /home/bballou/ffmpeg-worker && python -m pytest tests/test_store.py -v`

Expected: FAIL with ModuleNotFoundError

**Step 3: Write the implementation**

Create `app/store.py`:

```python
from app.models import Job, JobStatus


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}

    def add(self, job: Job) -> None:
        self._jobs[job.id] = job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list_all(self) -> list[Job]:
        return list(self._jobs.values())

    def list_by_status(self, status: JobStatus) -> list[Job]:
        return [job for job in self._jobs.values() if job.status == status]
```

**Step 4: Run test to verify it passes**

Run: `cd /home/bballou/ffmpeg-worker && python -m pytest tests/test_store.py -v`

Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add app/store.py tests/test_store.py
git commit -m "feat: add in-memory job store"
```

---

## Task 4: API Endpoints

**Files:**
- Modify: `app/main.py`
- Create: `tests/test_api.py`

**Step 1: Write the failing tests**

Create `tests/test_api.py`:

```python
import pytest
from fastapi.testclient import TestClient

from app.main import app, job_store


@pytest.fixture(autouse=True)
def clear_store():
    job_store._jobs.clear()
    yield
    job_store._jobs.clear()


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_create_job():
    response = client.post("/jobs", json={"command": "-i input.mp4 output.mp4"})

    assert response.status_code == 201
    data = response.json()
    assert data["id"].startswith("job_")
    assert data["status"] == "queued"
    assert data["command"] == "-i input.mp4 output.mp4"


def test_get_job():
    # Create a job first
    create_response = client.post("/jobs", json={"command": "-i test.mp4 out.mp4"})
    job_id = create_response.json()["id"]

    # Get the job
    response = client.get(f"/jobs/{job_id}")

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == job_id
    assert data["status"] == "queued"


def test_get_nonexistent_job():
    response = client.get("/jobs/nonexistent")

    assert response.status_code == 404


def test_list_jobs():
    client.post("/jobs", json={"command": "-i a.mp4 b.mp4"})
    client.post("/jobs", json={"command": "-i c.mp4 d.mp4"})

    response = client.get("/jobs")

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2


def test_list_jobs_filter_by_status():
    client.post("/jobs", json={"command": "-i a.mp4 b.mp4"})

    response = client.get("/jobs?status=queued")

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["status"] == "queued"
```

**Step 2: Run test to verify it fails**

Run: `cd /home/bballou/ffmpeg-worker && python -m pytest tests/test_api.py -v`

Expected: FAIL (job_store not defined, endpoints missing)

**Step 3: Write the implementation**

Replace `app/main.py`:

```python
from fastapi import FastAPI, HTTPException, status

from app.models import CreateJobRequest, Job, JobResponse, JobStatus
from app.store import JobStore

app = FastAPI(title="FFmpeg Worker", version="1.0.0")

job_store = JobStore()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/jobs", status_code=status.HTTP_201_CREATED)
async def create_job(request: CreateJobRequest) -> JobResponse:
    job = Job(command=request.command)
    job_store.add(job)
    return JobResponse.model_validate(job.model_dump())


@app.get("/jobs/{job_id}")
async def get_job(job_id: str) -> JobResponse:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobResponse.model_validate(job.model_dump())


@app.get("/jobs")
async def list_jobs(status: JobStatus | None = None) -> list[JobResponse]:
    if status is not None:
        jobs = job_store.list_by_status(status)
    else:
        jobs = job_store.list_all()
    return [JobResponse.model_validate(job.model_dump()) for job in jobs]
```

**Step 4: Run test to verify it passes**

Run: `cd /home/bballou/ffmpeg-worker && python -m pytest tests/test_api.py -v`

Expected: PASS (6 tests)

**Step 5: Commit**

```bash
git add app/main.py tests/test_api.py
git commit -m "feat: add job CRUD API endpoints"
```

---

## Task 5: FFmpeg Command Parser

**Files:**
- Create: `app/ffmpeg.py`
- Create: `tests/test_ffmpeg.py`

**Step 1: Write the failing tests**

Create `tests/test_ffmpeg.py`:

```python
import pytest

from app.ffmpeg import parse_command, resolve_paths


def test_parse_simple_command():
    command = "-i input.mp4 output.mp4"

    args = parse_command(command)

    assert args == ["-i", "input.mp4", "output.mp4"]


def test_parse_command_with_options():
    command = "-i input.mp4 -c:v libx264 -crf 23 output.mp4"

    args = parse_command(command)

    assert args == ["-i", "input.mp4", "-c:v", "libx264", "-crf", "23", "output.mp4"]


def test_parse_command_with_quotes():
    command = '-i "input file.mp4" output.mp4'

    args = parse_command(command)

    assert args == ["-i", "input file.mp4", "output.mp4"]


def test_resolve_paths():
    args = ["-i", "input/video.mp4", "-c:v", "libx264", "output/result.mp4"]
    data_path = "/data"

    resolved = resolve_paths(args, data_path)

    assert resolved == [
        "-i", "/data/input/video.mp4",
        "-c:v", "libx264",
        "/data/output/result.mp4"
    ]


def test_resolve_paths_preserves_absolute():
    args = ["-i", "/already/absolute.mp4", "output.mp4"]
    data_path = "/data"

    resolved = resolve_paths(args, data_path)

    assert resolved == ["-i", "/already/absolute.mp4", "/data/output.mp4"]


def test_resolve_paths_skips_options():
    args = ["-c:v", "libx264", "-preset", "fast"]
    data_path = "/data"

    resolved = resolve_paths(args, data_path)

    # Options and their values should not be resolved as paths
    assert resolved == ["-c:v", "libx264", "-preset", "fast"]
```

**Step 2: Run test to verify it fails**

Run: `cd /home/bballou/ffmpeg-worker && python -m pytest tests/test_ffmpeg.py -v`

Expected: FAIL with ModuleNotFoundError

**Step 3: Write the implementation**

Create `app/ffmpeg.py`:

```python
import shlex
from pathlib import Path

# FFmpeg options that take a value (not exhaustive, covers common ones)
OPTIONS_WITH_VALUES = {
    "-c", "-c:v", "-c:a", "-b", "-b:v", "-b:a", "-r", "-s", "-ar", "-ac",
    "-f", "-t", "-ss", "-to", "-vf", "-af", "-filter:v", "-filter:a",
    "-preset", "-crf", "-qp", "-profile", "-level", "-pix_fmt", "-map",
    "-metadata", "-disposition", "-threads", "-filter_complex",
}


def parse_command(command: str) -> list[str]:
    """Parse FFmpeg command string into argument list."""
    return shlex.split(command)


def resolve_paths(args: list[str], data_path: str) -> list[str]:
    """Resolve relative paths against the data directory."""
    resolved = []
    skip_next = False

    for i, arg in enumerate(args):
        if skip_next:
            resolved.append(arg)
            skip_next = False
            continue

        # Check if this is an option that takes a value
        if arg in OPTIONS_WITH_VALUES or arg.startswith("-"):
            resolved.append(arg)
            if arg in OPTIONS_WITH_VALUES:
                skip_next = True
            continue

        # This looks like a file path - resolve if relative
        path = Path(arg)
        if not path.is_absolute():
            resolved.append(str(Path(data_path) / arg))
        else:
            resolved.append(arg)

    return resolved
```

**Step 4: Run test to verify it passes**

Run: `cd /home/bballou/ffmpeg-worker && python -m pytest tests/test_ffmpeg.py -v`

Expected: PASS (6 tests)

**Step 5: Commit**

```bash
git add app/ffmpeg.py tests/test_ffmpeg.py
git commit -m "feat: add FFmpeg command parser with path resolution"
```

---

## Task 6: FFmpeg Progress Parser

**Files:**
- Modify: `app/ffmpeg.py`
- Modify: `tests/test_ffmpeg.py`

**Step 1: Write the failing tests**

Add to `tests/test_ffmpeg.py`:

```python
from app.ffmpeg import parse_progress, extract_output_path


def test_parse_progress():
    output = """frame=1234
fps=30.24
total_size=5678900
out_time_ms=83450000
bitrate=1250.5kbits/s
progress=continue
"""
    progress = parse_progress(output, duration_seconds=120.0)

    assert progress.frame == 1234
    assert progress.fps == 30.24
    assert progress.time == "00:01:23.45"
    assert progress.bitrate == "1250.5kbits/s"
    assert progress.percent == pytest.approx(69.54, rel=0.01)


def test_parse_progress_no_duration():
    output = "frame=100\nfps=25.0\nout_time_ms=4000000\nbitrate=500kbits/s\n"

    progress = parse_progress(output, duration_seconds=None)

    assert progress.frame == 100
    assert progress.percent is None


def test_extract_output_path():
    args = ["-i", "input.mp4", "-c:v", "libx264", "output.mp4"]

    output_path = extract_output_path(args)

    assert output_path == "output.mp4"


def test_extract_output_path_complex():
    args = ["-i", "a.mp4", "-i", "b.mp4", "-filter_complex", "[0:v][1:v]concat", "out.mp4"]

    output_path = extract_output_path(args)

    assert output_path == "out.mp4"
```

**Step 2: Run test to verify it fails**

Run: `cd /home/bballou/ffmpeg-worker && python -m pytest tests/test_ffmpeg.py::test_parse_progress -v`

Expected: FAIL (parse_progress not defined)

**Step 3: Write the implementation**

Add to `app/ffmpeg.py`:

```python
from app.models import Progress


def parse_progress(output: str, duration_seconds: float | None) -> Progress:
    """Parse FFmpeg progress output into Progress model."""
    data: dict[str, str] = {}
    for line in output.strip().split("\n"):
        if "=" in line:
            key, value = line.split("=", 1)
            data[key.strip()] = value.strip()

    frame = int(data.get("frame", 0))
    fps = float(data.get("fps", 0.0))
    out_time_ms = int(data.get("out_time_ms", 0))
    bitrate = data.get("bitrate", "0kbits/s")

    # Convert out_time_ms to HH:MM:SS.mm format
    total_seconds = out_time_ms / 1_000_000
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    time_str = f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"

    # Calculate percent if duration is known
    percent = None
    if duration_seconds and duration_seconds > 0:
        percent = (total_seconds / duration_seconds) * 100
        percent = min(percent, 100.0)  # Cap at 100%

    return Progress(
        frame=frame,
        fps=fps,
        time=time_str,
        bitrate=bitrate,
        percent=percent,
    )


def extract_output_path(args: list[str]) -> str | None:
    """Extract output file path from FFmpeg arguments (last non-option argument)."""
    # Work backwards to find the last argument that isn't an option or option value
    i = len(args) - 1
    while i >= 0:
        arg = args[i]
        # Skip if it's an option
        if arg.startswith("-"):
            i -= 1
            continue
        # Check if previous arg is an option that takes a value
        if i > 0 and args[i - 1] in OPTIONS_WITH_VALUES:
            i -= 1
            continue
        # Check if it looks like a file path (has extension or contains /)
        if "." in arg or "/" in arg:
            return arg
        i -= 1
    return None
```

Also update the imports at the top of `app/ffmpeg.py`:

```python
import shlex
from pathlib import Path

from app.models import Progress
```

**Step 4: Run test to verify it passes**

Run: `cd /home/bballou/ffmpeg-worker && python -m pytest tests/test_ffmpeg.py -v`

Expected: PASS (10 tests)

**Step 5: Commit**

```bash
git add app/ffmpeg.py tests/test_ffmpeg.py
git commit -m "feat: add FFmpeg progress parser and output path extraction"
```

---

## Task 7: Job Queue

**Files:**
- Create: `app/queue.py`
- Create: `tests/test_queue.py`

**Step 1: Write the failing tests**

Create `tests/test_queue.py`:

```python
import pytest
import asyncio

from app.queue import JobQueue


@pytest.mark.asyncio
async def test_enqueue_and_dequeue():
    queue = JobQueue()

    await queue.enqueue("job_123")
    job_id = await queue.dequeue()

    assert job_id == "job_123"


@pytest.mark.asyncio
async def test_queue_ordering():
    queue = JobQueue()

    await queue.enqueue("job_1")
    await queue.enqueue("job_2")
    await queue.enqueue("job_3")

    assert await queue.dequeue() == "job_1"
    assert await queue.dequeue() == "job_2"
    assert await queue.dequeue() == "job_3"


@pytest.mark.asyncio
async def test_queue_size():
    queue = JobQueue()

    assert queue.size() == 0
    await queue.enqueue("job_1")
    assert queue.size() == 1
    await queue.enqueue("job_2")
    assert queue.size() == 2
    await queue.dequeue()
    assert queue.size() == 1
```

**Step 2: Add pytest-asyncio to requirements**

Add to `requirements.txt`:

```txt
fastapi==0.115.0
uvicorn[standard]==0.32.0
pydantic==2.9.0
pytest==8.3.0
pytest-asyncio==0.24.0
```

**Step 3: Run test to verify it fails**

Run: `cd /home/bballou/ffmpeg-worker && pip install pytest-asyncio && python -m pytest tests/test_queue.py -v`

Expected: FAIL with ModuleNotFoundError

**Step 4: Write the implementation**

Create `app/queue.py`:

```python
import asyncio


class JobQueue:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[str] = asyncio.Queue()

    async def enqueue(self, job_id: str) -> None:
        await self._queue.put(job_id)

    async def dequeue(self) -> str:
        return await self._queue.get()

    def size(self) -> int:
        return self._queue.qsize()
```

**Step 5: Run test to verify it passes**

Run: `cd /home/bballou/ffmpeg-worker && python -m pytest tests/test_queue.py -v`

Expected: PASS (3 tests)

**Step 6: Commit**

```bash
git add app/queue.py tests/test_queue.py requirements.txt
git commit -m "feat: add async job queue"
```

---

## Task 8: FFmpeg Runner

**Files:**
- Modify: `app/ffmpeg.py`
- Modify: `tests/test_ffmpeg.py`

**Step 1: Write the failing test**

Add to `tests/test_ffmpeg.py`:

```python
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from app.models import Job


@pytest.mark.asyncio
async def test_get_duration():
    from app.ffmpeg import get_duration

    # Mock ffprobe output
    mock_process = MagicMock()
    mock_process.communicate = AsyncMock(return_value=(b"120.5\n", b""))
    mock_process.returncode = 0

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        duration = await get_duration("/data/input.mp4")

    assert duration == 120.5


@pytest.mark.asyncio
async def test_get_duration_failure():
    from app.ffmpeg import get_duration

    mock_process = MagicMock()
    mock_process.communicate = AsyncMock(return_value=(b"", b"error"))
    mock_process.returncode = 1

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        duration = await get_duration("/data/input.mp4")

    assert duration is None
```

**Step 2: Run test to verify it fails**

Run: `cd /home/bballou/ffmpeg-worker && python -m pytest tests/test_ffmpeg.py::test_get_duration -v`

Expected: FAIL (get_duration not defined)

**Step 3: Write the implementation**

Add to `app/ffmpeg.py`:

```python
import asyncio


async def get_duration(input_path: str) -> float | None:
    """Get duration of media file using ffprobe."""
    try:
        process = await asyncio.create_subprocess_exec(
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            input_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await process.communicate()

        if process.returncode == 0:
            return float(stdout.decode().strip())
    except (ValueError, OSError):
        pass
    return None
```

**Step 4: Run test to verify it passes**

Run: `cd /home/bballou/ffmpeg-worker && python -m pytest tests/test_ffmpeg.py::test_get_duration tests/test_ffmpeg.py::test_get_duration_failure -v`

Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add app/ffmpeg.py tests/test_ffmpeg.py
git commit -m "feat: add ffprobe duration detection"
```

---

## Task 9: Worker Loop

**Files:**
- Modify: `app/queue.py`
- Modify: `app/main.py`

**Step 1: Add worker loop to queue.py**

Add to `app/queue.py`:

```python
import asyncio
import os
from datetime import datetime, timezone

from app.ffmpeg import (
    parse_command,
    resolve_paths,
    get_duration,
    parse_progress,
    extract_output_path,
)
from app.models import Job, JobStatus, Progress
from app.store import JobStore


class JobQueue:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[str] = asyncio.Queue()

    async def enqueue(self, job_id: str) -> None:
        await self._queue.put(job_id)

    async def dequeue(self) -> str:
        return await self._queue.get()

    def size(self) -> int:
        return self._queue.qsize()


async def run_ffmpeg(job: Job, data_path: str, timeout: int) -> None:
    """Execute FFmpeg command and update job with progress."""
    args = parse_command(job.command)
    resolved_args = resolve_paths(args, data_path)

    # Find input file for duration
    input_path = None
    for i, arg in enumerate(resolved_args):
        if arg == "-i" and i + 1 < len(resolved_args):
            input_path = resolved_args[i + 1]
            break

    duration = None
    if input_path:
        duration = await get_duration(input_path)

    # Build FFmpeg command with progress output
    ffmpeg_args = ["ffmpeg", "-y", "-progress", "pipe:1", "-nostats"] + resolved_args

    process = await asyncio.create_subprocess_exec(
        *ffmpeg_args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    # Read progress output
    progress_buffer = ""
    try:
        async with asyncio.timeout(timeout):
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                progress_buffer += line.decode()

                # Parse progress when we get a complete block
                if "progress=" in progress_buffer:
                    job.progress = parse_progress(progress_buffer, duration)
                    progress_buffer = ""

            _, stderr = await process.communicate()
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        raise TimeoutError(f"FFmpeg timed out after {timeout} seconds")

    if process.returncode != 0:
        raise RuntimeError(stderr.decode())

    # Extract output path
    output_path = extract_output_path(args)
    if output_path:
        job.output_files = [output_path]


async def worker_loop(job_queue: "JobQueue", job_store: JobStore) -> None:
    """Main worker loop that processes jobs from the queue."""
    data_path = os.environ.get("DATA_PATH", "/data")
    timeout = int(os.environ.get("FFMPEG_TIMEOUT", "3600"))

    while True:
        job_id = await job_queue.dequeue()
        job = job_store.get(job_id)

        if job is None:
            continue

        job.status = JobStatus.RUNNING
        job.started_at = datetime.now(timezone.utc)
        job.progress = Progress()

        try:
            await run_ffmpeg(job, data_path, timeout)
            job.status = JobStatus.COMPLETED
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
        finally:
            job.completed_at = datetime.now(timezone.utc)
```

**Step 2: Wire up worker in main.py**

Replace `app/main.py`:

```python
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status

from app.models import CreateJobRequest, Job, JobResponse, JobStatus
from app.queue import JobQueue, worker_loop
from app.store import JobStore

job_store = JobStore()
job_queue = JobQueue()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start worker on startup
    worker_task = asyncio.create_task(worker_loop(job_queue, job_store))
    yield
    # Cancel worker on shutdown
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="FFmpeg Worker", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/jobs", status_code=status.HTTP_201_CREATED)
async def create_job(request: CreateJobRequest) -> JobResponse:
    job = Job(command=request.command)
    job_store.add(job)
    await job_queue.enqueue(job.id)
    return JobResponse.model_validate(job.model_dump())


@app.get("/jobs/{job_id}")
async def get_job(job_id: str) -> JobResponse:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobResponse.model_validate(job.model_dump())


@app.get("/jobs")
async def list_jobs(status: JobStatus | None = None) -> list[JobResponse]:
    if status is not None:
        jobs = job_store.list_by_status(status)
    else:
        jobs = job_store.list_all()
    return [JobResponse.model_validate(job.model_dump()) for job in jobs]
```

**Step 3: Run all tests**

Run: `cd /home/bballou/ffmpeg-worker && python -m pytest tests/ -v`

Expected: All tests PASS

**Step 4: Commit**

```bash
git add app/queue.py app/main.py
git commit -m "feat: add worker loop with FFmpeg execution"
```

---

## Task 10: Dockerfile

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yml`
- Create: `.dockerignore`

**Step 1: Create Dockerfile**

Create `Dockerfile`:

```dockerfile
FROM python:3.14-slim

# Install FFmpeg
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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
```

**Step 2: Create docker-compose.yml**

Create `docker-compose.yml`:

```yaml
services:
  ffmpeg-worker:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/data
    environment:
      - DATA_PATH=/data
      - FFMPEG_TIMEOUT=3600
```

**Step 3: Create .dockerignore**

Create `.dockerignore`:

```
__pycache__/
*.pyc
.git/
.gitignore
tests/
docs/
data/
*.md
.env
.venv/
```

**Step 4: Create data directory**

```bash
mkdir -p /home/bballou/ffmpeg-worker/data
echo "data/" >> /home/bballou/ffmpeg-worker/.gitignore
```

**Step 5: Commit**

```bash
git add Dockerfile docker-compose.yml .dockerignore .gitignore
git commit -m "feat: add Docker configuration"
```

---

## Task 11: Integration Test

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write integration test**

Create `tests/test_integration.py`:

```python
"""
Integration test - requires Docker and FFmpeg.
Run with: pytest tests/test_integration.py -v -s
"""
import pytest
import subprocess
import time
import requests
import os

WORKER_URL = "http://localhost:8000"


@pytest.fixture(scope="module")
def docker_compose():
    """Start docker-compose for integration tests."""
    # Skip if not running integration tests
    if os.environ.get("SKIP_INTEGRATION"):
        pytest.skip("Skipping integration tests")

    # Create a test video file
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    test_input = os.path.join(data_dir, "test_input.mp4")

    # Generate test video with FFmpeg (2 seconds of color bars)
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=30",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", test_input
    ], check=True, capture_output=True)

    # Start docker-compose
    subprocess.run(["docker-compose", "up", "-d", "--build"], check=True)

    # Wait for service to be ready
    for _ in range(30):
        try:
            response = requests.get(f"{WORKER_URL}/health")
            if response.status_code == 200:
                break
        except requests.ConnectionError:
            pass
        time.sleep(1)
    else:
        pytest.fail("Service did not start in time")

    yield

    # Cleanup
    subprocess.run(["docker-compose", "down"], check=True)


def test_full_workflow(docker_compose):
    """Test complete job submission and processing workflow."""
    # Submit job
    response = requests.post(
        f"{WORKER_URL}/jobs",
        json={"command": "-i test_input.mp4 -c:v libx264 -crf 28 test_output.mp4"}
    )
    assert response.status_code == 201
    job = response.json()
    job_id = job["id"]
    assert job["status"] == "queued"

    # Poll for completion
    for _ in range(60):
        response = requests.get(f"{WORKER_URL}/jobs/{job_id}")
        assert response.status_code == 200
        job = response.json()

        if job["status"] in ("completed", "failed"):
            break
        time.sleep(1)

    assert job["status"] == "completed", f"Job failed: {job.get('error')}"
    assert "test_output.mp4" in job["output_files"]
```

**Step 2: Commit**

```bash
git add tests/test_integration.py
git commit -m "feat: add integration test"
```

---

## Summary

Tasks 1-11 build the complete FFmpeg worker:

1. Project setup with FastAPI
2. Job models (Pydantic)
3. In-memory job store
4. REST API endpoints
5. FFmpeg command parser
6. Progress parser
7. Async job queue
8. FFmpeg runner with ffprobe
9. Worker loop integration
10. Docker configuration
11. Integration test

All tasks follow TDD with specific file paths, complete code, and exact commands.
