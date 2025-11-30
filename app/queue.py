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
