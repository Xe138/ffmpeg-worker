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
