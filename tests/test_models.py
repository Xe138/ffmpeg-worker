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
