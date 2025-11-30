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
