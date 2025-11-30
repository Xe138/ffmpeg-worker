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
