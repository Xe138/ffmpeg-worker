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
