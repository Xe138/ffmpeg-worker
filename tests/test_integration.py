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
