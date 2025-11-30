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
