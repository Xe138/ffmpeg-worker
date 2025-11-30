.PHONY: test test-unit test-integration install dev build up down clean

# Install dependencies
install:
	uv sync

# Run unit tests (excludes integration tests)
test-unit:
	uv run pytest tests/ -v --ignore=tests/test_integration.py

# Run integration tests (requires Docker)
test-integration:
	uv run pytest tests/test_integration.py -v -s

# Run all tests
test: test-unit

# Run dev server locally
dev:
	uv run uvicorn app.main:app --reload

# Docker commands
build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down

# Clean up
clean:
	rm -rf .venv venv __pycache__ .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
