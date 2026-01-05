.PHONY: help install download train test lint format api docker-build docker-run clean

help:
	@echo "Available commands:"
	@echo "  make install      - Install dependencies"
	@echo "  make download     - Download dataset"
	@echo "  make train        - Train models"
	@echo "  make test         - Run tests"
	@echo "  make lint         - Run linters"
	@echo "  make format       - Format code with black"
	@echo "  make api          - Run API server"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run   - Run Docker container"
	@echo "  make clean        - Clean generated files"

install:
	pip install -r requirements.txt

download:
	python src/data/download_data.py

train:
	python src/models/train.py

test:
	pytest tests/ -v --cov=src --cov-report=html

lint:
	flake8 src/ tests/
	black --check src/ tests/

format:
	black src/ tests/

api:
	uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t heart-disease-api:latest .

docker-run:
	docker run -d -p 8000:8000 --name heart-disease-api heart-disease-api:latest

clean:
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage

