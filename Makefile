.PHONY: help install install-dev test lint format clean build docs run-example run-api

help:  ## Show this help message
	@echo "AetherSpeak Protocol - Development Commands"
	@echo "=========================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package in development mode
	pip install -e .

install-dev:  ## Install development dependencies
	pip install -e ".[dev]"

test:  ## Run tests
	pytest tests/ -v

test-cov:  ## Run tests with coverage
	pytest tests/ --cov=aetherspeak --cov-report=html --cov-report=term

lint:  ## Run linting checks
	flake8 aetherspeak/ tests/
	mypy aetherspeak/

format:  ## Format code with black and isort
	black aetherspeak/ tests/
	isort aetherspeak/ tests/

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build:  ## Build the package
	python setup.py sdist bdist_wheel

docs:  ## Build documentation
	cd docs && make html

run-example:  ## Run the basic usage example
	python examples/basic_usage.py

run-api:  ## Run the API server example
	python examples/api_server.py --server-only

run-api-test:  ## Run the API server and test endpoints
	python examples/api_server.py

install-pre-commit:  ## Install pre-commit hooks
	pre-commit install

check-all: format lint test  ## Run all checks (format, lint, test)

dev-setup: install-dev install-pre-commit  ## Complete development setup
