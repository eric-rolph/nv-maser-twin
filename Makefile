# NV Maser Digital Twin — developer task runner
# Usage: make <target>
# Requires: make (from Git for Windows or WSL), Python venv activated

.PHONY: help install install-all test lint format benchmark export serve clean

help:   ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install:  ## Install package with dev+api extras
	pip install -e ".[dev,api]"

install-all:  ## Install package with all optional extras
	pip install -e ".[dev,api,onnx]"

test:  ## Run full test suite
	python -m pytest -q

test-cov:  ## Run tests with HTML coverage report
	python -m pytest --cov=src/nv_maser --cov-report=html --cov-report=term-missing -q

lint:  ## Run ruff linter
	ruff check src/ tests/ scripts/

format:  ## Auto-fix ruff lint issues
	ruff check --fix src/ tests/ scripts/

benchmark:  ## Run inference benchmark
	python benchmarks/benchmark_inference.py

export:  ## Export model to ONNX (requires trained checkpoint)
	python scripts/export_onnx.py

sweep:  ## Run hyperparameter sweep (5 epochs, fast)
	python scripts/run_sweep.py --epochs 5

serve:  ## Start the FastAPI server on localhost:8000
	python -m nv_maser serve

dataset:  ## Build and cache the training dataset
	python -m nv_maser dataset

train:  ## Train the shimming controller
	python -m nv_maser train

clean:  ## Remove generated artifacts
	rm -rf dataset_cache/ training_curves.png .pytest_cache/ htmlcov/ .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
