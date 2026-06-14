.PHONY: clean install test build help

VENV := .venv
.DEFAULT_GOAL := help

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  install       Create .venv, install in editable mode (light mode, no JAX)"
	@echo "  install-jax   Create .venv, install in editable mode (full mode, with JAX)"
	@echo "  test          Run all tests (light mode)"
	@echo "  test-jax      Run all tests (full mode, with JAX)"
	@echo "  build         Build distribution wheel"
	@echo "  lint          Run ruff check + black"
	@echo "  clean         Remove build artifacts and cache files"
	@echo "  help          Show this help message"

install:
	@if [ ! -d "$(VENV)" ]; then \
		echo "Creating virtual environment with uv..."; \
		uv venv; \
	fi
	@echo "Installing package in editable mode with dev dependencies (light mode)..."
	uv pip install -e ".[dev]"

install-jax:
	@if [ ! -d "$(VENV)" ]; then \
		echo "Creating virtual environment with uv..."; \
		uv venv; \
	fi
	@echo "Installing package in editable mode with dev + jax dependencies (full mode)..."
	uv pip install -e ".[dev,jax]"

test:
	pytest tests

test-jax:
	pytest tests

lint:
	ruff check .
	black --check .

build:
	uv build --wheel -o dist ./

clean:
	@echo "Cleaning build artifacts and cache files..."
	find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .coverage dist build 2>/dev/null || true
	@echo "Clean complete!"
