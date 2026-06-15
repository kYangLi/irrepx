.PHONY: clean install test tests build help lint

VENV := .venv
.DEFAULT_GOAL := help

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  install       Create .venv, install in editable mode (no JAX)"
	@echo "  install-jax   Create .venv, install in editable mode (with JAX)"
	@echo "  install-test  Create .venv, install in editable mode with all test deps (jax + e3nn-jax + e3nn)"
	@echo "  test          Run tests"
	@echo "  build         Build distribution wheel"
	@echo "  lint          Run ruff check + black"
	@echo "  clean         Remove build artifacts and cache files"
	@echo "  help          Show this help message"

install:
	@if [ ! -d "$(VENV)" ]; then \
		echo "Creating virtual environment with uv..."; \
		uv venv; \
	fi
	@echo "Installing package in editable mode with dev dependencies (no JAX)..."
	uv pip install -e ".[dev]"

install-jax:
	@if [ ! -d "$(VENV)" ]; then \
		echo "Creating virtual environment with uv..."; \
		uv venv; \
	fi
	@echo "Installing package in editable mode with dev + jax dependencies..."
	uv pip install -e ".[dev,jax]"

install-test:
	@if [ ! -d "$(VENV)" ]; then \
		echo "Creating virtual environment with uv..."; \
		uv venv; \
	fi
	@echo "Installing package in editable mode with all test dependencies (jax + e3nn-jax + e3nn)..."
	uv pip install -e ".[test]"

test:
	.venv/bin/python -m pytest tests -q

lint:
	.venv/bin/ruff check .
	.venv/bin/black --check .

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
