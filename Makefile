# Trajecto Makefile - Convenient commands for development and testing

.PHONY: help test test-python test-firmware test-cov clean install

# Default target
help:
	@echo "Trajecto Development Commands"
	@echo "============================="
	@echo ""
	@echo "Testing:"
	@echo "  make test           - Run all Python tests"
	@echo "  make test-verbose   - Run tests with verbose output"
	@echo "  make test-cov       - Run tests with coverage report"
	@echo "  make test-watch     - Watch for changes and auto-run tests"
	@echo "  make test-firmware  - Build and flash firmware tests"
	@echo ""
	@echo "Development:"
	@echo "  make install        - Install dependencies"
	@echo "  make clean          - Clean build artifacts"
	@echo "  make format         - Format code (if configured)"
	@echo ""
	@echo "Training:"
	@echo "  make train          - Train ESKF-TCN model"
	@echo "  make validate       - Validate trained model"
	@echo ""

# ============================================================================
# Testing
# ============================================================================

test:
	@echo "Running Python tests..."
	pytest

test-verbose:
	@echo "Running Python tests (verbose)..."
	pytest -v

test-cov:
	@echo "Running tests with coverage..."
	pytest --cov=model --cov=utils --cov-report=html --cov-report=term
	@echo ""
	@echo "Coverage report: htmlcov/index.html"

test-watch:
	@echo "Watching for changes..."
	@which ptw > /dev/null || (echo "Installing pytest-watch..." && pip install pytest-watch)
	ptw

test-fast:
	@echo "Running fast tests only..."
	pytest -m "not slow"

test-unit:
	@echo "Running unit tests..."
	pytest -m unit

test-integration:
	@echo "Running integration tests..."
	pytest -m integration

test-parity:
	@echo "Running Python-C++ parity tests..."
	pytest -m parity -v

test-firmware:
	@echo "Building and flashing firmware tests..."
	cd firmware && idf.py build flash monitor

# ============================================================================
# Development
# ============================================================================

install:
	@echo "Installing dependencies with uv..."
	uv sync
	@echo ""
	@echo "Installing dev dependencies..."
	uv add --dev pytest pytest-cov pytest-watch
	@echo ""
	@echo "Done! Activate with: source .venv/bin/activate"

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "Clean complete!"

# ============================================================================
# Training & Validation
# ============================================================================

train:
	@echo "Training ESKF-TCN model..."
	python train.py --model eskf_tcn --epochs 200 --lr 1e-4 --batch_size 4

train-baseline:
	@echo "Training TCN-only baseline..."
	python train.py --model only_tcn --epochs 200

validate:
	@echo "Validating trained model..."
	python validate.py --model_type eskf_tcn --model_path eskf_tcn_model.pth

validate-baseline:
	@echo "Validating pure ESKF baseline..."
	python validate.py --model_type pure_eskf

# ============================================================================
# Firmware
# ============================================================================

firmware-build:
	@echo "Building firmware..."
	cd firmware && idf.py build

firmware-flash:
	@echo "Flashing firmware..."
	cd firmware && idf.py flash

firmware-monitor:
	@echo "Monitoring firmware output..."
	cd firmware && idf.py monitor

firmware-clean:
	@echo "Cleaning firmware build..."
	cd firmware && idf.py fullclean

# ============================================================================
# CI/CD (for automation)
# ============================================================================

ci-test:
	@echo "Running CI test suite..."
	pytest --cov=model --cov=utils --cov-report=xml --cov-report=term -v

ci-lint:
	@echo "Running linters..."
	@which ruff > /dev/null && ruff check . || echo "Ruff not installed (optional)"
	@which mypy > /dev/null && mypy model/ utils/ || echo "mypy not installed (optional)"
