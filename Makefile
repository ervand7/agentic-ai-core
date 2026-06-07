# Makefile for agentic-ai-core
# Run `make` or `make help` to see all available commands.

# Use the project virtualenv if it exists, otherwise fall back to system python.
VENV ?= venv
PYTHON := $(VENV)/bin/python
PIP := $(PYTHON) -m pip

# Default app entrypoint + dev server settings.
APP := app.main:app
HOST ?= 0.0.0.0
PORT ?= 8000

# Paths used by tooling.
SRC := app
TESTS := tests

.DEFAULT_GOAL := help

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
.PHONY: venv
venv: ## Create the virtualenv (.venv) using python3.12
	python3.12 -m venv $(VENV)
	$(PIP) install --upgrade pip

.PHONY: install
install: ## Install runtime dependencies
	$(PIP) install -r requirements.txt

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
.PHONY: run
run: ## Run the API with auto-reload (dev)
	$(PYTHON) -m uvicorn $(APP) --host $(HOST) --port $(PORT) --reload

.PHONY: serve
serve: ## Run the API without reload (prod-like)
	$(PYTHON) -m uvicorn $(APP) --host $(HOST) --port $(PORT)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
.PHONY: test
test: ## Run the full suite (unit + api + all integration tiers)
	RUN_LIVE_OPENAI=1 $(PYTHON) -m pytest

.PHONY: test-unit
test-unit: ## Run unit tests only
	$(PYTHON) -m pytest $(TESTS)/unit

.PHONY: test-api
test-api: ## Run API router tests only
	$(PYTHON) -m pytest $(TESTS)/api

.PHONY: test-integration
test-integration: ## Run all integration tests (Tier 1 + Qdrant + live OpenAI)
	RUN_LIVE_OPENAI=1 $(PYTHON) -m pytest $(TESTS)/integration -m integration

.PHONY: test-cov
test-cov: ## Run tests with a terminal coverage report
	$(PYTHON) -m pytest --cov=$(SRC) --cov-report=term-missing

.PHONY: test-cov-html
test-cov-html: ## Run tests and write an HTML coverage report to htmlcov/
	$(PYTHON) -m pytest --cov=$(SRC) --cov-report=html
	@echo "Open htmlcov/index.html"

.PHONY: test-watch
test-watch: ## Re-run the failed-first suite (quick feedback loop)
	$(PYTHON) -m pytest --ff -q

# ---------------------------------------------------------------------------
# Linting / formatting / types
# ---------------------------------------------------------------------------
.PHONY: lint
lint: ## Lint with ruff
	$(PYTHON) -m ruff check $(SRC) $(TESTS)

.PHONY: lint-fix
lint-fix: ## Lint and auto-fix with ruff
	$(PYTHON) -m ruff check --fix $(SRC) $(TESTS)

.PHONY: format
format: ## Format code with ruff
	$(PYTHON) -m ruff format $(SRC) $(TESTS)

.PHONY: format-check
format-check: ## Check formatting without writing changes
	$(PYTHON) -m ruff format --check $(SRC) $(TESTS)

.PHONY: typecheck
typecheck: ## Static type-check with mypy
	$(PYTHON) -m mypy $(SRC)

# ---------------------------------------------------------------------------
# Aggregate / CI
# ---------------------------------------------------------------------------
.PHONY: check
check: lint format-check typecheck test ## Run all checks (lint + format + types + tests)

.PHONY: ci
ci: install-dev check ## Install dev deps then run all checks (for CI)

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------
.PHONY: docker-up
docker-up: ## Start services (app + Qdrant) via docker compose
	docker compose up --build

.PHONY: docker-down
docker-down: ## Stop and remove docker compose services
	docker compose down

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------
.PHONY: clean
clean: ## Remove caches and coverage artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
