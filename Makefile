# ==============================================================================
# Makefile — Web RAG Chatbot
# Usage: make <target>
# ==============================================================================

.PHONY: help install install-playwright setup run-api run-frontend \
        test lint format type-check clean

help:          ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | \
	awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:       ## Install Python dependencies
	pip install --upgrade pip
	pip install -r requirements.txt

install-playwright: ## Install Playwright browser (run once after pip install)
	python -m playwright install chromium

setup: install install-playwright  ## Full first-time setup
	cp -n .env.example .env || true
	@echo "\n✅  Setup complete. Edit .env and add your GROQ_API_KEY.\n"

run-api:       ## Start the FastAPI backend
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

run-frontend:  ## Start the Streamlit frontend
	streamlit run src/frontend/app.py --server.port 8501

run:           ## Start backend + frontend (requires two terminals or tmux)
	@echo "Open two terminals and run:"
	@echo "  make run-api"
	@echo "  make run-frontend"

test:          ## Run all tests
	pytest tests/ -v --tb=short

test-unit:     ## Run unit tests only
	pytest tests/unit/ -v

test-integration: ## Run integration tests only
	pytest tests/integration/ -v

lint:          ## Lint with ruff
	ruff check src/ tests/

format:        ## Auto-format with black + isort
	black src/ tests/ config/ app.py
	isort src/ tests/ config/ app.py

type-check:    ## Run mypy type checking
	mypy src/ --ignore-missing-imports

clean:         ## Remove caches, logs, and compiled files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	rm -rf logs/*.log