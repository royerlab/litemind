.PHONY: help setup install install-dev test test-cov check format lint typecheck build clean publish publish-patch

# Default target
help:
	@echo "Available commands:"
	@echo "  make setup        - Install project with all development dependencies"
	@echo "  make install      - Install project with minimal dependencies"
	@echo "  make install-dev  - Install project with dev dependencies only"
	@echo "  make test         - Run all tests"
	@echo "  make test-cov     - Run tests with coverage report"
	@echo "  make check        - Run all code checks (format check + lint + typecheck)"
	@echo "  make format       - Format code with black and isort"
	@echo "  make lint         - Run flake8 linter"
	@echo "  make typecheck    - Run mypy type checker"
	@echo "  make build        - Build package"
	@echo "  make clean        - Clean build artifacts"
	@echo "  make publish      - Bump version, commit, tag, and push (triggers PyPI release)"
	@echo "  make publish-patch- Publish patch version (same day increment)"

# =============================================================================
# Setup & Installation
# =============================================================================

setup:
	python -m pip install --upgrade pip
	pip install -e ".[dev,rag,whisper,documents,tables,videos,audio,remote,tasks]"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# =============================================================================
# Testing
# =============================================================================

test:
	pytest src/

test-cov:
	@mkdir -p reports
	pytest --cov=src --cov-report=html:reports/coverage --cov-report=xml:reports/coverage.xml src/

# =============================================================================
# Code Quality
# =============================================================================

check: format-check lint typecheck
	@echo "All checks passed!"

format:
	isort .
	black .

format-check:
	@echo "Checking code formatting..."
	black --check .
	isort --check-only .

lint:
	@echo "Running flake8..."
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

typecheck:
	@echo "Running mypy..."
	mypy src/litemind --ignore-missing-imports

# =============================================================================
# Building
# =============================================================================

build: clean
	hatch build

clean:
	hatch clean
	rm -rf dist/ build/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# =============================================================================
# Publishing
# =============================================================================

# Get current version from __init__.py
CURRENT_VERSION := $(shell grep -o '__version__ = "[^"]*"' src/litemind/__init__.py | cut -d'"' -f2)
TODAY := $(shell date +%Y.%-m.%-d)

# Bump version to today's date and publish
publish:
	@echo "Current version: $(CURRENT_VERSION)"
	@if echo "$(CURRENT_VERSION)" | grep -q "^$(TODAY)"; then \
		echo "Error: Version $(CURRENT_VERSION) is already today's date."; \
		echo "Use 'make publish-patch' for same-day releases."; \
		exit 1; \
	fi
	@echo "Updating version to: $(TODAY)"
	@sed -i.bak 's/__version__ = "[^"]*"/__version__ = "$(TODAY)"/' src/litemind/__init__.py && rm -f src/litemind/__init__.py.bak
	@echo "Committing version bump..."
	git add src/litemind/__init__.py
	git commit -m "chore: bump version to $(TODAY)"
	@echo "Creating tag v$(TODAY)..."
	git tag "v$(TODAY)"
	@echo "Pushing to origin..."
	git push origin main --tags
	@echo "Done! GitHub Actions will publish to PyPI."

# Publish patch version (for same-day releases)
publish-patch:
	@echo "Current version: $(CURRENT_VERSION)"
	@if echo "$(CURRENT_VERSION)" | grep -q "^$(TODAY)\."; then \
		PATCH=$$(echo "$(CURRENT_VERSION)" | sed 's/$(TODAY)\.\([0-9]*\)/\1/'); \
		NEW_PATCH=$$((PATCH + 1)); \
		NEW_VERSION="$(TODAY).$$NEW_PATCH"; \
	elif [ "$(CURRENT_VERSION)" = "$(TODAY)" ]; then \
		NEW_VERSION="$(TODAY).1"; \
	else \
		NEW_VERSION="$(TODAY)"; \
	fi; \
	echo "Updating version to: $$NEW_VERSION"; \
	sed -i.bak "s/__version__ = \"[^\"]*\"/__version__ = \"$$NEW_VERSION\"/" src/litemind/__init__.py && rm -f src/litemind/__init__.py.bak; \
	echo "Committing version bump..."; \
	git add src/litemind/__init__.py; \
	git commit -m "chore: bump version to $$NEW_VERSION"; \
	echo "Creating tag v$$NEW_VERSION..."; \
	git tag "v$$NEW_VERSION"; \
	echo "Pushing to origin..."; \
	git push origin main --tags; \
	echo "Done! GitHub Actions will publish to PyPI."
