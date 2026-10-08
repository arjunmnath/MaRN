.PHONY: help install update shell test test-cov lint format typecheck check clean build patch minor major release docs serve-docs docs-clean docs-live

help:
	@echo "Available commands:"
	@echo "  make install     Install dependencies"
	@echo "  make update      Update dependencies"
	@echo "  make shell       Activate poetry shell"
	@echo "  make test        Run tests"
	@echo "  make test-cov    Run tests with coverage"
	@echo "  make lint        Run Ruff linter"
	@echo "  make format      Format code"
	@echo "  make typecheck   Run mypy"
	@echo "  make check       Run all checks"
	@echo "  make build       Build package"
	@echo "  make clean       Remove caches/build artifacts"
	@echo "  make docs        Build documentation"
	@echo "  make serve-docs  Serve documentation locally"
	@echo "  make docs-clean  Clean documentation build artifacts"
	@echo "  make docs-live   Live rebuild/serve documentation"
	@echo "  make release     Format code and run all checks"
	@echo "  make patch       Release a patch version bump"
	@echo "  make minor       Release a minor version bump"
	@echo "  make major       Release a major version bump"

docs:
	poetry run sphinx-build -b html docs docs/_build/html

serve-docs:
	poetry run python -m http.server 8000 -d docs/_build/html

docs-clean:
	rm -rf docs/_build docs/api/generated

docs-live:
	poetry run sphinx-autobuild docs docs/_build/html

release:
	poetry run ruff format .
	$(MAKE) check

patch:
	$(MAKE) release
	poetry version patch
	$(MAKE) docs
	poetry build
	$(MAKE) clean
	@echo "New version: $$(poetry version -s)"

minor:
	$(MAKE) release
	poetry version minor
	$(MAKE) docs
	poetry build
	$(MAKE) clean
	@echo "New version: $$(poetry version -s)"

major:
	$(MAKE) release
	poetry version major
	$(MAKE) docs
	poetry build
	$(MAKE) clean
	@echo "New version: $$(poetry version -s)"

install:
	poetry install

update:
	poetry update

shell:
	poetry shell

test:
	poetry run pytest

test-cov:
	poetry run pytest --cov=marn --cov-report=term-missing

lint:
	poetry run ruff check .

format:
	poetry run ruff format .

typecheck:
	poetry run mypy src tests benchmarks cookbook

check: lint typecheck test

build:
	poetry build

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf dist build .coverage htmlcov