# Set the shell explicitly to make sure the Makefile runs on different platforms
SHELL := $(shell which bash)

# Root directory of sig-gitops
GIT_ROOT := "$(shell git rev-parse --show-toplevel)"

help:
	@echo "$$(grep -hE '^\S+:.*## .*$$|(^#--)' $(MAKEFILE_LIST))" \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[32m %-43s\033[0m %s\n", $$1, $$2}' \
		| sed -e 's/\[32m #-- /[33m/'

#-- general

init: ## Project Setup
	${MAKE} git/pre-commit
	${MAKE} uv/sync

#-- git

git/pre-commit: ## Install pre-commit hooks
	uv run pre-commit install

#-- uv

uv/sync: ## Install dependencies
	uv sync

#-- pytest

pytest/run: ## Run tests
	uv run pytest

pytest/coverage: ## Run test coverage
	uv run pytest --cov=src

#-- ruff

ruff/format: ## Run formatting
	uv run ruff format ${GIT_ROOT}

uv/lint: ## Run linting
	uv run ruff check ${GIT_ROOT}

#-- mypy

mypy/typecheck: ## Type check
	uv run mypy ${GIT_ROOT}