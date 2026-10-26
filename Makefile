SHELL := /bin/bash
UV := uv
ARGS ?=

GOINFRE := /goinfre/$(USER)
ifneq ($(wildcard $(GOINFRE)),)
export HF_HOME ?= $(GOINFRE)/.cache/huggingface
export UV_CACHE_DIR ?= $(GOINFRE)/.cache/uv
export UV_PROJECT_ENVIRONMENT ?= $(GOINFRE)/venvs/$(notdir $(CURDIR))
endif
VENV := $(or $(UV_PROJECT_ENVIRONMENT),.venv)

.DEFAULT_GOAL := run
.PHONY: install run debug test clean fclean lint lint-strict

install:
	@command -v $(UV) >/dev/null 2>&1 || { \
		echo "uv not found, installing it for the current user..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
		echo "Add \$$HOME/.local/bin to PATH if 'uv' is still not found."; \
	}
	$(UV) sync

# Extra CLI options: make run ARGS="--input data/input/x.json"
run:
	$(UV) run python -m src $(ARGS)

debug:
	$(UV) run python -m pdb -m src $(ARGS)

test:
	$(UV) run python -m unittest discover -s tests -t . -v

clean:
	find . \( -name .venv -o -name .git \) -prune -o \
		\( -name __pycache__ -o -name .mypy_cache -o -name .pytest_cache \) \
		-type d -exec rm -rf {} +
	rm -rf data/output

fclean: clean
	rm -rf .venv $(VENV)

lint:
	$(UV) run flake8 .
	$(UV) run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	$(UV) run flake8 .
	$(UV) run mypy . --strict
