SHELL := /bin/bash
UV := uv
ARGS ?=

# 42 machines have a small home quota: the model (1.5 GB) and the uv cache
# go to /goinfre when it exists. Values already set in the environment win.
GOINFRE := /goinfre/$(USER)
ifneq ($(wildcard $(GOINFRE)),)
export HF_HOME ?= $(GOINFRE)/.cache/huggingface
export UV_CACHE_DIR ?= $(GOINFRE)/.cache/uv
endif

.DEFAULT_GOAL := run
.PHONY: install run debug test clean fclean lint lint-strict

# Installs uv (Ubuntu: no apt package, use the official installer) if missing.
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
	rm -rf .venv

lint:
	$(UV) run flake8 .
	$(UV) run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	$(UV) run flake8 .
	$(UV) run mypy . --strict
