.PHONY: install run debug clean lint lint-strict test

install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

test:
	uv run python -m unittest discover -s tests -t . -v

clean:
	find . \( -name .venv -o -name .git \) -prune -o \
		\( -name __pycache__ -o -name .mypy_cache \) -type d \
		-exec rm -rf {} +
	rm -rf data/output

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict
