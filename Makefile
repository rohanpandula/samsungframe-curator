.PHONY: install test lint type all

install:
	uv sync

test:
	uv run pytest -q

lint:
	uv run ruff check .

type:
	uv run mypy src

all: install lint type test
