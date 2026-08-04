.PHONY: install test lint type all acceptance

install:
	uv sync

test:
	uv run pytest -q

# Deterministic, air-gapped acceptance gate (S05 + M002/S04): the ingest /
# connector contract scenarios plus the T3/T4 analysis -> propose -> manifest
# lifecycle. Fails non-zero on any regression across ingest / analyize / propose /
# manifest; later scenarios (scan/exit-codes, consolidation) are appended here.
acceptance:
	uv run pytest tests/test_acceptance.py tests/test_connector_contract.py tests/test_acceptance_analysis.py -q

lint:
	uv run ruff check .

type:
	uv run mypy src

all: install lint type test
