.PHONY: install test lint type all acceptance

install:
	uv sync

test:
	uv run pytest -q

# Deterministic, air-gapped acceptance gate (S05): Scenario 1 + the connector
# contract suite. Fails non-zero on any regression across ingest / contracts;
# later scenarios (scan/exit-codes, consolidation) are appended to this target.
acceptance:
	uv run pytest tests/test_acceptance.py tests/test_connector_contract.py -q

lint:
	uv run ruff check .

type:
	uv run mypy src

all: install lint type test
