.PHONY: install test lint type all acceptance

install:
	uv sync

test:
	uv run pytest -q

# Deterministic, air-gapped acceptance gate (S05 + M002/S04): the ingest /
# connector contract scenarios plus the T3/T4 analysis -> propose -> manifest
# lifecycle. Fails non-zero on any regression across ingest / analyize / propose /
# manifest; later scenarios (scan/exit-codes, consolidation) are appended here,
# through the M007 Taste Lens gate, the M008 Taste Dialogue gate (R032-R038), and
# the M009 Embedding Taste Head gate (R039-R043: vote capture, embedding provider,
# nonparametric head, attribution/exemplars, uncertainty-aware comparison, and the
# reachability check that every symbol/table this milestone added is wired up).
# The M010 Arbitrary Packing gate closes the set (R044-R047: N-source region
# geometry with its three manifest invariants, N-cell rendering with no silent
# drop and no silent upscale through either fit path, deterministic weighted
# packing proved byte-identical at both render targets, bounded-pool embedding
# grouping that degrades honestly to "unavailable" rather than inventing a group,
# and a reachability check strictly stronger than M009's — satisfiable only from
# cli.py / webui/app.js, never from api.py).
acceptance:
	uv run pytest tests/test_acceptance.py tests/test_connector_contract.py tests/test_acceptance_analysis.py tests/test_acceptance_render.py tests/test_acceptance_webui.py tests/test_acceptance_adapters.py tests/test_acceptance_ops.py tests/test_acceptance_taste.py tests/test_acceptance_taste_dialogue.py tests/test_acceptance_taste_embedding.py tests/test_acceptance_packing.py -q

lint:
	uv run ruff check .

type:
	uv run mypy src

all: install lint type test
