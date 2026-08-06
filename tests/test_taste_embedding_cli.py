"""Tests for the ``curator taste embedding-head``/``embed-status --backfill`` CLI
commands (M009/S02, S03) — previously exercised by no test at all (``run_cli``
never invoked either command anywhere in this suite).
"""

from __future__ import annotations

import json
from pathlib import Path

from acceptance_harness import run_cli
from curator.cli import EXIT_OK

FIXTURE_MODEL = Path(__file__).parent / "fixtures" / "tiny_embedding_model.onnx"


def _use_fixture_model(monkeypatch) -> None:
    monkeypatch.setenv("CURATOR_TASTE_EMBEDDING_MODEL_PATH", str(FIXTURE_MODEL))


# ---------------------------------------------------------------------------
# WR-05: zero-vote parity self-check now exercises the real fit_embedding_head
# ---------------------------------------------------------------------------


def test_cli_embedding_head_zero_vote_parity_ok_end_to_end(data_root, monkeypatch):
    _use_fixture_model(monkeypatch)
    rc, out = run_cli(["taste", "embedding-head", "--json"])
    assert rc == EXIT_OK
    payload = json.loads(out)
    assert payload["ok"] is True
    assert payload["capacity"] == 0
    assert payload["retained_parameters"] == 0
    assert payload["zero_vote_parity_ok"] is True
