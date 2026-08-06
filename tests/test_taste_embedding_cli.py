"""Tests for the ``curator taste embedding-head``/``embed-status --backfill`` CLI
commands (M009/S02, S03) — previously exercised by no test at all (``run_cli``
never invoked either command anywhere in this suite).
"""

from __future__ import annotations

import io
import json
from pathlib import Path

from PIL import Image

from acceptance_harness import run_cli
from curator.catalog import Catalog
from curator.cli import EXIT_OK
from curator.connectors.local import LocalConnector

FIXTURE_MODEL = Path(__file__).parent / "fixtures" / "tiny_embedding_model.onnx"


def _use_fixture_model(monkeypatch) -> None:
    monkeypatch.setenv("CURATOR_TASTE_EMBEDDING_MODEL_PATH", str(FIXTURE_MODEL))


def _cataloged(catalog: Catalog, tmp_path, name: str, color) -> str:
    """Register one real PNG's bytes via the ContentStore; return its sha256."""
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(buf, format="PNG")
    data = buf.getvalue()
    folder = tmp_path / "fixture"
    folder.mkdir(exist_ok=True)
    asset = folder / name
    asset.write_bytes(data)
    connector = LocalConnector(folder)
    return catalog.add_source(connector.connector_id, str(asset.resolve()), data)


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


# ---------------------------------------------------------------------------
# WR-06: a per-entry backfill failure must not abort the whole loop
# ---------------------------------------------------------------------------


def test_cli_embed_status_backfill_continues_past_one_bad_entry_and_reports_it(
    data_root, tmp_path, monkeypatch
):
    """A StorageError on one entry (content row present in the DB, but its blob
    was never written to disk) must not abort the whole --backfill loop — every
    OTHER entry still gets backfilled, and the failure is reported, not swallowed
    into a bare stack trace that discards every already-successful entry."""
    _use_fixture_model(monkeypatch)
    catalog = Catalog(data_root=data_root)
    try:
        # Two entries with REAL content bytes — these must successfully backfill.
        _cataloged(catalog, tmp_path, "a.png", (200, 30, 30))
        _cataloged(catalog, tmp_path, "b.png", (30, 30, 200))

        # A third entry whose `content` row exists (satisfies catalog_entries'
        # FK) but whose blob was never written to disk — catalog.content.get()
        # raises StorageError for this one, deterministically, mirroring a
        # corrupted/evicted content blob rather than anything embedding-specific.
        missing_sha = "f" * 64
        catalog.db.execute("INSERT INTO content(sha256, size) VALUES (?, ?)", (missing_sha, 1))
        catalog.db.execute(
            "INSERT INTO catalog_entries(connector_id, asset_id, revision, sha256)"
            " SELECT connector_id, 'missing-blob', '1', ? FROM source_connectors LIMIT 1",
            (missing_sha,),
        )
        catalog.db.commit()
    finally:
        catalog.db.close()

    rc, out = run_cli(["taste", "embed-status", "--backfill", "--json"])
    assert rc == EXIT_OK
    payload = json.loads(out)
    assert payload["ok"] is True
    # The two good entries still backfilled — the bad one didn't abort the loop.
    assert payload["backfilled"] == 2
    assert payload["backfill_error_count"] == 1
    assert len(payload["backfill_errors"]) == 1
    assert payload["backfill_errors"][0]["sha256"] == missing_sha
    assert "not found" in payload["backfill_errors"][0]["error"]
