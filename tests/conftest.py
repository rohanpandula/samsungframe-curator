"""Shared test fixtures for the Curator test suite."""

from __future__ import annotations

import pytest


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    """A shared tmp CURATOR_DATA_ROOT.

    Sets ``CURATOR_DATA_ROOT`` to a per-test temporary directory and returns the
    path, so any code path that resolves config from environment sees an isolated
    data root.
    """
    root = tmp_path / "curator-data"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("CURATOR_DATA_ROOT", str(root))
    return root
