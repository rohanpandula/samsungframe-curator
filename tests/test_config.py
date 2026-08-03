"""Tests for src/curator/config.py — six-axis configuration (R022)."""

from __future__ import annotations

from pathlib import Path

from curator.config import CuratorConfig


def test_data_root_from_env(data_root):
    cfg = CuratorConfig()
    assert cfg.data_root == Path(data_root)


def test_source_type_default_is_local():
    cfg = CuratorConfig()
    assert cfg.source.type == "local"


def test_source_type_override_via_env(monkeypatch):
    monkeypatch.setenv("CURATOR_SOURCE__TYPE", "immich")
    cfg = CuratorConfig()
    assert cfg.source.type == "immich"


def test_source_path_override_via_env(monkeypatch):
    monkeypatch.setenv("CURATOR_SOURCE__PATH", "/tmp/my-photos")
    cfg = CuratorConfig()
    assert cfg.source.path == Path("/tmp/my-photos")


def test_data_root_defaults_to_curator_home(monkeypatch):
    monkeypatch.delenv("CURATOR_DATA_ROOT", raising=False)
    cfg = CuratorConfig()
    assert cfg.data_root == Path.home() / ".curator"


def test_all_six_axes_present_with_defaults():
    cfg = CuratorConfig()
    # Every axis is a typed placeholder with a default; no axis may be missing.
    assert cfg.intelligence_provider.provider == "local"
    assert cfg.interface.mode == "cli"
    assert cfg.runtime.mode == "one-shot"
    assert cfg.render_target.profile == "1080p"
    assert cfg.destination.type == "filesystem"
    assert cfg.source.type == "local"


def test_unknown_env_var_is_ignored(monkeypatch):
    # R022: config must not break on unknown/extra vars (no bespoke axis code paths).
    monkeypatch.setenv("CURATOR_SOMETHING_ELSE", "whatever")
    monkeypatch.setenv("CURATOR_DATA_ROOT", "/tmp/ignored-env-test")
    cfg = CuratorConfig()
    assert cfg.data_root == Path("/tmp/ignored-env-test")
