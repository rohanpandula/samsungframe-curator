"""Tests for the M006/S03 packaging surface (launchd + Docker/Compose + headless start).

These tests run without Docker: they statically parse the packaging artifacts
(plist via :mod:`plistlib`; compose via an environment-clean structural check since
``yaml`` is not a dependency) and exercise the ``curator --headless start --check``
dry path in-process via :func:`curator.cli.main`, which resolves configuration and
emits the machine-readable JSON status without binding a server.
"""

from __future__ import annotations

import json
import os
import plistlib
from pathlib import Path

from curator import cli

REPO = Path(__file__).resolve().parents[1]
PLIST = REPO / "packaging" / "launchd" / "com.rohan.curator.plist"
DOCKERFILE = REPO / "packaging" / "docker" / "Dockerfile"
DOCKERFILE_CUDA = REPO / "packaging" / "docker" / "Dockerfile.cuda"
COMPOSE = REPO / "packaging" / "docker" / "docker-compose.yml"

_SECRET_ENV_SUFFIXES = ("_TOKEN", "_API_KEY")


def _clear_env_secrets(monkeypatch):
    """Remove any ambient ``*_TOKEN`` / ``*_API_KEY`` vars so tests are deterministic."""
    for key in list(os.environ):
        if key.endswith(_SECRET_ENV_SUFFIXES):
            monkeypatch.delenv(key, raising=False)


def test_launchd_plist_parses_and_runs_headless_start():
    """The launchd plist is valid and ProgramArguments run the headless start."""
    plist = plistlib.loads(PLIST.read_bytes())
    assert plist["Label"] == "com.rohan.curator"
    assert plist["KeepAlive"] is True
    args = plist["ProgramArguments"]
    assert any("curator" in os.path.basename(arg) for arg in args)
    assert "--headless" in args
    assert "start" in args
    assert args.index("--headless") < args.index("start")


def test_compose_structural_services_curator_ports():
    """docker-compose.yml has the expected service, ports and health wiring.

    ``yaml`` is not installed (no new deps), so this is a structural check for the
    documented top-level keys rather than a full YAML parse.
    """
    text = COMPOSE.read_text()
    assert "services:" in text
    assert "curator:" in text
    assert "127.0.0.1:8765:8765" in text  # loopback-bound port mapping
    assert "CURATOR_DATA_ROOT" in text
    assert "env_file" in text
    assert "volumes" in _service_block(text)


def _service_block(text: str) -> str | None:
    """Return the ``curator:`` service block text, or None if absent."""
    start = text.find("curator:")
    if start == -1:
        return None
    return text[start:]


def test_dockerfiles_contain_expected_directives():
    """Both Dockerfiles are non-empty and carry the documented directives."""
    for path in (DOCKERFILE, DOCKERFILE_CUDA):
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        assert lines, f"{path.name} must not be empty"
        joined = "\n".join(lines)
        assert any(ln.startswith("FROM") for ln in lines)
        assert any("curator" in ln and "--headless" in ln and "start" in ln for ln in lines)
        assert "COPY" in joined
        assert "ENTRYPOINT" in joined
        assert "HEALTHCHECK" in joined


def test_headless_start_check_ok_with_secret(data_root, monkeypatch, capsys):
    """With an env-backed secret set, --check returns 0 + JSON ready/data_root."""
    _clear_env_secrets(monkeypatch)
    monkeypatch.setenv("SAMSUNG_API_TOKEN", "test-token")

    rc = cli.main(["--headless", "start", "--check"])
    assert rc == 0

    doc = json.loads(capsys.readouterr().out)
    assert doc["status"] == "ok"
    assert doc["ready"] is True
    assert doc["data_root"] == str(data_root)
    assert doc["api"] == "http://127.0.0.1:8765"


def test_headless_start_missing_secret_nonzero(data_root, monkeypatch, capsys):
    """Without any required secret, --check exits non-zero with a clear error."""
    _clear_env_secrets(monkeypatch)

    rc = cli.main(["--headless", "start", "--check"])
    assert rc == 2

    err = capsys.readouterr().err
    assert "secret" in err
    assert "API_KEY" in err or "TOKEN" in err


def test_headless_start_cuda_fallback_cpu(data_root, monkeypatch, capsys):
    """Requesting CUDA when unavailable reports a CPU fallback + clear message."""
    _clear_env_secrets(monkeypatch)
    monkeypatch.setenv("SAMSUNG_API_TOKEN", "test-token")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")  # still no nvidia-smi -> unavailable

    rc = cli.main(["--headless", "start", "--check", "--accelerator", "cuda"])
    assert rc == 0

    captured = capsys.readouterr()
    doc = json.loads(captured.out)
    assert doc["ready"] is True
    assert doc["accelerator"] == "cpu (fallback)"
    assert "degrading to CPU" in captured.err


def test_headless_start_config_file_overrides_secrets(data_root, monkeypatch, capsys, tmp_path):
    """--config supplies secrets/data_root fallback; env still wins for data_root."""
    _clear_env_secrets(monkeypatch)
    monkeypatch.setenv("CURATOR_DATA_ROOT", str(data_root))
    cfg = tmp_path / "curator.json"
    cfg.write_text(
        json.dumps({"data_root": "/ignored", "secrets": {"CURATOR_API_KEY": "cfg-secret"}}),
        encoding="utf-8",
    )

    rc = cli.main(["--headless", "--config", str(cfg), "start", "--check"])
    assert rc == 0

    doc = json.loads(capsys.readouterr().out)
    assert doc["data_root"] == str(data_root)  # env overrides config-file fallback
    assert doc["ready"] is True
