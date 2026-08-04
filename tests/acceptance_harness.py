"""Shared in-process helpers for the S05 acceptance test suite.

These helpers let every scenario test bootstrap its own deterministic fixture
and drive the CLI in-process (no subprocess, no network) so the ``make
acceptance`` gate is repeatable and self-contained. They intentionally avoid
coupling to pytest's ``capsys`` so a test can capture stdout at will.

The ``assert_no_network_imports`` audit is the air-gap regression gate for
Scenario 1: it statically proves the ingest import closure references no network
client module, so a future change cannot accidentally pull ``urllib`` /
``requests`` / ``socket`` / ``httpx`` / ``aiohttp`` / ``http`` into the ingest
path without the acceptance gate failing.
"""

from __future__ import annotations

import ast
import importlib.util
import io
from contextlib import redirect_stdout
from pathlib import Path

from consolidate_fixture import build_consolidation_fixture
from curator.cli import main as cli_main
from curator.hashing import sha256_hex
from fixture_library import build_fixture

# Top-level modules that would pull a network client into the ingest path. A
# static-audit regression gate (S05 Scenario 1): ``socket`` makes code network-
# reachable and ``http`` is the stdlib HTTP client surface.
NETWORK_BANNED = {"urllib", "requests", "socket", "httpx", "aiohttp", "http"}

# Entry points whose transitive in-package import closure is audited.
_AUDIT_ROOTS = ("curator.ingest", "curator.connectors.local", "curator.hashing")


def build_ingest_tree(tmp_dir) -> Path:
    """Build the deterministic 50-file / 30-cluster fixture; return its folder."""
    return build_fixture(Path(tmp_dir) / "ingest-fixture").root


def build_consolidation_tree(tmp_dir) -> Path:
    """Build the deterministic legacy-ssd consolidation fixture; return its folder."""
    return build_consolidation_fixture(Path(tmp_dir) / "src").root


def run_cli(argv) -> tuple[int, str]:
    """Run ``curator.cli.main(argv)`` in-process, returning ``(rc, stdout)``."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli_main(list(argv))
    return rc, buf.getvalue()


def sha256_file(path) -> str:
    """Return the SHA-256 hex digest of the file at *path*."""
    return sha256_hex(Path(path).read_bytes())


def _top_level(name: str) -> str:
    return name.split(".")[0]


def assert_no_network_imports() -> None:
    """Assert the ingest import closure references no network client module.

    Walks the transitive in-package import graph reachable from ``curator.ingest``,
    ``curator.connectors.local`` and ``curator.hashing`` by parsing each module's
    AST for ``import`` / ``from ... import`` statements. Only in-package
    (``curator``) targets are followed transitively; every non-``curator`` absolute
    top-level import name is checked directly against the banned set. This is a
    static audit — no runtime network-enforcement code is added.
    """
    banned_hit: set[str] = set()
    seen: set[str] = set()
    frontier = list(_AUDIT_ROOTS)
    while frontier:
        mod = frontier.pop()
        if mod in seen:
            continue
        seen.add(mod)
        spec = importlib.util.find_spec(mod)
        if spec is None or spec.origin is None:
            continue
        try:
            tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = _top_level(alias.name)
                    if top in NETWORK_BANNED:
                        banned_hit.add(top)
                    elif top == "curator" and alias.name not in seen:
                        frontier.append(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                base = _top_level(node.module)
                if base in NETWORK_BANNED:
                    banned_hit.add(base)
                elif base == "curator" and node.module not in seen:
                    frontier.append(node.module)
    assert not banned_hit, (
        "network import(s) reachable from ingest closure: "
        + ", ".join(sorted(banned_hit))
    )
