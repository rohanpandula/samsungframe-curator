"""Non-destructive consolidation executor (S03-T3 / R002 execute).

:class:`ConsolidationExecutor` performs the **execute** half of R002 on the same
legacy-ssd directory the planner inventories (:mod:`curator.consolidate.plan`).
For every source file it:

1. **stages** the bytes into a content-addressed staging area
   ``<root>/staging/ab/cd/<sha256>`` (atomic temp + fsync + ``os.replace``, the
   S01 :class:`~curator.content_store.ContentStore` idiom),
2. **verifies** the staged SHA-256 equals the source SHA-256,
3. **promotes** the staged blob atomically into the canonical library root
   ``<root>/library/ab/cd/<sha256>`` (content-addressed, so byte-dupes converge
   on one library file).

Sources are **read-only** throughout :meth:`execute` — the only step that moves
the legacy folder is the explicitly-approved :meth:`archive`, which relocates a
fully-consolidated (every file ``promoted``) source folder intact beneath
``<root>/archive/``.

Every file's progress is recorded in the per-file ``consolidation_journal``
(schema v3, ``started -> staged -> verified -> promoted/error``) via the
:class:`~curator.catalog.Catalog` journal helpers, so a mid-run interrupt (a
process crash, not a recoverable I/O error) leaves a durable checkpoint that
:meth:`execute(resume=True)` continues from — already-``promoted`` files are
skipped, ``verified``/``staged`` files are completed, and ``error`` files are
re-attempted (mirroring the S02 IngestPipeline resume semantics).
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from curator.catalog import (
    CONSOLIDATION_ERROR,
    CONSOLIDATION_PROMOTED,
    CONSOLIDATION_STAGED,
    CONSOLIDATION_VERIFIED,
    Catalog,
)
from curator.config import CuratorConfig
from curator.connectors.local import is_supported_suffix
from curator.consolidate.plan import SIDECAR_SUFFIXES
from curator.errors import ConsolidationError
from curator.hashing import sha256_hex


@dataclass
class ConsolidationResult:
    """Structured, JSON-serializable outcome of one :meth:`execute` run.

    ``connector_id`` is the resolved legacy source path (the consolidation run
    identity in ``consolidation_journal``) and ``source_path`` its canonical form.
    ``staged`` / ``verified`` / ``promoted`` count files advanced this run;
    ``skipped`` counts files already ``promoted`` by a prior run and left alone on
    resume. ``errors`` preserves per-file failure detail. ``unique_library_files``
    is the content-addressed convergence count — the number of distinct blobs on
    disk under ``<root>/library/`` (equal to the distinct source hashes, so byte
    dupes collapse to one library file).
    """

    connector_id: str
    source_path: str
    staged: int = 0
    verified: int = 0
    promoted: int = 0
    skipped: int = 0
    unique_library_files: int = 0
    errors: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-encodable dict (mirrors IngestReport's asdict pattern)."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Return this result serialized as a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class ConsolidationExecutor:
    """Stage / verify / promote / resume / archive a legacy source folder.

    All deterministic artefacts live under one canonical data root (from the
    six-axis config, ``CURATOR_DATA_ROOT``): ``<root>/staging/`` for in-flight
    blobs, ``<root>/library/`` for the content-addressed canonical store, and
    ``<root>/archive/`` for explicitly-approved relocations of fully-consolidated
    source folders.
    """

    def __init__(
        self,
        source: Path | str,
        catalog: Catalog | None = None,
        data_root: Path | None = None,
    ) -> None:
        self.source = Path(source)
        if data_root is None:
            data_root = CuratorConfig().data_root
        root = Path(data_root)
        self.library_root = root / "library"
        self.staging_root = root / "staging"
        self.archive_root = root / "archive"
        self.catalog = catalog if catalog is not None else Catalog(data_root=root)

    # -- layout helpers --------------------------------------------------------

    def staged_path(self, sha256: str) -> Path:
        """Content-addressed staging path for *sha256* (two-level hex shard)."""
        return self.staging_root / sha256[:2] / sha256[2:4] / sha256

    def _library_path(self, sha256: str) -> Path:
        """Canonical content-addressed library path for *sha256*."""
        return self.library_root / sha256[:2] / sha256[2:4] / sha256

    def _count_library_files(self) -> int:
        """Return the number of distinct blobs under the canonical library root."""
        if not self.library_root.is_dir():
            return 0
        return sum(1 for p in self.library_root.rglob("*") if p.is_file())

    # -- execute ---------------------------------------------------------------

    def execute(self, resume: bool = False) -> ConsolidationResult:
        """Consolidate every source file, journaled and non-destructively.

        Stages each file, verifies its staged SHA-256, and atomically promotes it
        into the content-addressed library. With *resume=True* an already-``promoted``
        file (from a prior run) is skipped, a ``verified`` or ``staged`` file is
        completed from its staged bytes, and an ``error``/``started``/untouched file
        is processed fresh. Sources remain byte-for-byte untouched throughout.
        """
        source = Path(self.source)
        if not source.is_dir():
            raise ConsolidationError(
                f"consolidate source is not a directory: {source}"
            )
        self.staging_root.mkdir(parents=True, exist_ok=True)
        self.library_root.mkdir(parents=True, exist_ok=True)
        connector_id = str(source.resolve())
        result = ConsolidationResult(
            connector_id=connector_id, source_path=str(source.resolve())
        )

        for path in _iter_consolidated_files(source):
            asset_id = str(path.resolve())
            if resume:
                checkpoint = self.catalog.consolidation_checkpoint(
                    connector_id, asset_id
                )
                if checkpoint == CONSOLIDATION_PROMOTED:
                    result.skipped += 1
                    continue
            else:
                checkpoint = None

            row_id = self.catalog.consolidation_journal_start(
                connector_id, asset_id
            )
            try:
                data = path.read_bytes()
                source_sha = sha256_hex(data)
                staged = self.staged_path(source_sha)

                # A file already verified in a prior run needs no re-copy/verify;
                # a staged (or fresh/error) file goes through the full path.
                if checkpoint != CONSOLIDATION_VERIFIED:
                    self._stage(source_sha, staged, data)
                    self.catalog.consolidation_journal_update(
                        row_id, CONSOLIDATION_STAGED, sha256=source_sha
                    )
                    self._verify(source_sha, staged)
                    self.catalog.consolidation_journal_update(
                        row_id, CONSOLIDATION_VERIFIED
                    )

                self._promote(source_sha, staged)
                self.catalog.consolidation_journal_update(
                    row_id, CONSOLIDATION_PROMOTED
                )
                result.staged += 1
                result.verified += 1
                result.promoted += 1
            except OSError as exc:
                # Recoverable I/O failure: record per-file 'error' and continue so
                # the run is observable; a crash (e.g. KeyboardInterrupt) is NOT
                # caught here and aborts execute mid-run (the resume seam).
                self.catalog.consolidation_journal_update(
                    row_id, CONSOLIDATION_ERROR, error=str(exc)
                )
                result.errors.append({"asset_id": asset_id, "error": str(exc)})

        result.unique_library_files = self._count_library_files()
        return result

    # -- stage / verify / promote (S01 atomic-write idiom) ---------------------

    def _stage(self, source_sha: str, staged: Path, data: bytes) -> None:
        """Atomically write *data* to the content-addressed staging path."""
        staged.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.staging_root / "tmp" / secrets.token_hex(16)
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        # Atomic move into the staging shard; replaces any stale/identical blob.
        os.replace(tmp, staged)

    def _verify(self, source_sha: str, staged: Path) -> None:
        """Assert the staged file's SHA-256 equals the source SHA-256."""
        actual = sha256_hex(staged.read_bytes())
        if actual != source_sha:
            raise ConsolidationError(
                f"staged SHA-256 mismatch for {staged}:"
                f" expected {source_sha}, got {actual}"
            )

    def _promote(self, source_sha: str, staged: Path) -> None:
        """Atomically promote the staged blob into the canonical library root."""
        final = self._library_path(source_sha)
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged, final)

    # -- archive ---------------------------------------------------------------

    def archive(self, source: Path | str | None = None) -> Path:
        """Move a fully-consolidated source folder intact under ``<root>/archive/``.

        Only allowed once every file in the run's ``consolidation_journal`` has
        reached ``promoted``; otherwise raises :class:`ConsolidationError`. Also
        refuses to overwrite a target that already exists (already archived).
        """
        source = Path(self.source) if source is None else Path(source)
        if not source.is_dir():
            raise ConsolidationError(
                f"consolidate source is not a directory: {source}"
            )
        connector_id = str(source.resolve())
        rows = self.catalog.consolidation_journal_rows(connector_id)
        if not rows:
            raise ConsolidationError(
                f"nothing to archive: no consolidated files for {connector_id}"
            )
        not_done = [r for r in rows if r["status"] != CONSOLIDATION_PROMOTED]
        if not_done:
            pending = ", ".join(sorted(r["asset_id"] for r in not_done))
            raise ConsolidationError(
                "archive blocked: not every file reached 'promoted';"
                f" pending: {pending}"
            )
        target = self.archive_root / source.name
        if target.exists():
            raise ConsolidationError(
                f"source folder already archived: {target}"
            )
        self.archive_root.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        return target


def _iter_consolidated_files(source: Path) -> list[Path]:
    """Return every source file the executor consolidates, sorted by path.

    The consolidated surface is the union of supported-suffix media (inline image
    formats, incl. bytes that fail to decode -> corrupt) and non-media sidecar
    companions that must move with their media. Unknown/RAW suffixes are outside
    the S03 inventory surface and are left untouched.
    """
    result: list[Path] = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in SIDECAR_SUFFIXES or is_supported_suffix(suffix):
            result.append(path)
    return result
