"""Orchestration layer that turns a SourceConnector's enumerated assets into
clustered, cataloged entries (S02 boundary map 'produces').

:class:`IngestPipeline` is the S02 core deliverable. It walks
:meth:`connector.enumerate`, reads + decodes each asset, classifies it
(exact/near content via the pure phash clusterer, or an explicit unsupported /
corrupt / error failure), writes catalog entries plus per-content-hash image
signatures through :class:`~curator.catalog.Catalog`, and records every per-file
transition in the ``ingest_journal``:

    ``started`` -> ``indexed`` | ``unsupported`` | ``corrupt`` | ``error``

The journal is the resumable checkpoint: on a subsequent ``resume=True`` run a
previously ``indexed`` asset is not re-read or re-decoded from the source; its
stored signature and content bytes are reused, while its catalog entry is
idempotently re-asserted. This also keeps re-ingest idempotent (same
``(connector_id, asset_id, revision)`` upserts one row) so S05's acceptance
(SELECT count of rows stable across runs) holds.
"""

from __future__ import annotations

from pathlib import Path

from curator.catalog import Catalog
from curator.connectors.base import SourceConnector
from curator.connectors.local import UNSUPPORTED_EXTRA_KEY
from curator.errors import ConnectorError, IngestError, StorageError
from curator.ingest.clustering import (
    ImageItem,
    cluster_images,
    hamming_distance,
)
from curator.ingest.decode import DecodedImage, DecodeError, decode_image
from curator.ingest.report import IngestReport, ReportEntry, ReportIssue

_TIMESTAMP = "strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"

# Journal status vocabulary (started -> indexed/unsupported/corrupt/error).
_JOURNAL_STARTED = "started"
_JOURNAL_INDEXED = "indexed"
_JOURNAL_UNSUPPORTED = "unsupported"
_JOURNAL_CORRUPT = "corrupt"
_JOURNAL_ERROR = "error"


class IngestPipeline:
    """Ingests one connector's assets into the catalog and reports the result."""

    def __init__(
        self,
        connector: SourceConnector,
        catalog: Catalog | None = None,
        data_root: Path | None = None,
    ) -> None:
        self.connector = connector
        self.catalog = catalog if catalog is not None else Catalog(data_root=data_root)

    # -- public API -----------------------------------------------------------

    def run(self, resume: bool = False) -> IngestReport:
        """Ingest all enumerated assets, cluster, catalog, and report.

        With *resume=True*, assets already recorded as ``indexed`` in the journal
        are not re-read/decoded from the source — their stored content and image
        signature are reused (pipeline can continue an interrupted run cheaply).
        Returns a fully JSON-serializable :class:`IngestReport`.
        """
        report = IngestReport(connector_id=self.connector.connector_id)
        # asset_id -> sha256 for assets checkpointed as successfully indexed.
        checkpointed = self._indexed_checkpoints() if resume else {}

        items: list[ImageItem] = []
        bytes_map: dict[str, bytes] = {}
        sig_map: dict[str, DecodedImage] = {}

        for meta in self.connector.enumerate():
            report.total_enumerated += 1
            row_id = self._journal_start(meta)
            asset = meta.asset_id

            if meta.extra.get(UNSUPPORTED_EXTRA_KEY):
                # Recognized RAW: explicit-unsupported surface, never dropped (R003).
                report.failures.append(
                    ReportIssue(
                        status="unsupported",
                        asset_id=asset,
                        connector_id=self.connector.connector_id,
                        media_type=meta.media_type,
                    )
                )
                report.unsupported_count += 1
                self._journal_finish(row_id, _JOURNAL_UNSUPPORTED)
                continue

            try:
                data, sig = self._obtain(meta, checkpointed)
            except DecodeError as exc:
                report.failures.append(
                    ReportIssue(
                        status="corrupt",
                        asset_id=asset,
                        connector_id=self.connector.connector_id,
                        media_type=meta.media_type,
                        error=str(exc),
                    )
                )
                report.corrupt_count += 1
                self._journal_finish(row_id, _JOURNAL_CORRUPT, error=str(exc))
                continue
            except (ConnectorError, StorageError) as exc:
                report.failures.append(
                    ReportIssue(
                        status="error",
                        asset_id=asset,
                        connector_id=self.connector.connector_id,
                        media_type=meta.media_type,
                        error=str(exc),
                    )
                )
                report.error_count += 1
                self._journal_finish(row_id, _JOURNAL_ERROR, error=str(exc))
                continue

            bytes_map[asset] = data
            sig_map[asset] = sig
            items.append(
                ImageItem(
                    key=asset,
                    sha256=sig.sha256,
                    phash=sig.phash,
                    width=sig.width,
                    height=sig.height,
                    metadata={
                        "connector_id": self.connector.connector_id,
                        "media_type": meta.media_type,
                    },
                )
            )
            report.indexed_count += 1
            self._journal_finish(row_id, _JOURNAL_INDEXED, sha=sig.sha256)

        self._cluster_and_catalog(report, items, bytes_map, sig_map)
        return report

    # -- impl: acquisition ----------------------------------------------------

    def _obtain(
        self, meta, checkpointed: dict[str, str]
    ) -> tuple[bytes, DecodedImage]:
        """Return ``(bytes, signature)`` for *meta*.

        With *resume=True* and a matching ``indexed`` checkpoint, reuse the stored
        content bytes + persisted image signature instead of re-reading/decoding
        the source.
        """
        sha = checkpointed.get(meta.asset_id)
        if sha is not None:
            data = self.catalog.content.get(sha)
            sig_row = self.catalog.get_image_signature(sha)
            if sig_row is not None:
                sig = DecodedImage(
                    sha, sig_row["width"], sig_row["height"], sig_row["phash"]
                )
                return data, sig
            # Signature row missing (legacy) — fall through to a fresh decode.
        data = self.connector.read_original(meta.asset_id)
        return data, decode_image(data)

    def _indexed_checkpoints(self) -> dict[str, str]:
        """Return ``{asset_id: sha256}`` for assets last recorded as ``indexed``."""
        cur = self.catalog.db.execute(
            "SELECT asset_id, sha256 FROM ingest_journal"
            " WHERE connector_id = ? AND status = ? ORDER BY id",
            (self.connector.connector_id, _JOURNAL_INDEXED),
        )
        result: dict[str, str] = {}
        for asset_id, sha in cur.fetchall():
            if sha:
                result[asset_id] = sha  # later rows (newer) win
        return result

    # -- impl: clustering + catalog write -------------------------------------

    def _cluster_and_catalog(
        self,
        report: IngestReport,
        items: list[ImageItem],
        bytes_map: dict[str, bytes],
        sig_map: dict[str, DecodedImage],
    ) -> None:
        clusters = cluster_images(items)
        report.unique_clusters = len(clusters)

        for cluster in clusters:
            members = cluster.members
            distinct_sha = {m.sha256 for m in members}
            if len(members) > 1:
                if len(distinct_sha) == 1:
                    report.exact_clusters += 1
                else:
                    report.near_clusters += 1
            best = cluster.best
            best_phash = best.phash

            for member in members:
                is_best = member.key == cluster.best_key
                dupe_of = None if is_best else best.key
                sig = sig_map[member.key]
                distance = _phash_distance(member.phash, best_phash)
                self.catalog.add_source(
                    self.connector.connector_id,
                    member.key,
                    data=bytes_map[member.key],
                    metadata={
                        "connector_type": "local",
                        "cluster_id": cluster.cluster_id,
                        "dupe_of": dupe_of,
                        "best_original": is_best,
                        "quality_flags": {
                            "highest_res": is_best,
                            "crop_candidate": not is_best,
                            "phash_distance": distance,
                        },
                    },
                )
                self.catalog.set_image_signature(
                    sig.sha256, sig.width, sig.height, sig.phash
                )
                report.entries.append(
                    ReportEntry(
                        asset_id=member.key,
                        connector_id=self.connector.connector_id,
                        sha256=sig.sha256,
                        cluster_id=cluster.cluster_id,
                        best_original=is_best,
                        dupe_of=dupe_of,
                        width=sig.width,
                        height=sig.height,
                        phash=sig.phash,
                        phash_distance=distance,
                    )
                )

    # -- impl: journal --------------------------------------------------------

    def _journal_start(self, meta) -> int:
        """Upsert the connector row (journal FK) and record a ``started`` entry."""
        db = self.catalog.db
        db.execute(
            "INSERT OR IGNORE INTO source_connectors(connector_id, connector_type)"
            " VALUES (?, ?)",
            (meta.connector_id, "local"),
        )
        cur = db.execute(
            "INSERT INTO ingest_journal(connector_id, asset_id, status)"
            " VALUES (?, ?, ?)",
            (meta.connector_id, meta.asset_id, _JOURNAL_STARTED),
        )
        db.commit()
        row_id = cur.lastrowid
        if row_id is None:
            raise IngestError("failed to obtain ingest_journal row id")
        return int(row_id)

    def _journal_finish(
        self,
        row_id: int,
        status: str,
        sha: str | None = None,
        error: str | None = None,
    ) -> None:
        """Finalize a journal row with its terminal status."""
        self.catalog.db.execute(
            f"UPDATE ingest_journal SET sha256 = ?, status = ?, error = ?,"
            f" finished_at = {_TIMESTAMP} WHERE id = ?",
            (sha, status, error, row_id),
        )
        self.catalog.db.commit()


def _phash_distance(phash_a: str | None, phash_b: str | None) -> int | None:
    """Hamming distance between two hashes, or ``None`` when either is absent."""
    if phash_a is None or phash_b is None:
        return None
    return hamming_distance(phash_a, phash_b)
