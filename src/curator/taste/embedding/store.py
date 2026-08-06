"""Content-scoped, model-versioned embedding vector storage (M009/S02).

:class:`EmbeddingStore` mirrors :meth:`~curator.catalog.Catalog.set_image_signature`/
:meth:`~curator.catalog.Catalog.get_image_signature`'s upsert/read idiom over the
new ``content_embeddings`` table (schema v17) — the first BLOB column in this
schema. Vectors are keyed by ``(sha256, model_version)`` so a model upgrade
appends a new row rather than silently overwriting or colliding with the old
one; :func:`cosine_similarity` refuses to compare vectors from different
``model_version`` values rather than returning a plausible-looking but
meaningless float (T-09-07).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np

from curator.catalog import Catalog
from curator.taste.embedding.errors import EmbeddingError, EmbeddingVersionError
from curator.taste.embedding.provider import EMBEDDING_DIM

# ISO-8601 UTC timestamp used for the column this module writes explicitly
# (mirrors ``curator.catalog``'s ``_TIMESTAMP`` upsert idiom).
_TIMESTAMP = "strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"


@dataclass(frozen=True)
class StoredEmbedding:
    """One content-scoped, model-versioned embedding vector.

    Not intended for ``==``/hashing — carries a numpy array (unhashable, and
    ``==`` on two arrays returns an elementwise array, not a bool). Tests compare
    ``.vector`` via ``np.allclose``/``np.array_equal`` directly, never dataclass
    equality.
    """

    sha256: str
    model_version: str
    dim: int
    vector: np.ndarray
    created_at: str


class EmbeddingStore:
    """Persist + read :class:`StoredEmbedding` rows in ``content_embeddings``.

    Takes a :class:`~curator.catalog.Catalog` (reusing its shared ``.db``) or a
    raw ``sqlite3.Connection``, mirroring
    :class:`~curator.taste.store.TasteVoteStore`'s coercion idiom.
    """

    def __init__(self, db: sqlite3.Connection | Catalog) -> None:
        self.db = db.db if isinstance(db, Catalog) else db

    def set(self, sha256: str, model_version: str, vector: np.ndarray) -> None:
        """Upsert *vector* for *(sha256, model_version)*.

        Re-computing an embedding for the same content under the same model
        version overwrites in place (idempotent) — mirrors
        :meth:`~curator.catalog.Catalog.set_image_signature`'s
        ``ON CONFLICT DO UPDATE`` upsert idiom. ``created_at`` is computed once
        (the ``strftime(...)`` in ``VALUES``) and reused via ``excluded.created_at``
        on conflict, so both branches of one call see the same timestamp.
        """
        blob = vector.astype(np.float32).tobytes()
        try:
            self.db.execute(
                "INSERT INTO content_embeddings(sha256, model_version, dim, vector, created_at)"
                f" VALUES (?, ?, ?, ?, {_TIMESTAMP})"
                " ON CONFLICT(sha256, model_version) DO UPDATE SET"
                "   vector = excluded.vector,"
                "   dim = excluded.dim,"
                "   created_at = excluded.created_at",
                (sha256, model_version, int(vector.shape[0]), blob),
            )
            self.db.commit()
        except sqlite3.Error as exc:
            self.db.rollback()
            raise EmbeddingError(
                f"failed to store embedding for sha256={sha256!r}"
                f" model_version={model_version!r}: {exc}"
            ) from exc

    def get(self, sha256: str, model_version: str) -> StoredEmbedding | None:
        """Return the stored embedding for *(sha256, model_version)*, or ``None``."""
        row = self.db.execute(
            "SELECT dim, vector, created_at FROM content_embeddings"
            " WHERE sha256 = ? AND model_version = ?",
            (sha256, model_version),
        ).fetchone()
        if row is None:
            return None
        dim, blob, created_at = row
        vector = np.frombuffer(blob, dtype=np.float32).copy()
        return StoredEmbedding(
            sha256=sha256,
            model_version=model_version,
            dim=int(dim),
            vector=vector,
            created_at=str(created_at),
        )

    def get_matrix(self, model_version: str) -> tuple[list[str], np.ndarray]:
        """Return ``(shas, matrix)`` for every vector stored under *model_version*.

        ``matrix.shape == (len(shas), EMBEDDING_DIM)``, ordered by ``sha256`` —
        the bulk read S03 (head fitting) and S04 (exemplar lookup) both need.
        Brute-force numpy over this matrix is the retrieval strategy at this
        library's scale (research measured 9ms for a top-20 query over 100k
        vectors — no vector index needed).
        """
        rows = self.db.execute(
            "SELECT sha256, vector FROM content_embeddings"
            " WHERE model_version = ? ORDER BY sha256",
            (model_version,),
        ).fetchall()
        if not rows:
            return [], np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        shas = [str(sha) for sha, _ in rows]
        matrix = np.stack([np.frombuffer(blob, dtype=np.float32) for _, blob in rows])
        return shas, matrix


def cosine_similarity(a: StoredEmbedding, b: StoredEmbedding) -> float:
    """Return the cosine similarity between two embeddings.

    Raises :class:`EmbeddingVersionError` when *a* and *b* come from different
    ``model_version`` values — a mismatched pair never silently returns a
    plausible-but-meaningless float (T-09-07).
    """
    if a.model_version != b.model_version:
        raise EmbeddingVersionError(
            f"cannot compare embeddings from different model versions: "
            f"{a.model_version!r} vs {b.model_version!r}"
        )
    return float(
        np.dot(a.vector, b.vector) / (np.linalg.norm(a.vector) * np.linalg.norm(b.vector))
    )
