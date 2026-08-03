"""Hashing helpers.

Content-addressed hashing is the single convergence point of the catalog: the
same bytes always produce the same SHA-256, across every connector and ingest
path.
"""

from __future__ import annotations

import hashlib


def sha256_hex(data: bytes) -> str:
    """Return the lowercase hex SHA-256 digest of *data*."""
    return hashlib.sha256(data).hexdigest()
