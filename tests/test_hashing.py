"""Tests for src/curator/hashing.py."""

from __future__ import annotations

import hashlib

from curator.hashing import sha256_hex


def test_sha256_hex_matches_stdlib():
    data = b"hello curator"
    assert sha256_hex(data) == hashlib.sha256(data).hexdigest()


def test_sha256_hex_is_deterministic_and_lowercase():
    data = b"some image bytes \x00\x01\xff"
    a = sha256_hex(data)
    b = sha256_hex(data)
    assert a == b
    assert a == a.lower()
    assert len(a) == 64


def test_sha256_hex_distinguishes_content():
    assert sha256_hex(b"a") != sha256_hex(b"b")
    assert sha256_hex(b"") != sha256_hex(b" ")
