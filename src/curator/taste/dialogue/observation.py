"""Observation surface for the taste dialogue subsystem (M008/S01/T1).

A :class:`TasteObservation` captures one user statement during a taste
dialogue: the verbatim text, extracted attribute tags, a polarity, a
confidence, and any images the statement referenced. Observations and their
:class:`ImageRef` images are frozen and round-trip losslessly through JSON via
``to_dict`` / ``from_dict``.
"""

from __future__ import annotations

import enum
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from curator.errors import CuratorError


class ObservationError(CuratorError):
    """Raised when a taste observation is malformed or out of range."""


class Polarity(enum.Enum):
    """The user's disposition toward the discussed art."""

    LIKE = "like"
    DISLIKE = "dislike"
    CONFLICTED = "conflicted"


@dataclass(frozen=True)
class ImageRef:
    """A reference to one image a taste observation pointed at.

    ``thumb_path`` is the absolute path to a retained thumbnail when one exists
    (``None`` otherwise); ``ephemeral`` marks thumb-only evidence retention;
    ``catalog_saved`` marks explicit promotion of the full-resolution image into
    the catalog.
    """

    sha256: str
    thumb_path: str | Path | None = None
    ephemeral: bool = False
    catalog_saved: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "sha256": self.sha256,
            "thumb_path": str(self.thumb_path) if self.thumb_path is not None else None,
            "ephemeral": self.ephemeral,
            "catalog_saved": self.catalog_saved,
        }

    @classmethod
    def from_dict(cls, data: Any) -> ImageRef:
        if isinstance(data, ImageRef):
            return data
        fields = dict(data)
        return cls(
            sha256=fields["sha256"],
            thumb_path=fields.get("thumb_path"),
            ephemeral=bool(fields.get("ephemeral", False)),
            catalog_saved=bool(fields.get("catalog_saved", False)),
        )


@dataclass(frozen=True)
class TasteObservation:
    """One user statement recorded during a taste dialogue.

    ``id`` and ``created_at`` are opaque metadata filled by the store layer when
    an observation is persisted.
    """

    session_id: str
    verbatim: str
    attributes: list[str]
    polarity: Polarity
    confidence: float
    images: list[ImageRef]
    id: int | None = None
    created_at: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ObservationError(
                f"confidence {self.confidence!r} out of range [0.0, 1.0]"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "verbatim": self.verbatim,
            "attributes": list(self.attributes),
            "polarity": self.polarity.value,
            "confidence": self.confidence,
            "images": [img.to_dict() for img in self.images],
            "id": self.id,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Any) -> TasteObservation:
        if isinstance(data, TasteObservation):
            return data
        fields = dict(data)
        return cls(
            session_id=fields["session_id"],
            verbatim=fields["verbatim"],
            attributes=list(fields.get("attributes", ())),
            polarity=_coerce_polarity(fields.get("polarity", Polarity.LIKE.value)),
            confidence=float(fields.get("confidence", 0.5)),
            images=[ImageRef.from_dict(img) for img in fields.get("images", ())],
            id=fields.get("id"),
            created_at=fields.get("created_at", ""),
        )


def create_observation(
    session_id: str,
    verbatim: str,
    *,
    attributes: Iterable[str] = (),
    polarity: Polarity = Polarity.LIKE,
    confidence: float = 0.5,
    images: Iterable[ImageRef] = (),
    id: int | None = None,
    created_at: str = "",
) -> TasteObservation:
    """Convenience constructor that accepts iterables for list-typed fields."""
    return TasteObservation(
        session_id=session_id,
        verbatim=verbatim,
        attributes=list(attributes),
        polarity=polarity,
        confidence=confidence,
        images=list(images),
        id=id,
        created_at=created_at,
    )


def _coerce_polarity(value: Any) -> Polarity:
    """Return *value* as a :class:`Polarity`, coercing strings or enum instances."""
    if isinstance(value, Polarity):
        return value
    try:
        return Polarity(value)
    except ValueError as exc:
        raise ObservationError(f"invalid polarity {value!r}") from exc
