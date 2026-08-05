"""TasteSession — one taste dialogue session (M008/S01/T2).

A :class:`TasteSession` groups the observations of a single dialogue run: an
opaque string ``id`` (the ``taste_sessions`` primary key), the surface
``kind`` that opened it (e.g. ``"reaction-room"`` or ``"cli"``), the images the
session surfaced, and ``started_at``/``closed_at`` timestamps. Sessions are
frozen and round-trip losslessly through JSON via ``to_dict`` / ``from_dict``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from curator.taste.dialogue.observation import ImageRef


@dataclass(frozen=True)
class TasteSession:
    """One dialogue session; ``closed_at`` is ``None`` until :meth:`close`."""

    id: str
    kind: str
    images: list[ImageRef]
    started_at: str
    closed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "images": [img.to_dict() for img in self.images],
            "started_at": self.started_at,
            "closed_at": self.closed_at,
        }

    @classmethod
    def from_dict(cls, data: Any) -> TasteSession:
        if isinstance(data, TasteSession):
            return data
        fields = dict(data)
        return cls(
            id=fields["id"],
            kind=fields["kind"],
            images=[ImageRef.from_dict(img) for img in fields.get("images", ())],
            started_at=fields["started_at"],
            closed_at=fields.get("closed_at"),
        )
