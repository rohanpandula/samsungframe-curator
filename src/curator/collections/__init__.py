"""Collections & rotation subsystem (M005/S04).

Exposes deterministic playlist rotation: the JSON-serializable
:class:`~curator.collections.rotation.Playlist` /
:class:`~curator.collections.rotation.ScheduleWindow` /
:class:`~curator.collections.rotation.RotationState` /
:class:`~curator.collections.rotation.RotationStep` data model, the stateless
:class:`~curator.collections.rotation.RotationEngine` decision-maker, and the
:class:`~curator.collections.rotation.RotationStore` persistence layer over the
schema v10 ``playlists`` / ``playlist_members`` / ``rotation_state`` tables.
"""

from curator.collections.rotation import (
    Playlist,
    RotationEngine,
    RotationState,
    RotationStep,
    RotationStore,
    ScheduleWindow,
)

__all__ = [
    "Playlist",
    "RotationEngine",
    "RotationState",
    "RotationStep",
    "RotationStore",
    "ScheduleWindow",
]
