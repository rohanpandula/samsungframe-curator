"""Render subsystem — deterministic frame output (M003/S01).

Turns an :class:`~curator.artdirection.manifest.ArtDirectionManifest` plus
content-addressed source bytes into a target-sized, sRGB-tagged PNG via
:class:`~curator.render.renderer.DeterministicRenderer`. Rendering is fully
deterministic: the same manifest + sources + target always yield byte-identical
output (:class:`~curator.render.renderer.RenderResult.sha256`).
"""

from __future__ import annotations

from curator.render.renderer import (
    RENDERER_VERSION,
    DeterministicRenderer,
    RenderError,
    RenderResult,
)

__all__ = [
    "RENDERER_VERSION",
    "DeterministicRenderer",
    "RenderError",
    "RenderResult",
]
