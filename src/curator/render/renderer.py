"""Deterministic frame renderer (M003/S01, T1+T2).

:class:`DeterministicRenderer` renders an
:class:`~curator.artdirection.manifest.ArtDirectionManifest` into a
target-sized (e.g. 1920x1080 / 3840x2160) sRGB image. Sources are provided as a
``sha256 -> bytes`` map; output is encoded losslessly as **PNG** so the
:class:`RenderResult.sha256` is a byte-deterministic content hash: rendering the
same manifest/sources/target twice produces identical bytes and hash.

Treatments implemented here:

- ``SINGLE_FULLBLEED`` — center-crop the source to the target aspect ratio,
  then scale to fill the exact target, no letterbox.
- ``CONTAIN_MATTE`` — fit the source within the target preserving aspect and
  letterbox with the manifest background, with an optional matte border.
- ``PANORAMIC`` — fit a wide source, centered, with neutral sides.
- ``SQUARE`` — center a square canvas (side = min dimension) with balanced matte.
- ``DIPTYCH`` — compose the first two sources side-by-side for landscape/wide
  targets or stacked for portrait targets, each letterbox-fit within its half
  separated by a thin deterministic gap.

Upscaling a source region relative to the target is never silent: it is blocked
(``RenderError``) unless ``processing_intent.upscale_warning`` explicitly
approves it, in which case the result records ``upscaled_warning=True``.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from types import UnionType
from typing import Any, Union, cast, get_args, get_origin, get_type_hints

from PIL import Image, ImageDraw, ImageOps
from PIL.ImageColor import getrgb

from curator.artdirection.manifest import (
    ArtDirectionManifest,
    BackgroundSpec,
    LayoutTreatment,
)
from curator.errors import CuratorError
from curator.hashing import sha256_hex

#: Version of the renderer implementation; recorded on every RenderResult.
RENDERER_VERSION = "renderer-1.0.0"

# Neutral letterbox default when the manifest supplies no background color.
_DEFAULT_BACKGROUND = (0, 0, 0)


class RenderError(CuratorError):
    """Raised when a manifest or set of sources cannot be rendered."""


@dataclass(frozen=True)
class RenderResult:
    """JSON-serializable, byte-deterministic result of one render (T1/T2)."""

    target_width: int
    target_height: int
    treatment: str  # LayoutTreatment value, e.g. "single_fullbleed"
    renderer_version: str
    sha256: str
    size_bytes: int
    sources: list[str] = field(default_factory=list)
    color_mode: str = "RGB"
    color_profile: str = "sRGB"
    upscaled_warning: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-encodable dict (nested dataclasses expanded)."""
        return {f.name: _to_plain(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RenderResult:
        """Build a :class:`RenderResult` from a dict, coercing an enum value."""
        if isinstance(data, cls):
            return data
        return _build(cls, data)


class DeterministicRenderer:
    """Stateless renderer: deterministic given manifest + sources + target."""

    def render_bytes(
        self,
        manifest: ArtDirectionManifest,
        sources: dict[str, bytes],
        target: tuple[int, int],
    ) -> bytes:
        """Encode the rendered image as lossless PNG bytes.

        Byte-deterministic: identical manifest/sources/target produce identical
        bytes. Callers that need the exact payload (e.g. storage) use this.
        """
        return self._render(manifest, sources, target)[0]

    def render(
        self,
        manifest: ArtDirectionManifest,
        sources: dict[str, bytes],
        target: tuple[int, int],
    ) -> RenderResult:
        """Render *manifest* using *sources* to *target* (width, height).

        Every source sha referenced by the manifest must be present in
        *sources*. Output is a lossless PNG whose SHA-256 is returned on the
        :class:`RenderResult`, guaranteeing byte determinism.
        """
        payload, upscaled = self._render(manifest, sources, target)
        return self._result(
            manifest, payload, upscaled, target, self._source_order(manifest)
        )

    def _render(
        self,
        manifest: ArtDirectionManifest,
        sources: dict[str, bytes],
        target: tuple[int, int],
    ) -> tuple[bytes, bool]:
        manifest.validate()
        target_width, target_height = target
        if target_width <= 0 or target_height <= 0:
            raise RenderError(
                f"target dimensions must be positive, got {target!r}"
            )

        if not manifest.sources:
            raise RenderError("manifest requires at least one source sha")

        bg = _background_color(manifest.background)

        if manifest.layout_treatment is LayoutTreatment.SINGLE_FULLBLEED:
            primary = manifest.sources[0]
            if primary not in sources:
                raise RenderError(f"missing source bytes for {primary!r}")
            img, sw, sh = _open_rgb(sources[primary])
            art = _center_crop_fill(img, target_width, target_height)
            offset = (0, 0)
            upscaled = max(target_width / sw, target_height / sh) > 1.0
            border = None
        elif manifest.layout_treatment is LayoutTreatment.CONTAIN_MATTE:
            primary = manifest.sources[0]
            if primary not in sources:
                raise RenderError(f"missing source bytes for {primary!r}")
            img, sw, sh = _open_rgb(sources[primary])
            scale = min(target_width / sw, target_height / sh)
            upscaled = scale > 1.0
            art = _fit_resize(img, target_width, target_height)
            offset = _centered_offset(art, target_width, target_height)
            border = manifest.background
        elif manifest.layout_treatment is LayoutTreatment.PANORAMIC:
            primary = manifest.sources[0]
            if primary not in sources:
                raise RenderError(f"missing source bytes for {primary!r}")
            img, sw, sh = _open_rgb(sources[primary])
            scale = min(target_width / sw, target_height / sh)
            upscaled = scale > 1.0
            art = _fit_resize(img, target_width, target_height)
            offset = _centered_offset(art, target_width, target_height)
            border = None
        elif manifest.layout_treatment is LayoutTreatment.SQUARE:
            primary = manifest.sources[0]
            if primary not in sources:
                raise RenderError(f"missing source bytes for {primary!r}")
            img, sw, sh = _open_rgb(sources[primary])
            side = min(target_width, target_height)
            scale = side / min(sw, sh)
            upscaled = scale > 1.0
            art = _fit_resize(img, side, side)
            offset = (
                (target_width - art.width) // 2,
                (target_height - art.height) // 2,
            )
            border = None
        elif manifest.layout_treatment is LayoutTreatment.DIPTYCH:
            _require_two_sources(manifest, sources)
            art, upscaled = _diptych(
                sources, manifest, (target_width, target_height)
            )
            offset = (0, 0)
            border = None
        else:
            raise RenderError(
                f"treatment not implemented: {manifest.layout_treatment.value!r}"
            )

        if upscaled and not manifest.processing_intent.upscale_warning:
            raise RenderError(
                f"refusing to upscale source(s) into target "
                f"{(target_width, target_height)} without "
                "processing_intent.upscale_warning approval (R008)"
            )

        canvas = Image.new("RGB", (target_width, target_height), bg)
        canvas.paste(art, offset)
        if border is not None and border.width:
            _draw_border(canvas, border, bg)

        payload = _encode_png(canvas)
        return payload, upscaled

    def _source_order(self, manifest: ArtDirectionManifest) -> list[str]:
        """Return the source shas in the order they appear in the render.

        Diptych resolves to its pairing order (falling back to source order);
        other treatments use the manifest source order.
        """
        if manifest.layout_treatment is LayoutTreatment.DIPTYCH:
            order = manifest.pairing_order or list(manifest.sources)
            return order[:2]
        return list(manifest.sources)

    def _result(
        self,
        manifest: ArtDirectionManifest,
        payload: bytes,
        upscaled: bool,
        target: tuple[int, int],
        source_shas: list[str],
    ) -> RenderResult:
        return RenderResult(
            target_width=target[0],
            target_height=target[1],
            treatment=manifest.layout_treatment.value,
            renderer_version=RENDERER_VERSION,
            sha256=sha256_hex(payload),
            size_bytes=len(payload),
            sources=source_shas,
            color_mode="RGB",
            color_profile="sRGB",
            upscaled_warning=upscaled,
            notes=[],
        )


# ---------------------------------------------------------------------------
# rendering helpers
# ---------------------------------------------------------------------------


def _open_rgb(data: bytes) -> tuple[Image.Image, int, int]:
    """Decode *data* into an orientation-corrected RGB image + its size."""
    try:
        with Image.open(io.BytesIO(data)) as img:
            oriented = ImageOps.exif_transpose(img)
            rgb = oriented.convert("RGB")
    except Exception as exc:  # noqa: BLE001 - PIL raises varied types
        raise RenderError(
            f"failed to decode source ({type(exc).__name__}): {exc}"
        ) from exc
    return rgb, rgb.width, rgb.height


def _center_crop_fill(img: Image.Image, tw: int, th: int) -> Image.Image:
    """Center-crop *img* to *tw* x *th*, scaling to fill then cropping."""
    sw, sh = img.size
    scale = max(tw / sw, th / sh)
    nw, nh = max(1, round(sw * scale)), max(1, round(sh * scale))
    resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
    left = (nw - tw) // 2
    top = (nh - th) // 2
    return resized.crop((left, top, left + tw, top + th))


def _fit_resize(img: Image.Image, box_w: int, box_h: int) -> Image.Image:
    """Resize *img* to fit within *box_w* x *box_h* preserving aspect."""
    sw, sh = img.size
    scale = min(box_w / sw, box_h / sh)
    nw, nh = max(1, round(sw * scale)), max(1, round(sh * scale))
    return img.resize((nw, nh), Image.Resampling.LANCZOS)


def _centered_offset(art: Image.Image, tw: int, th: int) -> tuple[int, int]:
    """Return the (left, top) offset to center *art* on a tw x th canvas."""
    return ((tw - art.width) // 2, (th - art.height) // 2)


def _require_two_sources(
    manifest: ArtDirectionManifest, sources: dict[str, bytes]
) -> None:
    """Ensure the diptych has two present sources (pairing or source order)."""
    order = manifest.pairing_order or list(manifest.sources)
    pairs = order[:2]
    if len(pairs) < 2:
        raise RenderError("diptych requires at least two sources")
    for sha in pairs:
        if sha not in sources:
            raise RenderError(f"missing source bytes for {sha!r}")


def _diptych(
    sources: dict[str, bytes],
    manifest: ArtDirectionManifest,
    target: tuple[int, int],
) -> tuple[Image.Image, bool]:
    """Compose a diptych canvas from the first two sources.

    Side-by-side when ``target_width >= target_height``, stacked otherwise.
    Each panel is letterbox-fit within its half and centered, separated by a
    thin gap; returns the composed canvas and whether any panel is upscaled.
    """
    tw, th = target
    order = manifest.pairing_order or list(manifest.sources)
    a_img, _, _ = _open_rgb(sources[order[0]])
    b_img, _, _ = _open_rgb(sources[order[1]])

    bg = _background_color(manifest.background)
    canvas = Image.new("RGB", target, bg)

    gap = max(1, min(tw, th) // 32)
    if tw >= th:
        panel_w = max(1, (tw - gap) // 2)
        box_a = (0, 0, panel_w, th)
        box_b = (tw - panel_w, 0, panel_w, th)
    else:
        panel_h = max(1, (th - gap) // 2)
        box_a = (0, 0, tw, panel_h)
        box_b = (0, th - panel_h, tw, panel_h)

    up_a = _paste_panel(canvas, a_img, box_a)
    up_b = _paste_panel(canvas, b_img, box_b)
    return canvas, up_a or up_b


def _paste_panel(
    canvas: Image.Image,
    img: Image.Image,
    box: tuple[int, int, int, int],
) -> bool:
    """Letterbox-fit *img* into *box* on *canvas*; return whether upscaled."""
    bx, by, bw, bh = box
    sw, sh = img.size
    scale = min(bw / sw, bh / sh)
    art = _fit_resize(img, bw, bh)
    ox = bx + (bw - art.width) // 2
    oy = by + (bh - art.height) // 2
    canvas.paste(art, (ox, oy))
    return scale > 1.0


def _background_color(bg: BackgroundSpec) -> tuple[int, int, int]:
    """Resolve the letterbox background RGB, defaulting to neutral black."""
    if bg.color:
        try:
            return getrgb(bg.color)[:3]
        except ValueError:
            return _DEFAULT_BACKGROUND
    return _DEFAULT_BACKGROUND


def _draw_border(
    canvas: Image.Image,
    bg: BackgroundSpec,
    color: tuple[int, int, int],
) -> None:
    """Stamp a solid matte border of ``bg.width`` px inside the canvas."""
    width = bg.width if bg.width else 1
    draw = ImageDraw.Draw(canvas)
    w, h = canvas.size
    draw.rectangle(
        [(0, 0), (w - 1, h - 1)],
        outline=color,
        width=width,
    )


def _encode_png(img: Image.Image) -> bytes:
    """Encode *img* losslessly as PNG bytes (byte-deterministic)."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# generic (de)serialization helpers
# ---------------------------------------------------------------------------


def _to_plain(value: Any) -> Any:
    """Recursively convert a value into JSON-encodable primitives."""
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {f.name: _to_plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    return value


def _build(cls: Any, data: dict[str, Any]) -> Any:
    """Construct *cls* from *data*, coercing enum fields by type hint."""
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name in data:
            kwargs[f.name] = _convert(data[f.name], hints.get(f.name, Any))
    return cls(**kwargs)


def _convert(value: Any, hint: Any) -> Any:
    if value is None:
        return None
    if isinstance(hint, type) and is_dataclass(hint):
        return cast(Any, hint).from_dict(value)
    if isinstance(hint, type) and issubclass(hint, Enum):
        if isinstance(value, hint):
            return value
        try:
            return hint(value)
        except (ValueError, KeyError):
            raise RenderError(
                f"invalid {hint.__name__} value: {value!r}"
            ) from None
    origin = get_origin(hint)
    if origin is None:
        return value
    args = get_args(hint)
    if origin in (Union, UnionType):
        for arg in args:
            try:
                return _convert(value, arg)
            except RenderError:
                continue
        return value
    if origin in (list, set):
        elem = args[0] if args else Any
        if not isinstance(value, list):
            return value
        converted = [_convert(item, elem) for item in value]
        return set(converted) if origin is set else converted
    return value
