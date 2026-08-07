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
- ``DIPTYCH`` / ``TRIPTYCH`` / ``QUAD`` / ``PACKED`` — every
  :data:`~curator.artdirection.manifest.MULTI_CELL_TREATMENTS` member renders
  through one region-iterating loop (:func:`_multi_cell`, M010/S02): the
  manifest's cells are resolved for the target by
  :func:`~curator.artdirection.packing.resolve_regions` and each source is fit
  into its own cell — letterboxed by default, center-cropped to fill when its
  region opts in (M010/S05) — separated by a thin deterministic gutter. A
  manifest whose source count does not match its named template is **rejected**,
  never truncated; ``PACKED`` (M010/S03) has no fixed count and needed no new
  branch here — only the 2..:data:`~curator.artdirection.manifest.MAX_LAYOUT_SOURCES`
  bound every treatment gets.

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
    _TREATMENT_SOURCE_COUNT,
    CROP_FILL,
    MAX_LAYOUT_SOURCES,
    MULTI_CELL_TREATMENTS,
    VALID_CROP_MODES,
    ArtDirectionManifest,
    BackgroundSpec,
    LayoutTreatment,
)
from curator.artdirection.packing import PackingError, resolve_regions
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
        elif manifest.layout_treatment in MULTI_CELL_TREATMENTS:
            art, upscaled = _multi_cell(
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

        A multi-cell treatment resolves to its pairing order (falling back to
        source order) — **every** source it lays out, not the first two
        (M010/S02): a ``RenderResult`` that under-reported its own sources was
        the reporting half of the same silent-truncation bug ``_multi_cell``
        closes. Other treatments use the manifest source order.
        """
        if manifest.layout_treatment in MULTI_CELL_TREATMENTS:
            return list(manifest.pairing_order or manifest.sources)
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


#: Spelled-out counts, so an under-count error reads as prose (M010/S02).
_COUNT_WORDS = {2: "two", 3: "three", 4: "four"}


def _multi_cell(
    sources: dict[str, bytes],
    manifest: ArtDirectionManifest,
    target: tuple[int, int],
) -> tuple[Image.Image, bool]:
    """Compose any multi-cell treatment as one region-iterating loop (M010/S02).

    Replaces the diptych special case: the cells come from
    :func:`~curator.artdirection.packing.resolve_regions` (stored geometry when
    it exactly tiles this target, freshly packed cells otherwise) and each source
    is letterbox-fit into its own cell by the already-generic
    :func:`_paste_panel`. Because ``equal_cells`` at N=2 reproduces the old
    diptych box math exactly, every pre-existing diptych render stays
    byte-identical.

    **Per-cell fit (M010/S05).** A region's ``crop`` selects how its source meets
    its cell: falsy (``None`` / ``""``) letterboxes, which is the default and the
    only fit any pre-S05 manifest can carry, and
    :data:`~curator.artdirection.manifest.CROP_FILL` fills the cell through
    :func:`_center_crop_fill` — the codebase's single crop, reused verbatim.
    Anything else raises. The renderer holds no ``AnalysisResult`` and therefore
    never decides *whether* a cell may be cropped; it only honors the decision
    ``policy.materialize_manifest`` already gated on that source's own
    ``crop_safety``.

    **Reject, never truncate.** A treatment with an entry in
    :data:`~curator.artdirection.manifest._TREATMENT_SOURCE_COUNT` must be handed
    exactly that many sources; an over-count used to render silently, dropping
    every source past the second. A treatment with **no** entry has a variable
    source count by design (M010/S03's ``PACKED``) and is bounded on both sides
    instead: at least two cells, at most
    :data:`~curator.artdirection.manifest.MAX_LAYOUT_SOURCES`. Returns the
    composed canvas plus ``any(per-cell upscaled)`` — the whole-manifest
    generalization of the old ``up_a or up_b``, which the caller's R008 gate then
    approves or blocks once.
    """
    order = manifest.pairing_order or list(manifest.sources)
    treatment = manifest.layout_treatment
    required = _TREATMENT_SOURCE_COUNT.get(treatment)
    if required is None:
        if not 2 <= len(order) <= MAX_LAYOUT_SOURCES:
            raise RenderError(
                f"{treatment.value} lays out 2 to {MAX_LAYOUT_SOURCES} sources, "
                f"got {len(order)} — an out-of-bounds request is rejected, never "
                f"truncated"
            )
    elif len(order) != required:
        if len(order) < required:
            raise RenderError(
                f"{treatment.value} requires at least "
                f"{_COUNT_WORDS[required]} sources, got {len(order)}"
            )
        raise RenderError(
            f"{treatment.value} requires exactly {required} sources, got "
            f"{len(order)} — an over-count request is rejected, never truncated"
        )
    for sha in order:
        if sha not in sources:
            raise RenderError(f"missing source bytes for {sha!r}")

    try:
        regions = resolve_regions(manifest, target)
    except PackingError as exc:
        raise RenderError(str(exc)) from exc

    canvas = Image.new("RGB", target, _background_color(manifest.background))
    upscaled: list[bool] = []
    for region in regions:
        img, sw, sh = _open_rgb(sources[region.source_sha256])
        cell_x, cell_y = int(region.x), int(region.y)
        cell_w, cell_h = int(region.w), int(region.h)
        mode = region.crop
        if not mode:
            # The default, and every manifest written before M010/S05.
            upscaled.append(_paste_panel(canvas, img, (cell_x, cell_y, cell_w, cell_h)))
        elif mode == CROP_FILL:
            canvas.paste(_center_crop_fill(img, cell_w, cell_h), (cell_x, cell_y))
            # The FILL scale, exactly as `_render` computes it for
            # SINGLE_FULLBLEED — never `_paste_panel`'s min(...) fit scale, which
            # would let a cropped upscale slip past the R008 gate below.
            upscaled.append(max(cell_w / sw, cell_h / sh) > 1.0)
        else:
            # Clear status, never silent: letterboxing an unrecognized directive
            # would discard a caller's stated intent. (tests/test_manifest.py's
            # full_manifest() fixture carries crop="center" purely to exercise the
            # JSON round trip; it is never rendered.)
            raise RenderError(
                f"unknown crop mode {mode!r} on the region for "
                f"{region.source_sha256!r} — accepted values are "
                f"{sorted(VALID_CROP_MODES)} or null (letterbox); an unrecognized "
                f"fit directive is rejected, never silently letterboxed"
            )
    return canvas, any(upscaled)


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
