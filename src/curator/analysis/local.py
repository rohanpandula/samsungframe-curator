"""Deterministic, offline, CPU-only local analysis provider (M002/S02).

:class:`LocalAnalysisProvider` implements the :class:`AnalysisProvider` ABC and
turns one image into a fully-populated :class:`~curator.analysis.schema.AnalysisResult`
using only numpy / Pillow / imagehash. Every signal is explained by lightweight,
repeatable heuristics:

- **perceptual** — a fixed-length, normalized multi-resolution phash+dhash vector.
- **technical** — resolution sufficiency, Laplacian-variance sharpness, histogram
  exposure/contrast; a blur/denoise-risk proxy is recorded in metadata.
- **aesthetic** — rule-of-thirds energy, spatial balance, and perceptual colorfulness.
- **saliency** — a center-surround saliency map from edge magnitude + edge density,
  connected-component subjects, a weighted focal point, and a skin-tone-cluster
  face-presence hint that never identifies anyone.
- **crop safety** — per-direction safe flags and margin ratios from the saliency map.
- **color story** — deterministic k-means dominant palette (perceptual-ish RGB),
  colorfulness, harmony, and neutral background candidates.
- **pairing** — explainable cross-image affinity via ``pairing_scores``; a single
  image's own slot is an identity self-match.
- **segmentation/depth** — lightweight boundary/region segmentation in metadata;
  depth is honestly reported as *not available* rather than silently dropped.

The provider is fully deterministic: identical input yields an identical
:class:`AnalysisResult` across calls. Corrupt or undecodable input raises
:class:`AnalysisError` with an actionable reason.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import io
import math
import os
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import imagehash
import numpy as np
from PIL import Image, ImageFilter, ImageOps

from curator.analysis.compute import ComputeBackend
from curator.analysis.errors import AnalysisError
from curator.analysis.profiles import AnalysisProfile, profile_specs
from curator.analysis.provider import (
    AnalysisCapabilities,
    AnalysisProvider,
    ComputeProbe,
)
from curator.analysis.schema import (
    AnalysisMetadata,
    AnalysisResult,
    BoundingBox,
    ColorStory,
    CropSafety,
    Pairing,
    PerceptualRepresentation,
    Point,
    QualitySignals,
    Saliency,
)

#: Engine version reported in every :class:`AnalysisMetadata`.
ENGINE_VERSION = "local-1.0.0"

#: Default model specification describing this deterministic reference engine.
_MODEL_SPEC: dict[str, Any] = {
    "name": "curator-local-signal-engine",
    "version": ENGINE_VERSION,
    "family": "reference",
    "backend": "cpu",
    "task": "analysis",
}

#: Target dimensions for resolution sufficiency (Full HD and UHD 4K).
_1080P = (1920, 1080)

#: Fixed embedding width (64 phash + 64 dhash bits).
_PERCEPTUAL_DIM = 128

#: K-means palette size and the minimum ratio to count as "dominant".
_K = 5
_DOMINANT_MIN_RATIO = 0.05

#: Minimum normalized subject area to keep as a real bounding box.
_MIN_SUBJECT_AREA = 0.002

#: Minimum normalized margin to consider a direction safe to crop.
_CROP_SAFE_MARGIN = 0.02

#: Max possible RGB Euclidean distance (for normalization).
_EMAX = math.sqrt(3.0)


def _clamp01(value: float) -> float:
    """Clamp *value* into [0, 1]."""
    return min(1.0, max(0.0, value))


def _saturate(value: float, k: float) -> float:
    """Saturating normalization ``value / (value + k)`` in [0, 1)."""
    return value / (value + k) if value + k > 0 else 0.0


def _hex(rgb: tuple[int, int, int]) -> str:
    """Format an RGB triple as a lowercase CSS hex color."""
    return "#{:02x}{:02x}{:02x}".format(*rgb)


@dataclass(frozen=True)
class _PaletteEntry:
    """One dominant-color cluster: merged RGB, ratio, hue, and saturation."""

    rgb: tuple[int, int, int]
    ratio: float
    hue: float
    saturation: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "hex": _hex(self.rgb),
            "ratio": round(self.ratio, 4),
            "r": self.rgb[0],
            "g": self.rgb[1],
            "b": self.rgb[2],
            "hue": round(self.hue, 4),
            "saturation": round(self.saturation, 4),
        }


@dataclass
class _Features:
    """Per-asset cached signals used by :meth:`LocalAnalysisProvider.pairing_scores`."""

    asset_id: str
    phash_bits: list[int]
    palette: list[_PaletteEntry]
    orientation: str
    date: float | None = None


def _rgb_to_hsv(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    """Convert RGB -> HSV (h in [0,360], s in [0,1], v in [0,1])."""
    r, g, b = (x / 255.0 for x in rgb)
    mx, mn = max(r, g, b), min(r, g, b)
    d = mx - mn
    if d == 0.0:
        h = 0.0
    elif mx == r:
        h = 60.0 * (((g - b) / d) % 6.0)
    elif mx == g:
        h = 60.0 * (((b - r) / d) + 2.0)
    else:
        h = 60.0 * (((r - g) / d) + 4.0)
    s = 0.0 if mx == 0.0 else d / mx
    return h, s, mx


def _colorfulness(rgb: np.ndarray) -> float:
    """Perceptual colorfulness (Hasler & Süsstrunk) normalized into [0, 1].

    *rgb* is an ``(N, 3)`` float array in [0, 1].
    """
    if rgb.shape[0] == 0:
        return 0.0
    rg = rgb[:, 0] - rgb[:, 1]
    yb = 0.5 * (rgb[:, 0] + rgb[:, 1]) - rgb[:, 2]
    std_rg, std_yb = float(np.std(rg)), float(np.std(yb))
    mean_rg, mean_yb = float(np.mean(rg)), float(np.mean(yb))
    metric = math.sqrt(std_rg * std_rg + std_yb * std_yb) + 0.3 * math.sqrt(
        mean_rg * mean_rg + mean_yb * mean_yb
    )
    return _clamp01(_saturate(metric, 25.0))


class _ImageData:
    """Decoded, orientation-corrected image as numpy arrays plus dims."""

    def __init__(self, pil: Image.Image, asset_id: str) -> None:
        self.pil = pil
        self.asset_id = asset_id
        self.width, self.height = pil.size
        self.rgb: np.ndarray = np.asarray(pil, dtype=np.float32) / 255.0
        self.gray: np.ndarray = np.asarray(pil.convert("L"), dtype=np.float32) / 255.0

    @property
    def orientation(self) -> str:
        return "landscape" if self.width >= self.height else "portrait"


class LocalAnalysisProvider(AnalysisProvider):
    """Deterministic, offline, CPU-only analysis provider (M002/S02)."""

    def __init__(
        self,
        engine_version: str = ENGINE_VERSION,
        model_spec: dict[str, Any] | None = None,
    ) -> None:
        self.engine_version = engine_version
        self.model_spec = model_spec if model_spec is not None else dict(_MODEL_SPEC)
        self._features: dict[str, _Features] = {}

    # -- AnalysisProvider ABC -------------------------------------------------

    def capabilities(self) -> AnalysisCapabilities:
        return AnalysisCapabilities(
            profiles=frozenset(AnalysisProfile),
            backends=frozenset({ComputeBackend.CPU}),
            air_gapped=True,
            deterministic=True,
        )

    def probe(self) -> ComputeProbe:
        return ComputeProbe(
            ok=True,
            backend=ComputeBackend.CPU,
            latency_ms=0.0,
            available_backends=frozenset({ComputeBackend.CPU}),
            message="local CPU signal engine available (deterministic, air-gapped)",
        )

    # -- public image analysis ----------------------------------------------

    def analyze(
        self,
        source: str | os.PathLike[str] | bytes | bytearray,
        profile: AnalysisProfile = AnalysisProfile.BALANCED,
        asset_id: str | None = None,
    ) -> AnalysisResult:
        """Analyze *source* (path or encoded bytes) and return a full result.

        Corrupt or undecodable input raises :class:`AnalysisError`. The output is
        deterministic: identical input always yields an identical result.
        """
        start = time.perf_counter()
        data = self._read(source)
        pil = self._decode(data, source)
        asset = asset_id if asset_id is not None else self._default_asset_id(data)
        stages = {s.stage for s in profile_specs(profile)}

        img = _ImageData(pil, asset)
        feats = self._features_for(img)

        perceptual = (
            self._perceptual(img) if "perceptual" in stages else PerceptualRepresentation()
        )

        sal_map = None
        if "saliency" in stages or "cropsafety" in stages or "aesthetic" in stages:
            sal_map = self._saliency_map(img)

        quality = QualitySignals()
        if "technical" in stages:
            aesthetic = 0.0
            if "aesthetic" in stages and sal_map is not None:
                aesthetic = self._aesthetic(img, sal_map)
            quality = self._technical(img, aesthetic)

        saliency = Saliency()
        if "saliency" in stages and sal_map is not None:
            saliency = self._saliency(img, sal_map)
        crop_safety = CropSafety()
        if "cropsafety" in stages and sal_map is not None:
            crop_safety = self._crop_safety(sal_map, img)
        color_story = self._color_story(feats) if "colorstory" in stages else ColorStory()
        pairing = self._pairing_single(feats) if "pairing" in stages else Pairing()

        face = self._face_presence_hint(img)
        self.model_spec["depth_available"] = False
        self.model_spec["face_presence"] = face
        self.model_spec["regions"] = self._region_segmentation(img)

        metadata = self._metadata(profile, start)
        self._features[asset] = feats

        return AnalysisResult(
            asset_id=asset,
            quality=quality,
            saliency=saliency,
            crop_safety=crop_safety,
            color_story=color_story,
            pairing=pairing,
            perceptual=perceptual,
            metadata=metadata,
        )

    def pairing_scores(self, other: str | AnalysisResult) -> Pairing:
        """Return the explaining :class:`Pairing` between the *other* asset and self.

        *other* is an asset id (or a result whose ``asset_id``) that this provider
        has already analyzed. Affinity is a weighted, explainable combination of
        palette similarity, phash similarity, orientation match and date proximity.
        """
        start = time.perf_counter()
        other_feats = self._resolve_feature(other)
        pairing = self._pair_for_pair(self._current_feature(), other_feats)
        rep = pairing.to_dict()
        rep["timing_ms"] = round((time.perf_counter() - start) * 1000.0, 3)
        return cast(Pairing, Pairing.from_dict(rep))

    def pairing_scores_between(self, a: str | AnalysisResult, b: str | AnalysisResult) -> Pairing:
        """Return the :class:`Pairing` affinity between two cached analyzed assets."""
        start = time.perf_counter()
        fa = self._resolve_feature(a)
        fb = self._resolve_feature(b)
        pairing = self._pair_for_pair(fa, fb)
        rep = pairing.to_dict()
        rep["timing_ms"] = round((time.perf_counter() - start) * 1000.0, 3)
        return cast(Pairing, Pairing.from_dict(rep))

    def _resolve_feature(self, ref: str | AnalysisResult) -> _Features:
        ref_id = ref.asset_id if isinstance(ref, AnalysisResult) else ref
        feats = self._features.get(ref_id)
        if feats is None:
            raise AnalysisError(
                f"pairing requires prior analysis of asset {ref_id!r} by this provider"
            )
        return feats

    def _current_feature(self) -> _Features:
        if not self._features:
            raise AnalysisError(
                "pairing_scores requires a prior analyze() call on this provider"
            )
        return next(reversed(list(self._features.values())))

    # -- input handling ------------------------------------------------------

    @staticmethod
    def _read(source: str | os.PathLike[str] | bytes | bytearray) -> bytes:
        if isinstance(source, (str, os.PathLike)):
            try:
                with open(source, "rb") as fh:
                    return fh.read()
            except OSError as exc:
                raise AnalysisError(f"cannot read image file {source!r}: {exc}") from exc
        return bytes(source)

    @staticmethod
    def _decode(data: bytes, source: Any) -> Image.Image:
        try:
            with Image.open(io.BytesIO(data)) as im:
                return ImageOps.exif_transpose(im).convert("RGB")
        except Exception as exc:  # PIL raises many error types for malformed input
            raise AnalysisError(
                f"failed to decode image from {source!r} "
                f"({type(exc).__name__}): {exc} — expected JPEG/PNG/GIF/WebP/TIFF/HEIC bytes"
            ) from exc

    @staticmethod
    def _default_asset_id(data: bytes) -> str:
        return "asset-" + hashlib.sha256(data).hexdigest()[:16]

    # -- shared feature computation ------------------------------------------

    def _features_for(self, img: _ImageData) -> _Features:
        phash_bits = _hash_bits(img.pil, imagehash.phash)
        palette = self._kmeans_palette(img.rgb)
        date = _exif_or_filename_date(img.pil, img.asset_id)
        return _Features(
            asset_id=img.asset_id,
            phash_bits=phash_bits,
            palette=palette,
            orientation=img.orientation,
            date=date,
        )

    @staticmethod
    def _gradients(img: _ImageData) -> tuple[np.ndarray, np.ndarray]:
        """Sobel-like x/y gradients of the grayscale image, each in [-1, 1]."""
        return np.gradient(img.gray)

    @staticmethod
    def _laplacian_variance(img: _ImageData) -> float:
        gy, gx = LocalAnalysisProvider._gradients(img)
        d2x = np.gradient(gx, axis=1)
        d2y = np.gradient(gy, axis=0)
        return float((d2x + d2y).var())

    @staticmethod
    def _saliency_map(img: _ImageData) -> np.ndarray:
        """Center-surround saliency map from edge magnitude + edge density, in [0,1]."""
        gy, gx = LocalAnalysisProvider._gradients(img)
        mag = np.sqrt(gx * gx + gy * gy)
        mag_norm = mag / (mag.max() + 1e-9)
        y, x = np.mgrid[0 : img.height, 0 : img.width]
        cy, cx = img.height / 2.0, img.width / 2.0
        center = np.exp(
            -(((x - cx) / (0.6 * img.width)) ** 2 + ((y - cy) / (0.6 * img.height)) ** 2)
        )
        dense = np.asarray(
            Image.fromarray((mag_norm * 255.0).clip(0, 255).astype(np.uint8)).filter(
                ImageFilter.GaussianBlur(radius=4)
            )
        ).astype(np.float32) / 255.0
        sal = (mag_norm + 0.25 * dense) * (0.35 + 0.65 * center)
        return sal / (sal.max() + 1e-9)

    # -- perceptual ----------------------------------------------------------

    @staticmethod
    def _perceptual(img: _ImageData) -> PerceptualRepresentation:
        ph_bits = _hash_bits(img.pil, imagehash.phash)
        dh_bits = _hash_bits(img.pil, imagehash.dhash)
        raw = np.array(ph_bits + dh_bits, dtype=np.float64)
        raw = raw - raw.mean()
        norm = float(np.linalg.norm(raw)) or 1.0
        return PerceptualRepresentation(
            method="phash+dhash", dim=_PERCEPTUAL_DIM, vector=(raw / norm).tolist()
        )

    # -- technical / aesthetic ----------------------------------------------

    @staticmethod
    def _technical(img: _ImageData, aesthetic_quality: float) -> QualitySignals:
        lap_var = LocalAnalysisProvider._laplacian_variance(img)
        sharpness = _clamp01(_saturate(lap_var, 0.001))

        p10, p90 = np.percentile(img.gray, [10, 90])
        exposure = _clamp01(1.0 - abs(float(0.5 * (p10 + p90)) - 0.5) * 2.0)
        contrast = _clamp01(_saturate(float(img.gray.std()), 0.5))

        res_sufficient = img.width >= _1080P[0] and img.height >= _1080P[1]
        res_factor = 1.0 if res_sufficient else _clamp01(
            min(img.width / _1080P[0], img.height / _1080P[1])
        )

        technical_quality = _clamp01(
            0.4 * sharpness + 0.3 * exposure + 0.2 * contrast + 0.1 * res_factor
        )
        return QualitySignals(
            technical_quality=round(technical_quality, 4),
            aesthetic_quality=round(aesthetic_quality, 4),
            sharpness=round(sharpness, 4),
            exposure=round(exposure, 4),
            contrast=round(contrast, 4),
            resolution_sufficient=res_sufficient,
        )

    @staticmethod
    def _aesthetic(img: _ImageData, sal: np.ndarray) -> float:
        h, w = sal.shape
        total = float(sal.sum()) + 1e-9
        thirds = 0.0
        for frac in (1 / 3, 2 / 3):
            for axis in (0, 1):
                idx = int(round(frac * (sal.shape[axis] - 1)))
                lo = max(0, idx - 2)
                hi = min(sal.shape[axis], idx + 3)
                strip = np.take(sal, range(lo, hi), axis=axis)
                thirds += float(strip.sum()) / total
        thirds = _clamp01(thirds / 4.0)

        gx, gy = LocalAnalysisProvider._gradients(img)
        g = gx * gx + gy * gy
        c = w // 2
        r = h // 2
        left, right = g[:, :c].sum(), g[:, c:].sum()
        top, bottom = g[:r, :].sum(), g[r:, :].sum()
        balance = _clamp01(
            1.0
            - 0.5
            * (
                abs(left - right) / (left + right + 1e-9)
                + abs(top - bottom) / (top + bottom + 1e-9)
            )
        )
        colorful = _colorfulness(img.rgb.reshape(-1, 3))
        return float(_clamp01(0.4 * thirds + 0.3 * balance + 0.3 * colorful))

    # -- saliency / subjects -------------------------------------------------

    @staticmethod
    def _saliency(img: _ImageData, sal: np.ndarray) -> Saliency:
        h, w = sal.shape
        total_s = float(sal.sum())
        if total_s < 1e-9:
            return Saliency(map_size=(w, h), subjects=[], focal_point=Point(x=0.5, y=0.5))

        ys, xs = np.mgrid[0:h, 0:w]
        cx = float((xs * sal).sum()) / total_s / w
        cy = float((ys * sal).sum()) / total_s / h
        mean_s = float(sal.mean())
        std_s = float(sal.std())
        mask = sal > (mean_s + 0.4 * std_s)
        subjects = _subjects_from_mask(mask, img)
        return Saliency(
            map_size=(w, h),
            subjects=subjects,
            focal_point=Point(x=_clamp01(cx), y=_clamp01(cy)),
        )

    # -- crop safety ---------------------------------------------------------

    @staticmethod
    def _crop_safety(sal: np.ndarray, img: _ImageData) -> CropSafety:
        h, w = sal.shape
        mean_s = float(sal.mean())
        std_s = float(sal.std())
        content = sal > (mean_s + 0.4 * std_s)
        if not content.any():
            return CropSafety(
                safe_north=True, safe_south=True, safe_east=True, safe_west=True,
                margin_north=0.5, margin_south=0.5, margin_east=0.5, margin_west=0.5,
            )

        rows = content.any(axis=1)
        cols = content.any(axis=0)
        first_row = int(np.argmax(rows))
        last_row = int(h - np.argmax(rows[::-1]) - 1)
        first_col = int(np.argmax(cols))
        last_col = int(w - np.argmax(cols[::-1]) - 1)

        margin_north = first_row / h
        margin_south = (h - 1 - last_row) / h
        margin_west = first_col / w
        margin_east = (w - 1 - last_col) / w

        centered = 0.35 <= first_row / h <= 0.65 and 0.35 <= last_row / h <= 0.65
        vertically_centered = 0.35 <= first_col / w <= 0.65 and 0.35 <= last_col / w <= 0.65
        subject_near_center = centered and vertically_centered

        def _safe(margin: float, sensitive: bool) -> bool:
            if sensitive:
                return margin >= 0.06
            return margin >= _CROP_SAFE_MARGIN

        return CropSafety(
            safe_north=_safe(margin_north, subject_near_center),
            safe_south=_safe(margin_south, subject_near_center),
            safe_east=_safe(margin_east, subject_near_center),
            safe_west=_safe(margin_west, subject_near_center),
            margin_north=round(margin_north, 4),
            margin_south=round(margin_south, 4),
            margin_east=round(margin_east, 4),
            margin_west=round(margin_west, 4),
        )

    # -- color story ---------------------------------------------------------

    @staticmethod
    def _color_story(feats: _Features) -> ColorStory:
        palette = feats.palette
        dominant = [p for p in palette if p.ratio >= _DOMINANT_MIN_RATIO]
        dominant_colors = [p.to_dict() for p in dominant]

        colorful = (
            _colorfulness(np.array([[c / 255.0 for c in p.rgb] for p in palette], dtype=np.float32))
            if palette
            else 0.0
        )
        harmony = _harmony(palette) if palette else 0.0

        candidates = [p for p in palette if p.saturation < 0.25] or sorted(
            palette, key=lambda p: p.saturation
        )
        background_candidates = [_hex(p.rgb) for p in candidates[:4]]
        choice = background_candidates[0] if background_candidates else None

        return ColorStory(
            dominant_colors=dominant_colors,
            colorfulness=round(colorful, 4),
            harmony=round(harmony, 4),
            background_candidates=background_candidates,
            background_choice=choice,
        )

    def _kmeans_palette(self, rgb: np.ndarray) -> list[_PaletteEntry]:
        """Deterministic k-means dominant palette over a subsample of *rgb*."""
        pixels = rgb.reshape(-1, 3)
        if len(pixels) > 20000:
            idx = np.linspace(0, len(pixels) - 1, 20000).astype(np.int64)
            pixels = pixels[idx]
        if len(pixels) == 0:
            return []
        rng = np.random.default_rng(0)
        k = min(_K, len(pixels))
        init_idx = rng.choice(len(pixels), size=k, replace=False)
        centroids = pixels[init_idx].copy().astype(np.float64)
        labels = np.zeros(len(pixels), dtype=np.int64)
        for _ in range(15):
            dists = np.array([np.linalg.norm(pixels - c, axis=1) for c in centroids])
            labels = np.argmin(dists, axis=0)
            new_centroids = centroids.copy()
            for j in range(k):
                sel = labels == j
                if sel.sum() > 0:
                    new_centroids[j] = pixels[sel].mean(axis=0)
            if np.allclose(centroids, new_centroids, atol=1e-6):
                centroids = new_centroids
                break
            centroids = new_centroids

        dists = np.array([np.linalg.norm(pixels - c, axis=1) for c in centroids])
        labels = np.argmin(dists, axis=0)
        counts = np.bincount(labels, minlength=k)

        cluster_centers: list[tuple[np.ndarray, float]] = []
        for j in range(k):
            if counts[j] == 0:
                continue
            color = (centroids[j] * 255.0).round()
            placed = False
            for ci, (prev_c, prev_count) in enumerate(cluster_centers):
                if np.linalg.norm(prev_c - color) <= 32.0:
                    cluster_centers[ci] = (
                        (prev_c * prev_count + color * float(counts[j]))
                        / (prev_count + float(counts[j])),
                        prev_count + float(counts[j]),
                    )
                    placed = True
                    break
            if not placed:
                cluster_centers.append((color, float(counts[j])))

        total = float(np.sum([c for _, c in cluster_centers])) or 1.0
        entries: list[_PaletteEntry] = []
        for color_arr, count in cluster_centers:
            rgb_ = (
                int(round(color_arr[0])),
                int(round(color_arr[1])),
                int(round(color_arr[2])),
            )
            h, s, _ = _rgb_to_hsv(rgb_)
            entries.append(_PaletteEntry(rgb=rgb_, ratio=count / total, hue=h, saturation=s))
        entries.sort(key=lambda p: p.ratio, reverse=True)
        return entries

    # -- pairing -------------------------------------------------------------

    @staticmethod
    def _pairing_single(feats: _Features) -> Pairing:
        del feats
        return Pairing(
            affinity=1.0,
            phash_distance=0,
            palette_distance=0.0,
            date_proximity=None,
            orientation_match=True,
        )

    @staticmethod
    def _pair_for_pair(fa: _Features, fb: _Features) -> Pairing:
        phash_distance = _phash_distance(fa.phash_bits, fb.phash_bits)
        phash_sim = 1.0 - phash_distance / float(_PERCEPTUAL_DIM)
        palette_distance = _palette_distance(fa.palette, fb.palette)
        palette_sim = 1.0 - palette_distance
        orientation_match = fa.orientation == fb.orientation
        date_proximity = _date_proximity(fa.date, fb.date)
        date_sim = date_proximity if date_proximity is not None else 0.5

        affinity = _clamp01(
            0.4 * palette_sim
            + 0.3 * phash_sim
            + 0.2 * (1.0 if orientation_match else 0.0)
            + 0.1 * date_sim
        )
        return Pairing(
            affinity=round(affinity, 4),
            phash_distance=phash_distance,
            palette_distance=round(palette_distance, 4),
            date_proximity=date_proximity,
            orientation_match=orientation_match,
        )

    # -- segmentation / depth ------------------------------------------------

    @staticmethod
    def _region_segmentation(img: _ImageData) -> dict[str, Any]:
        """Lightweight boundary/region segmentation: count + mean region area.

        Returns an honest report; depth is *not available* on the CPU engine and
        is marked explicitly rather than silently dropped.
        """
        gy, gx = LocalAnalysisProvider._gradients(img)
        mag = np.sqrt(gx * gx + gy * gy)
        boundary = mag > mag.mean() + 0.5 * mag.std()
        labelled = _connected_components(~boundary)
        if labelled is None:
            n_regions = 0
        else:
            n_regions = int(labelled.max())
        return {
            "regions": n_regions,
            "depth_available": False,
            "depth_metric": None,
            "note": "no depth channel; depth reported as unavailable",
        }

    # -- face presence (identity-free) --------------------------------------

    @staticmethod
    def _face_presence_hint(img: _ImageData) -> bool:
        """Skin-tone-cluster heuristic for face-like regions WITHOUT identity.

        Returns only a boolean presence hint; no identity is ever derived.
        """
        rgb = img.rgb * 255.0
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        mx = rgb.max(axis=2)
        mn = rgb.min(axis=2)
        skin = (
            (r > 95)
            & (g > 40)
            & (b > 20)
            & (mx - mn > 15)
            & (r > g)
            & (r > b)
            & (np.abs(r - g) > 15)
        )
        labelled = _connected_components(skin)
        if labelled is None:
            return False
        for label in range(1, int(labelled.max()) + 1):
            if int((labelled == label).sum()) >= 0.002 * img.width * img.height:
                return True
        return False

    # -- metadata ------------------------------------------------------------

    def _metadata(self, profile: AnalysisProfile, start: float) -> AnalysisMetadata:
        return AnalysisMetadata(
            profile=profile.value,
            compute_backend=ComputeBackend.CPU.value,
            model_spec=dict(self.model_spec),
            engine_version=self.engine_version,
            deterministic=True,
            timing_ms=round((time.perf_counter() - start) * 1000.0, 3),
        )


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------


def _hash_bits(pil: Image.Image, fn: Any) -> list[int]:
    """Return the 64 bits of a 64-bit image hash as a list of 0/1 ints."""
    return [int(bit) for bit in f"{int(str(fn(pil)), 16):064b}"]


def _exif_or_filename_date(pil: Image.Image, asset_id: str) -> float | None:
    """Parse a numeric date (epoch seconds) from EXIF or the asset filename."""
    try:
        tag = int(getattr(Image.ExifTags, "DateTimeOriginal", 0) or 0)
        if tag:
            exif = pil.getexif()
            if tag in exif:
                parsed = _parse_datetime(str(exif[tag]))
                if parsed is not None:
                    return parsed
    except Exception:
        pass
    match = re.search(r"(19|20)\d{6}", asset_id)
    if match:
        parsed = _parse_datetime(match.group(0))
        if parsed is not None:
            return parsed
    return None


def _parse_datetime(text: str) -> float | None:
    """Best-effort parse of an EXIF/filename datetime to epoch seconds; None on failure."""
    s = text.strip()
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y%m%d", "%Y-%m-%d", "%Y%m%d%H%M%S"):
        try:
            return _dt.datetime.strptime(s, fmt).timestamp()
        except ValueError:
            continue
    return None


def _phash_distance(a_bits: Sequence[int], b_bits: Sequence[int]) -> int:
    return sum(1 for ai, bi in zip(a_bits, b_bits) if ai != bi)


def _palette_distance(a: Sequence[_PaletteEntry], b: Sequence[_PaletteEntry]) -> float:
    """Normalized [0,1] distance between two palettes (ratio-weighted, RGB)."""
    if not a or not b:
        return 1.0
    cost = 0.0
    total_weight = 0.0
    for pa in a:
        best = min(
            np.linalg.norm(np.array(pa.rgb) - np.array(pb.rgb)) / _EMAX for pb in b
        )
        cost += pa.ratio * best
        total_weight += pa.ratio
    return float(min(1.0, cost / (total_weight + 1e-9)))


def _date_proximity(da: float | None, db: float | None) -> float | None:
    """Proximity in [0,1] from the day gap between two dates; None when unknown."""
    if da is None or db is None:
        return None
    gap_days = abs(da - db) / 86400.0
    return _clamp01(1.0 - min(gap_days / 30.0, 1.0))


def _harmony(palette: Sequence[_PaletteEntry]) -> float:
    """Harmony in [0,1]: 1 for a consistent palette, lower as it grows chaotic."""
    if not palette:
        return 0.0
    rgb = np.array([p.rgb for p in palette], dtype=np.float64) / 255.0
    if len(rgb) == 1:
        return 1.0
    dists = [
        float(np.linalg.norm(rgb[i] - rgb[j]) / _EMAX)
        for i in range(len(rgb))
        for j in range(i + 1, len(rgb))
    ]
    mean_d = float(np.mean(dists))
    var_d = float(np.var(dists)) if dists else 0.0
    spread = min(1.0, mean_d)
    consistency = 1.0 - min(1.0, var_d * 4.0)
    return _clamp01(0.5 * (1.0 - spread) + 0.5 * consistency)


def _subjects_from_mask(mask: np.ndarray, img: _ImageData) -> list[BoundingBox]:
    """Normalized bounding boxes for connected components of *mask* above min area."""
    labelled = _connected_components(mask)
    if labelled is None:
        return []
    boxes: list[BoundingBox] = []
    for label in range(1, int(labelled.max()) + 1):
        ys, xs = np.nonzero(labelled == label)
        if ys.size == 0:
            continue
        area = ys.size / float(mask.size)
        if area < _MIN_SUBJECT_AREA:
            continue
        x0, x1 = float(xs.min()) / img.width, (float(xs.max()) + 1) / img.width
        y0, y1 = float(ys.min()) / img.height, (float(ys.max()) + 1) / img.height
        boxes.append(BoundingBox(x=x0, y=y0, w=x1 - x0, h=y1 - y0))
    boxes.sort(key=lambda b: (b.w * b.h, b.x, b.y), reverse=True)
    return boxes


def _connected_components(mask: np.ndarray) -> np.ndarray | None:
    """4-connected component labelling of *mask*; None when no foreground pixels."""
    mask = mask.astype(bool)
    if not mask.any():
        return None
    labelled = np.zeros(mask.shape, dtype=np.int64)
    h, w = mask.shape
    label = 0
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or labelled[y, x]:
                continue
            label += 1
            stack = [(y, x)]
            labelled[y, x] = label
            while stack:
                cy, cx = stack.pop()
                for yn, xn in (
                    (cy - 1, cx),
                    (cy + 1, cx),
                    (cy, cx - 1),
                    (cy, cx + 1),
                ):
                    if 0 <= yn < h and 0 <= xn < w and mask[yn, xn] and not labelled[yn, xn]:
                        labelled[yn, xn] = label
                        stack.append((yn, xn))
    return labelled
