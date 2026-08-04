"""Deterministic taste-based reranking over baseline scores (M007/S01).

:class:`TasteRanker` is pure and stateless: given candidates that already carry a
baseline score plus their M002 analysis, it optionally adds a profile's personal
delta (``sum(weight * signal_value)``) and re-sorts deterministically. When the
profile is disabled (``None`` or all-zero weights, see :meth:`is_enabled`) the
returned order is **exactly** the baseline order (isolation — no reordering).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from curator.analysis.schema import AnalysisResult
from curator.taste.profiles import SIGNAL_NAMES, TasteProfile


class TasteRanker:
    """Pure, deterministic reranker over baseline-scored candidates."""

    def signal_values(self, analysis: AnalysisResult) -> dict[str, float]:
        """Extract the M002 signal values weighted by profiles, keyed by name.

        Every canonical :data:`SIGNAL_NAMES` key is present. ``vibrancy`` is a
        derived proxy equal to the color-story colorfulness signal.
        """
        return {
            "aesthetic_quality": analysis.quality.aesthetic_quality,
            "technical_quality": analysis.quality.technical_quality,
            "colorfulness": analysis.color_story.colorfulness,
            "harmony": analysis.color_story.harmony,
            "pairing_affinity": analysis.pairing.affinity,
            "vibrancy": analysis.color_story.colorfulness,
        }

    def personal_delta(
        self, analysis: AnalysisResult, profile: TasteProfile
    ) -> tuple[float, list[dict[str, Any]]]:
        """Return ``(delta, contributions)`` explaining the profile's rerank.

        ``delta = sum(weight * value)`` over every canonical signal; each
        contribution dict records ``{signal, weight, value, contribution}`` so the
        terms are individually explainable and sum to ``delta`` exactly.
        """
        values = self.signal_values(analysis)
        contributions: list[dict[str, Any]] = []
        total = 0.0
        for signal in SIGNAL_NAMES:
            weight = profile.weights.get(signal, 0.0)
            value = values[signal]
            contribution = weight * value
            total += contribution
            contributions.append(
                {
                    "signal": signal,
                    "weight": weight,
                    "value": value,
                    "contribution": contribution,
                }
            )
        return total, contributions

    def is_enabled(self, profile: TasteProfile | None) -> bool:
        """True unless *profile* is ``None`` or carries all-zero weights."""
        if profile is None:
            return False
        return any(weight != 0.0 for weight in profile.weights.values())

    def rank(
        self,
        candidates: Sequence[dict[str, Any]],
        profile: TasteProfile | None = None,
        *,
        analysis_map: dict[Any, AnalysisResult],
    ) -> list[dict[str, Any]]:
        """Return *candidates* ranked by ``baseline ⊕ personal_delta``.

        ``candidates`` are ``{"id": ..., "baseline": <float>}`` dicts;
        ``analysis_map`` maps a candidate id to its :class:`AnalysisResult`. A
        disabled profile returns the input order **exactly** (baseline isolation).
        An enabled profile scores each candidate as ``baseline + delta`` and sorts
        deterministically — descending score, with input order as the stable
        tie-break so identical scores never reorder arbitrarily.
        """
        if not self.is_enabled(profile):
            return list(candidates)
        assert profile is not None
        keyed: list[tuple[float, int, dict[str, Any]]] = []
        for index, cand in enumerate(candidates):
            analysis = analysis_map[cand["id"]]
            delta, _ = self.personal_delta(analysis, profile)
            score = float(cand["baseline"]) + delta
            keyed.append((score, index, cand))
        keyed.sort(key=lambda row: (-row[0], row[1]))
        return [cand for _, _, cand in keyed]
