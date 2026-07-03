"""Data structures for the detection metrics.

``Detection3DSample`` is the per-frame input. ``DetectionState`` is what the
suite hands to each metric: the accumulated samples plus the shared config the
metrics need, with a memoized ``match_curve`` so the expensive center-distance
matching runs once per ``(label, threshold)`` and is shared across every metric.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

ERROR_NAMES = ("ATE", "AOE", "ASE", "AVE", "AAE")


@dataclass(frozen=True)
class Detection3DSample:
    """Prediction and ground-truth tensors for one detection frame."""

    pred_boxes: torch.Tensor
    pred_scores: torch.Tensor
    pred_labels: torch.Tensor
    gt_boxes: torch.Tensor
    gt_labels: torch.Tensor


@dataclass(frozen=True)
class MatchCurve:
    """Per-class, per-threshold score-ordered matching results."""

    total_gt: int
    scores: np.ndarray
    true_positive: np.ndarray
    false_positive: np.ndarray
    heading_score: np.ndarray
    translation_error: np.ndarray
    orientation_error: np.ndarray
    scale_error: np.ndarray
    velocity_error: np.ndarray
    attribute_error: np.ndarray

    @property
    def num_predictions(self) -> int:
        """Return the number of scored predictions in this match curve."""
        return int(self.scores.shape[0])

    @property
    def num_match(self) -> int:
        """Return the number of true-positive matches in this curve."""
        return int(np.sum(self.true_positive))

    @property
    def cumulative_tp(self) -> np.ndarray:
        """Return cumulative true positives in descending score order."""
        return np.cumsum(self.true_positive)

    @property
    def cumulative_fp(self) -> np.ndarray:
        """Return cumulative false positives in descending score order."""
        return np.cumsum(self.false_positive)

    @property
    def cumulative_heading_tp(self) -> np.ndarray:
        """Return cumulative heading-weighted true positives in score order."""
        return np.cumsum(self.heading_score)


@dataclass(frozen=True)
class PredictionRecord:
    """Score-sorted prediction candidate used during center-distance matching."""

    score: float
    sample_index: int
    box: np.ndarray = field(compare=False)


@dataclass(frozen=True)
class CurveMetrics:
    """AP-style summary values derived from one match curve."""

    ap: float
    aph: float
    max_f1: float
    optimal_conf: float
    optimal_index: int
    optimal_recall: float
    optimal_precision: float


@dataclass(frozen=True)
class SelectedTpErrors:
    """True-positive error values selected at one operating point."""

    count: int
    errors: dict[str, float]


@dataclass
class DetectionState:
    """Synced detection state the suite hands to each metric.

    Attributes:
        samples: Per-frame samples (already GT-filtered, optionally range-clipped).
        class_names: Ordered class names for metric keys, or ``None``.
        thresholds: Center-distance match thresholds in meters.
    """

    samples: list[Detection3DSample]
    class_names: tuple[str, ...] | None
    thresholds: tuple[float, ...]
    _curve_cache: dict[tuple[int, float], MatchCurve] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def labels(self, full: bool) -> list[int]:
        """Class labels to report.

        At test (``full``) every configured class is reported; otherwise only
        classes that actually have ground truth, matching the lightweight
        validation report.
        """
        from autoware_ml.metrics.detection3d.matching import labels_to_evaluate

        return labels_to_evaluate(self.samples, self.class_names if full else None)

    def match_curve(self, label: int, threshold: float) -> MatchCurve:
        """Return the score-ordered match curve for a class and threshold.

        Memoized, so the matching runs once per ``(label, threshold)`` and is
        shared by every metric that asks for it.
        """
        key = (label, float(threshold))
        curve = self._curve_cache.get(key)
        if curve is None:
            from autoware_ml.metrics.detection3d.matching import match_center_distance

            curve = match_center_distance(self.samples, label, threshold)
            self._curve_cache[key] = curve
        return curve
