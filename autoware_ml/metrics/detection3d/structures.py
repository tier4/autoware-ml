"""Pure data structures for the detection metrics.

``Detection3DSample`` is the per-frame input. ``MatchCurve`` carries one
class's score-ordered matching results and ``CurveMetrics`` its AP-style
summary. The memoizing ``DetectionState`` that orchestrates the matching lives
in :mod:`autoware_ml.metrics.detection3d.matching`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

ERROR_NAMES = ("ATE", "AOE", "ASE", "AVE", "AAE")

# nuScenes center-distance AP thresholds (meters). The AP-family metrics own this
# set. Metrics that need a single operating point share DEFAULT_TP_THRESHOLD (the
# nuScenes dist_th_tp), so the whole report moves together when it changes. Nds
# and TpErrors additionally require tp_threshold to be a configured threshold, so
# the aggregate operating point always appears among the detail keys.
DEFAULT_MATCH_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)
DEFAULT_TP_THRESHOLD = 2.0


@dataclass(frozen=True)
class Detection3DSample:
    """Prediction and ground-truth tensors for one detection frame.

    ``gt_ttc`` / ``pred_ttc`` are the per-box reachability time-to-collision to
    ego in seconds (``inf`` means unreachable), present only when the suite is
    built with a collision provider, and read by the criticality
    metrics. ``ttc_covered`` is False for frames whose scene has no lanelet map
    (their TTC is all-``inf`` by construction): the criticality metrics exclude
    such frames from their denominators, mirroring the region-filter coverage
    rule.
    """

    pred_boxes: torch.Tensor
    pred_scores: torch.Tensor
    pred_labels: torch.Tensor
    gt_boxes: torch.Tensor
    gt_labels: torch.Tensor
    gt_ttc: torch.Tensor | None = None
    pred_ttc: torch.Tensor | None = None
    ttc_covered: bool = True


@dataclass(frozen=True)
class MatchCurve:
    """Per-class, per-threshold score-ordered matching results.

    ``corner_error`` and ``nearest_surface_error`` are per-prediction arrays
    (``NaN`` for non-matches), filled for true positives alongside the nuScenes
    error channels so the driving-aware components read them without re-matching.
    """

    total_gt: int
    scores: np.ndarray
    true_positive: np.ndarray  # (N,) bool, per-prediction match flag
    false_positive: np.ndarray  # (N,) bool, complement of true_positive
    heading_score: np.ndarray
    translation_error: np.ndarray
    orientation_error: np.ndarray
    scale_error: np.ndarray
    velocity_error: np.ndarray
    attribute_error: np.ndarray
    corner_error: np.ndarray
    nearest_surface_error: np.ndarray

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
