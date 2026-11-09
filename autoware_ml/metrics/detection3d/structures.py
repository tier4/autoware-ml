"""Pure data structures for the detection metrics.

``Detection3DSample`` is the per-frame input. ``MatchCurve`` carries one
class's score-ordered matching results and ``CurveMetrics`` its AP-style
summary. ``TpErrorSet`` holds the nuScenes true-positive errors of one selection
of matches and ``RecallTarget`` names a recall operating point to select them at.
The memoizing ``Detection3DState`` that orchestrates the matching lives in
:mod:`autoware_ml.metrics.detection3d.matching`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, model_validator

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


class TpErrorSet(BaseModel):
    """
    The nuScenes true-positive errors of one selection of matches.

    Attributes:
      ate: Average translation error, the BEV center distance in meters.
      aoe: Average orientation error, the absolute yaw gap in radians.
      ase: Average scale error, one minus the aligned 3D IoU.
      ave: Average velocity error in meters per second.
      aae: Average attribute error, the neutral 1.0 while attributes are not evaluated.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    ate: float
    aoe: float
    ase: float
    ave: float
    aae: float

    @classmethod
    def nan(cls) -> TpErrorSet:
        """
        The error set of a selection without a measurable match.

        Returns:
          TpErrorSet: Every error NaN.
        """
        nan = float("nan")
        return cls(ate=nan, aoe=nan, ase=nan, ave=nan, aae=nan)

    def named(self) -> tuple[tuple[str, float], ...]:
        """
        The errors with their nuScenes names, in the nuScenes order.

        Returns:
          tuple[tuple[str, float], ...]: ``(name, value)`` pairs, ``ATE`` first.
        """
        return (
            ("ATE", self.ate),
            ("AOE", self.aoe),
            ("ASE", self.ase),
            ("AVE", self.ave),
            ("AAE", self.aae),
        )


class SelectedTpErrors(BaseModel):
    """
    True-positive errors selected at one operating point.

    Attributes:
      count: Number of true positives the errors are averaged over.
      errors: The mean errors over them, every error 1.0 when no true positive was selected.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    count: int
    errors: TpErrorSet


class RecallTarget(BaseModel):
    """
    A recall operating point the true-positive errors are selected up to.

    Attributes:
      name: Variant name carried by the metric keys, for example ``default``. ``optimal`` is
        reserved for the max-F1 operating point that is always reported.
      recall: Recall level in ``[0, 1]`` up to which the matches are pooled.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    name: str
    recall: float

    @model_validator(mode="after")
    def _check_name_and_range(self) -> RecallTarget:
        """Reject the reserved name and a recall outside the unit range."""
        if self.name == "optimal":
            raise ValueError('"optimal" is the reserved max-F1 operating point, pick another name.')
        if not 0.0 <= self.recall <= 1.0:
            raise ValueError(f"recall target {self.name!r} must be in [0, 1], got {self.recall}.")
        return self
