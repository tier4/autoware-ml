"""Unit tests for the driving-aware detection metrics (A1, A2, A3).

Geometry helpers are pinned against hand-computed values, then each component is
exercised on a hand-built ``DetectionState`` with a known true-positive pair.
"""

from __future__ import annotations

from math import pi

import numpy as np
import pytest
import torch

from autoware_ml.metrics.base import EvalStage
from autoware_ml.metrics.detection3d.corner_error import CornerError
from autoware_ml.metrics.detection3d.geometry import (
    bev_corners,
    corner_displacement,
    nearest_surface_distance,
    signed_nearest_surface_error,
)
from autoware_ml.metrics.detection3d.heading_flip import HeadingFlipRate
from autoware_ml.metrics.detection3d.nearest_surface_error import NearestSurfaceError
from autoware_ml.metrics.detection3d.matching import DetectionState
from autoware_ml.metrics.detection3d.structures import Detection3DSample


def _box(cx=10.0, cy=0.0, dx=4.0, dy=2.0, yaw=0.0) -> np.ndarray:
    return np.array([cx, cy, 0.0, dx, dy, 1.5, yaw, 0.0, 0.0], dtype=np.float64)


def test_bev_corners_axis_aligned() -> None:
    corners = bev_corners(_box(cx=0.0, cy=0.0, dx=2.0, dy=2.0, yaw=0.0))
    expected = {(1.0, 1.0), (1.0, -1.0), (-1.0, -1.0), (-1.0, 1.0)}
    assert {tuple(np.round(c, 6)) for c in corners} == expected


def test_corner_displacement_identity_and_translation() -> None:
    assert corner_displacement(_box(), _box()) == pytest.approx(0.0)
    # Pure 1 m translation shifts every corner by exactly 1 m.
    assert corner_displacement(_box(cx=11.0), _box(cx=10.0)) == pytest.approx(1.0)


def test_nearest_surface_distance_and_sign() -> None:
    # Box spans x in [8, 12], y in [-1, 1], nearest face to the origin is x = 8.
    assert nearest_surface_distance(_box()) == pytest.approx(8.0)
    # Predicting the box 1 m farther => +1 m signed near-face error (brakes late).
    assert signed_nearest_surface_error(_box(cx=11.0), _box(cx=10.0)) == pytest.approx(1.0)


def _yaw_state(yaw_err: float, match_cost: str = "center") -> DetectionState:
    gt = torch.tensor([[10.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0, 0.0, 0.0]])
    pred = gt.clone()
    pred[0, 6] = yaw_err
    sample = Detection3DSample(
        pred_boxes=pred,
        pred_scores=torch.tensor([0.9]),
        pred_labels=torch.tensor([0]),
        gt_boxes=gt,
        gt_labels=torch.tensor([0]),
    )
    return DetectionState(
        samples=[sample], class_names=("car",), match_cost=match_cost
    )


def test_heading_flip_rate_no_flip_for_small_error() -> None:
    out = HeadingFlipRate().evaluate(_yaw_state(0.1), EvalStage.TEST)
    assert out["flip_count_car"] == 0.0
    assert out["flip_rate_car"] == pytest.approx(0.0)


def test_heading_flip_rate_counts_a_reversal() -> None:
    # A near-180 deg error is a flip: uniform +1 regardless of exact angle.
    out = HeadingFlipRate().evaluate(_yaw_state(pi - 0.1), EvalStage.TEST)
    assert out["flip_count_car"] == 1.0
    assert out["flip_rate_car"] == pytest.approx(1.0)
    assert out["mflip_rate"] == pytest.approx(1.0)


def test_corner_error_positive_under_yaw() -> None:
    out = CornerError().evaluate(_yaw_state(0.1), EvalStage.TEST)
    assert out["corner_mean_car"] > 0.0
    assert out["corner_max_car"] >= out["corner_mean_car"]


def test_nearest_surface_error_present() -> None:
    out = NearestSurfaceError().evaluate(_yaw_state(0.1), EvalStage.TEST)
    assert "nsurf_absmax_car" in out
    assert out["nsurf_absmax_car"] >= 0.0


def test_corner_matching_mode_still_matches() -> None:
    # A near-identical box must remain a true positive under corner-distance matching.
    out = CornerError().evaluate(_yaw_state(0.05, match_cost="corner"), EvalStage.TEST)
    assert not np.isnan(out["corner_mean_car"])
