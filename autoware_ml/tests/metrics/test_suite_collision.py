"""End-to-end: the detection suite computes per-box reachability TTC at update and
the criticality metrics (B1, B2) consume it."""

from __future__ import annotations

from math import inf

import numpy as np
import torch
from shapely.geometry import box

from autoware_ml.metrics.base import EvalStage
from autoware_ml.metrics.detection3d.collision import CollisionTTC
from autoware_ml.metrics.detection3d.collision_weighted_map import CollisionWeightedMeanAP
from autoware_ml.metrics.detection3d.critical_fp_fn import CriticalFPFN
from autoware_ml.metrics.detection3d.suite import Detection3DMetricSuite
from autoware_ml.metrics.geometry.reachability import ReachabilityParams

CLASS_NAMES = (
    "car", "truck", "bus", "train", "motorcycle", "bicycle", "pedestrian",
    "animal", "barrier", "traffic_cone", "debris", "bicycle_rack", "vehicle_extension",
)


class _FakeMap:
    def __init__(self, polygon):
        self._polygon = polygon

    def region_union(self, tokens):
        return self._polygon

    def speed_at(self, x, y, default):
        return default


class _FakeProvider:
    def __init__(self, polygon):
        self._map = _FakeMap(polygon)

    def get(self, scene_token):
        return self._map

    def available(self, scene_token):
        return True


def _b(cx, cy, yaw=0.0):
    return [cx, cy, 0.0, 4.0, 2.0, 1.5, yaw, 0.0, 0.0]


def test_suite_computes_ttc_and_criticality_metrics() -> None:
    collision = CollisionTTC(
        CLASS_NAMES,
        _FakeProvider(box(-80.0, -60.0, 500.0, 60.0)),
        params=ReachabilityParams(horizon_s=4.0, dt_s=0.1),
        max_speed_mps=10.0,
    )
    suite = Detection3DMetricSuite(
        components=[
            CriticalFPFN(confidences=(0.5,), match_threshold=2.0),
            CollisionWeightedMeanAP(thresholds=(2.0,), decay=0.5),
        ],
        class_names=CLASS_NAMES,
        collision=collision,
    )

    # GT: a lead car and a barrier ahead, both reachable in the worst case.
    gt_boxes = torch.tensor([_b(25.0, 0.0), _b(30.0, 0.0)], dtype=torch.float32)
    gt_labels = torch.tensor([0, 8], dtype=torch.long)  # car, barrier
    # Preds: match both GT + a phantom barrier in the path (critical FP).
    preds = {
        "bboxes_3d": torch.tensor([_b(25.0, 0.0), _b(30.0, 0.0), _b(18.0, 0.0)], dtype=torch.float32),
        "scores_3d": torch.tensor([0.9, 0.9, 0.9], dtype=torch.float32),
        "labels_3d": torch.tensor([0, 8, 8], dtype=torch.long),  # car, barrier, phantom barrier
    }
    suite.update({
        "predictions": [preds],
        "gt_boxes": [gt_boxes],
        "gt_labels": [gt_labels],
        "ego2global": [np.eye(4)],
        "scene_token": ["scene-0"],
    })

    state = suite.state_for(None)
    sample = state.samples[0]
    assert sample.gt_ttc is not None and sample.pred_ttc is not None
    gt_ttc = sample.gt_ttc.numpy()
    assert gt_ttc[0] != inf          # the lead can brake, so it is reachable
    assert gt_ttc[1] != inf          # barrier ahead reachable

    b1 = CriticalFPFN(confidences=(0.5,), match_threshold=2.0).evaluate(state, EvalStage.TEST)
    assert b1["critical_fp_0p5m"] == 1.0   # the phantom barrier in path
    assert b1["critical_fn_0p5m"] == 0.0   # both GT are matched

    b2 = CollisionWeightedMeanAP(thresholds=(2.0,), decay=0.5).evaluate(state, EvalStage.TEST)
    assert not np.isnan(b2["cw_mAP"])      # the reachable barrier carries the weight


def test_collision_provider_declares_context_keys() -> None:
    # The provider's frame context is part of the fail-loud required-keys chain,
    # exactly like the region filters' keys.
    suite = Detection3DMetricSuite(
        components=[CriticalFPFN(confidences=(0.5,))],
        class_names=CLASS_NAMES,
        collision=CollisionTTC(
            CLASS_NAMES,
            _FakeProvider(box(-80.0, -60.0, 500.0, 60.0)),
            params=ReachabilityParams(horizon_s=4.0, dt_s=0.1),
        ),
    )
    keys = suite.required_keys()
    assert "ego2global" in keys and "scene_token" in keys


def test_min_num_points_without_counts_raises() -> None:
    import pytest

    suite = Detection3DMetricSuite(
        components=[CriticalFPFN(confidences=(0.5,))],
        class_names=CLASS_NAMES,
        min_num_points=2,
    )
    with pytest.raises(ValueError, match="gt_num_points"):
        suite.update({
            "predictions": [
                {
                    "bboxes_3d": torch.zeros((0, 9)),
                    "scores_3d": torch.zeros(0),
                    "labels_3d": torch.zeros(0, dtype=torch.long),
                }
            ],
            "gt_boxes": [torch.zeros((0, 9))],
            "gt_labels": [torch.zeros(0, dtype=torch.long)],
        })
