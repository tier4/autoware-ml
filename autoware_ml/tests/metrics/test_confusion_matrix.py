"""Tests for the detection (matched-pair) and segmentation (point) confusion metrics."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from autoware_ml.metrics.base import EvalStage
from autoware_ml.metrics.confusion_report import confusion_cells
from autoware_ml.metrics.detection3d.confusion_matrix import ConfusionMatrix as DetConfusion
from autoware_ml.metrics.detection3d.matching import DetectionState
from autoware_ml.metrics.detection3d.structures import Detection3DSample
from autoware_ml.metrics.segmentation3d.confusion import ConfusionState
from autoware_ml.metrics.segmentation3d.confusion_matrix import ConfusionMatrix as SegConfusion
from autoware_ml.metrics.segmentation3d.suite import Segmentation3DConfusionMatrixMetricSuite


def test_confusion_cells_flattens_true_by_pred() -> None:
    matrix = np.array([[5, 1], [2, 7]], dtype=np.int64)
    cells = confusion_cells(matrix, ("road", "obstacle"))
    assert cells["confusion_road__road"] == 5.0
    assert cells["confusion_road__obstacle"] == 1.0
    assert cells["confusion_obstacle__road"] == 2.0
    assert cells["confusion_obstacle__obstacle"] == 7.0


def _det_sample(pred_xy_label, gt_xy_label) -> Detection3DSample:
    def box(x, y):
        return [float(x), float(y), 0.0, 4.0, 2.0, 1.5, 0.0]

    return Detection3DSample(
        pred_boxes=torch.tensor([box(x, y) for x, y, _ in pred_xy_label]),
        pred_scores=torch.tensor([0.9] * len(pred_xy_label)),
        pred_labels=torch.tensor([lbl for _, _, lbl in pred_xy_label], dtype=torch.long),
        gt_boxes=torch.tensor([box(x, y) for x, y, _ in gt_xy_label]),
        gt_labels=torch.tensor([lbl for _, _, lbl in gt_xy_label], dtype=torch.long),
    )


def test_det_confusion_counts_matched_label_pair() -> None:
    # A car prediction (label 0) sits on a truck GT (label 1) -> confusion[truck, car].
    sample = _det_sample([(10.0, 0.0, 0)], [(10.0, 0.0, 1)])
    state = DetectionState(samples=[sample], class_names=("car", "truck"), match_cost="center")
    out = DetConfusion(match_threshold=2.0).evaluate(state, EvalStage.TEST)
    assert out["confusion_truck__car"] == 1.0
    assert out["confusion_truck__truck"] == 0.0
    assert out["confusion_car__car"] == 0.0


def test_det_confusion_drops_unmatched() -> None:
    # Prediction 100 m from the only GT -> no match -> empty matrix.
    sample = _det_sample([(100.0, 0.0, 0)], [(10.0, 0.0, 0)])
    state = DetectionState(samples=[sample], class_names=("car", "truck"), match_cost="center")
    out = DetConfusion(match_threshold=2.0).evaluate(state, EvalStage.TEST)
    assert all(v == 0.0 for v in out.values())


def test_seg_confusion_reads_state_matrix() -> None:
    state = ConfusionState(
        confusion=torch.tensor([[5, 1], [2, 7]]), class_names=("road", "obstacle"), num_classes=2
    )
    out = SegConfusion().evaluate(state, EvalStage.TEST)
    assert out["confusion_road__obstacle"] == 1.0
    assert out["confusion_obstacle__road"] == 2.0


def test_seg_confusion_grouped_folds_intra_group_pairs() -> None:
    # car<->truck confusion folds into the grouped_vehicle diagonal.
    names = ("car", "truck", "road", "sidewalk")
    groups = {"grouped_vehicle": ["car", "truck"], "grouped_flat": ["road", "sidewalk"]}
    suite = Segmentation3DConfusionMatrixMetricSuite(
        components=[SegConfusion(stages=["test"])],
        num_classes=4,
        class_names=names,
        ranges=(),
        class_groups=groups,
    )
    suite.update(
        {
            "seg_frames": [
                {
                    "pred": torch.tensor([0, 1, 2, 3]),
                    "target": torch.tensor([1, 0, 3, 2]),  # all intra-group swaps
                    "coord": torch.zeros((4, 3)),
                }
            ]
        }
    )
    report = suite.result(EvalStage.TEST)
    # Both vehicle points land in the grouped_vehicle diagonal, none leak to flat.
    assert report["confusion_grouped_vehicle__grouped_vehicle"] == 2.0
    assert report["confusion_grouped_vehicle__grouped_flat"] == 0.0
    assert report["confusion_grouped_flat__grouped_flat"] == 2.0
    assert "confusion_car__truck" not in report
