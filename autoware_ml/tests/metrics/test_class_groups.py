"""Tests for behaviour-taxonomy class grouping (resolve + fold, confusion + point suites)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from autoware_ml.metrics.base import EvalStage
from autoware_ml.metrics.class_groups import fold_labels, resolve_class_groups
from autoware_ml.metrics.segmentation3d.error_clusters import ErrorClusters
from autoware_ml.metrics.segmentation3d.iou import IoU
from autoware_ml.metrics.segmentation3d.point_cloud import Segmentation3DPointCloudMetricSuite
from autoware_ml.metrics.segmentation3d.suite import Segmentation3DConfusionMatrixMetricSuite

NAMES = ("car", "truck", "road", "sidewalk")
GROUPS = {"grouped_vehicle": ["car", "truck"], "grouped_flat": ["road", "sidewalk"]}


def test_resolve_class_groups_full_taxonomy() -> None:
    lut, names = resolve_class_groups(NAMES, GROUPS)
    assert names == ("grouped_vehicle", "grouped_flat")
    assert lut.tolist() == [0, 0, 1, 1]


def test_resolve_class_groups_singleton_and_forward_compat_member() -> None:
    # A singleton group, plus a not-yet-trained member (ghost_point) skipped.
    lut, names = resolve_class_groups(
        NAMES,
        {"grouped_vehicle": ["car", "truck"], "grouped_road": ["road"],
         "grouped_walk": ["sidewalk", "ghost_point"]},
    )
    assert names == ("grouped_vehicle", "grouped_road", "grouped_walk")
    assert lut.tolist() == [0, 0, 1, 2]


def test_resolve_class_groups_requires_full_partition() -> None:
    with pytest.raises(ValueError, match="cover every trained class"):
        resolve_class_groups(NAMES, {"grouped_vehicle": ["car", "truck"]})  # road/sidewalk unassigned


def test_resolve_class_groups_rejects_absent_only_group() -> None:
    with pytest.raises(ValueError, match="no trained member"):
        resolve_class_groups(
            NAMES,
            {"grouped_vehicle": ["car", "truck"], "grouped_flat": ["road", "sidewalk"],
             "grouped_ghost": ["ghost_point", "phantom"]},
        )


def test_fold_labels_leaves_ignore_untouched() -> None:
    lut, _ = resolve_class_groups(NAMES, GROUPS)
    assert fold_labels(np.array([0, 1, 2, 3, -1]), lut).tolist() == [0, 0, 1, 1, -1]


def _confusion_suite(class_groups) -> Segmentation3DConfusionMatrixMetricSuite:
    return Segmentation3DConfusionMatrixMetricSuite(
        components=[IoU(stages=["test"])],
        num_classes=4,
        class_names=NAMES,
        ranges=(),
        class_groups=class_groups,
    )


def _update_confusion(suite) -> None:
    # car/truck confused with each other, road/sidewalk confused with each other.
    suite.update(
        {
            "seg_frames": [
                {
                    "pred": torch.tensor([0, 1, 2, 3]),
                    "target": torch.tensor([1, 0, 3, 2]),
                    "coord": torch.zeros((4, 3)),
                }
            ]
        }
    )


def test_confusion_grouped_folds_intra_group_confusion_to_tp() -> None:
    suite = _confusion_suite(GROUPS)
    _update_confusion(suite)
    report = suite.result(EvalStage.TEST)
    assert report["iou_grouped_vehicle"] == pytest.approx(1.0)  # car/truck confusion is a hit
    assert report["iou_grouped_flat"] == pytest.approx(1.0)
    assert report["mIoU"] == pytest.approx(1.0)
    assert "iou_car" not in report  # trained classes are gone in the grouped suite


def test_confusion_per_class_still_penalizes_confusion() -> None:
    suite = _confusion_suite(None)
    _update_confusion(suite)
    report = suite.result(EvalStage.TEST)
    assert report["iou_car"] == pytest.approx(0.0)  # every car predicted as truck
    assert "iou_grouped_vehicle" not in report


def test_point_suite_serves_both_taxonomies_from_one_cache() -> None:
    # One suite, one accumulation: per-class components see the trained space,
    # grouped_components the folded view under the grouped/ key prefix.
    suite = Segmentation3DPointCloudMetricSuite(
        components=[ErrorClusters(stages=["test"])],
        grouped_components=[ErrorClusters(stages=["test"])],
        num_classes=4,
        class_names=NAMES,
        ranges=(),
        class_groups=GROUPS,
    )
    suite.update(
        {
            "seg_frames": [
                {
                    "coord": torch.tensor([[0.0, 0, 0], [10.0, 0, 0], [20.0, 0, 0], [30.0, 0, 0]]),
                    "target": torch.tensor([0, 1, 2, 3]),  # car, truck, road, sidewalk
                    "pred": torch.tensor([1, 0, 3, 2]),  # all intra-group confusion
                    "scores": torch.full((4, 4), 0.25),
                }
            ]
        }
    )
    report = suite.result(EvalStage.TEST)
    # Trained view: every point is wrong (intra-group confusion counts).
    assert report["error_rate"] == pytest.approx(1.0)
    # Grouped view: the fold makes every point correct.
    assert report["grouped/error_rate"] == pytest.approx(0.0)
    assert report["grouped/error_rate_grouped_vehicle"] == pytest.approx(0.0)
    assert report["grouped/error_rate_grouped_flat"] == pytest.approx(0.0)


def test_point_suite_requires_grouped_components_with_class_groups() -> None:
    with pytest.raises(ValueError, match="come together"):
        Segmentation3DPointCloudMetricSuite(
            components=[ErrorClusters(stages=["test"])],
            num_classes=4,
            class_names=NAMES,
            ranges=(),
            class_groups=GROUPS,
        )


def test_point_suite_grouped_confidence_is_the_predicted_groups_mass() -> None:
    suite = Segmentation3DPointCloudMetricSuite(
        components=[ErrorClusters(stages=["test"])],
        grouped_components=[ErrorClusters(stages=["test"])],
        num_classes=4,
        class_names=NAMES,
        ranges=(),
        class_groups=GROUPS,
    )
    # Raw scores: car 0.4 (argmax), truck 0.0, road 0.35, sidewalk 0.25.
    # Folded: vehicle 0.4, flat 0.6, the reported group is vehicle, so the
    # grouped confidence must be 0.4, never the 0.6 max.
    suite.update(
        {
            "seg_frames": [
                {
                    "coord": torch.zeros((1, 3)),
                    "pred": torch.tensor([0]),
                    "target": torch.tensor([2]),
                    "scores": torch.tensor([[0.4, 0.0, 0.35, 0.25]]),
                }
            ]
        }
    )
    trained = suite.state_for(None)
    grouped = suite._grouped_state_for(None)
    assert trained.frames[0].confidence[0] == pytest.approx(0.4)
    assert grouped.frames[0].pred[0] == 0  # grouped_vehicle
    assert grouped.frames[0].confidence[0] == pytest.approx(0.4)
    assert grouped.frames[0].target[0] == 1  # grouped_flat
