"""Distributed (multi-rank) tests for the metric sync paths.

These spawn real CPU process groups with the gloo backend, so the actual
collectives run: torchmetrics list-state gather for detection samples and
``all_reduce`` for the segmentation confusion matrices. No GPU is required.

The correctness property checked is the design guarantee: after sync every rank
holds the global state, so each rank's report must equal the single-process
report computed over the union of all per-rank shards. The single-process
reference is built on rank 0 with ``sync_on_compute=False`` so it issues no
collectives and cannot deadlock the group.
"""

from __future__ import annotations

import pytest
import torch

from autoware_ml.metrics.base import EvalStage, MetricRange
from autoware_ml.metrics.detection3d.heading_ap import HeadingAP
from autoware_ml.metrics.detection3d.mean_ap import MeanAP
from autoware_ml.metrics.detection3d.nds import Nds
from autoware_ml.metrics.detection3d.suite import Detection3DMetricSuite
from autoware_ml.metrics.detection3d.tp_errors import TpErrors
from autoware_ml.metrics.report import MetricReport
from autoware_ml.metrics.segmentation3d.accuracy import Accuracy
from autoware_ml.metrics.segmentation3d.iou import IoU
from autoware_ml.metrics.segmentation3d.precision_recall_f1 import PrecisionRecallF1
from autoware_ml.metrics.segmentation3d.suite import Segmentation3DConfusionMatrixMetricSuite
from autoware_ml.utils.tests.distributed_fixtures import (
    GLOO_AVAILABLE,
    spawn_gloo_workers,
)


def _det3d_suite(thresholds=(0.5, 1.0, 2.0, 4.0), **kwargs):
    return Detection3DMetricSuite(
        components=[
            MeanAP(thresholds=thresholds),
            HeadingAP(thresholds=thresholds),
            Nds(thresholds=thresholds, tp_threshold=thresholds[0]),
            TpErrors(thresholds=thresholds, tp_threshold=thresholds[0]),
        ],
        **kwargs,
    )


def _seg3d_suite(**kwargs):
    return Segmentation3DConfusionMatrixMetricSuite(
        components=[IoU(), Accuracy(), PrecisionRecallF1()], **kwargs
    )


pytestmark = pytest.mark.skipif(not GLOO_AVAILABLE, reason="gloo backend unavailable")

_WORLD_SIZES = [2, 8]
_SEG3D_RANGES = (MetricRange("0-50m", 0.0, 50.0), MetricRange("50-90m", 50.0, 90.0))


def _seg3d_shard(rank: int) -> dict[str, object]:
    """Deterministic per-rank points, two near (10 m) and one far (60 m)."""
    return {
        "seg3d_frames": [
            {
                "pred": torch.tensor([0, 1, rank % 2]),
                "target": torch.tensor([0, 1, 0]),
                "coord": torch.tensor([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0], [60.0, 0.0, 0.0]]),
            }
        ]
    }


def _det3d_shard(rank: int) -> dict[str, object]:
    """One frame per rank with a single matching prediction at x = rank."""
    boxes = torch.zeros((1, 7), dtype=torch.float32)
    boxes[0, 0] = float(rank)
    prediction = {
        "bboxes_3d": boxes.clone(),
        "scores_3d": torch.tensor([0.9]),
        "labels_3d": torch.tensor([0]),
    }
    return {
        "predictions": [prediction],
        "gt_boxes": [boxes.clone()],
        "gt_labels": [torch.tensor([0])],
    }


def _assert_reports_equal(actual: MetricReport, expected: MetricReport) -> None:
    actual_keys = set(actual.metric_names())
    expected_keys = set(expected.metric_names())
    assert actual_keys == expected_keys, actual_keys ^ expected_keys
    for key in expected.metric_names():
        expected_value = expected.metric_value(key)
        actual_value = actual.metric_value(key)
        if expected_value != expected_value:  # NaN
            assert actual_value != actual_value, (key, actual_value)
        else:
            assert abs(actual_value - expected_value) < 1e-6, (key, actual_value, expected_value)


def _seg3d_worker(rank: int, world_size: int) -> None:
    metric = _seg3d_suite(num_classes=2, ranges=_SEG3D_RANGES)
    metric.update(_seg3d_shard(rank))
    report = metric.result(EvalStage.TEST)  # all_reduce inside compute, then build report

    if rank == 0:
        reference = _seg3d_suite(num_classes=2, ranges=_SEG3D_RANGES, sync_on_compute=False)
        for other in range(world_size):
            reference.update(_seg3d_shard(other))
        _assert_reports_equal(report, reference.result(EvalStage.TEST))


def _det3d_worker(rank: int, world_size: int) -> None:
    metric = _det3d_suite(thresholds=(0.5,))
    metric.update(_det3d_shard(rank))
    report = metric.result(EvalStage.VAL)  # list-state gather inside compute, then build report

    if rank == 0:
        reference = _det3d_suite(thresholds=(0.5,), sync_on_compute=False)
        for other in range(world_size):
            reference.update(_det3d_shard(other))
        _assert_reports_equal(report, reference.result(EvalStage.VAL))


@pytest.mark.parametrize("world_size", _WORLD_SIZES)
def test_seg3d_metric_sync_matches_single_process(world_size: int) -> None:
    spawn_gloo_workers(_seg3d_worker, world_size=world_size)


@pytest.mark.parametrize("world_size", _WORLD_SIZES)
def test_det3d_metric_sync_matches_single_process(world_size: int) -> None:
    spawn_gloo_workers(_det3d_worker, world_size=world_size)
