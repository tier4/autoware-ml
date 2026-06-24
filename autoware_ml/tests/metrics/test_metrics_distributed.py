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

import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from autoware_ml.metrics.base import EvalStage, MetricRange
from autoware_ml.metrics.detection3d.heading_ap import HeadingAP
from autoware_ml.metrics.detection3d.mean_ap import MeanAP
from autoware_ml.metrics.detection3d.nds import Nds
from autoware_ml.metrics.detection3d.suite import Detection3DMetricSuite
from autoware_ml.metrics.detection3d.tp_errors import TpErrors
from autoware_ml.metrics.segmentation3d.accuracy import Accuracy
from autoware_ml.metrics.segmentation3d.iou import IoU
from autoware_ml.metrics.segmentation3d.precision_recall_f1 import PrecisionRecallF1
from autoware_ml.metrics.segmentation3d.suite import Segmentation3DMetricSuite


def _det_suite(**kwargs):
    return Detection3DMetricSuite(components=[MeanAP(), HeadingAP(), Nds(), TpErrors()], **kwargs)


def _seg_suite(**kwargs):
    return Segmentation3DMetricSuite(components=[IoU(), Accuracy(), PrecisionRecallF1()], **kwargs)


pytestmark = pytest.mark.skipif(
    not (dist.is_available() and dist.is_gloo_available()),
    reason="gloo backend unavailable",
)

_WORLD_SIZES = [2, 8]
_SEG_RANGES = (MetricRange("0-50m", 0.0, 50.0), MetricRange("50-90m", 50.0, 90.0))


def _seg_shard(rank: int) -> dict[str, torch.Tensor]:
    """Deterministic per-rank points, two near (10 m) and one far (60 m)."""
    return {
        "seg_pred_labels": torch.tensor([0, 1, rank % 2]),
        "seg_target_labels": torch.tensor([0, 1, 0]),
        "seg_coord": torch.tensor([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0], [60.0, 0.0, 0.0]]),
    }


def _det_shard(rank: int) -> dict[str, object]:
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


def _assert_reports_equal(actual: dict[str, float], expected: dict[str, float]) -> None:
    assert set(actual) == set(expected), set(actual) ^ set(expected)
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if expected_value != expected_value:  # NaN
            assert actual_value != actual_value, (key, actual_value)
        else:
            assert abs(actual_value - expected_value) < 1e-6, (key, actual_value, expected_value)


def _seg_worker(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    try:
        metric = _seg_suite(num_classes=2, ranges=_SEG_RANGES)
        metric.update(_seg_shard(rank))
        report = metric.result(EvalStage.TEST)  # all_reduce inside compute, then build report

        if rank == 0:
            reference = _seg_suite(num_classes=2, ranges=_SEG_RANGES, sync_on_compute=False)
            for other in range(world_size):
                reference.update(_seg_shard(other))
            _assert_reports_equal(report, reference.result(EvalStage.TEST))
    finally:
        dist.destroy_process_group()


def _det_worker(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    try:
        metric = _det_suite(thresholds=(0.5,))
        metric.update(_det_shard(rank))
        report = metric.result(EvalStage.VAL)  # list-state gather inside compute, then build report

        if rank == 0:
            reference = _det_suite(thresholds=(0.5,), sync_on_compute=False)
            for other in range(world_size):
                reference.update(_det_shard(other))
            _assert_reports_equal(report, reference.result(EvalStage.VAL))
    finally:
        dist.destroy_process_group()


def _spawn(worker, world_size: int) -> None:
    with tempfile.TemporaryDirectory() as directory:
        # mp.spawn re-raises any child assertion in the parent, failing the test.
        mp.spawn(
            worker,
            args=(world_size, os.path.join(directory, "store")),
            nprocs=world_size,
            join=True,
        )


@pytest.mark.parametrize("world_size", _WORLD_SIZES)
def test_seg_metric_sync_matches_single_process(world_size: int) -> None:
    _spawn(_seg_worker, world_size)


@pytest.mark.parametrize("world_size", _WORLD_SIZES)
def test_det_metric_sync_matches_single_process(world_size: int) -> None:
    _spawn(_det_worker, world_size)
