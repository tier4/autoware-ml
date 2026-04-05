"""Tests for COCO-style detection metrics."""

from __future__ import annotations

import pytest
import torch

from autoware_ml.metrics.detection2d import (
    build_coco_api_from_dataset_dict,
    evaluate_class_agnostic_localization,
    evaluate_coco_predictions,
)


pytest.importorskip("pycocotools")


def test_coco_metrics_report_perfect_match_for_identical_box() -> None:
    coco_gt = build_coco_api_from_dataset_dict(
        {
            "images": [{"id": 1, "width": 40, "height": 20, "file_name": "sample.jpg"}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 5, "bbox": [10, 5, 20, 10], "area": 200, "iscrowd": 0}
            ],
            "categories": [{"id": 5, "name": "car"}],
        }
    )

    metrics = evaluate_coco_predictions(
        coco_gt=coco_gt,
        predictions=[
            {
                "image_id": 1,
                "boxes": torch.tensor([[10.0, 5.0, 30.0, 15.0]]),
                "scores": torch.tensor([0.99]),
                "labels": torch.tensor([0]),
            }
        ],
        label_to_category_id={0: 5},
    )

    assert metrics["AP50"] > 0.99


def test_class_agnostic_localization_reports_recall_for_overlapping_box() -> None:
    coco_gt = build_coco_api_from_dataset_dict(
        {
            "images": [{"id": 1, "width": 40, "height": 20, "file_name": "sample.jpg"}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 5, "bbox": [10, 5, 20, 10], "area": 200, "iscrowd": 0}
            ],
            "categories": [{"id": 5, "name": "car"}],
        }
    )

    metrics = evaluate_class_agnostic_localization(
        coco_gt=coco_gt,
        predictions=[
            {
                "image_id": 1,
                "boxes": torch.tensor([[10.0, 5.0, 30.0, 15.0]]),
                "scores": torch.tensor([0.1]),
                "labels": torch.tensor([7]),
            }
        ],
    )

    assert metrics["mean_best_iou"] > 0.99
    assert metrics["gt_recall_0p5"] == pytest.approx(1.0)
