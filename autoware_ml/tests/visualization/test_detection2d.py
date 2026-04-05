"""Tests for detection2d visualization helpers."""

from __future__ import annotations

import torch

from autoware_ml.visualization.detection2d import (
    build_label_names,
    targets_to_absolute_xyxy,
)


def test_targets_to_absolute_xyxy_rescales_normalized_boxes() -> None:
    boxes, labels = targets_to_absolute_xyxy(
        {
            "boxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=torch.float32),
            "labels": torch.tensor([3], dtype=torch.int64),
        },
        torch.tensor([10, 20], dtype=torch.int64),
    )

    assert labels.tolist() == [3]
    assert boxes.tolist() == [[5.0, 2.5, 15.0, 7.5]]


def test_build_label_names_uses_generic_labels_for_class_mismatch() -> None:
    label_names = build_label_names(
        [{"name": "car"}, {"name": "truck"}],
        num_classes=4,
    )

    assert label_names[0] == "class_0"
    assert label_names[3] == "class_3"
