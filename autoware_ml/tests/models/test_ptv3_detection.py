"""Unit tests for PTv3-based detection models."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from autoware_ml.models.segmentation3d.ptv3 import PTv3SegmentationModel
from autoware_ml.ops.spconv.availability import IS_SPCONV_AVAILABLE
from autoware_ml.tests.models.ptv3_detection_fixtures import (
    build_inputs,
    build_ptv3_backbone,
    build_targets,
    build_trans_model,
    move_batch_to_device,
    move_targets_to_device,
)
from autoware_ml.utils.checkpoints import apply_matching_weights


@pytest.mark.skipif(
    not IS_SPCONV_AVAILABLE or not torch.cuda.is_available(),
    reason="PTv3 sparse-convolution tests require CUDA spconv",
)
def test_ptv3_transhead_detection_runs_loss_and_predict() -> None:
    device = torch.device("cuda")
    model = build_trans_model().to(device)
    inputs = move_batch_to_device(build_inputs(), device)
    gt_boxes, gt_labels = build_targets()
    gt_boxes, gt_labels = move_targets_to_device(gt_boxes, gt_labels, device)

    outputs = model(**inputs)
    metrics = model.compute_metrics({"gt_boxes": gt_boxes, "gt_labels": gt_labels}, outputs)
    predictions = model.bbox_head.predict(outputs)

    assert "loss" in metrics
    assert outputs["dense_heatmap"].shape[:2] == (1, 2)
    assert outputs["query_labels"].shape == (1, 8)
    assert isinstance(predictions, list)
    assert set(predictions[0]) == {"bboxes_3d", "scores_3d", "labels_3d"}


def test_ptv3_detection_loads_backbone_from_seg_checkpoint_via_matching_weights(
    tmp_path: Path,
) -> None:
    segmentation_model = PTv3SegmentationModel(
        backbone=build_ptv3_backbone(),
        num_classes=3,
        backbone_out_channels=8,
        ignore_index=-1,
        optimizer=lambda params: torch.optim.AdamW(params, lr=1e-3),
        grid_size=1.0,
        point_cloud_range=[0.0, 0.0, -2.0, 8.0, 8.0, 2.0],
    )
    checkpoint_path = tmp_path / "ptv3_segmentation.ckpt"
    torch.save({"state_dict": segmentation_model.state_dict()}, checkpoint_path)

    model = build_trans_model(freeze_backbone=True)
    apply_matching_weights(model, (checkpoint_path,))
    model.train()

    reference_state = segmentation_model.backbone.state_dict()
    loaded_state = model.backbone.state_dict()
    first_key = next(iter(reference_state))

    assert torch.allclose(loaded_state[first_key], reference_state[first_key])
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
    assert model.backbone.training is False
