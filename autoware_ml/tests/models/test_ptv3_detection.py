"""Unit tests for PTv3-based detection models."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from autoware_ml.models.detection3d.heads.centerpoint import CenterHead
from autoware_ml.models.segmentation3d.ptv3 import PTv3SegmentationModel
from autoware_ml.ops.spconv.availability import IS_SPCONV_AVAILABLE
from autoware_ml.tests.models.ptv3_detection_fixtures import (
    build_center_model,
    build_inputs,
    build_ptv3_backbone,
    build_targets,
    build_trans_model,
    move_batch_to_device,
    move_targets_to_device,
)
from autoware_ml.utils.checkpoints import apply_matching_weights


def test_centerhead_uses_natural_dimension_order() -> None:
    head = CenterHead(
        in_channels=4,
        num_classes=2,
        shared_channels=4,
        point_cloud_range=[0.0, 0.0, -2.0, 8.0, 8.0, 2.0],
        voxel_size=[0.5, 0.5, 4.0],
        out_size_factor=2,
        max_objs=16,
        min_radius=1,
        score_threshold=0.1,
        post_max_size=10,
        nms_min_radius=1.0,
        use_velocity=False,
    )
    gt_boxes = [torch.tensor([[2.0, 3.0, 0.2, 4.0, 1.6, 1.5, 0.25]], dtype=torch.float32)]
    gt_labels = [torch.tensor([0], dtype=torch.long)]

    targets = head.get_targets(
        gt_boxes,
        gt_labels,
        feature_map_size=(4, 4),
        device=torch.device("cpu"),
    )

    assert torch.allclose(
        targets.anno_boxes[0, 0, 3:6],
        torch.tensor([4.0, 1.6, 1.5]).log(),
    )

    flat_index = int(targets.indices[0, 0].item())
    y_index, x_index = divmod(flat_index, 4)
    target_box = targets.anno_boxes[0, 0]

    outputs = {
        "heatmap": torch.full((1, 2, 4, 4), -20.0),
        "reg": torch.zeros((1, 2, 4, 4)),
        "height": torch.zeros((1, 1, 4, 4)),
        "dim": torch.zeros((1, 3, 4, 4)),
        "rot": torch.zeros((1, 2, 4, 4)),
    }
    outputs["heatmap"][0, 0, y_index, x_index] = 20.0
    outputs["reg"][0, :, y_index, x_index] = target_box[0:2]
    outputs["height"][0, 0, y_index, x_index] = target_box[2]
    outputs["dim"][0, :, y_index, x_index] = target_box[3:6]
    outputs["rot"][0, :, y_index, x_index] = target_box[6:8]

    predictions = head.predict(outputs)

    assert torch.allclose(
        predictions[0]["bboxes_3d"][0, 3:6],
        torch.tensor([4.0, 1.6, 1.5]),
    )


@pytest.mark.skipif(
    not IS_SPCONV_AVAILABLE or not torch.cuda.is_available(),
    reason="PTv3 sparse-convolution tests require CUDA spconv",
)
def test_ptv3_centerhead_detection_runs_loss_and_predict() -> None:
    device = torch.device("cuda")
    model = build_center_model().to(device)
    inputs = move_batch_to_device(build_inputs(), device)
    gt_boxes, gt_labels = build_targets()
    gt_boxes, gt_labels = move_targets_to_device(gt_boxes, gt_labels, device)

    outputs = model(**inputs)
    metrics = model.compute_metrics({"gt_boxes": gt_boxes, "gt_labels": gt_labels}, outputs)
    predictions = model.bbox_head.predict(outputs)

    assert "loss" in metrics
    assert outputs["heatmap"].shape[:2] == (1, 2)
    assert isinstance(predictions, list)
    assert set(predictions[0]) == {"bboxes_3d", "scores_3d", "labels_3d"}


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

    model = build_center_model(freeze_backbone=True)
    apply_matching_weights(model, (checkpoint_path,))
    model.train()

    reference_state = segmentation_model.backbone.state_dict()
    loaded_state = model.backbone.state_dict()
    first_key = next(iter(reference_state))

    assert torch.allclose(loaded_state[first_key], reference_state[first_key])
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
    assert model.backbone.training is False
