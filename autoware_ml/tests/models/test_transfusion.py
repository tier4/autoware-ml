"""Unit tests for the native TransFusion detector."""

from __future__ import annotations

import math

import torch
import pytest

from autoware_ml.models.detection3d.backbones.second import SECONDBackbone
from autoware_ml.models.detection3d.encoders.pillar import PillarFeatureNet, PointPillarsScatter
from autoware_ml.models.detection3d.heads.transfusion import TransFusionHead
from autoware_ml.models.detection3d.necks.second_fpn import SECONDFPN
from autoware_ml.models.detection3d.task_modules.assigners import AssignResult, HungarianAssigner3D
from autoware_ml.models.detection3d.task_modules.bbox_coders import TransFusionBBoxCoder
from autoware_ml.models.detection3d.task_modules.match_costs import (
    BBoxBEVL1Cost,
    ClassificationCost,
    IoU3DCost,
)
from autoware_ml.models.detection3d.transfusion import TransFusionDetectionModel


def _build_model() -> TransFusionDetectionModel:
    return TransFusionDetectionModel(
        pts_voxel_encoder=PillarFeatureNet(
            in_channels=4,
            feat_channels=[64],
            voxel_size=[0.5, 0.5, 4.0],
            point_cloud_range=[0.0, 0.0, -2.0, 8.0, 8.0, 2.0],
        ),
        pts_middle_encoder=PointPillarsScatter(in_channels=64, output_shape=[16, 16]),
        pts_backbone=SECONDBackbone(
            in_channels=64,
            out_channels=[64, 128, 256],
            layer_nums=[1, 1, 1],
            layer_strides=[2, 2, 2],
        ),
        pts_neck=SECONDFPN(
            in_channels=[64, 128, 256],
            out_channels=[128, 128, 128],
            upsample_strides=[1, 2, 4],
        ),
        bbox_head=TransFusionHead(
            num_proposals=8,
            auxiliary=True,
            in_channels=384,
            hidden_channel=128,
            num_classes=2,
            num_decoder_layers=1,
            num_heads=4,
            feedforward_channels=128,
            common_heads={
                "center": (2, 2),
                "height": (1, 2),
                "dim": (3, 2),
                "rot": (2, 2),
                "vel": (2, 2),
            },
            bbox_coder=TransFusionBBoxCoder(
                pc_range=[0.0, 0.0],
                out_size_factor=2,
                voxel_size=[0.5, 0.5],
                post_center_range=[-1.0, -1.0, -5.0, 10.0, 10.0, 5.0],
                code_size=10,
            ),
            assigner=HungarianAssigner3D(
                cls_cost=ClassificationCost(weight=0.15),
                reg_cost=BBoxBEVL1Cost(weight=0.25),
                iou_cost=IoU3DCost(weight=0.25),
            ),
            point_cloud_range=[0.0, 0.0, -2.0, 8.0, 8.0, 2.0],
            voxel_size=[0.5, 0.5, 4.0],
            out_size_factor=2,
            code_weights=[1.0] * 8 + [0.2, 0.2],
            min_radius=1,
            gaussian_overlap=0.1,
            score_threshold=0.1,
            post_max_size=8,
            nms_min_radius=1.0,
        ),
    )


def _build_head(**kwargs) -> TransFusionHead:
    assigner = kwargs.pop(
        "assigner",
        HungarianAssigner3D(
            cls_cost=ClassificationCost(weight=0.15),
            reg_cost=BBoxBEVL1Cost(weight=0.25),
            iou_cost=IoU3DCost(weight=0.25),
        ),
    )
    return TransFusionHead(
        num_proposals=2,
        auxiliary=False,
        in_channels=32,
        hidden_channel=16,
        num_classes=2,
        num_decoder_layers=1,
        num_heads=2,
        feedforward_channels=32,
        common_heads={
            "center": (2, 2),
            "height": (1, 2),
            "dim": (3, 2),
            "rot": (2, 2),
            "vel": (2, 2),
        },
        bbox_coder=TransFusionBBoxCoder(
            pc_range=[0.0, 0.0],
            out_size_factor=1,
            voxel_size=[1.0, 1.0],
            post_center_range=[-10.0, -10.0, -10.0, 10.0, 10.0, 10.0],
            score_threshold=0.0,
            code_size=10,
        ),
        assigner=assigner,
        point_cloud_range=[0.0, 0.0, -2.0, 8.0, 8.0, 2.0],
        voxel_size=[1.0, 1.0, 4.0],
        out_size_factor=1,
        code_weights=[1.0] * 8 + [0.2, 0.2],
        min_radius=1,
        gaussian_overlap=0.1,
        score_threshold=0.0,
        post_max_size=8,
        nms_min_radius=1.0,
        **kwargs,
    )


def test_transfusion_forward_returns_query_predictions() -> None:
    model = _build_model()
    voxels = torch.randn(12, 5, 4)
    num_points = torch.randint(1, 5, (12,), dtype=torch.int32)
    voxel_coords = torch.randint(0, 8, (12, 4), dtype=torch.int32)
    voxel_coords[:, 0] = 0

    outputs = model(voxels=voxels, num_points=num_points, voxel_coords=voxel_coords)

    assert "dense_heatmap" in outputs
    assert "query_heatmap_score" in outputs
    assert "query_labels" in outputs
    assert outputs["heatmap"].shape[-1] == 8
    assert outputs["center"].shape[-1] == 8


def test_transfusion_predict_reweights_scores_by_query_labels() -> None:
    head = _build_head()
    outputs = {
        "heatmap": torch.tensor([[[0.0, 9.0], [9.0, 0.0]]], dtype=torch.float32),
        "query_heatmap_score": torch.ones((1, 2, 2), dtype=torch.float32),
        "query_labels": torch.tensor([[0, 1]], dtype=torch.long),
        "center": torch.tensor([[[1.0, 2.0], [1.0, 2.0]]], dtype=torch.float32),
        "height": torch.zeros((1, 1, 2), dtype=torch.float32),
        "dim": torch.zeros((1, 3, 2), dtype=torch.float32),
        "rot": torch.tensor([[[0.0, 0.0], [1.0, 1.0]]], dtype=torch.float32),
        "vel": torch.zeros((1, 2, 2), dtype=torch.float32),
    }

    predictions = head.predict(outputs)

    assert predictions[0]["labels_3d"].tolist() == [0, 1]


def test_transfusion_predict_skips_circle_nms_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    head = _build_head()
    outputs = {
        "heatmap": torch.tensor([[[8.0, 8.0], [0.0, 0.0]]], dtype=torch.float32),
        "query_heatmap_score": torch.ones((1, 2, 2), dtype=torch.float32),
        "query_labels": torch.tensor([[0, 0]], dtype=torch.long),
        "center": torch.tensor([[[1.0, 1.0], [1.0, 1.0]]], dtype=torch.float32),
        "height": torch.zeros((1, 1, 2), dtype=torch.float32),
        "dim": torch.zeros((1, 3, 2), dtype=torch.float32),
        "rot": torch.tensor([[[0.0, 0.0], [1.0, 1.0]]], dtype=torch.float32),
        "vel": torch.zeros((1, 2, 2), dtype=torch.float32),
    }

    def fail_circle_nms(*args, **kwargs):
        raise AssertionError("circle_nms should not run when nms_type is None")

    monkeypatch.setattr(
        "autoware_ml.models.detection3d.heads.transfusion.circle_nms", fail_circle_nms
    )

    predictions = head.predict(outputs)

    assert predictions[0]["scores_3d"].shape[0] == 2


def test_transfusion_predict_applies_circle_nms_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    head = _build_head(nms_type="circle")
    outputs = {
        "heatmap": torch.tensor([[[8.0, 8.0], [0.0, 0.0]]], dtype=torch.float32),
        "query_heatmap_score": torch.ones((1, 2, 2), dtype=torch.float32),
        "query_labels": torch.tensor([[0, 0]], dtype=torch.long),
        "center": torch.tensor([[[1.0, 1.1], [1.0, 1.1]]], dtype=torch.float32),
        "height": torch.zeros((1, 1, 2), dtype=torch.float32),
        "dim": torch.zeros((1, 3, 2), dtype=torch.float32),
        "rot": torch.tensor([[[0.0, 0.0], [1.0, 1.0]]], dtype=torch.float32),
        "vel": torch.zeros((1, 2, 2), dtype=torch.float32),
    }

    def fake_circle_nms(boxes, scores, min_radius, post_max_size):
        del boxes, scores, min_radius, post_max_size
        return torch.tensor([0], dtype=torch.long)

    monkeypatch.setattr(
        "autoware_ml.models.detection3d.heads.transfusion.circle_nms", fake_circle_nms
    )

    predictions = head.predict(outputs)

    assert predictions[0]["scores_3d"].shape[0] == 1


def test_transfusion_targets_use_raw_logits_for_assignment() -> None:
    captured_cls_pred: list[torch.Tensor] = []

    class RecordingAssigner:
        def assign(self, bboxes, gt_bboxes, gt_labels, cls_pred, point_cloud_range):
            del bboxes, gt_bboxes, gt_labels, point_cloud_range
            captured_cls_pred.append(cls_pred.detach().clone())
            return AssignResult(
                num_gts=1,
                gt_inds=torch.tensor([1, 0], dtype=torch.long),
                max_overlaps=torch.tensor([1.0, 0.0], dtype=torch.float32),
                labels=torch.tensor([0, -1], dtype=torch.long),
            )

    head = _build_head(assigner=RecordingAssigner())
    outputs = {
        "heatmap": torch.tensor([[[0.2, -1.1], [1.3, -0.7]]], dtype=torch.float32),
        "dense_heatmap": torch.zeros((1, 2, 4, 4), dtype=torch.float32),
        "center": torch.zeros((1, 2, 2), dtype=torch.float32),
        "height": torch.zeros((1, 1, 2), dtype=torch.float32),
        "dim": torch.zeros((1, 3, 2), dtype=torch.float32),
        "rot": torch.tensor([[[0.0, 0.0], [1.0, 1.0]]], dtype=torch.float32),
        "vel": torch.zeros((1, 2, 2), dtype=torch.float32),
    }
    gt_boxes = [torch.tensor([[1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)]
    gt_labels = [torch.tensor([0], dtype=torch.long)]

    head.get_targets(gt_boxes, gt_labels, outputs)

    assert captured_cls_pred
    assert torch.allclose(captured_cls_pred[0], outputs["heatmap"][0])


def test_transfusion_heatmap_loss_receives_raw_logits() -> None:
    captured_prediction: list[torch.Tensor] = []

    class RecordingHeatmapLoss(torch.nn.Module):
        def forward(self, prediction, target):
            del target
            captured_prediction.append(prediction.detach().clone())
            return prediction.new_tensor(0.0)

    class RecordingAssigner:
        def assign(self, bboxes, gt_bboxes, gt_labels, cls_pred, point_cloud_range):
            del bboxes, gt_bboxes, gt_labels, cls_pred, point_cloud_range
            return AssignResult(
                num_gts=1,
                gt_inds=torch.tensor([1, 0], dtype=torch.long),
                max_overlaps=torch.tensor([1.0, 0.0], dtype=torch.float32),
                labels=torch.tensor([0, -1], dtype=torch.long),
            )

    head = _build_head(assigner=RecordingAssigner())
    head.loss_heatmap = RecordingHeatmapLoss()
    outputs = {
        "heatmap": torch.zeros((1, 2, 2), dtype=torch.float32),
        "dense_heatmap": torch.full((1, 2, 4, 4), -2.19, dtype=torch.float32),
        "center": torch.zeros((1, 2, 2), dtype=torch.float32),
        "height": torch.zeros((1, 1, 2), dtype=torch.float32),
        "dim": torch.zeros((1, 3, 2), dtype=torch.float32),
        "rot": torch.tensor([[[0.0, 0.0], [1.0, 1.0]]], dtype=torch.float32),
        "vel": torch.zeros((1, 2, 2), dtype=torch.float32),
    }
    gt_boxes = [torch.tensor([[1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)]
    gt_labels = [torch.tensor([0], dtype=torch.long)]

    head.loss(outputs, gt_boxes, gt_labels)

    assert captured_prediction
    assert torch.allclose(captured_prediction[0], outputs["dense_heatmap"])


def test_transfusion_bbox_loss_normalizes_by_positive_count() -> None:
    class OnePositiveAssigner:
        def assign(self, bboxes, gt_bboxes, gt_labels, cls_pred, point_cloud_range):
            del bboxes, gt_bboxes, gt_labels, cls_pred, point_cloud_range
            return AssignResult(
                num_gts=1,
                gt_inds=torch.tensor([1, 0], dtype=torch.long),
                max_overlaps=torch.tensor([1.0, 0.0], dtype=torch.float32),
                labels=torch.tensor([0, -1], dtype=torch.long),
            )

    head = _build_head(assigner=OnePositiveAssigner())
    outputs = {
        "heatmap": torch.zeros((1, 2, 2), dtype=torch.float32),
        "dense_heatmap": torch.zeros((1, 2, 4, 4), dtype=torch.float32),
        "center": torch.zeros((1, 2, 2), dtype=torch.float32),
        "height": torch.zeros((1, 1, 2), dtype=torch.float32),
        "dim": torch.zeros((1, 3, 2), dtype=torch.float32),
        "rot": torch.zeros((1, 2, 2), dtype=torch.float32),
        "vel": torch.zeros((1, 2, 2), dtype=torch.float32),
    }
    gt_boxes = [torch.tensor([[1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)]
    gt_labels = [torch.tensor([0], dtype=torch.long)]

    losses = head.loss(outputs, gt_boxes, gt_labels)

    encoded_target = head.bbox_coder.encode(gt_boxes[0])[0]
    expected = (
        encoded_target.abs() * torch.tensor(head.code_weights)
    ).sum() * head.loss_bbox_weight
    assert torch.allclose(losses["layer_-1_loss_bbox"], expected)


def _heatmap_for_box(
    head: TransFusionHead, length: float, width: float, yaw: float
) -> torch.Tensor:
    """Build a single-box dense heatmap target and return its [H, W] class map."""
    grid = 24
    box = torch.tensor([[12.0, 12.0, 0.0, length, width, 2.0, yaw, 0.0, 0.0]], dtype=torch.float32)
    labels = torch.tensor([0], dtype=torch.long)
    heatmap = head._build_heatmap_targets([box], [labels], (grid, grid), torch.device("cpu"))
    return heatmap[0, 0]


def test_oriented_heatmap_spreads_along_box_length() -> None:
    head = _build_head(heatmap_target="oriented")
    heatmap = _heatmap_for_box(head, length=12.0, width=2.0, yaw=0.0)
    center_x, center_y = 12, 12
    along_length = heatmap[center_y, center_x + 3]
    across_width = heatmap[center_y + 3, center_x]
    assert along_length > 0.2
    assert across_width < 1e-2
    assert along_length > across_width


def test_oriented_heatmap_follows_yaw() -> None:
    head = _build_head(heatmap_target="oriented")
    heatmap = _heatmap_for_box(head, length=12.0, width=2.0, yaw=math.pi / 2)
    center_x, center_y = 12, 12
    along_x = heatmap[center_y, center_x + 3]
    along_y = heatmap[center_y + 3, center_x]
    # A 90 degree yaw rotates the long axis from x to y.
    assert along_y > 0.2
    assert along_x < 1e-2


def test_round_heatmap_is_isotropic_and_default() -> None:
    head = _build_head()
    assert head.heatmap_target == "round"
    heatmap = _heatmap_for_box(head, length=12.0, width=2.0, yaw=0.0)
    center_x, center_y = 12, 12
    # A round blob ignores orientation, so equal offsets are equal.
    assert torch.isclose(
        heatmap[center_y, center_x + 1], heatmap[center_y + 1, center_x], atol=1e-4
    )


def test_invalid_heatmap_target_raises() -> None:
    with pytest.raises(ValueError):
        _build_head(heatmap_target="square")
