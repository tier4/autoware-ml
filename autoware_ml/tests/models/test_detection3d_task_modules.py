"""Tests for reusable detection3d task modules."""

from __future__ import annotations

import torch

from autoware_ml.models.detection3d.heads.transfusion import TransFusionHead
from autoware_ml.models.detection3d.task_modules.assigners import HungarianAssigner3D
from autoware_ml.models.detection3d.task_modules.bbox_coders import TransFusionBBoxCoder
from autoware_ml.models.detection3d.task_modules.match_costs import (
    BBoxBEVL1Cost,
    ClassificationCost,
    IoU3DCost,
)


def _build_transfusion_head(heatmap_target: str) -> TransFusionHead:
    return TransFusionHead(
        num_proposals=2,
        auxiliary=False,
        in_channels=4,
        hidden_channel=8,
        num_classes=1,
        num_decoder_layers=1,
        num_heads=1,
        feedforward_channels=8,
        common_heads={
            "center": (2, 1),
            "height": (1, 1),
            "dim": (3, 1),
            "rot": (2, 1),
            "vel": (2, 1),
        },
        bbox_coder=TransFusionBBoxCoder(
            pc_range=[0.0, 0.0],
            out_size_factor=1,
            voxel_size=[1.0, 1.0],
            post_center_range=[-20.0, -20.0, -5.0, 20.0, 20.0, 5.0],
            code_size=10,
        ),
        assigner=HungarianAssigner3D(
            cls_cost=ClassificationCost(weight=0.15),
            reg_cost=BBoxBEVL1Cost(weight=0.25),
            iou_cost=IoU3DCost(weight=0.25),
        ),
        point_cloud_range=[0.0, 0.0, -2.0, 16.0, 16.0, 2.0],
        voxel_size=[1.0, 1.0, 4.0],
        out_size_factor=1,
        code_weights=[1.0] * 8 + [0.2, 0.2],
        min_radius=1,
        gaussian_overlap=0.1,
        score_threshold=0.1,
        post_max_size=10,
        nms_min_radius=1.0,
        heatmap_target=heatmap_target,
    )


def test_transfusion_oriented_heatmap_target_runs_through_get_targets() -> None:
    head = _build_transfusion_head(heatmap_target="oriented")
    outputs = {
        "dense_heatmap": torch.zeros((1, 1, 16, 16), dtype=torch.float32),
        "heatmap": torch.zeros((1, 1, 2), dtype=torch.float32),
        "center": torch.zeros((1, 2, 2), dtype=torch.float32),
        "height": torch.zeros((1, 1, 2), dtype=torch.float32),
        "dim": torch.zeros((1, 3, 2), dtype=torch.float32),
        "rot": torch.zeros((1, 2, 2), dtype=torch.float32),
        "vel": torch.zeros((1, 2, 2), dtype=torch.float32),
    }
    gt_boxes = [
        torch.tensor([[8.0, 8.0, 0.0, 12.0, 2.0, 1.5, 0.0, 0.0, 0.0]], dtype=torch.float32)
    ]
    gt_labels = [torch.tensor([0], dtype=torch.long)]

    targets = head.get_targets(gt_boxes, gt_labels, outputs)
    heatmap = targets.heatmap[0, 0]

    assert heatmap[8, 8] == torch.tensor(1.0)
    assert heatmap[8, 12] > heatmap[12, 8]


def test_transfusion_bbox_coder_encode_decode_round_trip_with_full_geometry_vectors() -> None:
    coder = TransFusionBBoxCoder(
        pc_range=[-10.0, -20.0, -2.0, 10.0, 20.0, 4.0],
        out_size_factor=2,
        voxel_size=[0.5, 0.25, 0.2],
        post_center_range=[-1.0, -1.0, -5.0, 10.0, 10.0, 5.0],
        code_size=10,
    )
    boxes = torch.tensor([[2.0, 4.0, 1.0, 4.0, 2.0, 1.5, 0.25, 0.1, -0.2]], dtype=torch.float32)

    encoded = coder.encode(boxes)
    decoded = coder.decode(
        heatmap=torch.tensor([[[0.2], [0.8]]], dtype=torch.float32),
        rot=encoded[:, 6:8].T.unsqueeze(0),
        dim=encoded[:, 3:6].T.unsqueeze(0),
        center=encoded[:, 0:2].T.unsqueeze(0),
        height=encoded[:, 2:3].T.unsqueeze(0),
        vel=encoded[:, 8:10].T.unsqueeze(0),
    )[0]["bboxes"]

    assert encoded.shape == (1, 10)
    assert decoded.shape == (1, 9)
    assert torch.allclose(decoded[0, :6], boxes[0, :6], atol=1e-4)
    assert torch.allclose(decoded[0, 6:9], boxes[0, 6:9], atol=1e-4)


def test_hungarian_assigner_matches_best_query() -> None:
    assigner = HungarianAssigner3D(
        cls_cost=ClassificationCost(weight=0.15),
        reg_cost=BBoxBEVL1Cost(weight=0.25),
        iou_cost=IoU3DCost(weight=0.25),
    )
    bboxes = torch.tensor(
        [
            [2.0, 2.0, 0.5, 4.0, 2.0, 1.5, 0.0],
            [20.0, 20.0, 0.5, 4.0, 2.0, 1.5, 0.0],
        ],
        dtype=torch.float32,
    )
    gt_bboxes = torch.tensor([[2.1, 2.0, 0.5, 4.0, 2.0, 1.5, 0.0]], dtype=torch.float32)
    gt_labels = torch.tensor([1], dtype=torch.long)
    cls_pred = torch.tensor([[0.1, 4.0], [3.0, 0.1]], dtype=torch.float32)

    result = assigner.assign(
        bboxes=bboxes,
        gt_bboxes=gt_bboxes,
        gt_labels=gt_labels,
        cls_pred=cls_pred,
        point_cloud_range=[0.0, 0.0, -1.0, 40.0, 40.0, 3.0],
    )

    assert result.gt_inds.tolist() == [1, 0]
    assert result.labels.tolist() == [1, -1]
