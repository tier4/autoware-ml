"""Tests for reusable detection3d task modules."""

from __future__ import annotations

import torch

from autoware_ml.models.detection3d.task_modules.assigners import HungarianAssigner3D
from autoware_ml.models.detection3d.task_modules.bbox_coders import TransFusionBBoxCoder
from autoware_ml.models.detection3d.task_modules.match_costs import (
    BBoxBEVL1Cost,
    ClassificationCost,
    IoU3DCost,
)


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
