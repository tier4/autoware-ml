"""Shared test fixtures for PTv3 detection models."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from autoware_ml.models.detection3d.heads.transfusion import TransFusionHead
from autoware_ml.models.detection3d.task_modules.assigners import HungarianAssigner3D
from autoware_ml.models.detection3d.task_modules.bbox_coders import TransFusionBBoxCoder
from autoware_ml.models.detection3d.task_modules.match_costs import (
    BBoxBEVL1Cost,
    ClassificationCost,
    IoU3DCost,
)
from autoware_ml.models.detection3d.ptv3 import (
    PTv3BEVEncoder,
    PTv3BEVProjection,
    PTv3DetectionModel,
)
from autoware_ml.models.segmentation3d.backbones.ptv3 import PointTransformerV3Backbone


def build_ptv3_backbone() -> PointTransformerV3Backbone:
    """Return a small PTv3 backbone suitable for unit tests."""
    return PointTransformerV3Backbone(
        in_channels=4,
        order=("z",),
        stride=(2,),
        enc_depths=(1, 1),
        enc_channels=(8, 16),
        enc_num_head=(1, 2),
        enc_patch_size=(4, 4),
        dec_depths=(1,),
        dec_channels=(8,),
        dec_num_head=(1,),
        dec_patch_size=(4,),
        mlp_ratio=2.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        pre_norm=True,
        shuffle_orders=False,
        enable_rpe=False,
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
    )


def build_bev_encoder() -> PTv3BEVEncoder:
    """Return a lightweight BEV encoder for PTv3 detection tests."""
    return PTv3BEVEncoder(
        in_channels=16,
        hidden_channels=32,
        out_channels=64,
        dilations=(1, 2, 1),
    )


def build_trans_model(
    freeze_backbone: bool = False,
) -> PTv3DetectionModel:
    """Return a PTv3 + TransFusionHead detection model for tests."""
    return PTv3DetectionModel(
        backbone=build_ptv3_backbone(),
        bev_projector=PTv3BEVProjection(
            in_channels=8,
            out_channels=16,
            output_shape=[8, 8],
            bev_stride=1,
        ),
        bev_encoder=build_bev_encoder(),
        bbox_head=TransFusionHead(
            num_proposals=8,
            auxiliary=False,
            in_channels=64,
            hidden_channel=32,
            num_classes=2,
            num_decoder_layers=1,
            num_heads=4,
            feedforward_channels=64,
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
                post_center_range=[-1.0, -1.0, -5.0, 10.0, 10.0, 5.0],
                code_size=10,
            ),
            assigner=HungarianAssigner3D(
                cls_cost=ClassificationCost(weight=0.15),
                reg_cost=BBoxBEVL1Cost(weight=0.25),
                iou_cost=IoU3DCost(weight=0.25),
            ),
            point_cloud_range=[0.0, 0.0, -2.0, 8.0, 8.0, 2.0],
            voxel_size=[1.0, 1.0, 4.0],
            out_size_factor=1,
            code_weights=[1.0] * 8 + [0.2, 0.2],
            min_radius=1,
            gaussian_overlap=0.1,
            score_threshold=0.1,
            post_max_size=10,
            nms_min_radius=1.0,
        ),
        export_output_names=[
            "dense_heatmap",
            "query_heatmap_score",
            "query_labels",
            "heatmap",
            "center",
            "height",
            "dim",
            "rot",
            "vel",
        ],
        freeze_backbone=freeze_backbone,
        grid_size=1.0,
        point_cloud_range=[0.0, 0.0, -2.0, 8.0, 8.0, 2.0],
        optimizer=lambda params: torch.optim.AdamW(params, lr=1e-3),
    )


def build_inputs() -> dict[str, torch.Tensor]:
    """Return one small PTv3 detection input batch."""
    coord = torch.tensor(
        [
            [0.2, 0.5, 0.0],
            [1.1, 1.3, 0.2],
            [2.0, 1.5, 0.4],
            [3.2, 2.1, 0.1],
            [4.4, 3.0, 0.0],
            [5.1, 4.4, 0.3],
            [6.5, 5.0, 0.2],
            [7.0, 6.1, 0.1],
        ],
        dtype=torch.float32,
    )
    feat = torch.cat([coord, torch.linspace(0.1, 0.8, steps=coord.shape[0]).unsqueeze(1)], dim=1)
    grid_coord = coord.floor().to(dtype=torch.int32)
    grid_coord[:, 2] += 2
    offset = torch.tensor([coord.shape[0]], dtype=torch.long)
    return {"coord": coord, "feat": feat, "grid_coord": grid_coord, "offset": offset}


def build_targets() -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Return one simple detection target batch."""
    gt_boxes = [
        torch.tensor(
            [[2.5, 3.0, 0.2, 1.8, 0.9, 1.6, 0.1, 0.0, 0.0]],
            dtype=torch.float32,
        )
    ]
    gt_labels = [torch.tensor([0], dtype=torch.long)]
    return gt_boxes, gt_labels


def move_batch_to_device(
    batch: Mapping[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Copy a PTv3 input batch onto one device."""
    return {name: value.to(device) for name, value in batch.items()}


def move_targets_to_device(
    gt_boxes: list[torch.Tensor],
    gt_labels: list[torch.Tensor],
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Copy detection targets onto one device."""
    return (
        [boxes.to(device) for boxes in gt_boxes],
        [labels.to(device) for labels in gt_labels],
    )
