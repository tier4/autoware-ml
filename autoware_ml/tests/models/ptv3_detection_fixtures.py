"""Shared test fixtures for PTv3 task models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from autoware_ml.models.detection3d.heads.transfusion import TransFusionHead
from autoware_ml.models.detection3d.ptv3 import (
    PTv3BEVEncoder,
    PTv3BEVProjection,
    PTv3DetBEVNeck,
    PTv3DetectionModel,
    PTv3DetFeatureFusion,
)
from autoware_ml.models.detection3d.task_modules.assigners import HungarianAssigner3D
from autoware_ml.models.detection3d.task_modules.bbox_coders import TransFusionBBoxCoder
from autoware_ml.models.detection3d.task_modules.match_costs import (
    BBoxBEVL1Cost,
    ClassificationCost,
    IoU3DCost,
)
from autoware_ml.models.segmentation3d.encoders.ptv3 import (
    LitePTEncoder,
    PointTransformerV3Encoder,
)
from autoware_ml.models.segmentation3d.heads.ptv3 import PTv3SegDecoderHead
from autoware_ml.models.segmentation3d.ptv3 import PTv3SegmentationModel
from autoware_ml.preprocessing.detection3d.point_pillar import PointPillarPreprocessor


def build_ptv3_encoder() -> PointTransformerV3Encoder:
    """Return a small PTv3 encoder suitable for unit tests."""
    return PointTransformerV3Encoder(
        in_channels=5,
        order=("z",),
        stride=(2,),
        enc_depths=(1, 1),
        enc_channels=(8, 16),
        enc_num_head=(1, 2),
        enc_patch_size=(4, 4),
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


def build_seg_head(num_classes: int = 3, dec_depths: Sequence[int] = (1,)) -> PTv3SegDecoderHead:
    """Return a small PTv3 segmentation decoder head matching the test encoder."""
    return PTv3SegDecoderHead(
        num_classes=num_classes,
        ignore_index=-1,
        order=("z",),
        enc_channels=(8, 16),
        dec_depths=dec_depths,
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
        enable_rpe=False,
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
    )


def build_seg_model() -> PTv3SegmentationModel:
    """Return a small PTv3 segmentation model for tests."""
    return PTv3SegmentationModel(
        encoder=build_ptv3_encoder(),
        seg3d_head=build_seg_head(),
        optimizer=lambda params: torch.optim.AdamW(params, lr=1e-3),
        grid_size=1.0,
        time_lag_dim=4,
        point_cloud_range=[0.0, 0.0, -2.0, 8.0, 8.0, 2.0],
    )


def build_bev_encoder() -> PTv3BEVEncoder:
    """Return a lightweight BEV encoder for PTv3 detection tests."""
    return PTv3BEVEncoder(
        in_channels=16,
        hidden_channels=32,
        out_channels=64,
        dilations=(1, 2, 1),
    )


def build_bev_neck() -> PTv3DetBEVNeck:
    """Return a lightweight detection BEV neck matching the test encoder."""
    return PTv3DetBEVNeck(
        fusion=PTv3DetFeatureFusion(in_channels=16, skip_channels=8, out_channels=16),
        bev_projector=PTv3BEVProjection(
            in_channels=16,
            out_channels=16,
            output_shape=[8, 8],
        ),
        bev_encoder=build_bev_encoder(),
    )


def build_transfusion_head() -> TransFusionHead:
    """Return a lightweight TransFusion head for PTv3 tests."""
    return TransFusionHead(
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
    )


def build_trans_model(
    freeze_encoder: bool = False,
) -> PTv3DetectionModel:
    """Return a PTv3 + TransFusionHead detection model for tests."""
    return PTv3DetectionModel(
        encoder=build_ptv3_encoder(),
        bev_neck=build_bev_neck(),
        bbox_head=build_transfusion_head(),
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
        freeze_encoder=freeze_encoder,
        grid_size=1.0,
        point_cloud_range=[0.0, 0.0, -2.0, 8.0, 8.0, 2.0],
        optimizer=lambda params: torch.optim.AdamW(params, lr=1e-3),
    )


POINT_CLOUD_RANGE = [0.0, 0.0, -2.0, 8.0, 8.0, 2.0]


def build_preprocessor() -> PointPillarPreprocessor:
    """Return the voxelizer matching the test geometry (1 m voxels, 8 m range)."""
    return PointPillarPreprocessor(
        voxel_size=[1.0, 1.0, 1.0],
        point_cloud_range=POINT_CLOUD_RANGE,
        max_num_points=4,
        max_voxels=64,
        eval_max_voxels=64,
        voxelization_z_order_first=False,
    )


def build_points() -> torch.Tensor:
    """Return one small ``(x, y, z, intensity, time_lag)`` point cloud.

    Two points share the first voxel and the last point comes from an earlier
    sweep (positive time lag).
    """
    coord = torch.tensor(
        [
            [0.2, 0.5, 0.0],
            [0.4, 0.6, 0.1],
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
    intensity = torch.linspace(0.1, 0.9, steps=coord.shape[0]).unsqueeze(1)
    time_lag = torch.zeros((coord.shape[0], 1), dtype=torch.float32)
    time_lag[-1] = 0.1
    return torch.cat([coord, intensity, time_lag], dim=1)


def build_batch() -> dict[str, Any]:
    """Return one preprocessed single-frame PTv3 batch with segmentation targets."""
    points = build_points()
    segment = torch.arange(points.shape[0], dtype=torch.long) % 3
    segment[-1] = -1
    batch = {"points": [points], "segment": segment}
    return build_preprocessor()(batch, is_training=True)


def build_inputs() -> dict[str, torch.Tensor]:
    """Return the forward inputs of one preprocessed single-frame PTv3 batch."""
    batch = build_batch()
    return {key: batch[key] for key in ("voxels", "num_points", "voxel_coords")}


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
    batch: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Copy a PTv3 input batch onto one device."""
    return {
        name: [item.to(device) for item in value] if isinstance(value, list) else value.to(device)
        for name, value in batch.items()
    }


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


def build_litept_encoder() -> LitePTEncoder:
    """Return a small LitePT encoder exercising both gating directions.

    Stages 0 and 1 are convolution-only and stage 2 is attention-only, so the
    encoder covers a pooling stage that needs no serialization order (0) and one
    that does (1), with no base-level order at all.
    """
    return LitePTEncoder(
        in_channels=5,
        order=("z",),
        stride=(2, 2),
        enc_depths=(1, 1, 1),
        enc_channels=(12, 24, 24),
        enc_num_head=(1, 2, 2),
        enc_patch_size=(2, 2, 2),
        mlp_ratio=2.0,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        shuffle_orders=False,
        enable_flash=False,
        enc_conv=(True, True, False),
        enc_attn=(False, False, True),
        enc_rope_base=100.0,
    )


def build_litept_seg_head(num_classes: int = 3) -> PTv3SegDecoderHead:
    """Return a LitePT-style decoder head: unpooling only, no decoder blocks."""
    return PTv3SegDecoderHead(
        num_classes=num_classes,
        ignore_index=-1,
        order=("z",),
        enc_channels=(12, 24, 24),
        dec_depths=(0, 0),
        dec_channels=(12, 24),
        dec_num_head=(1, 2),
        dec_patch_size=(2, 2),
        mlp_ratio=2.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
    )


def build_litept_seg_model() -> PTv3SegmentationModel:
    """Return a LitePT segmentation model using the unchanged PTv3 task wrapper."""
    return PTv3SegmentationModel(
        encoder=build_litept_encoder(),
        seg3d_head=build_litept_seg_head(),
        optimizer=lambda params: torch.optim.AdamW(params, lr=1e-3),
        grid_size=1.0,
        time_lag_dim=4,
        point_cloud_range=[0.0, 0.0, -2.0, 8.0, 8.0, 2.0],
    )
