# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for BEVFusion modules."""

from __future__ import annotations

import torch
import torch.nn as nn

from autoware_ml.models.common.backbones.resnet import ResNet18MultiScale
from autoware_ml.models.common.necks.lss_fpn import GeneralizedLSSFPN
from autoware_ml.models.detection3d.bevfusion import (
    BEVFusionDetectionModel,
    _BEVFusionImageBackboneExportWrapper,
    _export_detection_outputs,
    _runtime_coors_to_voxel_coords,
)
from autoware_ml.models.detection3d.fusion import ConvFuser
from autoware_ml.transforms.camera.resize import ResizeMultiviewImages
from autoware_ml.models.detection3d.view_transforms.depth_lss import DepthLSSTransform


class _IdentityHead(nn.Module):
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"bev": x}

    def loss(self, outputs, gt_boxes, gt_labels):
        return {"loss": outputs["bev"].sum() * 0.0}

    def predict(self, outputs):
        return outputs


def test_resnet18_multiscale_returns_three_feature_levels() -> None:
    model = ResNet18MultiScale(in_channels=3)
    x = torch.randn(2, 3, 128, 128)
    c3, c4, c5 = model(x)
    assert c3.shape == (2, 128, 16, 16)
    assert c4.shape == (2, 256, 8, 8)
    assert c5.shape == (2, 512, 4, 4)


def test_generalized_lss_fpn_projects_feature_pyramid() -> None:
    neck = GeneralizedLSSFPN(in_channels=[128, 256, 512], out_channels=64)
    outputs = neck(
        (
            torch.randn(2, 128, 16, 16),
            torch.randn(2, 256, 8, 8),
            torch.randn(2, 512, 4, 4),
        )
    )
    assert len(outputs) == 2
    assert outputs[0].shape == (2, 64, 16, 16)
    assert outputs[1].shape == (2, 64, 8, 8)


def test_depth_lss_transform_uses_bev_pool(monkeypatch) -> None:
    calls = {}

    def fake_bev_pool(feats, coords, ranks, batch_size, depth, height, width, is_training):
        calls["shape"] = (
            feats.shape,
            coords.shape,
            ranks.shape,
            batch_size,
            depth,
            height,
            width,
            is_training,
        )
        channels = feats.shape[1]
        return feats.new_zeros((batch_size, channels, depth, height, width))

    monkeypatch.setattr(
        "autoware_ml.models.detection3d.view_transforms.depth_lss.bev_pool", fake_bev_pool
    )

    transform = DepthLSSTransform(
        in_channels=32,
        out_channels=8,
        image_size=[64, 64],
        feature_size=[8, 8],
        xbound=[-8.0, 8.0, 1.0],
        ybound=[-8.0, 8.0, 1.0],
        zbound=[-5.0, 3.0, 8.0],
        dbound=[1.0, 5.0, 1.0],
        downsample=1,
    )

    image_features = torch.randn(2, 3, 32, 8, 8)
    points = [torch.rand(50, 4) * 8.0 for _ in range(2)]
    lidar2image = torch.eye(4).view(1, 1, 4, 4).repeat(2, 3, 1, 1)
    intrinsics = torch.eye(4).view(1, 1, 4, 4).repeat(2, 3, 1, 1)
    camera2lidar = torch.eye(4).view(1, 1, 4, 4).repeat(2, 3, 1, 1)
    img_aug = torch.eye(4).view(1, 1, 4, 4).repeat(2, 3, 1, 1)
    lidar_aug = torch.eye(4).view(1, 4, 4).repeat(2, 1, 1)

    bev = transform(
        image_features, points, lidar2image, intrinsics, camera2lidar, img_aug, lidar_aug
    )
    assert bev.shape[0] == 2
    assert bev.shape[1] == 8 * 1
    assert transform.nx == (16, 16, 1)
    assert "shape" in calls


def test_depth_lss_transform_supports_precomputed_pool_metadata(monkeypatch) -> None:
    calls = {}

    def fake_bev_pool(feats, coords, ranks, batch_size, depth, height, width, is_training):
        calls["coords"] = coords
        channels = feats.shape[1]
        return feats.new_zeros((batch_size, channels, depth, height, width))

    monkeypatch.setattr(
        "autoware_ml.models.detection3d.view_transforms.depth_lss.bev_pool", fake_bev_pool
    )

    transform = DepthLSSTransform(
        in_channels=16,
        out_channels=4,
        image_size=[32, 32],
        feature_size=[4, 4],
        xbound=[-4.0, 4.0, 1.0],
        ybound=[-4.0, 4.0, 1.0],
        zbound=[-2.0, 2.0, 4.0],
        dbound=[1.0, 3.0, 1.0],
        downsample=1,
    )

    image_features = torch.randn(1, 2, 16, 4, 4)
    points = [torch.rand(30, 4) * 4.0]
    lidar2image = torch.eye(4).view(1, 1, 4, 4).repeat(1, 2, 1, 1)
    intrinsics = torch.eye(4).view(1, 1, 4, 4).repeat(1, 2, 1, 1)
    camera2lidar = torch.eye(4).view(1, 1, 4, 4).repeat(1, 2, 1, 1)
    img_aug = torch.eye(4).view(1, 1, 4, 4).repeat(1, 2, 1, 1)
    lidar_aug = torch.eye(4).view(1, 4, 4)

    geom = transform.camera_to_lidar_geometry(camera2lidar, intrinsics, lidar_aug, img_aug)
    geom_feats, kept, ranks, indices = transform.bev_pool_aux(geom)
    bev = transform.forward_precomputed(
        image_features, points, lidar2image, img_aug, geom_feats, kept, ranks, indices
    )

    assert bev.shape == (1, 4, 8, 8)
    assert calls["coords"].shape[1] == 4


def test_resize_multiview_images_updates_intrinsics() -> None:
    transform = ResizeMultiviewImages(target_size=[4, 8])
    input_dict = {
        "img": torch.ones(2, 3, 2, 4).numpy(),
        "camera_intrinsics": torch.eye(4).view(1, 4, 4).repeat(2, 1, 1).numpy(),
        "lidar2cam": torch.eye(4).view(1, 4, 4).repeat(2, 1, 1).numpy(),
        "lidar2img": torch.eye(4).view(1, 4, 4).repeat(2, 1, 1).numpy(),
    }

    output = transform(input_dict)

    assert output["img"].shape == (2, 3, 4, 8)
    assert output["camera_intrinsics"][0, 0, 0] == 2.0
    assert output["camera_intrinsics"][0, 1, 1] == 2.0


def test_bevfusion_model_fuses_camera_and_lidar_branches() -> None:
    class FakeVoxelEncoder(nn.Module):
        def forward(self, voxels, num_points, voxel_coords):
            return voxels.mean(dim=1)

    class FakeMiddleEncoder(nn.Module):
        def forward(self, features, coords, batch_size):
            return features.new_ones((batch_size, 64, 16, 16))

    class FakeImageBackbone(nn.Module):
        def forward(self, x):
            batch = x.shape[0]
            return (
                x.new_ones((batch, 128, 16, 16)),
                x.new_ones((batch, 256, 8, 8)),
                x.new_ones((batch, 512, 4, 4)),
            )

    class FakeImageNeck(nn.Module):
        def forward(self, features):
            batch = features[0].shape[0]
            return (features[0].new_ones((batch, 128, 8, 8)),)

    class FakeViewTransform(nn.Module):
        def forward(
            self,
            x,
            points,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            geom_feats_precomputed=None,
        ):
            del (
                points,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                geom_feats_precomputed,
            )
            batch = x.shape[0]
            return x.new_ones((batch, 80, 16, 16))

    model = BEVFusionDetectionModel(
        pts_voxel_encoder=FakeVoxelEncoder(),
        pts_middle_encoder=FakeMiddleEncoder(),
        pts_backbone=None,
        pts_neck=None,
        bbox_head=_IdentityHead(),
        img_backbone=FakeImageBackbone(),
        img_neck=FakeImageNeck(),
        view_transform=FakeViewTransform(),
        fusion_layer=ConvFuser(in_channels=[80, 64], out_channels=64),
    )

    batch_size = 2
    outputs = model(
        voxels=torch.randn(8, 5, 5),
        num_points=torch.ones(8, dtype=torch.int32),
        voxel_coords=torch.tensor([[0, 0, 0, 0], [1, 0, 0, 0]], dtype=torch.int32),
        img=[torch.randn(6, 3, 64, 64) for _ in range(batch_size)],
        points=[torch.rand(40, 4) for _ in range(batch_size)],
        lidar2img=[torch.eye(4).repeat(6, 1, 1) for _ in range(batch_size)],
        camera_intrinsics=[torch.eye(4).repeat(6, 1, 1) for _ in range(batch_size)],
        lidar2cam=[torch.eye(4).repeat(6, 1, 1) for _ in range(batch_size)],
    )
    assert outputs["bev"].shape == (batch_size, 64, 16, 16)


def test_bevfusion_validates_static_geometry_contract() -> None:
    class FakeMiddleEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.output_shape = (16, 16)

        def forward(self, features, coords, batch_size):
            return features.new_ones((batch_size, 64, 16, 16))

    class FakeViewTransform(nn.Module):
        expected_bev_shape = (8, 8)

    try:
        BEVFusionDetectionModel(
            pts_voxel_encoder=nn.Identity(),
            pts_middle_encoder=FakeMiddleEncoder(),
            pts_backbone=None,
            pts_neck=None,
            bbox_head=_IdentityHead(),
            img_backbone=nn.Identity(),
            img_neck=nn.Identity(),
            view_transform=FakeViewTransform(),
            fusion_layer=ConvFuser(in_channels=[80, 64], out_channels=64),
        )
    except ValueError as error:
        assert "must share the same BEV shape" in str(error)
    else:
        raise AssertionError("Expected BEV geometry validation to reject mismatched shapes.")


def test_bevfusion_validates_runtime_bev_shapes() -> None:
    class FakeViewTransform(nn.Module):
        def forward(
            self,
            x,
            points,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            geom_feats_precomputed=None,
        ):
            del (
                points,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                geom_feats_precomputed,
            )
            return x.new_ones((x.shape[0], 80, 8, 8))

    model = BEVFusionDetectionModel(
        pts_voxel_encoder=None,
        pts_middle_encoder=None,
        pts_backbone=None,
        pts_neck=None,
        bbox_head=_IdentityHead(),
        img_backbone=None,
        img_neck=None,
        view_transform=None,
        fusion_layer=ConvFuser(in_channels=[80, 64], out_channels=64),
    )

    try:
        model._validate_runtime_bev_shapes([torch.ones(1, 64, 16, 16), torch.ones(1, 80, 8, 8)])
    except ValueError as error:
        assert "must share the same runtime BEV shape" in str(error)
    else:
        raise AssertionError(
            "Expected runtime BEV geometry validation to reject mismatched shapes."
        )


def test_runtime_coors_conversion_keeps_zyx_order_and_adds_batch_column() -> None:
    coors = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)

    voxel_coords = _runtime_coors_to_voxel_coords(coors)

    assert voxel_coords.dtype == torch.int32
    assert voxel_coords.tolist() == [[0, 1, 2, 3], [0, 4, 5, 6]]


def test_first_sample_voxel_inputs_round_trip_to_internal_layout() -> None:
    batch_inputs_dict = {
        "voxel_coords": torch.tensor([[0, 1, 2, 3], [0, 4, 5, 6], [1, 7, 8, 9]], dtype=torch.int64),
        "voxels": torch.arange(3 * 2 * 5, dtype=torch.float32).view(3, 2, 5),
        "num_points": torch.tensor([2, 1, 2], dtype=torch.int64),
    }

    voxels, coors, num_points = BEVFusionDetectionModel._first_sample_voxel_inputs(
        batch_inputs_dict
    )

    assert coors.tolist() == [[1, 2, 3], [4, 5, 6]]
    assert voxels.shape[0] == 2
    assert num_points.tolist() == [2, 1]
    assert _runtime_coors_to_voxel_coords(coors).tolist() == [[0, 1, 2, 3], [0, 4, 5, 6]]


def test_export_detection_outputs_packs_runtime_tensors() -> None:
    class FakeHead(nn.Module):
        num_proposals = 4
        num_classes = 3

    num_proposals, num_classes = 4, 3
    query_labels = torch.tensor([[0, 2, 1, 0]])
    outputs = {
        "query_labels": query_labels,
        "heatmap": torch.randn(1, num_classes, num_proposals),
        "query_heatmap_score": torch.rand(1, num_classes, num_proposals),
        "center": torch.randn(1, 2, num_proposals),
        "height": torch.randn(1, 1, num_proposals),
        "dim": torch.randn(1, 3, num_proposals),
        "rot": torch.randn(1, 2, num_proposals),
        "vel": torch.randn(1, 2, num_proposals),
    }

    bbox_pred, score, label_pred = _export_detection_outputs(FakeHead(), outputs)

    assert bbox_pred.shape == (10, num_proposals)
    assert torch.equal(
        bbox_pred,
        torch.cat([outputs[key][0] for key in ("center", "height", "dim", "rot", "vel")], dim=0),
    )
    assert label_pred.dtype == torch.int64
    assert torch.equal(label_pred, query_labels[0])
    expected_score = (outputs["heatmap"].sigmoid() * outputs["query_heatmap_score"])[0].gather(
        0, query_labels
    )[0]
    assert torch.allclose(score, expected_score)


def test_export_detection_outputs_requires_velocity_branch() -> None:
    class FakeHead(nn.Module):
        num_proposals = 2
        num_classes = 1

    outputs = {
        "query_labels": torch.zeros(1, 2, dtype=torch.int64),
        "heatmap": torch.randn(1, 1, 2),
        "query_heatmap_score": torch.rand(1, 1, 2),
        "center": torch.randn(1, 2, 2),
        "height": torch.randn(1, 1, 2),
        "dim": torch.randn(1, 3, 2),
        "rot": torch.randn(1, 2, 2),
    }

    try:
        _export_detection_outputs(FakeHead(), outputs)
    except ValueError as error:
        assert "velocity branch" in str(error)
    else:
        raise AssertionError("Expected missing velocity branch to fail loudly.")


def test_image_backbone_export_wrapper_normalizes_uint8_images() -> None:
    received = {}

    class FakeImageBackbone(nn.Module):
        def forward(self, x):
            received["images"] = x
            return (x.new_ones((x.shape[0], 8, 4, 4)),)

    class FakeImageNeck(nn.Module):
        def forward(self, features):
            return (features[0],)

    model = BEVFusionDetectionModel(
        pts_voxel_encoder=None,
        pts_middle_encoder=None,
        pts_backbone=None,
        pts_neck=None,
        bbox_head=_IdentityHead(),
        img_backbone=FakeImageBackbone(),
        img_neck=FakeImageNeck(),
        view_transform=None,
        fusion_layer=None,
    )

    imgs = torch.full((6, 3, 32, 32), 255, dtype=torch.uint8)
    image_feats = _BEVFusionImageBackboneExportWrapper(model)(imgs)

    assert image_feats.shape == (6, 8, 4, 4)
    assert received["images"].dtype == torch.float32
    assert torch.allclose(received["images"].max(), torch.tensor(1.0))
