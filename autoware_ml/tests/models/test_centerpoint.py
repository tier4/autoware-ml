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

"""Unit tests for the native CenterPoint detector."""

from __future__ import annotations

import math

import torch

from autoware_ml.models.detection3d.backbones.second import SECONDBackbone
from autoware_ml.models.detection3d.centerpoint import CenterPointDetectionModel
from autoware_ml.models.detection3d.encoders.pillar import PillarFeatureNet, PointPillarsScatter
from autoware_ml.models.detection3d.heads.centerpoint import CenterHead
from autoware_ml.models.detection3d.necks.second_fpn import SECONDFPN


def _build_model(use_velocity: bool = True) -> CenterPointDetectionModel:
    return CenterPointDetectionModel(
        pts_voxel_encoder=PillarFeatureNet(
            in_channels=5,
            feat_channels=[32, 32],
            voxel_size=[0.5, 0.5, 4.0],
            point_cloud_range=[0.0, 0.0, -2.0, 8.0, 8.0, 2.0],
        ),
        pts_middle_encoder=PointPillarsScatter(in_channels=32, output_shape=[16, 16]),
        pts_backbone=SECONDBackbone(
            in_channels=32,
            out_channels=[64, 128, 256],
            layer_nums=[1, 1, 1],
            layer_strides=[2, 2, 2],
        ),
        pts_neck=SECONDFPN(
            in_channels=[64, 128, 256],
            out_channels=[128, 128, 128],
            upsample_strides=[1, 2, 4],
        ),
        bbox_head=CenterHead(
            in_channels=384,
            num_classes=2,
            shared_channels=64,
            point_cloud_range=[0.0, 0.0, -2.0, 8.0, 8.0, 2.0],
            voxel_size=[0.5, 0.5, 4.0],
            out_size_factor=2,
            max_objs=16,
            min_radius=1,
            score_threshold=0.1,
            post_max_size=10,
            nms_min_radius=1.0,
            use_velocity=use_velocity,
        ),
    )


class TestCenterPointTargets:
    def test_build_targets_populates_heatmap_and_boxes(self) -> None:
        model = _build_model()
        gt_boxes = [
            torch.tensor([[2.0, 3.0, 0.2, 4.0, 1.6, 1.5, 0.25, 0.5, -0.1]], dtype=torch.float32)
        ]
        gt_labels = [torch.tensor([0], dtype=torch.long)]

        targets = model.bbox_head.get_targets(
            gt_boxes,
            gt_labels,
            feature_map_size=(4, 4),
            device=torch.device("cpu"),
        )

        assert targets.heatmap.shape == (1, 2, 4, 4)
        assert targets.mask[0, 0].item() is True
        assert targets.indices[0, 0].item() == 14
        assert targets.heatmap[0, 0, 3, 2].item() == 1.0
        assert torch.allclose(
            targets.anno_boxes[0, 0, 3:6],
            torch.tensor([4.0, 1.6, 1.5]).log(),
        )
        assert torch.allclose(
            targets.anno_boxes[0, 0, 6:8],
            torch.tensor([math.sin(0.25), math.cos(0.25)]),
        )
        assert torch.allclose(targets.anno_boxes[0, 0, 8:], torch.tensor([0.5, -0.1]))

    def test_predict_returns_length_width_height_after_unified_dim_order(self) -> None:
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
        outputs = {
            "heatmap": torch.full((1, 2, 4, 4), -20.0),
            "reg": torch.zeros((1, 2, 4, 4)),
            "height": torch.zeros((1, 1, 4, 4)),
            "dim": torch.zeros((1, 3, 4, 4)),
            "rot": torch.zeros((1, 2, 4, 4)),
        }
        outputs["heatmap"][0, 0, 3, 2] = 20.0
        outputs["height"][0, 0, 3, 2] = 0.2
        outputs["dim"][0, :, 3, 2] = torch.tensor([4.0, 1.6, 1.5]).log()
        outputs["rot"][0, 1, 3, 2] = 1.0

        predictions = head.predict(outputs)

        assert predictions[0]["bboxes_3d"].shape == (1, 7)
        assert torch.allclose(
            predictions[0]["bboxes_3d"][0, 3:6],
            torch.tensor([4.0, 1.6, 1.5]),
        )

    def test_centerpoint_loss_and_predict_run(self) -> None:
        model = _build_model()
        voxels = torch.randn(12, 5, 5)
        num_points = torch.randint(1, 5, (12,), dtype=torch.int32)
        voxel_coords = torch.randint(0, 8, (12, 4), dtype=torch.int32)
        voxel_coords[:, 0] = 0
        outputs = model(voxels=voxels, num_points=num_points, voxel_coords=voxel_coords)
        gt_boxes = [
            torch.tensor([[2.0, 3.0, 0.2, 4.0, 1.6, 1.5, 0.25, 0.5, -0.1]], dtype=torch.float32)
        ]
        gt_labels = [torch.tensor([0], dtype=torch.long)]

        metrics = model.compute_metrics({"gt_boxes": gt_boxes, "gt_labels": gt_labels}, outputs)
        predictions = model.bbox_head.predict(outputs)

        assert "loss" in metrics
        assert outputs["heatmap"].shape[:2] == (1, 2)
        assert isinstance(predictions, list)
        assert set(predictions[0]) == {"bboxes_3d", "scores_3d", "labels_3d"}

    def test_centerpoint_builds_split_deployment_specs(self) -> None:
        model = _build_model().eval()
        voxels = torch.randn(12, 5, 5)
        num_points = torch.randint(1, 5, (12,), dtype=torch.int32)
        voxel_coords = torch.randint(0, 8, (12, 4), dtype=torch.int32)
        voxel_coords[:, 0] = 0

        specs = model.build_export_specs(
            {"voxels": voxels, "num_points": num_points, "voxel_coords": voxel_coords}
        )

        assert list(specs) == [
            "pts_voxel_encoder_centerpoint",
            "pts_backbone_neck_head_centerpoint",
        ]
        voxel_spec = specs["pts_voxel_encoder_centerpoint"]
        assert voxel_spec.input_param_names == ["input_features"]
        assert voxel_spec.output_names == ["pillar_features"]
        assert voxel_spec.args[0].shape == (12, 5, 11)
        pillar_features = voxel_spec.module(*voxel_spec.args)
        assert pillar_features.shape == (12, 1, 32)

        head_spec = specs["pts_backbone_neck_head_centerpoint"]
        assert head_spec.input_param_names == ["spatial_features"]
        assert head_spec.output_names == ["heatmap", "reg", "height", "dim", "rot", "vel"]
        outputs = head_spec.module(*head_spec.args)
        assert [output.shape[1] for output in outputs] == [2, 2, 1, 3, 2, 2]

    def test_export_specs_follow_head_velocity_configuration(self) -> None:
        model = _build_model(use_velocity=False).eval()
        voxels = torch.randn(12, 5, 5)
        num_points = torch.randint(1, 5, (12,), dtype=torch.int32)
        voxel_coords = torch.randint(0, 8, (12, 4), dtype=torch.int32)
        voxel_coords[:, 0] = 0

        specs = model.build_export_specs(
            {"voxels": voxels, "num_points": num_points, "voxel_coords": voxel_coords}
        )

        head_spec = specs["pts_backbone_neck_head_centerpoint"]
        assert head_spec.output_names == ["heatmap", "reg", "height", "dim", "rot"]
        outputs = head_spec.module(*head_spec.args)
        assert [output.shape[1] for output in outputs] == [2, 2, 1, 3, 2]

    def test_pillar_decoration_matches_runtime_feature_layout(self) -> None:
        # Pin the 11-feature layout generated by the autoware_lidar_centerpoint
        # preprocess kernel for XYZIT models:
        # [x, y, z, i, t, cluster-offset xyz, voxel-center-offset xyz].
        encoder = PillarFeatureNet(
            in_channels=5,
            feat_channels=[8],
            voxel_size=[0.5, 0.5, 4.0],
            point_cloud_range=[0.0, 0.0, -2.0, 8.0, 8.0, 2.0],
        )
        voxels = torch.tensor([[[1.25, 0.9, -0.5, 7.0, 0.05]]], dtype=torch.float32)
        num_points = torch.tensor([1], dtype=torch.int32)
        coords = torch.tensor([[0, 0, 1, 2]], dtype=torch.int32)  # (batch, z, y, x)

        decorated = encoder.decorate(voxels, num_points, coords)

        expected = torch.tensor([1.25, 0.9, -0.5, 7.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.15, -0.5])
        assert decorated.shape == (1, 1, 11)
        assert torch.allclose(decorated[0, 0], expected, atol=1e-6)

    def test_point_pillars_scatter_places_features_at_expected_cells(self) -> None:
        scatter = PointPillarsScatter(in_channels=2, output_shape=[3, 4])
        pillar_features = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        coords = torch.tensor(
            [
                [0, 0, 1, 2],
                [0, 0, 2, 1],
            ],
            dtype=torch.int32,
        )

        canvas = scatter(pillar_features, coords, batch_size=1)

        assert canvas.shape == (1, 2, 3, 4)
        assert torch.allclose(canvas[0, :, 1, 2], pillar_features[0])
        assert torch.allclose(canvas[0, :, 2, 1], pillar_features[1])

    def test_point_pillars_scatter_handles_empty_inputs(self) -> None:
        scatter = PointPillarsScatter(in_channels=2, output_shape=[2, 2])
        pillar_features = torch.zeros((0, 2), dtype=torch.float32)
        coords = torch.zeros((0, 4), dtype=torch.int32)

        canvas = scatter(pillar_features, coords, batch_size=1)

        assert canvas.shape == (1, 2, 2, 2)
        assert torch.count_nonzero(canvas).item() == 0


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
