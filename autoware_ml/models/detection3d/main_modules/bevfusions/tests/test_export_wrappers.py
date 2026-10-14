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

"""Unit tests for the BEVFusion export output packing."""

from __future__ import annotations

from types import MappingProxyType
import unittest

from jaxtyping import Float32
import torch
import torch.nn as nn

from autoware_ml.dataclasses.models.detection3d.head_outputs import (
    TransFusionHeadOutputs,
    TransFusionSeparateHeadOutputs,
)
from autoware_ml.models.detection3d.heads.transfusions.transfusion_head import (
    ScoreThresholdConfig,
    TransFusionHead,
)
from autoware_ml.models.detection3d.main_modules.bevfusions.export_wrappers import (
    export_detection_outputs,
)
from autoware_ml.models.detection3d.task_modules.assigners import HungarianAssigner3D
from autoware_ml.models.detection3d.task_modules.bbox_coders import TransFusionBBoxCoder
from autoware_ml.models.detection3d.task_modules.match_costs import (
    BBoxBEVL1Cost,
    ClassificationCost,
    IoU3DCost,
)


class _FakeHead(nn.Module):
    """Head stand-in exposing only the attributes the export packing reads."""

    def __init__(self, num_proposals: int, num_classes: int) -> None:
        super().__init__()
        self.num_proposals = num_proposals
        self.num_classes = num_classes


class _ExportDetectionOutputsTestCase(unittest.TestCase):
    """Shared proposal layout and helpers for the export packing test cases."""

    def setUp(self) -> None:
        """Set up the device and the class and proposal layout shared by every test."""
        torch.manual_seed(0)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.class_names = ["car", "pedestrian", "cyclist"]
        self.num_classes = len(self.class_names)
        self.num_proposals = 4
        self.num_box_channels = 10

    def _build_outputs(
        self,
        batch_size: int,
        num_decoder_layers: int,
        heatmap_size: tuple[int, int] = (6, 6),
        with_velocity: bool = True,
    ) -> TransFusionHeadOutputs:
        """Build random head outputs in the layout the TransFusion head produces.

        With more than one decoder layer the separate head tensors carry ``num_decoder_layers *
        num_proposals`` columns, mirroring an auxiliary head that concatenates every layer along
        the proposal axis. The dense heatmap keeps its own ``(H, W)`` grid.
        """
        num_queries = num_decoder_layers * self.num_proposals
        separate_head_outputs = TransFusionSeparateHeadOutputs(
            heatmap=torch.randn(batch_size, self.num_classes, num_queries, device=self.device),
            center=torch.randn(batch_size, 2, num_queries, device=self.device),
            height=torch.randn(batch_size, 1, num_queries, device=self.device),
            dim=torch.randn(batch_size, 3, num_queries, device=self.device),
            rot=torch.randn(batch_size, 2, num_queries, device=self.device),
            vel=torch.randn(batch_size, 2, num_queries, device=self.device)
            if with_velocity
            else None,
        )
        return TransFusionHeadOutputs(
            dense_heatmap=torch.randn(
                batch_size, self.num_classes, *heatmap_size, device=self.device
            ),
            query_heatmap_score=torch.rand(
                batch_size, self.num_classes, self.num_proposals, device=self.device
            ),
            query_labels=torch.randint(
                0, self.num_classes, (batch_size, self.num_proposals), device=self.device
            ),
            separate_head_outputs=separate_head_outputs,
        )

    def _expected_bbox_pred(
        self, separate_head_outputs: TransFusionSeparateHeadOutputs
    ) -> Float32[torch.Tensor, "num_box_channels num_proposals"]:
        """Stack the last-layer regression channels of sample 0 in the runtime order."""
        assert separate_head_outputs.vel is not None
        return torch.cat(
            [
                separate_head_outputs.center[0, :, -self.num_proposals :],
                separate_head_outputs.height[0, :, -self.num_proposals :],
                separate_head_outputs.dim[0, :, -self.num_proposals :],
                separate_head_outputs.rot[0, :, -self.num_proposals :],
                separate_head_outputs.vel[0, :, -self.num_proposals :],
            ],
            dim=0,
        )

    def _expected_score(
        self, outputs: TransFusionHeadOutputs
    ) -> Float32[torch.Tensor, " num_proposals"]:
        """Score sample 0 as the predicted-class heatmap weighted by the query heatmap score."""
        class_scores = (
            outputs.separate_head_outputs.heatmap[0, :, -self.num_proposals :].sigmoid()
            * outputs.query_heatmap_score[0]
        )
        return class_scores.gather(0, outputs.query_labels[:1])[0]


class TestExportDetectionOutputs(_ExportDetectionOutputsTestCase):
    """Unit tests for ``export_detection_outputs`` on hand-built head outputs."""

    def setUp(self) -> None:
        """Set up a fake head with the shared proposal layout."""
        super().setUp()
        self.head = _FakeHead(num_proposals=self.num_proposals, num_classes=self.num_classes).to(
            self.device
        )

    def test_packs_last_layer_proposals_into_runtime_tensors(self) -> None:
        """
        Test that the regression channels of the last decoder layer are stacked in runtime order,
        the labels are passed through, and the score picks the predicted class of each proposal.
        """
        for num_decoder_layers in (1, 2):
            with self.subTest(num_decoder_layers=num_decoder_layers):
                outputs = self._build_outputs(batch_size=1, num_decoder_layers=num_decoder_layers)

                bbox_pred, score, label_pred = export_detection_outputs(self.head, outputs)

                self.assertEqual(bbox_pred.shape, (self.num_box_channels, self.num_proposals))
                torch.testing.assert_close(
                    bbox_pred, self._expected_bbox_pred(outputs.separate_head_outputs)
                )
                self.assertEqual(label_pred.dtype, torch.int64)
                torch.testing.assert_close(label_pred, outputs.query_labels[0])
                self.assertEqual(score.shape, (self.num_proposals,))
                torch.testing.assert_close(score, self._expected_score(outputs))

    def test_score_ignores_dense_heatmap_grid(self) -> None:
        """Test that the dense heatmap resolution does not leak into the exported score shape."""
        for heatmap_size in ((2, 2), (6, 6), (16, 32)):
            with self.subTest(heatmap_size=heatmap_size):
                outputs = self._build_outputs(
                    batch_size=1, num_decoder_layers=2, heatmap_size=heatmap_size
                )

                _, score, _ = export_detection_outputs(self.head, outputs)

                self.assertEqual(score.shape, (self.num_proposals,))

    def test_uses_first_sample_only(self) -> None:
        """Test that a batched input exports the first sample."""
        outputs = self._build_outputs(batch_size=2, num_decoder_layers=1)

        bbox_pred, score, label_pred = export_detection_outputs(self.head, outputs)

        self.assertEqual(bbox_pred.shape, (self.num_box_channels, self.num_proposals))
        self.assertEqual(score.shape, (self.num_proposals,))
        torch.testing.assert_close(
            bbox_pred, self._expected_bbox_pred(outputs.separate_head_outputs)
        )
        torch.testing.assert_close(label_pred, outputs.query_labels[0])
        torch.testing.assert_close(score, self._expected_score(outputs))

    def test_requires_velocity_branch(self) -> None:
        """Test that outputs without a velocity branch are rejected."""
        outputs = self._build_outputs(batch_size=1, num_decoder_layers=1, with_velocity=False)

        with self.assertRaisesRegex(ValueError, "velocity branch"):
            export_detection_outputs(self.head, outputs)


class TestExportDetectionOutputsWithTransFusionHead(_ExportDetectionOutputsTestCase):
    """Integration tests packing the outputs of a real, small TransFusion head."""

    def setUp(self) -> None:
        """Set up a small auxiliary TransFusion head on a 16x16 BEV grid."""
        super().setUp()
        self.num_decoder_layers = 2
        self.in_channels = 8
        self.bev_size = 16
        self.point_cloud_range = [0.0, 0.0, -2.0, 16.0, 16.0, 2.0]
        self.voxel_size = [1.0, 1.0, 4.0]
        self.out_size_factor = 1
        self.head = (
            TransFusionHead(
                num_proposals=self.num_proposals,
                auxiliary=True,
                in_channels=self.in_channels,
                hidden_channel=8,
                class_names=self.class_names,
                num_decoder_layers=self.num_decoder_layers,
                num_heads=2,
                feedforward_channels=16,
                common_heads=MappingProxyType(
                    {
                        "centers": (2, 2),
                        "heights": (1, 2),
                        "dims": (3, 2),
                        "rots": (2, 2),
                        "vels": (2, 2),
                    }
                ),
                bbox_coder=TransFusionBBoxCoder(
                    pc_range=self.point_cloud_range,
                    out_size_factor=self.out_size_factor,
                    voxel_size=self.voxel_size,
                    score_threshold_groups=None,
                    post_center_range=[-10.0, -10.0, -10.0, 26.0, 26.0, 10.0],
                    code_size=self.num_box_channels,
                ),
                assigner=HungarianAssigner3D(
                    cls_cost=ClassificationCost(weight=0.15),
                    reg_cost=BBoxBEVL1Cost(weight=0.25),
                    iou_cost=IoU3DCost(weight=0.25),
                    point_cloud_range=self.point_cloud_range,
                ),
                point_cloud_range=self.point_cloud_range,
                voxel_size=self.voxel_size,
                out_size_factor=self.out_size_factor,
                code_weights=[1.0] * 8 + [0.2, 0.2],
                min_radius=1,
                gaussian_overlap=0.1,
                score_threshold_group_configs=[
                    ScoreThresholdConfig(class_names=self.class_names, score_threshold=0.0)
                ],
                post_max_size=8,
                nms_min_radius=1.0,
                dense_heatmap_pooling_class_names=[],
            )
            .to(self.device)
            .eval()
        )
        self.bev_features: Float32[torch.Tensor, "batch_size channels height width"] = torch.randn(
            1, self.in_channels, self.bev_size, self.bev_size, device=self.device
        )

    def test_packs_real_head_outputs(self) -> None:
        """Test that outputs of the auxiliary head export as single-sample runtime tensors."""
        with torch.no_grad():
            outputs = self.head(self.bev_features)
            bbox_pred, score, label_pred = export_detection_outputs(self.head, outputs)

        self.assertEqual(
            outputs.separate_head_outputs.heatmap.shape[-1],
            self.num_decoder_layers * self.num_proposals,
        )
        self.assertEqual(bbox_pred.shape, (self.num_box_channels, self.num_proposals))
        self.assertEqual(score.shape, (self.num_proposals,))
        self.assertEqual(label_pred.shape, (self.num_proposals,))
        self.assertEqual(label_pred.dtype, torch.int64)
        self.assertTrue(torch.all((score >= 0.0) & (score <= 1.0)))
        self.assertTrue(torch.all((label_pred >= 0) & (label_pred < self.num_classes)))
        torch.testing.assert_close(
            bbox_pred, self._expected_bbox_pred(outputs.separate_head_outputs)
        )
        torch.testing.assert_close(score, self._expected_score(outputs))


if __name__ == "__main__":
    unittest.main()
