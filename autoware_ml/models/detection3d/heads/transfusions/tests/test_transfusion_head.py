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

"""Unit tests for TransFusionHead."""

import math
import tempfile
import unittest
from pathlib import Path
from types import MappingProxyType
from typing import Sequence

from jaxtyping import Bool, Float32, Int32
from omegaconf import OmegaConf
from onnx import TensorProto
import onnx
import torch

from autoware_ml.dataclasses.models.detection3d.head_targets import TransFusionHeadTargets
from autoware_ml.dataclasses.models.detection3d.predictions import Detection3DSamplePredictions
from autoware_ml.dataclasses.models.detection3d.head_outputs import (
    Detection3DHeadOutputs,
    TransFusionHeadOutputs,
)
from autoware_ml.models.detection3d.heads.transfusions.exportable_multi_head_attention import (
    ExportableMultiheadAttention,
)
from autoware_ml.losses.detection3d.rotated_iou_loss import Rotated2DIouLoss
from autoware_ml.models.detection3d.heads.transfusions.transfusion_head import (
    NMSGroupConfig,
    ScoreThresholdConfig,
    TransFusionHead,
)
from autoware_ml.models.detection3d.task_modules.assigners import HungarianAssigner3D
from autoware_ml.models.detection3d.task_modules.bbox_coders import TransFusionBBoxCoder
from autoware_ml.models.detection3d.task_modules.match_costs import (
    BBoxBEVL1Cost,
    ClassificationCost,
    IoU3DCost,
)
from autoware_ml.utils.onnx_precision import validate_module_onnx_precision


class TestTransFusionHead(unittest.TestCase):
    """Unit tests for the TransFusionHead."""

    def setUp(self) -> None:
        """Set up the common classes/inputs for the tests."""
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(0)

        # One BEV cell is one metre in this fixture, because voxel_size[0] * out_size_factor is
        # 1.0, so a metric center maps to the grid cell of the same number and the expectations
        # stay readable.
        self.class_names = ["car", "pedestrian"]
        self.num_classes = len(self.class_names)
        self.point_cloud_range = [0.0, 0.0, -2.0, 16.0, 16.0, 2.0]
        self.voxel_size = [1.0, 1.0, 4.0]
        self.out_size_factor = 1
        self.feature_map_size = (16, 16)
        self.post_center_range = [-10.0, -10.0, -10.0, 26.0, 26.0, 10.0]

        self.in_channels = 8
        self.hidden_channel = 8
        self.num_proposals = 4
        self.num_decoder_layers = 2
        self.num_heads = 2
        self.feedforward_channels = 16
        self.code_size = 10
        self.code_weights = [1.0] * 8 + [0.2, 0.2]
        self.min_radius = 1
        self.gaussian_overlap = 0.1
        self.post_max_size = 8
        self.nms_min_radius = 1.0
        self.common_heads = MappingProxyType(
            {
                "centers": (2, 2),
                "heights": (1, 2),
                "dims": (3, 2),
                "rots": (2, 2),
                "vels": (2, 2),
            }
        )
        self.score_threshold_group_configs = [
            ScoreThresholdConfig(class_names=self.class_names, score_threshold=0.0)
        ]
        # An auxiliary head concatenates every decoder layer along the query axis.
        self.num_queries = self.num_decoder_layers * self.num_proposals

        self.transfusion_head = self._build_transfusion_head()

        # Dummy inputs. Two samples with a different number of real boxes each, so the padded tail
        # of the second sample exercises the gt_valid_bboxes contract.
        self.gt_bboxes_3d = torch.tensor(
            [
                [
                    [2.0, 2.0, 0.0, 4.0, 1.6, 1.5, 0.25, 0.5, -0.1],
                    [8.0, 8.0, 0.0, 2.0, 2.0, 1.5, 0.0, 0.0, 0.0],
                    [12.0, 4.0, 0.0, 3.0, 1.8, 1.5, 1.0, -0.2, 0.3],
                ],
                [
                    [4.0, 10.0, 0.0, 4.0, 1.6, 1.5, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self.gt_labels_3d = torch.tensor(
            [[0, 1, 0], [1, -1, -1]], dtype=torch.int32, device=self.device
        )
        self.gt_valid_bboxes = torch.tensor([3, 1], dtype=torch.int32, device=self.device)
        self.batch_size = self.gt_bboxes_3d.shape[0]

    def _build_transfusion_head(
        self,
        auxiliary: bool = True,
        use_velocity: bool = True,
        heatmap_target: str = "round",
        dense_heatmap_pooling_class_names: Sequence[str] | None = None,
        score_threshold_group_configs: Sequence[ScoreThresholdConfig] | None = None,
        nms_type: str | None = None,
        nms_group_configs: Sequence[NMSGroupConfig] | None = None,
        post_center_range: Sequence[float] | None = None,
        use_bf16_cross_attention: bool = False,
        partial_ignore_labels: Sequence[str] | None = None,
        iou_loss: torch.nn.Module | None = None,
    ) -> TransFusionHead:
        """
        Build a TransFusionHead from the setUp parameters.

        Only the arguments a test actually varies are exposed. Passing None for one of them keeps
        the setUp default, except for nms_group_configs, where None is itself the default and means
        the head runs without grouped NMS.
        """
        return TransFusionHead(
            num_proposals=self.num_proposals,
            auxiliary=auxiliary,
            in_channels=self.in_channels,
            hidden_channel=self.hidden_channel,
            class_names=self.class_names,
            num_decoder_layers=self.num_decoder_layers,
            num_heads=self.num_heads,
            feedforward_channels=self.feedforward_channels,
            common_heads=self.common_heads,
            bbox_coder=TransFusionBBoxCoder(
                pc_range=self.point_cloud_range,
                out_size_factor=self.out_size_factor,
                voxel_size=self.voxel_size,
                score_threshold_groups=None,
                post_center_range=(
                    self.post_center_range if post_center_range is None else post_center_range
                ),
                code_size=self.code_size,
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
            code_weights=self.code_weights,
            min_radius=self.min_radius,
            gaussian_overlap=self.gaussian_overlap,
            score_threshold_group_configs=(
                self.score_threshold_group_configs
                if score_threshold_group_configs is None
                else score_threshold_group_configs
            ),
            post_max_size=self.post_max_size,
            nms_min_radius=self.nms_min_radius,
            dense_heatmap_pooling_class_names=(
                []
                if dense_heatmap_pooling_class_names is None
                else dense_heatmap_pooling_class_names
            ),
            heatmap_target=heatmap_target,
            nms_type=nms_type,
            nms_group_configs=nms_group_configs,
            use_velocity=use_velocity,
            use_bf16_cross_attention=use_bf16_cross_attention,
            partial_ignore_labels=partial_ignore_labels,
            iou_loss=iou_loss,
        ).to(self.device)

    def _export_attention(
        self, attention: ExportableMultiheadAttention, output_path: Path
    ) -> onnx.ModelProto:
        """Export one attention module on its own so the tests can inspect its ONNX graph."""
        query = torch.randn(1, 3, self.hidden_channel, device=self.device)
        key = torch.randn(1, 5, self.hidden_channel, device=self.device)
        torch.onnx.export(
            attention,
            (query, key, key),
            output_path,
            input_names=["query", "key", "value"],
            output_names=["output"],
            opset_version=17,
            dynamo=False,
        )
        return onnx.load(output_path)

    def _build_bev_features(
        self,
    ) -> Float32[torch.Tensor, "batch_size channels height width"]:
        """Build random BEV features of the shape the head expects."""
        height, width = self.feature_map_size
        return torch.randn(self.batch_size, self.in_channels, height, width, device=self.device)

    def _build_head_outputs(
        self, transfusion_head: TransFusionHead | None = None
    ) -> TransFusionHeadOutputs:
        """Run a forward pass, on the setUp head unless a test passes its own variant."""
        head = self.transfusion_head if transfusion_head is None else transfusion_head
        return head(self._build_bev_features())

    def _get_targets(
        self,
        transfusion_head: TransFusionHead | None = None,
        gt_valid_bboxes: Int32[torch.Tensor, " batch_size"] | None = None,
        gt_traffic_cone_barrier_bbox_status: Bool[torch.Tensor, " batch_size"] | None = None,
    ) -> TransFusionHeadTargets:
        """Build assignment targets from a fresh forward pass and the setUp ground truth."""
        head = self.transfusion_head if transfusion_head is None else transfusion_head
        return head.get_targets(
            outputs=self._build_head_outputs(head),
            gt_bboxes_3d=self.gt_bboxes_3d,
            gt_labels_3d=self.gt_labels_3d,
            gt_valid_bboxes=(self.gt_valid_bboxes if gt_valid_bboxes is None else gt_valid_bboxes),
            feature_map_size=self.feature_map_size,
            gt_traffic_cone_barrier_bbox_status=gt_traffic_cone_barrier_bbox_status,
        )

    def _compute_losses(
        self,
        transfusion_head: TransFusionHead | None = None,
        gt_valid_bboxes: Int32[torch.Tensor, " batch_size"] | None = None,
        gt_traffic_cone_barrier_bbox_status: Bool[torch.Tensor, " batch_size"] | None = None,
        transfusion_head_outputs: TransFusionHeadOutputs | None = None,
    ) -> MappingProxyType[str, torch.Tensor | float]:
        """
        Compute the losses from the setUp ground truth and a forward pass, fresh unless a test
        passes the outputs it wants to inspect afterwards.
        """
        head = self.transfusion_head if transfusion_head is None else transfusion_head
        if transfusion_head_outputs is None:
            transfusion_head_outputs = self._build_head_outputs(head)
        return head.loss(
            outputs=Detection3DHeadOutputs(
                center_head_outputs=None,
                transfusion_head_outputs=transfusion_head_outputs,
            ),
            gt_bboxes_3d=self.gt_bboxes_3d,
            gt_labels_3d=self.gt_labels_3d,
            gt_valid_bboxes=(self.gt_valid_bboxes if gt_valid_bboxes is None else gt_valid_bboxes),
            gt_traffic_cone_barrier_bbox_status=gt_traffic_cone_barrier_bbox_status,
        )

    def _decode_sample_predictions(
        self, transfusion_head: TransFusionHead | None = None
    ) -> Sequence[Detection3DSamplePredictions]:
        """Decode a fresh forward pass into the per-sample predictions, unwrapping the option."""
        head = self.transfusion_head if transfusion_head is None else transfusion_head
        sample_predictions = head.decode_outputs(
            Detection3DHeadOutputs(
                center_head_outputs=None,
                transfusion_head_outputs=self._build_head_outputs(head),
            )
        ).detection3d_predictions
        assert sample_predictions is not None
        return sample_predictions

    def _dense_heatmap_for_box(
        self,
        transfusion_head: TransFusionHead,
        length: float,
        width: float,
        yaw: float,
        center: float = 8.0,
    ) -> Float32[torch.Tensor, "height width"]:
        """Build a single-box dense heatmap target and return its class-0 map."""
        gt_bboxes_3d = torch.tensor(
            [[[center, center, 0.0, length, width, 1.5, yaw, 0.0, 0.0]]],
            dtype=torch.float32,
            device=self.device,
        )
        heatmaps = transfusion_head._build_dense_heatmap_targets(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=torch.zeros((1, 1), dtype=torch.int64, device=self.device),
            gt_valid_bboxes=torch.tensor([1], dtype=torch.int32, device=self.device),
            feature_map_size=self.feature_map_size,
            device=self.device,
        )
        return heatmaps[0, 0]

    def _build_filter_inputs(
        self, keep_masks: Bool[torch.Tensor, "batch_size num_proposals"]
    ) -> tuple[
        Float32[torch.Tensor, "batch_size num_proposals box_code_size"],
        Float32[torch.Tensor, "batch_size num_proposals"],
        Float32[torch.Tensor, "batch_size num_proposals"],
    ]:
        """
        Build the (batch_size=2, num_proposals=4) inputs shared by the _filter_bbox_predictions
        tests. Each box row is stamped with its flattened slot so the assertions can follow which
        prediction ends up where.
        """
        batch_size, num_proposals = keep_masks.shape
        scores = torch.tensor(
            [[0.1, 0.9, 0.5, 0.3], [0.7, 0.2, 0.6, 0.4]],
            dtype=torch.float32,
            device=self.device,
        )
        class_ids = (
            torch.arange(batch_size * num_proposals, dtype=torch.int64, device=self.device).view(
                batch_size, num_proposals
            )
            % self.num_classes
        )
        # Flattened slot index broadcast across the 9 box parameters.
        bbox_predictions = (
            torch.arange(batch_size * num_proposals, dtype=torch.float32, device=self.device)
            .view(batch_size, num_proposals, 1)
            .expand(batch_size, num_proposals, 9)
            .contiguous()
        )
        return bbox_predictions, scores, class_ids

    def test_transfusion_head_weight_init(self) -> None:
        """
        Test that the head weights are initialized around zero and that the dense heatmap head
        keeps its negative bias, which is what stops the untrained heatmap from saturating.
        """
        weights = torch.cat(
            [
                parameter.detach().flatten()
                for name, parameter in self.transfusion_head.named_parameters()
                if "weight" in name
            ]
        )

        # Normalization layers initialize their weight to 1.0, so the mean sits slightly above
        # zero rather than at it; the bound only has to catch a grossly wrong init.
        self.assertLess(abs(float(weights.mean())), 0.1)
        self.assertLess(float(weights.std()), 0.5)
        # The last conv of the dense heatmap head carries the configured init bias.
        dense_heatmap_bias = self.transfusion_head.dense_heatmap_head[-1].bias  # type: ignore
        self.assertTrue(
            torch.allclose(
                dense_heatmap_bias.detach(),
                torch.full_like(dense_heatmap_bias, self.transfusion_head.heatmap_init_bias),
            )
        )

    def test_forward_returns_outputs_of_expected_shape(self) -> None:
        """Test that a forward pass returns every output of TransFusionHeadOutputs correctly shaped."""
        outputs = self._build_head_outputs()
        separate_head_outputs = outputs.separate_head_outputs
        height, width = self.feature_map_size
        # An auxiliary head concatenates every decoder layer along the query axis.
        num_queries = self.num_decoder_layers * self.num_proposals

        self.assertEqual(
            outputs.dense_heatmap.shape, (self.batch_size, self.num_classes, height, width)
        )
        self.assertEqual(outputs.query_labels.shape, (self.batch_size, self.num_proposals))
        self.assertEqual(
            outputs.query_heatmap_score.shape,
            (self.batch_size, self.num_classes, self.num_proposals),
        )
        self.assertEqual(
            separate_head_outputs.heatmap.shape,
            (self.batch_size, self.num_classes, num_queries),
        )
        self.assertEqual(separate_head_outputs.center.shape, (self.batch_size, 2, num_queries))
        self.assertEqual(separate_head_outputs.height.shape, (self.batch_size, 1, num_queries))
        self.assertEqual(separate_head_outputs.dim.shape, (self.batch_size, 3, num_queries))
        self.assertEqual(separate_head_outputs.rot.shape, (self.batch_size, 2, num_queries))
        assert separate_head_outputs.vel is not None
        self.assertEqual(separate_head_outputs.vel.shape, (self.batch_size, 2, num_queries))
        self.assertTrue(torch.isfinite(separate_head_outputs.center).all())

    def test_forward_without_auxiliary_returns_only_the_last_layer(self) -> None:
        """
        Test that a head built without auxiliary supervision emits one layer's worth of queries,
        while the auxiliary head emits one per decoder layer.
        """
        transfusion_head = self._build_transfusion_head(auxiliary=False)

        outputs = self._build_head_outputs(transfusion_head)

        self.assertEqual(
            outputs.separate_head_outputs.heatmap.shape,
            (self.batch_size, self.num_classes, self.num_proposals),
        )

    def test_forward_query_labels_are_valid_class_ids(self) -> None:
        """Test that the proposals selected from the dense heatmap carry in-range class ids."""
        outputs = self._build_head_outputs()

        self.assertEqual(outputs.query_labels.dtype, torch.int64)
        self.assertTrue(bool((outputs.query_labels >= 0).all()))
        self.assertTrue(bool((outputs.query_labels < self.num_classes).all()))

    def test_forward_without_velocity_drops_the_velocity_branch(self) -> None:
        """Test that a head built without velocity emits no velocity channels."""
        transfusion_head = self._build_transfusion_head(use_velocity=False)

        outputs = self._build_head_outputs(transfusion_head)

        self.assertIsNone(outputs.separate_head_outputs.vel)

    def test_build_dense_heatmap_targets_peaks_at_the_box_center(self) -> None:
        """
        Test that a box's dense heatmap target peaks at the cell holding its center, on that box's
        own class channel only.
        """
        heatmaps = self.transfusion_head._build_dense_heatmap_targets(
            gt_bboxes_3d=self.gt_bboxes_3d,
            gt_labels_3d=self.gt_labels_3d.to(torch.int64),
            gt_valid_bboxes=self.gt_valid_bboxes,
            feature_map_size=self.feature_map_size,
            device=self.device,
        )
        height, width = self.feature_map_size

        self.assertEqual(heatmaps.shape, (self.batch_size, self.num_classes, height, width))
        # Sample 0 box 0 is a car at (2, 2) and box 1 a pedestrian at (8, 8).
        self.assertAlmostEqual(float(heatmaps[0, 0, 2, 2]), 1.0, places=5)
        self.assertAlmostEqual(float(heatmaps[0, 1, 8, 8]), 1.0, places=5)
        # The pedestrian's cell is empty on the car channel.
        self.assertAlmostEqual(float(heatmaps[0, 0, 8, 8]), 0.0, places=5)
        self.assertTrue(bool(((heatmaps >= 0.0) & (heatmaps <= 1.0)).all()))

    def test_build_dense_heatmap_targets_ignores_padded_boxes(self) -> None:
        """
        Test that boxes past gt_valid_bboxes contribute nothing, so the zero padding every batch
        carries is not drawn at the origin as if it were a real box.
        """
        heatmaps = self.transfusion_head._build_dense_heatmap_targets(
            gt_bboxes_3d=self.gt_bboxes_3d,
            gt_labels_3d=self.gt_labels_3d.to(torch.int64),
            gt_valid_bboxes=self.gt_valid_bboxes,
            feature_map_size=self.feature_map_size,
            device=self.device,
        )

        # Sample 1 has a single real box at (4, 10); its two padded rows sit at the origin.
        self.assertAlmostEqual(float(heatmaps[1, 1, 10, 4]), 1.0, places=5)
        self.assertAlmostEqual(float(heatmaps[1, 0, 0, 0]), 0.0, places=5)
        self.assertAlmostEqual(float(heatmaps[1, 1, 0, 0]), 0.0, places=5)

    def test_oriented_dense_heatmap_spreads_along_the_box_length(self) -> None:
        """Test that an oriented heatmap target stretches along the box length, not across it."""
        transfusion_head = self._build_transfusion_head(heatmap_target="oriented")

        heatmap = self._dense_heatmap_for_box(transfusion_head, length=14.0, width=2.0, yaw=0.0)

        center = 8
        along_length = float(heatmap[center, center + 3])
        across_width = float(heatmap[center + 3, center])
        self.assertGreater(along_length, 0.2)
        self.assertLess(across_width, 1e-2)

    def test_oriented_dense_heatmap_follows_yaw(self) -> None:
        """Test that a 90 degree yaw rotates the oriented blob's long axis from x to y."""
        transfusion_head = self._build_transfusion_head(heatmap_target="oriented")

        heatmap = self._dense_heatmap_for_box(
            transfusion_head, length=14.0, width=2.0, yaw=math.pi / 2
        )

        center = 8
        self.assertGreater(float(heatmap[center + 3, center]), 0.2)
        self.assertLess(float(heatmap[center, center + 3]), 1e-2)

    def test_round_dense_heatmap_is_isotropic_and_default(self) -> None:
        """Test that the default round heatmap target ignores orientation."""
        self.assertEqual(self.transfusion_head.heatmap_target, "round")

        heatmap = self._dense_heatmap_for_box(
            self.transfusion_head, length=10.0, width=2.0, yaw=0.0
        )

        center = 8
        self.assertAlmostEqual(
            float(heatmap[center, center + 1]), float(heatmap[center + 1, center]), places=4
        )

    def test_invalid_heatmap_target_raises(self) -> None:
        """Test that an unknown heatmap_target is rejected at construction."""
        with self.assertRaises(ValueError):
            self._build_transfusion_head(heatmap_target="square")

    def test_suppress_dense_heatmaps_keeps_only_local_maxima_of_pooled_classes(self) -> None:
        """
        Test that the classes named for local-max pooling keep only their peak while the classes
        left out are passed through untouched.
        """
        transfusion_head = self._build_transfusion_head(dense_heatmap_pooling_class_names=["car"])
        height, width = self.feature_map_size
        heatmaps = torch.zeros(
            (1, self.num_classes, height, width), dtype=torch.float32, device=self.device
        )
        # A peak with a weaker neighbour inside the 3x3 pooling window, on both channels.
        heatmaps[:, :, 5, 5] = 0.9
        heatmaps[:, :, 5, 6] = 0.4

        suppressed = transfusion_head._suppress_dense_heatmaps(heatmaps)

        # "car" is pooled: the peak survives and the neighbour is suppressed.
        self.assertAlmostEqual(float(suppressed[0, 0, 5, 5]), 0.9, places=5)
        self.assertAlmostEqual(float(suppressed[0, 0, 5, 6]), 0.0, places=5)
        # "pedestrian" is not pooled, so both cells are kept.
        self.assertAlmostEqual(float(suppressed[0, 1, 5, 5]), 0.9, places=5)
        self.assertAlmostEqual(float(suppressed[0, 1, 5, 6]), 0.4, places=5)

    def test_suppress_dense_heatmaps_is_a_no_op_without_pooling_classes(self) -> None:
        """Test that naming no pooling classes leaves the heatmaps exactly as they are."""
        heatmaps = torch.rand(
            (1, self.num_classes, *self.feature_map_size), dtype=torch.float32, device=self.device
        )

        suppressed = self.transfusion_head._suppress_dense_heatmaps(heatmaps)

        self.assertTrue(torch.equal(suppressed, heatmaps))

    def test_get_targets_populates_labels_and_boxes(self) -> None:
        """Test that get_targets returns targets shaped for every decoder layer's queries."""
        targets = self._get_targets()
        # Layers are concatenated along the query axis, matching the prediction layout.
        num_queries = self.num_decoder_layers * self.num_proposals

        self.assertEqual(targets.labels.shape, (self.batch_size, num_queries))
        self.assertEqual(
            targets.label_weights.shape, (self.batch_size, num_queries, self.num_classes)
        )
        self.assertEqual(targets.class_weights.shape, (self.batch_size, self.num_classes))
        self.assertEqual(targets.bbox_targets.shape, (self.batch_size, num_queries, self.code_size))
        self.assertEqual(targets.bbox_weights.shape, (self.batch_size, num_queries, self.code_size))
        self.assertEqual(
            targets.dense_heatmaps.shape,
            (self.batch_size, self.num_classes, *self.feature_map_size),
        )

    def test_get_targets_marks_positives_consistently(self) -> None:
        """
        Test that a positive query is exactly one carrying a real class label, that its regression
        weight is one, and that a negative query carries the background label and no weight.
        """
        targets = self._get_targets()

        positive_masks = targets.labels < self.num_classes
        self.assertEqual(int(positive_masks.sum()), targets.num_pos)
        self.assertTrue(bool((targets.bbox_weights[positive_masks] == 1.0).all()))
        self.assertTrue(bool((targets.bbox_weights[~positive_masks] == 0.0).all()))
        self.assertTrue(bool((targets.bbox_targets[~positive_masks] == 0.0).all()))
        self.assertTrue(torch.isfinite(targets.bbox_targets[positive_masks]).all())

    def test_get_targets_never_matches_more_than_the_valid_boxes(self) -> None:
        """
        Test that each sample's positives per layer are capped by its own gt_valid_bboxes, so the
        padded tail of a short sample is never assigned.
        """
        targets = self._get_targets()

        positive_masks = targets.labels < self.num_classes
        for batch_index in range(self.batch_size):
            for layer_index in range(self.num_decoder_layers):
                start = layer_index * self.num_proposals
                layer_positives = positive_masks[batch_index, start : start + self.num_proposals]
                self.assertLessEqual(
                    int(layer_positives.sum()), int(self.gt_valid_bboxes[batch_index])
                )

    def test_get_targets_with_no_valid_boxes_produces_no_positives(self) -> None:
        """Test that a batch whose boxes are all padding assigns nothing and reports zero IoU."""
        targets = self._get_targets(
            gt_valid_bboxes=torch.zeros(self.batch_size, dtype=torch.int32, device=self.device)
        )

        self.assertEqual(targets.num_pos, 0)
        self.assertEqual(targets.matched_iou, 0.0)
        self.assertTrue(bool((targets.labels == self.num_classes).all()))
        self.assertTrue(bool((targets.bbox_weights == 0.0).all()))

    def test_get_targets_supervises_every_class_without_partial_ignore(self) -> None:
        """
        Test that a head with no partially ignored classes weighs every class of every query,
        whatever the annotation flag says.
        """
        targets = self._get_targets(
            gt_traffic_cone_barrier_bbox_status=torch.tensor([True, False], device=self.device)
        )

        self.assertTrue(bool((targets.class_weights == 1.0).all()))
        self.assertTrue(bool((targets.label_weights == 1.0).all()))

    def test_get_targets_supervises_every_class_without_the_annotation_flag(self) -> None:
        """
        Test that partially ignored classes stay supervised when the dataset carries no annotation
        flag, since nothing then says their annotations are missing.
        """
        head = self._build_transfusion_head(partial_ignore_labels=["pedestrian"])

        targets = self._get_targets(head)

        self.assertTrue(bool((targets.class_weights == 1.0).all()))
        self.assertTrue(bool((targets.label_weights == 1.0).all()))

    def test_get_targets_drops_ignored_classes_of_an_unannotated_sample(self) -> None:
        """
        Test that a sample flagged as not annotated zeroes the ignored classes in its class weights
        and in its negative queries only, while positives and annotated samples keep full weights.
        """
        head = self._build_transfusion_head(partial_ignore_labels=["pedestrian"])
        pedestrian_id = self.class_names.index("pedestrian")
        car_id = self.class_names.index("car")

        targets = self._get_targets(
            head,
            gt_traffic_cone_barrier_bbox_status=torch.tensor([True, False], device=self.device),
        )

        expected_class_weights = torch.ones((self.batch_size, self.num_classes), device=self.device)
        expected_class_weights[1, pedestrian_id] = 0.0
        self.assertTrue(torch.equal(targets.class_weights, expected_class_weights))
        # The annotated sample is fully supervised.
        self.assertTrue(bool((targets.label_weights[0] == 1.0).all()))
        # The unannotated sample has one real box and four proposals per layer, so it has
        # negatives: they lose the pedestrian column only, positives keep every class.
        negative_masks = targets.labels[1] == self.num_classes
        self.assertGreater(int(negative_masks.sum()), 0)
        unannotated_negatives = targets.label_weights[1][negative_masks]
        self.assertTrue(bool((unannotated_negatives[:, pedestrian_id] == 0.0).all()))
        self.assertTrue(bool((unannotated_negatives[:, car_id] == 1.0).all()))
        self.assertTrue(bool((targets.label_weights[1][~negative_masks] == 1.0).all()))

    def test_loss_returns_one_entry_per_layer_and_a_finite_total(self) -> None:
        """Test that loss reports a heatmap term, a cls and bbox term per layer, and their sum."""
        losses = self._compute_losses()

        self.assertIn("loss_heatmap", losses)
        self.assertIn("loss", losses)
        self.assertIn("matched_ious", losses)
        # The last layer is logged as layer_-1, the rest by index.
        for prefix in ("layer_0", "layer_-1"):
            self.assertIn(f"{prefix}_loss_cls", losses)
            self.assertIn(f"{prefix}_loss_bbox", losses)
        for key, value in losses.items():
            self.assertTrue(torch.isfinite(torch.as_tensor(value)).all(), msg=key)
        component_total = sum(
            float(torch.as_tensor(value).detach())
            for key, value in losses.items()
            if "loss" in key and key != "loss"
        )
        total_loss = torch.as_tensor(losses["loss"]).detach()
        self.assertAlmostEqual(float(total_loss), component_total, places=4)

    def test_loss_backward_reaches_the_predictions(self) -> None:
        """
        Test that the reported total loss still carries a graph, since a detached total would
        silently train nothing.
        """
        losses = self._compute_losses()
        total_loss = losses["loss"]
        assert isinstance(total_loss, torch.Tensor)
        self.assertTrue(total_loss.requires_grad)

        total_loss.backward()

        gradients = [
            parameter.grad
            for parameter in self.transfusion_head.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        for gradient in gradients:
            self.assertTrue(torch.isfinite(gradient).all())

    def test_loss_with_no_valid_boxes_stays_finite(self) -> None:
        """
        Test that a batch with only padded boxes still produces a finite loss, since the box loss
        normalizes by the positive count and would divide by zero without a floor.
        """
        losses = self._compute_losses(
            gt_valid_bboxes=torch.zeros(self.batch_size, dtype=torch.int32, device=self.device)
        )

        for key, value in losses.items():
            self.assertTrue(torch.isfinite(torch.as_tensor(value)).all(), msg=key)
        self.assertAlmostEqual(float(losses["matched_ious"]), 0.0, places=6)

    def test_loss_heatmap_skips_ignored_classes_of_unannotated_samples(self) -> None:
        """
        Test that the dense heatmap of an ignored class gets no gradient in a sample flagged as not
        annotated for it, while it still does in an annotated sample and for the other classes.
        """
        head = self._build_transfusion_head(partial_ignore_labels=["pedestrian"])
        pedestrian_id = self.class_names.index("pedestrian")
        car_id = self.class_names.index("car")
        outputs = self._build_head_outputs(head)
        outputs.dense_heatmap.retain_grad()

        losses = self._compute_losses(
            head,
            gt_traffic_cone_barrier_bbox_status=torch.tensor([True, False], device=self.device),
            transfusion_head_outputs=outputs,
        )
        loss_heatmap = losses["loss_heatmap"]
        assert isinstance(loss_heatmap, torch.Tensor)
        loss_heatmap.backward()

        gradients = outputs.dense_heatmap.grad
        assert gradients is not None
        self.assertTrue(bool((gradients[1, pedestrian_id] == 0.0).all()))
        self.assertTrue(bool((gradients[0, pedestrian_id] != 0.0).any()))
        self.assertTrue(bool((gradients[1, car_id] != 0.0).any()))

    def test_loss_heatmap_of_an_unannotated_batch_equals_the_annotated_classes_alone(self) -> None:
        """
        Test that dropping a class from every sample yields the heatmap loss of the remaining
        classes computed on their own, normalization included.
        """
        head = self._build_transfusion_head(partial_ignore_labels=["pedestrian"])
        car_id = self.class_names.index("car")
        outputs = self._build_head_outputs(head)
        unannotated = torch.tensor([False, False], device=self.device)

        losses = self._compute_losses(
            head, gt_traffic_cone_barrier_bbox_status=unannotated, transfusion_head_outputs=outputs
        )
        targets = head.get_targets(
            outputs=outputs,
            gt_bboxes_3d=self.gt_bboxes_3d,
            gt_labels_3d=self.gt_labels_3d,
            gt_valid_bboxes=self.gt_valid_bboxes,
            feature_map_size=self.feature_map_size,
            gt_traffic_cone_barrier_bbox_status=unannotated,
        )
        expected = head.loss_heatmap(
            outputs.dense_heatmap[:, [car_id]], targets.dense_heatmaps[:, [car_id]]
        )

        loss_heatmap = losses["loss_heatmap"]
        assert isinstance(loss_heatmap, torch.Tensor)
        self.assertAlmostEqual(float(loss_heatmap.detach()), float(expected.detach()), places=5)

    def test_loss_without_iou_loss_reports_no_iou_term(self) -> None:
        """Test that a head built without an IoU loss skips the term entirely."""
        losses = self._compute_losses()

        self.assertIsNone(self.transfusion_head.iou_loss)
        self.assertFalse([key for key in losses if key.endswith("_loss_iou")])

    def _build_iou_loss(
        self,
        labels_to_ignore_rotation: Sequence[str] | None = None,
        loss_weight: float = 1.0,
    ) -> Rotated2DIouLoss:
        """Build a rotated IoU loss on the fixture's grid geometry."""
        return Rotated2DIouLoss(
            class_names=self.class_names,
            point_cloud_range=self.point_cloud_range,
            voxel_size=self.voxel_size,
            out_size_factor=self.out_size_factor,
            labels_to_ignore_rotation=labels_to_ignore_rotation,
            loss_weight=loss_weight,
        )

    def _encoded_gt_boxes(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the setUp gt boxes of the first sample, returning (encoded, labels)."""
        encoded = self.transfusion_head.bbox_coder.encode(self.gt_bboxes_3d[:1])
        labels = self.gt_labels_3d[:1].long()
        return encoded, labels

    def test_layer_iou_loss_without_positives_is_zero_but_attached(self) -> None:
        """Test that a layer with no matched proposals contributes a zero that keeps the graph."""
        head = self._build_transfusion_head(iou_loss=self._build_iou_loss())
        preds = torch.randn(
            self.batch_size,
            self.num_proposals,
            self.code_size,
            device=self.device,
            requires_grad=True,
        )
        pos_masks = torch.zeros(
            self.batch_size, self.num_proposals, dtype=torch.bool, device=self.device
        )

        loss_iou = head._build_layer_iou_loss(
            preds=preds,
            bbox_targets=torch.zeros_like(preds),
            labels=torch.full(pos_masks.shape, self.num_classes, device=self.device),
            pos_masks=pos_masks,
            num_pos=0,
        )

        self.assertEqual(float(loss_iou), 0.0)
        self.assertTrue(loss_iou.requires_grad)

    @unittest.skipUnless(torch.cuda.is_available(), "the rotated IoU op requires CUDA")
    def test_layer_iou_loss_is_zero_for_perfect_predictions(self) -> None:
        """Test that predicting the encoded target exactly gives an IoU of one and no loss."""
        head = self._build_transfusion_head(iou_loss=self._build_iou_loss())
        encoded, labels = self._encoded_gt_boxes()
        pos_masks = torch.ones(encoded.shape[:2], dtype=torch.bool, device=self.device)

        loss_iou = head._build_layer_iou_loss(
            preds=encoded.clone(),
            bbox_targets=encoded,
            labels=labels,
            pos_masks=pos_masks,
            num_pos=int(pos_masks.sum()),
        )

        self.assertAlmostEqual(float(loss_iou), 0.0, places=4)

    @unittest.skipUnless(torch.cuda.is_available(), "the rotated IoU op requires CUDA")
    def test_layer_iou_loss_grows_as_the_prediction_drifts_from_the_target(self) -> None:
        """Test that shifting the predicted center away from the gt raises the IoU loss."""
        head = self._build_transfusion_head(iou_loss=self._build_iou_loss())
        encoded, labels = self._encoded_gt_boxes()
        pos_masks = torch.ones(encoded.shape[:2], dtype=torch.bool, device=self.device)
        num_pos = int(pos_masks.sum())

        losses = []
        for shift_cells in (0.0, 0.5, 1.0):
            preds = encoded.clone()
            preds[..., 0] += shift_cells
            losses.append(
                float(
                    head._build_layer_iou_loss(
                        preds=preds,
                        bbox_targets=encoded,
                        labels=labels,
                        pos_masks=pos_masks,
                        num_pos=num_pos,
                    )
                )
            )

        self.assertLess(losses[0], losses[1])
        self.assertLess(losses[1], losses[2])

    @unittest.skipUnless(torch.cuda.is_available(), "the rotated IoU op requires CUDA")
    def test_loss_with_iou_loss_reports_a_finite_iou_term_per_layer(self) -> None:
        """Test that an IoU loss adds one finite, weighted term per decoder layer to the total."""
        head = self._build_transfusion_head(iou_loss=self._build_iou_loss(loss_weight=2.0))

        losses = self._compute_losses(transfusion_head=head)

        for prefix in ("layer_0", "layer_-1"):
            self.assertIn(f"{prefix}_loss_iou", losses)
            term = torch.as_tensor(losses[f"{prefix}_loss_iou"])
            self.assertTrue(torch.isfinite(term).all())
            self.assertGreaterEqual(float(term), 0.0)
        component_total = sum(
            float(torch.as_tensor(value).detach())
            for key, value in losses.items()
            if "loss" in key and key != "loss"
        )
        self.assertAlmostEqual(float(torch.as_tensor(losses["loss"])), component_total, places=4)

    @unittest.skipUnless(torch.cuda.is_available(), "the rotated IoU op requires CUDA")
    def test_loss_with_iou_loss_backward_reaches_the_predictions(self) -> None:
        """Test that the IoU term is differentiable back to the head parameters."""
        head = self._build_transfusion_head(iou_loss=self._build_iou_loss())
        losses = self._compute_losses(transfusion_head=head)
        iou_total = sum(value for key, value in losses.items() if key.endswith("_loss_iou"))
        assert isinstance(iou_total, torch.Tensor)

        iou_total.backward()

        gradients = [p.grad for p in head.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        for gradient in gradients:
            self.assertTrue(torch.isfinite(gradient).all())

    def test_loss_requires_transfusion_head_outputs(self) -> None:
        """Test that loss rejects outputs that carry no TransFusion branch."""
        with self.assertRaises(ValueError):
            self.transfusion_head.loss(
                outputs=Detection3DHeadOutputs(
                    center_head_outputs=None, transfusion_head_outputs=None
                ),
                gt_bboxes_3d=self.gt_bboxes_3d,
                gt_labels_3d=self.gt_labels_3d,
                gt_valid_bboxes=self.gt_valid_bboxes,
            )

    def test_decode_outputs_returns_one_entry_per_sample(self) -> None:
        """Test that decode_outputs returns ragged per-sample predictions of the right widths."""
        sample_predictions = self._decode_sample_predictions()

        self.assertEqual(len(sample_predictions), self.batch_size)
        for prediction in sample_predictions:
            num_boxes = prediction.bboxes_3d.shape[0]
            self.assertLessEqual(num_boxes, self.num_proposals)
            # 9 box parameters because the head is configured with velocity.
            self.assertEqual(prediction.bboxes_3d.shape[1:], (9,))
            self.assertEqual(prediction.scores_3d.shape, (num_boxes,))
            self.assertEqual(prediction.labels_3d.shape, (num_boxes,))
            self.assertTrue(torch.isfinite(prediction.bboxes_3d).all())

    def test_decode_outputs_drops_everything_under_the_score_threshold(self) -> None:
        """Test that a score threshold above every predicted score empties every sample."""
        transfusion_head = self._build_transfusion_head(
            score_threshold_group_configs=[
                ScoreThresholdConfig(class_names=self.class_names, score_threshold=1.0)
            ]
        )

        sample_predictions = self._decode_sample_predictions(transfusion_head)

        self.assertEqual(len(sample_predictions), self.batch_size)
        for prediction in sample_predictions:
            self.assertEqual(prediction.bboxes_3d.shape, (0, 9))
            self.assertEqual(prediction.scores_3d.shape, (0,))

    def test_decode_outputs_drops_boxes_outside_the_post_center_range(self) -> None:
        """Test that a post_center_range excluding the whole scene empties every sample."""
        transfusion_head = self._build_transfusion_head(
            post_center_range=[100.0, 100.0, 100.0, 200.0, 200.0, 200.0]
        )

        sample_predictions = self._decode_sample_predictions(transfusion_head)

        for prediction in sample_predictions:
            self.assertEqual(prediction.bboxes_3d.shape, (0, 9))

    def test_filter_bboxes_score_uses_the_group_of_each_class(self) -> None:
        """
        Test that each class is thresholded by its own group, so raising one group's threshold only
        drops proposals of that group's classes.
        """
        bbox_scores = torch.tensor([[0.1, 0.4, 0.1, 0.4]], device=self.device)
        bbox_classes = torch.tensor([[0, 0, 1, 1]], dtype=torch.int64, device=self.device)
        transfusion_head = self._build_transfusion_head(
            score_threshold_group_configs=[
                ScoreThresholdConfig(class_names=["car"], score_threshold=0.3),
                ScoreThresholdConfig(class_names=["pedestrian"], score_threshold=0.0),
            ]
        )

        keep_masks = transfusion_head.bbox_coder.filter_bboxes_score(
            bbox_scores=bbox_scores,
            bbox_classes=bbox_classes,
            num_classes=self.num_classes,
        )

        # Car needs 0.3, pedestrian only a positive score.
        self.assertEqual(keep_masks.tolist(), [[False, True, True, True]])

    def test_filter_bboxes_center_range_keeps_only_centers_inside_the_range(self) -> None:
        """Test that the center-range filter drops boxes whose centers fall outside the range."""
        bbox_centers = torch.tensor(
            [[[8.0, 8.0, 0.0], [100.0, 8.0, 0.0], [8.0, 8.0, 100.0]]], device=self.device
        )

        keep_masks = self.transfusion_head.bbox_coder.filter_bboxes_center_range(
            bbox_centers=bbox_centers
        )

        self.assertEqual(keep_masks.tolist(), [[True, False, False]])

    def test_filter_bboxes_nms_groups_suppresses_within_a_group_only(self) -> None:
        """
        Test that two classes sharing an NMS group suppress each other, while the same pair split
        across groups both survive, which is what putting a group in one channel buys.
        """
        bbox_centers = torch.tensor([[[8.0, 8.0], [8.5, 8.0]]], device=self.device)
        bbox_scores = torch.tensor([[0.9, 0.8]], device=self.device)
        bbox_classes = torch.tensor([[0, 1]], dtype=torch.int64, device=self.device)
        keep_masks = torch.ones((1, 2), dtype=torch.bool, device=self.device)

        together = self._build_transfusion_head(
            nms_type="circle",
            nms_group_configs=[
                NMSGroupConfig(class_names=self.class_names, nms_radius=1.0, max_size=8)
            ],
        )._filter_bboxes_nms_groups(bbox_centers, bbox_scores, bbox_classes, keep_masks)
        apart = self._build_transfusion_head(
            nms_type="circle",
            nms_group_configs=[
                NMSGroupConfig(class_names=["car"], nms_radius=1.0, max_size=8),
                NMSGroupConfig(class_names=["pedestrian"], nms_radius=1.0, max_size=8),
            ],
        )._filter_bboxes_nms_groups(bbox_centers, bbox_scores, bbox_classes, keep_masks)

        self.assertEqual(together.tolist(), [[True, False]])
        self.assertEqual(apart.tolist(), [[True, True]])

    def test_filter_bboxes_nms_groups_never_revives_a_rejected_proposal(self) -> None:
        """
        Test that a proposal already rejected upstream stays rejected and cannot suppress a
        survivor, since it takes no part in the NMS.
        """
        bbox_centers = torch.tensor([[[8.0, 8.0], [8.5, 8.0]]], device=self.device)
        bbox_scores = torch.tensor([[0.9, 0.8]], device=self.device)
        bbox_classes = torch.tensor([[0, 0]], dtype=torch.int64, device=self.device)
        transfusion_head = self._build_transfusion_head(
            nms_type="circle",
            nms_group_configs=[
                NMSGroupConfig(class_names=self.class_names, nms_radius=1.0, max_size=8)
            ],
        )

        keep_masks = transfusion_head._filter_bboxes_nms_groups(
            bbox_centers,
            bbox_scores,
            bbox_classes,
            torch.tensor([[False, True]], device=self.device),
        )

        self.assertEqual(keep_masks.tolist(), [[False, True]])

    def test_filter_bbox_predictions_keeps_survivors_in_proposal_order(self) -> None:
        """
        Test that _filter_bbox_predictions drops the masked proposals, keeps the survivors in
        proposal order rather than by score, and carries their scores and labels along.
        """
        keep_masks = torch.tensor(
            [[False, True, True, False], [True, False, False, True]],
            dtype=torch.bool,
            device=self.device,
        )
        bbox_predictions, scores, class_ids = self._build_filter_inputs(keep_masks)

        predictions = self.transfusion_head._filter_bbox_predictions(
            bbox_predictions=bbox_predictions,
            scores=scores,
            class_ids=class_ids,
            keep_masks=keep_masks,
        ).detection3d_predictions
        assert predictions is not None

        self.assertEqual(len(predictions), 2)
        # Slots 1 and 2 of sample 0, in proposal order, so the 0.9 and 0.5 scores, not sorted.
        self.assertTrue(torch.equal(predictions[0].bboxes_3d[:, 0], bbox_predictions[0, 1:3, 0]))
        self.assertTrue(torch.allclose(predictions[0].scores_3d, scores[0, 1:3]))
        self.assertTrue(torch.equal(predictions[0].labels_3d, class_ids[0, 1:3]))
        # Slots 0 and 3 of sample 1.
        self.assertTrue(torch.allclose(predictions[1].scores_3d, scores[1, [0, 3]]))
        self.assertTrue(torch.equal(predictions[1].labels_3d, class_ids[1, [0, 3]]))

    def test_filter_bbox_predictions_returns_empty_entry_per_sample_when_all_suppressed(
        self,
    ) -> None:
        """
        Test that _filter_bbox_predictions still returns one entry per sample when everything is
        suppressed, so the predictions stay aligned with the batch.
        """
        keep_masks = torch.zeros((2, self.num_proposals), dtype=torch.bool, device=self.device)
        bbox_predictions, scores, class_ids = self._build_filter_inputs(keep_masks)

        predictions = self.transfusion_head._filter_bbox_predictions(
            bbox_predictions=bbox_predictions,
            scores=scores,
            class_ids=class_ids,
            keep_masks=keep_masks,
        ).detection3d_predictions
        assert predictions is not None

        self.assertEqual(len(predictions), 2)
        for prediction in predictions:
            self.assertEqual(prediction.bboxes_3d.shape, (0, 9))
            self.assertEqual(prediction.scores_3d.shape, (0,))
            self.assertEqual(prediction.labels_3d.shape, (0,))

    def test_resolve_class_ids_maps_names_and_dedupes(self) -> None:
        """Test that class names resolve to sorted, deduplicated class ids."""
        self.assertEqual(
            list(self.transfusion_head._resolve_class_ids(["pedestrian", "car", "car"])), [0, 1]
        )

    def test_resolve_class_ids_rejects_an_unknown_name(self) -> None:
        """Test that a class name outside class_names is rejected rather than silently ignored."""
        with self.assertRaises(ValueError):
            self.transfusion_head._resolve_class_ids(["helicopter"])

    def test_resolve_nms_groups_fills_in_the_head_defaults(self) -> None:
        """Test that a group leaving nms_radius or max_size unset inherits the head's defaults."""
        transfusion_head = self._build_transfusion_head(
            nms_group_configs=[
                NMSGroupConfig(class_names=["car"], nms_radius=0.5, max_size=3),
                NMSGroupConfig(class_names=["pedestrian"], nms_radius=None, max_size=None),
            ]
        )
        groups = transfusion_head.nms_groups
        assert groups is not None

        self.assertEqual([list(group.class_ids) for group in groups], [[0], [1]])
        self.assertEqual((groups[0].nms_radius, groups[0].max_size), (0.5, 3))
        self.assertEqual(
            (groups[1].nms_radius, groups[1].max_size),
            (transfusion_head.nms_min_radius, transfusion_head.post_max_size),
        )

    def test_resolve_nms_groups_rejects_a_class_in_two_groups(self) -> None:
        """
        Test that a class claimed by two NMS groups is rejected, since it would otherwise be
        suppressed against two different radii.
        """
        with self.assertRaises(ValueError):
            self._build_transfusion_head(
                nms_group_configs=[
                    NMSGroupConfig(class_names=["car", "pedestrian"], nms_radius=1.0, max_size=8),
                    NMSGroupConfig(class_names=["pedestrian"], nms_radius=2.0, max_size=8),
                ]
            )

    def test_resolve_score_threshold_groups_rejects_a_class_in_two_groups(self) -> None:
        """Test that a class claimed by two score threshold groups is rejected as ambiguous."""
        with self.assertRaises(ValueError):
            self._build_transfusion_head(
                score_threshold_group_configs=[
                    ScoreThresholdConfig(class_names=["car"], score_threshold=0.1),
                    ScoreThresholdConfig(class_names=["car", "pedestrian"], score_threshold=0.5),
                ]
            )

    def test_bf16_export_emits_the_fusion_pattern(self) -> None:
        """
        Test that the bf16 export declares an fp16 requirement, fuses only the cross-attention
        core, and emits the bare MatMul-Softmax-MatMul pattern TensorRT recognizes.
        """
        transfusion_head = self._build_transfusion_head(
            use_bf16_cross_attention=True
        ).prepare_for_export()
        self_attention = transfusion_head.decoder[0].self_attn
        cross_attention = transfusion_head.decoder[0].cross_attn

        self.assertEqual(transfusion_head.required_onnx_precision, "fp16")
        self.assertTrue(self_attention.fuse_attention)
        self.assertFalse(self_attention.use_bf16)
        self.assertTrue(cross_attention.fuse_attention)
        self.assertTrue(cross_attention.use_bf16)
        validate_module_onnx_precision(transfusion_head, OmegaConf.create({"precision": "fp16"}))

        with tempfile.TemporaryDirectory() as export_dir:
            model = self._export_attention(
                cross_attention, Path(export_dir) / "cross_attention.onnx"
            )

        # The fused path drops the max-subtraction used to stabilize the explicit graph.
        self.assertFalse(any(node.op_type in {"ReduceMax", "Sub"} for node in model.graph.node))
        # One cast per attention input: query, key, and value.
        bf16_casts = sum(
            any(
                attribute.name == "to" and attribute.i == TensorProto.BFLOAT16
                for attribute in node.attribute
            )
            for node in model.graph.node
            if node.op_type == "Cast"
        )
        self.assertEqual(bf16_casts, 3)

        softmax = next(node for node in model.graph.node if node.op_type == "Softmax")
        producers = {output: node for node in model.graph.node for output in node.output}
        self.assertEqual(producers[softmax.input[0]].op_type, "MatMul")
        consumers = [node for node in model.graph.node if softmax.output[0] in node.input]
        self.assertEqual(len(consumers), 1)
        self.assertEqual(consumers[0].op_type, "MatMul")

    def test_bf16_export_rejects_a_non_fp16_precision(self) -> None:
        """Test that a bf16 head refuses to export under any precision other than fp16."""
        transfusion_head = self._build_transfusion_head(
            use_bf16_cross_attention=True
        ).prepare_for_export()

        with self.assertRaisesRegex(
            ValueError, "TransFusionHead requires deploy.onnx.precision='fp16'"
        ):
            validate_module_onnx_precision(
                transfusion_head, OmegaConf.create({"precision": "fp32"})
            )

    def test_default_export_keeps_the_explicit_attention(self) -> None:
        """
        Test that the default export declares no precision requirement and keeps the explicit,
        max-subtracted attention graph without any bf16 cast.
        """
        transfusion_head = self._build_transfusion_head().prepare_for_export()
        cross_attention = transfusion_head.decoder[0].cross_attn

        self.assertIsNone(transfusion_head.required_onnx_precision)
        self.assertFalse(cross_attention.fuse_attention)
        self.assertFalse(cross_attention.use_bf16)

        with tempfile.TemporaryDirectory() as export_dir:
            model = self._export_attention(
                cross_attention, Path(export_dir) / "explicit_cross_attention.onnx"
            )

        producers = {output: node for node in model.graph.node for output in node.output}
        softmax = next(node for node in model.graph.node if node.op_type == "Softmax")
        subtract = producers[softmax.input[0]]
        self.assertEqual(subtract.op_type, "Sub")
        self.assertEqual(producers[subtract.input[1]].op_type, "ReduceMax")

        consumers = [node for node in model.graph.node if softmax.output[0] in node.input]
        self.assertEqual(len(consumers), 1)
        self.assertEqual(consumers[0].op_type, "Cast")
        cast_consumers = [node for node in model.graph.node if consumers[0].output[0] in node.input]
        self.assertEqual(len(cast_consumers), 1)
        self.assertEqual(cast_consumers[0].op_type, "MatMul")
        self.assertFalse(
            any(
                any(
                    attribute.name == "to" and attribute.i == TensorProto.BFLOAT16
                    for attribute in node.attribute
                )
                for node in model.graph.node
                if node.op_type == "Cast"
            )
        )


if __name__ == "__main__":
    unittest.main()
