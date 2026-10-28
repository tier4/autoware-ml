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

from collections.abc import Sequence
import math
import tempfile
import unittest
from pathlib import Path
from types import MappingProxyType

from jaxtyping import Bool, Float32, Int32, Int64
from omegaconf import OmegaConf
from onnx import TensorProto
import onnx
import torch

from autoware_ml.dataclasses.models.detection3d.head_targets import TransFusionHeadTargets
from autoware_ml.dataclasses.models.detection3d.predictions import Detection3DSamplePredictions
from autoware_ml.dataclasses.models.detection3d.head_outputs import (
    Detection3DHeadOutputs,
    TransFusionHeadOutputs,
    TransFusionSeparateHeadOutputs,
)
from autoware_ml.models.detection3d.heads.transfusions.exportable_multi_head_attention import (
    ExportableMultiheadAttention,
)
from autoware_ml.models.detection3d.heads.transfusions.positional_encoding import (
    ConstantPositionalEncoding,
    LearnedPositionalEncoding,
)
from autoware_ml.losses.detection3d.rotated_iou_loss import Rotated2DIouLoss
from autoware_ml.models.detection3d.heads.transfusions.transfusion_head import (
    NMSGroupConfig,
    TransFusionHead,
)
from autoware_ml.models.detection3d.task_modules.assigners import HungarianAssigner3D
from autoware_ml.models.detection3d.task_modules.bbox_coders import (
    ScoreThresholdConfig,
    TransFusionBBoxCoder,
)
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
        self.score_threshold_configs = [
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
        score_threshold_configs: Sequence[ScoreThresholdConfig] | None = None,
        nms_type: str | None = None,
        coder_class_names: Sequence[str] | None = None,
        nms_group_configs: Sequence[NMSGroupConfig] | None = None,
        post_center_range: Sequence[float] | None = None,
        use_bf16_cross_attention: bool = False,
        iou_loss: torch.nn.Module | None = None,
    ) -> TransFusionHead:
        """
        Build a TransFusionHead from the setUp parameters.

        Only the arguments a test actually varies are exposed. Passing None for one of them keeps
        the setUp default, except for nms_type and nms_group_configs, where None is itself the
        default: no NMS at all, and with nms_type="circle" one group per class.
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
                class_names=self.class_names if coder_class_names is None else coder_class_names,
                score_threshold_configs=(
                    self.score_threshold_configs
                    if score_threshold_configs is None
                    else score_threshold_configs
                ),
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
        self, batch_size: int | None = None
    ) -> Float32[torch.Tensor, "batch_size channels height width"]:
        """Build random BEV features of the shape the head expects, for the setUp batch unless told."""
        height, width = self.feature_map_size
        batch_size = self.batch_size if batch_size is None else batch_size
        return torch.randn(batch_size, self.in_channels, height, width, device=self.device)

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
        gt_annotated_class_masks: Bool[torch.Tensor, "batch_size num_classes"] | None = None,
        transfusion_head_outputs: TransFusionHeadOutputs | None = None,
    ) -> TransFusionHeadTargets:
        """
        Build assignment targets from the setUp ground truth and a forward pass, fresh unless a
        test passes the outputs it built by hand.
        """
        head = self.transfusion_head if transfusion_head is None else transfusion_head
        if transfusion_head_outputs is None:
            transfusion_head_outputs = self._build_head_outputs(head)
        return head.get_targets(
            outputs=transfusion_head_outputs,
            gt_bboxes_3d=self.gt_bboxes_3d,
            gt_labels_3d=self.gt_labels_3d,
            gt_valid_bboxes=(self.gt_valid_bboxes if gt_valid_bboxes is None else gt_valid_bboxes),
            feature_map_size=self.feature_map_size,
            gt_annotated_class_masks=gt_annotated_class_masks,
        )

    def _compute_losses(
        self,
        transfusion_head: TransFusionHead | None = None,
        gt_valid_bboxes: Int32[torch.Tensor, " batch_size"] | None = None,
        gt_annotated_class_masks: Bool[torch.Tensor, "batch_size num_classes"] | None = None,
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
            gt_annotated_class_masks=gt_annotated_class_masks,
        )

    def _annotated_class_masks(
        self, unannotated: dict[int, Sequence[str]]
    ) -> Bool[torch.Tensor, "batch_size num_classes"]:
        """Build an all-annotated mask, except the named classes of the given sample indices."""
        masks = torch.ones(
            (self.batch_size, self.num_classes), dtype=torch.bool, device=self.device
        )
        for sample_index, class_names in unannotated.items():
            masks[sample_index, [self.class_names.index(name) for name in class_names]] = False
        return masks

    def _decode_sample_predictions(
        self,
        transfusion_head: TransFusionHead | None = None,
        transfusion_head_outputs: TransFusionHeadOutputs | None = None,
    ) -> Sequence[Detection3DSamplePredictions]:
        """
        Decode head outputs into the per-sample predictions, unwrapping the option. The outputs
        come from a fresh forward pass unless a test passes the ones it built by hand.
        """
        head = self.transfusion_head if transfusion_head is None else transfusion_head
        if transfusion_head_outputs is None:
            transfusion_head_outputs = self._build_head_outputs(head)
        # Decoding is inference only, and the hand-built outputs carry requires_grad leaves.
        with torch.no_grad():
            sample_predictions = head.decode_outputs(
                Detection3DHeadOutputs(
                    center_head_outputs=None,
                    transfusion_head_outputs=transfusion_head_outputs,
                )
            ).detection3d_predictions
        assert sample_predictions is not None
        return sample_predictions

    def _dense_heatmap_targets(
        self,
        transfusion_head: TransFusionHead,
        gt_bboxes_3d: Float32[torch.Tensor, "batch_size max_num_gt_bboxes 9"],
        gt_labels_3d: Int32[torch.Tensor, "batch_size max_num_gt_bboxes"],
        gt_valid_bboxes: Int32[torch.Tensor, " batch_size"],
    ) -> Float32[torch.Tensor, "batch_size num_classes height width"]:
        """
        Build the dense heatmap targets for the given ground truth through get_targets.

        The dense target depends on the ground truth and the grid only, so the forward pass it is
        paired with just has to carry the right batch size. Its proposals get assigned too, but
        that part of the targets is not read here.
        """
        outputs = transfusion_head(self._build_bev_features(batch_size=gt_bboxes_3d.shape[0]))
        return transfusion_head.get_targets(
            outputs=outputs,
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_valid_bboxes=gt_valid_bboxes,
            feature_map_size=self.feature_map_size,
        ).dense_heatmaps

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
        heatmaps = self._dense_heatmap_targets(
            transfusion_head,
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=torch.zeros((1, 1), dtype=torch.int32, device=self.device),
            gt_valid_bboxes=torch.tensor([1], dtype=torch.int32, device=self.device),
        )
        return heatmaps[0, 0]

    def _build_head_outputs_from_boxes(
        self,
        bboxes_3d: Float32[torch.Tensor, "batch_size num_queries 9"],
        labels: Int64[torch.Tensor, "batch_size num_queries"],
        scores: Float32[torch.Tensor, "batch_size num_queries"],
        transfusion_head: TransFusionHead | None = None,
    ) -> TransFusionHeadOutputs:
        """
        Hand-build head outputs that decode back to the given metric boxes, labels and scores.

        The boxes go through the head's own bbox coder, so the regression tensors are exactly
        what a perfect head would emit for them, and each query's logit on its label is the
        inverse sigmoid of its score, on a confident-background floor everywhere else. Every
        per-query tensor is a leaf with requires_grad, so a test can check which slots a loss
        reaches. ``num_queries`` may span several decoder layers, the way the auxiliary head
        concatenates them along the query axis. decode_outputs reads the last ``num_proposals``
        of them, which is the slice query_labels and query_heatmap_score describe.
        """
        head = self.transfusion_head if transfusion_head is None else transfusion_head
        batch_size, num_queries, _ = bboxes_3d.shape
        # (batch_size, num_queries, code_size) -> (batch_size, code_size, num_queries), laid out
        # as (center_x, center_y, z, log dims, sin, cos, velocities) like the regression branches.
        encoded = head.bbox_coder.encode(bboxes_3d).permute(0, 2, 1)
        background_logit = float(torch.logit(torch.tensor(0.02)))
        heatmap = torch.full(
            (batch_size, self.num_classes, num_queries), background_logit, device=self.device
        )
        heatmap.scatter_(1, labels.unsqueeze(1), torch.logit(scores).unsqueeze(1))
        height, width = self.feature_map_size
        return TransFusionHeadOutputs(
            dense_heatmap=torch.zeros(
                (batch_size, self.num_classes, height, width), device=self.device
            ),
            # A unit heatmap score leaves the decoded score equal to the sigmoid of the logit.
            query_heatmap_score=torch.ones(
                (batch_size, self.num_classes, self.num_proposals), device=self.device
            ),
            query_labels=labels[:, -self.num_proposals :].contiguous(),
            separate_head_outputs=TransFusionSeparateHeadOutputs(
                heatmap=heatmap.requires_grad_(True),
                center=encoded[:, 0:2].contiguous().requires_grad_(True),
                height=encoded[:, 2:3].contiguous().requires_grad_(True),
                dim=encoded[:, 3:6].contiguous().requires_grad_(True),
                rot=encoded[:, 6:8].contiguous().requires_grad_(True),
                vel=encoded[:, 8:10].contiguous().requires_grad_(True),
            ),
        )

    def _build_head_outputs_on_gt(
        self,
        transfusion_head: TransFusionHead | None = None,
        layer_center_shifts_metres: Sequence[float] | None = None,
    ) -> TransFusionHeadOutputs:
        """
        Hand-build auxiliary head outputs whose proposals sit on the setUp ground truth.

        In every decoder layer and sample, proposal ``i`` is placed exactly on that sample's real
        gt box ``i`` with a confident score on its class, so the assigner has an unambiguous
        positive for every real box. The proposals past the sample's real boxes are parked in the
        far corner of the grid as confident background. ``layer_center_shifts_metres`` moves one
        layer's on-gt proposals along x, so a test can give a layer a non-zero regression residual
        while its proposals still overlap their boxes and stay positives.
        """
        head = self.transfusion_head if transfusion_head is None else transfusion_head
        if layer_center_shifts_metres is None:
            layer_center_shifts_metres = [0.0 for _ in range(self.num_decoder_layers)]
        self.assertEqual(len(layer_center_shifts_metres), self.num_decoder_layers)
        self.assertGreaterEqual(self.num_proposals, int(self.gt_valid_bboxes.max()))

        num_queries = self.num_decoder_layers * self.num_proposals
        far_box = torch.tensor(
            [15.5, 15.5, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device
        )
        bboxes_3d = far_box.expand(self.batch_size, num_queries, 9).clone()
        labels = torch.zeros((self.batch_size, num_queries), dtype=torch.int64, device=self.device)
        scores = torch.full((self.batch_size, num_queries), 0.02, device=self.device)
        for batch_index in range(self.batch_size):
            for layer_index in range(self.num_decoder_layers):
                for gt_index in range(int(self.gt_valid_bboxes[batch_index])):
                    query_index = layer_index * self.num_proposals + gt_index
                    bboxes_3d[batch_index, query_index] = self.gt_bboxes_3d[batch_index, gt_index]
                    bboxes_3d[batch_index, query_index, 0] += layer_center_shifts_metres[
                        layer_index
                    ]
                    labels[batch_index, query_index] = self.gt_labels_3d[batch_index, gt_index]
                    scores[batch_index, query_index] = 0.98
        return self._build_head_outputs_from_boxes(bboxes_3d, labels, scores, head)

    def _build_single_sample_outputs(
        self,
        bboxes_3d: Sequence[Sequence[float]],
        labels: Sequence[int],
        scores: Sequence[float],
        transfusion_head: TransFusionHead | None = None,
    ) -> TransFusionHeadOutputs:
        """
        Hand-build single-layer outputs for one sample, padding to num_proposals with boxes far
        outside post_center_range so the padding never survives decode_outputs.
        """
        padding = self.num_proposals - len(bboxes_3d)
        self.assertGreaterEqual(padding, 0)
        far_outside = [100.0, 100.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        return self._build_head_outputs_from_boxes(
            bboxes_3d=torch.tensor(
                [list(bboxes_3d) + [far_outside for _ in range(padding)]],
                dtype=torch.float32,
                device=self.device,
            ),
            labels=torch.tensor(
                [list(labels) + [0 for _ in range(padding)]], dtype=torch.int64, device=self.device
            ),
            scores=torch.tensor(
                [list(scores) + [0.02 for _ in range(padding)]],
                dtype=torch.float32,
                device=self.device,
            ),
            transfusion_head=transfusion_head,
        )

    def test_common_heads_must_hold_every_regression_branch(self) -> None:
        """Test that a missing regression branch is rejected at construction."""
        self.common_heads = MappingProxyType(
            {name: shape for name, shape in self.common_heads.items() if name != "rots"}
        )

        with self.assertRaisesRegex(ValueError, "must hold the branches"):
            self._build_transfusion_head()

    def test_common_heads_reject_an_unknown_branch(self) -> None:
        """
        Test that a misspelt or extra branch is rejected instead of becoming an unused head, so a
        ``vel`` typo cannot leave the velocity branch silently unbuilt and the heatmaps branch
        cannot be configured by hand.
        """
        for extra in ("vel", "heatmaps"):
            with self.subTest(extra=extra):
                self.common_heads = MappingProxyType({**self.common_heads, extra: (2, 2)})

                with self.assertRaisesRegex(ValueError, "must hold the branches"):
                    self._build_transfusion_head()

    def _drop_velocity(self) -> None:
        """Strip the vels branch and the velocity channels from the setUp head parameters."""
        self.common_heads = MappingProxyType(
            {name: shape for name, shape in self.common_heads.items() if name != "vels"}
        )
        self.code_size = 8
        self.code_weights = self.code_weights[:8]

    def test_use_velocity_requires_the_vels_branch(self) -> None:
        """Test that predicting velocity without a ``vels`` branch is rejected at construction."""
        self.common_heads = MappingProxyType(
            {name: shape for name, shape in self.common_heads.items() if name != "vels"}
        )

        with self.assertRaisesRegex(ValueError, "exactly when use_velocity is set"):
            self._build_transfusion_head()

    def test_vels_branch_requires_use_velocity(self) -> None:
        """Test that a ``vels`` branch on a head without velocity is rejected at construction."""
        self.code_size = 8
        self.code_weights = self.code_weights[:8]

        with self.assertRaisesRegex(ValueError, "exactly when use_velocity is set"):
            self._build_transfusion_head(use_velocity=False)

    def test_code_size_and_code_weights_must_match_the_box_channels(self) -> None:
        """Test that the coder and the code weights must carry 10 channels with velocity."""
        for code_size, code_weights in ((8, self.code_weights), (10, self.code_weights[:8])):
            with self.subTest(code_size=code_size, num_code_weights=len(code_weights)):
                self.code_size = code_size
                self.code_weights = code_weights

                with self.assertRaisesRegex(ValueError, "regresses 10 channels"):
                    self._build_transfusion_head()

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
        # zero rather than at it. The bound only has to catch a grossly wrong init.
        self.assertLess(abs(float(weights.mean())), 0.1)
        self.assertLess(float(weights.std()), 0.5)
        # The last conv of the dense heatmap head carries the configured init bias.
        dense_heatmap_bias = self.transfusion_head.dense_heatmap_head[-1].bias
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

    @unittest.skipUnless(torch.cuda.is_available(), "float16 autocast runs on CUDA")
    def test_forward_and_decode_under_fp16_autocast_keep_fp32_heatmaps_and_boxes(self) -> None:
        """
        Test that half BEV features under fp16 autocast reach the fp32 dense heatmap head without
        a dtype mismatch, and that the outputs decode into float32 boxes.
        """
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = self.transfusion_head(self._build_bev_features().half())

        self.assertEqual(outputs.dense_heatmap.dtype, torch.float32)
        # Mixed precision evaluation hands half outputs to decode_outputs.
        branches = outputs.separate_head_outputs
        half_outputs = TransFusionHeadOutputs.model_validate(
            {
                "dense_heatmap": outputs.dense_heatmap.half(),
                "query_heatmap_score": outputs.query_heatmap_score.half(),
                "query_labels": outputs.query_labels,
                "separate_head_outputs": TransFusionSeparateHeadOutputs.model_validate(
                    {
                        "heatmap": branches.heatmap.half(),
                        "center": branches.center.half(),
                        "height": branches.height.half(),
                        "dim": branches.dim.half(),
                        "rot": branches.rot.half(),
                        "vel": None if branches.vel is None else branches.vel.half(),
                    }
                ),
            }
        )
        sample_predictions = self._decode_sample_predictions(transfusion_head_outputs=half_outputs)
        for prediction in sample_predictions:
            self.assertEqual(prediction.bboxes_3d.dtype, torch.float32)
            self.assertEqual(prediction.scores_3d.dtype, torch.float32)
            self.assertTrue(torch.isfinite(prediction.bboxes_3d).all())

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

    def test_forward_without_velocity_emits_no_velocity(self) -> None:
        """Test that a head built without velocity emits no velocity channels."""
        self._drop_velocity()
        transfusion_head = self._build_transfusion_head(use_velocity=False)

        outputs = self._build_head_outputs(transfusion_head)

        self.assertIsNone(outputs.separate_head_outputs.vel)

    def test_forward_chains_the_decoder_layers(self) -> None:
        """
        Test that decoder layer 2 refines the query features decoder layer 1 produced, at the
        centers layer 1 predicted, and that each layer's slice of the outputs is its own
        prediction head applied to its own output. A head feeding every layer the initial proposal
        features, or reading the wrong layer's output, passes every shape test while all but one
        decoder layer is wasted, so the whole chain is replayed by hand from the first layer's
        inputs with the layer modules themselves.
        """
        # Eval mode switches dropout off, so the manual replay below reproduces the forward pass
        # exactly. Forward hooks only observe the calls. Nothing is replaced.
        head = self.transfusion_head.eval()
        captured_calls: list[tuple[dict[str, torch.Tensor], torch.Tensor]] = []

        def capture(
            module: torch.nn.Module,
            args: tuple[torch.Tensor, ...],
            kwargs: dict[str, torch.Tensor],
            output: torch.Tensor,
        ) -> None:
            captured_calls.append((kwargs, output))

        handles = [
            decoder_layer.register_forward_hook(capture, with_kwargs=True)
            for decoder_layer in head.decoder
        ]
        try:
            with torch.no_grad():
                outputs = head(self._build_bev_features())
        finally:
            for handle in handles:
                handle.remove()

        self.assertEqual(len(captured_calls), self.num_decoder_layers)
        first_inputs, first_query_feats = captured_calls[0]
        second_inputs, second_query_feats = captured_calls[1]
        centers = outputs.separate_head_outputs.center
        first_layer_centers = centers[..., : self.num_proposals]
        second_layer_centers = centers[..., self.num_proposals :]

        # The first layer changes its queries, and the second layer gets exactly those, not the
        # initial proposal features, positioned at the centers the first layer predicted.
        self.assertFalse(torch.equal(first_query_feats, first_inputs["query"]))
        self.assertTrue(torch.equal(second_inputs["query"], first_query_feats))
        self.assertTrue(
            torch.equal(second_inputs["query_pos"], first_layer_centers.permute(0, 2, 1))
        )

        # Replay both steps from the first layer's inputs with the layer modules themselves.
        with torch.no_grad():
            replayed_first = head.decoder[0](
                query=first_inputs["query"],
                key=first_inputs["key"],
                query_pos=first_inputs["query_pos"],
                key_pos=first_inputs["key_pos"],
            )
            replayed_first_prediction = head.prediction_heads[0](replayed_first)
            replayed_first_centers = replayed_first_prediction.center + first_inputs[
                "query_pos"
            ].permute(0, 2, 1)
            replayed_second = head.decoder[1](
                query=replayed_first,
                key=first_inputs["key"],
                query_pos=replayed_first_centers.permute(0, 2, 1),
                key_pos=first_inputs["key_pos"],
            )
            replayed_second_prediction = head.prediction_heads[1](replayed_second)
            replayed_second_centers = replayed_second_prediction.center + replayed_first_centers

        self.assertTrue(torch.allclose(second_query_feats, replayed_second, atol=1e-6))
        self.assertTrue(torch.allclose(first_layer_centers, replayed_first_centers, atol=1e-6))
        self.assertTrue(torch.allclose(second_layer_centers, replayed_second_centers, atol=1e-6))
        self.assertTrue(
            torch.allclose(
                outputs.separate_head_outputs.heatmap[..., self.num_proposals :],
                replayed_second_prediction.heatmap,
                atol=1e-6,
            )
        )
        # The second layer is a refinement, not a copy, of the first.
        self.assertFalse(torch.allclose(second_layer_centers, first_layer_centers))

    def test_get_targets_dense_heatmap_peaks_at_the_box_center(self) -> None:
        """
        Test that a box's dense heatmap target peaks at the cell holding its center, on that box's
        own class channel only.
        """
        heatmaps = self._get_targets().dense_heatmaps
        height, width = self.feature_map_size

        self.assertEqual(heatmaps.shape, (self.batch_size, self.num_classes, height, width))
        # Sample 0 box 0 is a car at (2, 2) and box 1 a pedestrian at (8, 8).
        self.assertAlmostEqual(float(heatmaps[0, 0, 2, 2]), 1.0, places=5)
        self.assertAlmostEqual(float(heatmaps[0, 1, 8, 8]), 1.0, places=5)
        # The pedestrian's cell is empty on the car channel.
        self.assertAlmostEqual(float(heatmaps[0, 0, 8, 8]), 0.0, places=5)
        self.assertTrue(bool(((heatmaps >= 0.0) & (heatmaps <= 1.0)).all()))

    def test_get_targets_dense_heatmap_ignores_padded_boxes(self) -> None:
        """
        Test that boxes past gt_valid_bboxes contribute nothing, so the zero padding every batch
        carries is not drawn at the origin as if it were a real box.
        """
        heatmaps = self._get_targets().dense_heatmaps

        # Sample 1 has a single real box at (4, 10). Its two padded rows sit at the origin.
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
        Test that the classes named for local-max pooling keep only their peak, the classes left
        out are passed through untouched, and a head naming no class leaves the map as it is.

        This is the one test that reaches into a private method. The suppression only exists
        inside forward(), between the dense heatmap head and the top-k proposal selection, and
        the heatmap it consumes is a network output: driving a chosen peak-and-neighbour pattern
        through the shared block and the dense heatmap convolutions from BEV features is not
        possible without rewriting their weights, which would test the rewrite rather than the
        suppression. forward() exposes only the selected proposals, whose order top-k leaves
        unspecified, so the cell-level contract is asserted on the helper directly.
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
        untouched = self.transfusion_head._suppress_dense_heatmaps(heatmaps)

        # "car" is pooled: the peak survives and the neighbour is suppressed.
        self.assertAlmostEqual(float(suppressed[0, 0, 5, 5]), 0.9, places=5)
        self.assertAlmostEqual(float(suppressed[0, 0, 5, 6]), 0.0, places=5)
        # "pedestrian" is not pooled, so both cells are kept.
        self.assertAlmostEqual(float(suppressed[0, 1, 5, 5]), 0.9, places=5)
        self.assertAlmostEqual(float(suppressed[0, 1, 5, 6]), 0.4, places=5)
        # The setUp head names no pooling class, so nothing is suppressed at all.
        self.assertTrue(torch.equal(untouched, heatmaps))

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
        height, width = self.feature_map_size
        self.assertEqual(
            targets.dense_heatmaps.shape,
            (self.batch_size, self.num_classes, height, width),
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

    def test_get_targets_supervises_every_class_without_annotated_masks(self) -> None:
        """Test that without an annotated mask every class of every query is weighed."""
        targets = self._get_targets()

        self.assertTrue(bool((targets.class_weights == 1.0).all()))
        self.assertTrue(bool((targets.label_weights == 1.0).all()))

    def test_get_targets_supervises_every_class_of_an_all_true_mask(self) -> None:
        """Test that an all-true annotated mask weighs every class, like no mask at all."""
        targets = self._get_targets(gt_annotated_class_masks=self._annotated_class_masks({}))

        self.assertTrue(bool((targets.class_weights == 1.0).all()))
        self.assertTrue(bool((targets.label_weights == 1.0).all()))

    def test_get_targets_rejects_an_annotated_mask_of_the_wrong_shape(self) -> None:
        """Test that an annotated mask not shaped (batch_size, num_classes) is rejected."""
        with self.assertRaises(ValueError):
            self._get_targets(
                gt_annotated_class_masks=torch.ones(
                    (self.batch_size,), dtype=torch.bool, device=self.device
                )
            )

    def test_get_targets_drops_unannotated_classes_of_a_sample(self) -> None:
        """
        Test that a class marked unannotated in a sample is zeroed in that sample's class weights
        and in its negative queries only, while positives and the other sample keep full weights.
        """
        head = self.transfusion_head
        pedestrian_id = self.class_names.index("pedestrian")
        car_id = self.class_names.index("car")

        targets = self._get_targets(
            head, gt_annotated_class_masks=self._annotated_class_masks({1: ["pedestrian"]})
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
        """
        Test that loss reports a heatmap term, a cls and bbox term per layer, and their sum, on a
        batch where the assigner finds real positives: the proposals sit on their gt boxes, half a
        metre off along x, so every layer has matched proposals with a non-zero box residual.
        """
        outputs = self._build_head_outputs_on_gt(layer_center_shifts_metres=[0.5, 0.5])

        targets = self._get_targets(transfusion_head_outputs=outputs)
        losses = self._compute_losses(transfusion_head_outputs=outputs)

        # Every real box of every layer is matched, so the positives are real overlaps, not the
        # forced matches the Hungarian assigner makes between random proposals and boxes.
        self.assertEqual(targets.num_pos, self.num_decoder_layers * int(self.gt_valid_bboxes.sum()))
        self.assertGreater(targets.num_pos, 0)
        self.assertGreater(float(losses["matched_ious"]), 0.5)
        self.assertIn("loss_heatmap", losses)
        self.assertIn("loss", losses)
        # The last layer is logged as layer_-1, the rest by index.
        for prefix in ("layer_0", "layer_-1"):
            self.assertIn(f"{prefix}_loss_cls", losses)
            loss_bbox = losses[f"{prefix}_loss_bbox"]
            assert isinstance(loss_bbox, torch.Tensor)
            self.assertGreater(float(loss_bbox.detach()), 0.0)
            self.assertTrue(loss_bbox.requires_grad)
        for key, value in losses.items():
            self.assertTrue(torch.isfinite(torch.as_tensor(value)).all(), msg=key)
        component_total = sum(
            float(torch.as_tensor(value).detach())
            for key, value in losses.items()
            if "loss" in key and key != "loss"
        )
        total_loss = torch.as_tensor(losses["loss"]).detach()
        self.assertAlmostEqual(float(total_loss), component_total, places=4)

    def test_loss_bbox_reaches_only_the_positive_proposals(self) -> None:
        """
        Test that the box loss carries gradient back to every matched proposal's regression
        outputs and to no unmatched one, since the negatives carry a zero regression weight.
        """
        outputs = self._build_head_outputs_on_gt(layer_center_shifts_metres=[0.5, 0.5])
        targets = self._get_targets(transfusion_head_outputs=outputs)
        losses = self._compute_losses(transfusion_head_outputs=outputs)
        loss_bbox = sum(value for key, value in losses.items() if key.endswith("_loss_bbox"))
        assert isinstance(loss_bbox, torch.Tensor)
        self.assertGreater(float(loss_bbox.detach()), 0.0)

        loss_bbox.backward()

        # (batch_size, num_queries): True where the assigner matched the proposal to a real box.
        positive_masks = targets.labels < self.num_classes
        self.assertGreater(int(positive_masks.sum()), 0)
        center_gradients = outputs.separate_head_outputs.center.grad
        assert center_gradients is not None
        # (batch_size, 2, num_queries) -> (batch_size, num_queries): the x residual is non-zero
        # for every positive, so each one receives a gradient. No negative receives any.
        gradient_masks = center_gradients.abs().sum(dim=1) > 0.0
        self.assertTrue(bool(gradient_masks[positive_masks].all()))
        self.assertFalse(bool(gradient_masks[~positive_masks].any()))

    def test_loss_supervises_each_layer_with_its_own_slice(self) -> None:
        """
        Test that each layer's box loss reads that layer's slice of the auxiliary outputs: moving
        only the second layer's proposals off their boxes raises the layer_-1 term alone, and
        moving only the first layer's raises the layer_0 term alone.
        """
        second_layer_shifted = self._compute_losses(
            transfusion_head_outputs=self._build_head_outputs_on_gt(
                layer_center_shifts_metres=[0.0, 0.5]
            )
        )
        first_layer_shifted = self._compute_losses(
            transfusion_head_outputs=self._build_head_outputs_on_gt(
                layer_center_shifts_metres=[0.5, 0.0]
            )
        )

        self.assertAlmostEqual(float(second_layer_shifted["layer_0_loss_bbox"]), 0.0, places=5)
        self.assertGreater(float(second_layer_shifted["layer_-1_loss_bbox"]), 0.0)
        self.assertGreater(float(first_layer_shifted["layer_0_loss_bbox"]), 0.0)
        self.assertAlmostEqual(float(first_layer_shifted["layer_-1_loss_bbox"]), 0.0, places=5)

    def test_loss_backward_reaches_the_head_parameters(self) -> None:
        """
        Test that the total loss of a forward pass still carries a graph back to the head's
        parameters, since a detached total would silently train nothing.
        """
        outputs = self._build_head_outputs()
        targets = self._get_targets(transfusion_head_outputs=outputs)
        losses = self._compute_losses(transfusion_head_outputs=outputs)
        total_loss = losses["loss"]
        assert isinstance(total_loss, torch.Tensor)
        self.assertGreater(targets.num_pos, 0)
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

    def test_loss_heatmap_skips_unannotated_classes_of_a_sample(self) -> None:
        """
        Test that the dense heatmap of a class gets no gradient in a sample where it is marked
        unannotated, while it still does in the other sample and for the other classes.
        """
        head = self.transfusion_head
        pedestrian_id = self.class_names.index("pedestrian")
        car_id = self.class_names.index("car")
        outputs = self._build_head_outputs(head)
        outputs.dense_heatmap.retain_grad()

        losses = self._compute_losses(
            head,
            gt_annotated_class_masks=self._annotated_class_masks({1: ["pedestrian"]}),
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
        head = self.transfusion_head
        car_id = self.class_names.index("car")
        outputs = self._build_head_outputs(head)
        unannotated = self._annotated_class_masks({0: ["pedestrian"], 1: ["pedestrian"]})

        losses = self._compute_losses(
            head, gt_annotated_class_masks=unannotated, transfusion_head_outputs=outputs
        )
        targets = head.get_targets(
            outputs=outputs,
            gt_bboxes_3d=self.gt_bboxes_3d,
            gt_labels_3d=self.gt_labels_3d,
            gt_valid_bboxes=self.gt_valid_bboxes,
            feature_map_size=self.feature_map_size,
            gt_annotated_class_masks=unannotated,
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
            labels_to_ignore_rotation=labels_to_ignore_rotation,
            loss_weight=loss_weight,
        )

    def _iou_terms(
        self, losses: MappingProxyType[str, torch.Tensor | float]
    ) -> Sequence[torch.Tensor]:
        """Pick the per-layer IoU terms out of a loss mapping, one per decoder layer."""
        terms = [
            value
            for key, value in losses.items()
            if key.endswith("_loss_iou") and isinstance(value, torch.Tensor)
        ]
        self.assertEqual(len(terms), self.num_decoder_layers)
        return terms

    def test_loss_iou_without_positives_is_zero_but_attached(self) -> None:
        """
        Test that a batch with only padded boxes, hence no matched proposals, contributes an IoU
        term of zero per layer that still keeps the graph.
        """
        head = self._build_transfusion_head(iou_loss=self._build_iou_loss())

        losses = self._compute_losses(
            head,
            gt_valid_bboxes=torch.zeros(self.batch_size, dtype=torch.int32, device=self.device),
            transfusion_head_outputs=self._build_head_outputs_on_gt(head),
        )

        for loss_iou in self._iou_terms(losses):
            self.assertEqual(float(loss_iou.detach()), 0.0)
            self.assertTrue(loss_iou.requires_grad)

    @unittest.skipUnless(torch.cuda.is_available(), "the rotated IoU op requires CUDA")
    def test_loss_iou_is_zero_for_perfect_predictions(self) -> None:
        """
        Test that proposals sitting exactly on their gt boxes are matched with an IoU of one and
        carry no IoU loss in any layer.
        """
        head = self._build_transfusion_head(iou_loss=self._build_iou_loss())
        outputs = self._build_head_outputs_on_gt(head)

        targets = self._get_targets(head, transfusion_head_outputs=outputs)
        losses = self._compute_losses(head, transfusion_head_outputs=outputs)

        self.assertGreater(targets.num_pos, 0)
        self.assertAlmostEqual(targets.matched_iou, 1.0, places=4)
        for loss_iou in self._iou_terms(losses):
            self.assertAlmostEqual(float(loss_iou.detach()), 0.0, places=4)

    @unittest.skipUnless(torch.cuda.is_available(), "the rotated IoU op requires CUDA")
    def test_loss_iou_grows_as_the_prediction_drifts_from_the_target(self) -> None:
        """Test that shifting the matched proposals away from their gt boxes raises the IoU loss."""
        head = self._build_transfusion_head(iou_loss=self._build_iou_loss())

        iou_losses = []
        for shift_metres in (0.0, 0.5, 1.0):
            losses = self._compute_losses(
                head,
                transfusion_head_outputs=self._build_head_outputs_on_gt(
                    head, layer_center_shifts_metres=[shift_metres, shift_metres]
                ),
            )
            iou_losses.append(sum(float(term.detach()) for term in self._iou_terms(losses)))

        self.assertLess(iou_losses[0], iou_losses[1])
        self.assertLess(iou_losses[1], iou_losses[2])

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
    def test_loss_iou_under_fp16_autocast_matches_float32(self) -> None:
        """
        Test that the IoU term computed under fp16 autocast is finite, float32 and equal to the
        full precision term: the boxes are decoded and intersected outside autocast.
        """
        head = self._build_transfusion_head(iou_loss=self._build_iou_loss())
        outputs = self._build_head_outputs_on_gt(head, layer_center_shifts_metres=[0.5, 0.5])

        expected = self._iou_terms(self._compute_losses(head, transfusion_head_outputs=outputs))
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            actual = self._iou_terms(self._compute_losses(head, transfusion_head_outputs=outputs))

        for term, reference in zip(actual, expected):
            self.assertEqual(term.dtype, torch.float32)
            self.assertTrue(torch.isfinite(term).all())
            torch.testing.assert_close(term, reference, rtol=1e-3, atol=1e-4)

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
            score_threshold_configs=[
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

    def test_filter_bboxes_center_range_keeps_only_centers_inside_the_range(self) -> None:
        """Test that the center-range filter drops boxes whose centers fall outside the range."""
        bbox_centers = torch.tensor(
            [[[8.0, 8.0, 0.0], [100.0, 8.0, 0.0], [8.0, 8.0, 100.0]]], device=self.device
        )

        keep_masks = self.transfusion_head.bbox_coder.filter_bboxes_center_range(
            bbox_centers=bbox_centers
        )

        self.assertEqual(keep_masks.tolist(), [[True, False, False]])

    def test_decode_outputs_nms_groups_suppress_within_a_group_only(self) -> None:
        """
        Test that two classes sharing an NMS group suppress each other, while the same pair split
        across groups both survive, which is what putting a group in one channel buys.
        """
        # A car and a pedestrian half a metre apart, inside the one-metre NMS radius.
        bboxes_3d = [
            [8.0, 8.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [8.5, 8.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        ]
        together_head = self._build_transfusion_head(
            nms_type="circle",
            nms_group_configs=[
                NMSGroupConfig(class_names=self.class_names, nms_radius=1.0, max_size=8)
            ],
        )
        apart_head = self._build_transfusion_head(
            nms_type="circle",
            nms_group_configs=[
                NMSGroupConfig(class_names=["car"], nms_radius=1.0, max_size=8),
                NMSGroupConfig(class_names=["pedestrian"], nms_radius=1.0, max_size=8),
            ],
        )

        together = self._decode_sample_predictions(
            together_head,
            self._build_single_sample_outputs(
                bboxes_3d, labels=[0, 1], scores=[0.9, 0.8], transfusion_head=together_head
            ),
        )[0]
        apart = self._decode_sample_predictions(
            apart_head,
            self._build_single_sample_outputs(
                bboxes_3d, labels=[0, 1], scores=[0.9, 0.8], transfusion_head=apart_head
            ),
        )[0]

        # One group: the weaker pedestrian is suppressed by the car.
        self.assertEqual(together.labels_3d.tolist(), [0])
        self.assertAlmostEqual(float(together.scores_3d[0]), 0.9, places=5)
        # Separate groups: both survive, in proposal order.
        self.assertEqual(apart.labels_3d.tolist(), [0, 1])

    def test_decode_outputs_nms_never_revives_a_rejected_proposal(self) -> None:
        """
        Test that a proposal already rejected by the center-range filter stays rejected and cannot
        suppress a survivor, since it takes no part in the NMS.
        """
        transfusion_head = self._build_transfusion_head(
            nms_type="circle",
            nms_group_configs=[
                NMSGroupConfig(class_names=self.class_names, nms_radius=1.0, max_size=8)
            ],
        )
        # Two cars half a metre apart in BEV. The stronger one floats 50 m up, outside the
        # post_center_range, so the center-range filter drops it before the NMS.
        bboxes_3d = [
            [8.0, 8.0, 50.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [8.5, 8.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        ]

        prediction = self._decode_sample_predictions(
            transfusion_head,
            self._build_single_sample_outputs(
                bboxes_3d, labels=[0, 0], scores=[0.9, 0.8], transfusion_head=transfusion_head
            ),
        )[0]

        self.assertEqual(prediction.scores_3d.shape, (1,))
        self.assertAlmostEqual(float(prediction.scores_3d[0]), 0.8, places=5)
        self.assertAlmostEqual(float(prediction.bboxes_3d[0, 0]), 8.5, places=4)

    def test_decode_outputs_nms_type_none_runs_no_nms(self) -> None:
        """
        Test that nms_type=None skips NMS entirely, so two overlapping cars both survive even
        though the head default radius would suppress the weaker one.
        """
        bboxes_3d = [
            [8.0, 8.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [8.5, 8.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        ]
        transfusion_head = self._build_transfusion_head(nms_type=None)
        self.assertIsNone(transfusion_head.nms_groups)

        prediction = self._decode_sample_predictions(
            transfusion_head,
            self._build_single_sample_outputs(
                bboxes_3d, labels=[0, 0], scores=[0.9, 0.8], transfusion_head=transfusion_head
            ),
        )[0]

        self.assertEqual(prediction.scores_3d.shape, (2,))

    def test_decode_outputs_circle_nms_without_groups_suppresses_per_class(self) -> None:
        """
        Test that nms_type="circle" with no groups falls back to one group per class with the
        head defaults, as on main: two cars inside the radius collapse to the stronger one, while
        a car and a pedestrian at the same distance both survive.
        """
        bboxes_3d = [
            [8.0, 8.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [8.5, 8.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        ]
        transfusion_head = self._build_transfusion_head(nms_type="circle")
        groups = transfusion_head.nms_groups
        assert groups is not None
        self.assertEqual([list(group.class_ids) for group in groups], [[0], [1]])
        self.assertEqual(
            {(group.nms_radius, group.max_size) for group in groups},
            {(transfusion_head.nms_min_radius, transfusion_head.post_max_size)},
        )

        same_class = self._decode_sample_predictions(
            transfusion_head,
            self._build_single_sample_outputs(
                bboxes_3d, labels=[0, 0], scores=[0.9, 0.8], transfusion_head=transfusion_head
            ),
        )[0]
        different_classes = self._decode_sample_predictions(
            transfusion_head,
            self._build_single_sample_outputs(
                bboxes_3d, labels=[0, 1], scores=[0.9, 0.8], transfusion_head=transfusion_head
            ),
        )[0]

        self.assertEqual(same_class.scores_3d.shape, (1,))
        self.assertAlmostEqual(float(same_class.scores_3d[0]), 0.9, places=5)
        self.assertEqual(different_classes.scores_3d.shape, (2,))

    def test_nms_groups_without_nms_type_are_rejected(self) -> None:
        """
        Test that nms_group_configs with nms_type=None is rejected, since the groups would
        otherwise be configured and silently never applied.
        """
        with self.assertRaises(ValueError):
            self._build_transfusion_head(
                nms_type=None,
                nms_group_configs=[NMSGroupConfig(class_names=["car"], nms_radius=1.0, max_size=8)],
            )

    def test_transfusion_nms_groups_cap_zero_radius_groups_by_score(self) -> None:
        """
        Test that a group with an NMS radius of 0 applies no suppression and only caps its
        proposals by score, as on main: two cars half a metre apart both survive under a large
        cap, and only the stronger one survives under a cap of one.
        """
        bboxes_3d = [
            [8.0, 8.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [8.5, 8.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        ]
        uncapped_head = self._build_transfusion_head(
            nms_type="circle",
            nms_group_configs=[NMSGroupConfig(class_names=["car"], nms_radius=0.0, max_size=8)],
        )
        capped_head = self._build_transfusion_head(
            nms_type="circle",
            nms_group_configs=[NMSGroupConfig(class_names=["car"], nms_radius=0.0, max_size=1)],
        )

        uncapped = self._decode_sample_predictions(
            uncapped_head,
            self._build_single_sample_outputs(
                bboxes_3d, labels=[0, 0], scores=[0.8, 0.9], transfusion_head=uncapped_head
            ),
        )[0]
        capped = self._decode_sample_predictions(
            capped_head,
            self._build_single_sample_outputs(
                bboxes_3d, labels=[0, 0], scores=[0.8, 0.9], transfusion_head=capped_head
            ),
        )[0]

        # No suppression: both cars survive even though they are within any usual radius.
        self.assertEqual(uncapped.scores_3d.shape, (2,))
        # The cap keeps the highest score only, which is the second proposal.
        self.assertEqual(capped.scores_3d.shape, (1,))
        self.assertAlmostEqual(float(capped.scores_3d[0]), 0.9, places=5)
        self.assertAlmostEqual(float(capped.bboxes_3d[0, 0]), 8.5, places=4)

    def test_decode_outputs_keeps_survivors_in_proposal_order(self) -> None:
        """
        Test that decode_outputs drops the rejected proposals, keeps the survivors in proposal
        order rather than by score, and carries their scores and labels along.
        """
        inside = 8.0
        # Outside post_center_range, so the center-range filter rejects the proposal.
        outside = 100.0
        bboxes_3d = torch.tensor(
            [
                [
                    [outside, inside, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, inside, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    [2.0, inside, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    [outside, inside, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                ],
                [
                    [3.0, inside, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    [outside, inside, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    [outside, inside, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    [4.0, inside, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                ],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        labels = torch.tensor([[0, 1, 0, 1], [0, 1, 0, 1]], dtype=torch.int64, device=self.device)
        scores = torch.tensor(
            [[0.1, 0.9, 0.5, 0.3], [0.7, 0.2, 0.6, 0.4]], dtype=torch.float32, device=self.device
        )

        predictions = self._decode_sample_predictions(
            transfusion_head_outputs=self._build_head_outputs_from_boxes(bboxes_3d, labels, scores)
        )

        self.assertEqual(len(predictions), 2)
        # Slots 1 and 2 of sample 0, in proposal order, so the 0.9 and 0.5 scores, not sorted.
        self.assertTrue(torch.allclose(predictions[0].bboxes_3d[:, 0], bboxes_3d[0, 1:3, 0]))
        self.assertTrue(torch.allclose(predictions[0].scores_3d, scores[0, 1:3], atol=1e-5))
        self.assertTrue(torch.equal(predictions[0].labels_3d, labels[0, 1:3]))
        # Slots 0 and 3 of sample 1.
        self.assertTrue(torch.allclose(predictions[1].bboxes_3d[:, 0], bboxes_3d[1, [0, 3], 0]))
        self.assertTrue(torch.allclose(predictions[1].scores_3d, scores[1, [0, 3]], atol=1e-5))
        self.assertTrue(torch.equal(predictions[1].labels_3d, labels[1, [0, 3]]))

    def test_class_names_resolve_to_sorted_deduplicated_ids(self) -> None:
        """Test that class names given to the head resolve to sorted, deduplicated class ids."""
        transfusion_head = self._build_transfusion_head(
            dense_heatmap_pooling_class_names=["pedestrian", "car", "car"]
        )

        self.assertEqual(list(transfusion_head.dense_heatmap_pooling_class_ids), [0, 1])

    def test_unknown_class_name_is_rejected(self) -> None:
        """Test that a class name outside class_names is rejected rather than silently ignored."""
        with self.assertRaises(ValueError):
            self._build_transfusion_head(dense_heatmap_pooling_class_names=["helicopter"])

    def test_resolve_nms_groups_fills_in_the_head_defaults(self) -> None:
        """Test that a group leaving nms_radius or max_size unset inherits the head's defaults."""
        transfusion_head = self._build_transfusion_head(
            nms_type="circle",
            nms_group_configs=[
                NMSGroupConfig(class_names=["car"], nms_radius=0.5, max_size=3),
                NMSGroupConfig(class_names=["pedestrian"], nms_radius=None, max_size=None),
            ],
        )
        groups = transfusion_head.nms_groups
        assert groups is not None

        self.assertEqual([list(group.class_ids) for group in groups], [[0], [1]])
        self.assertEqual((groups[0].nms_radius, groups[0].max_size), (0.5, 3))
        self.assertEqual(
            (groups[1].nms_radius, groups[1].max_size),
            (transfusion_head.nms_min_radius, transfusion_head.post_max_size),
        )

    def test_resolve_nms_groups_keeps_a_zero_radius(self) -> None:
        """
        Test that an explicit nms_radius of 0 is kept rather than replaced by the head default,
        since 0 means no suppression for the group.
        """
        transfusion_head = self._build_transfusion_head(
            nms_type="circle",
            nms_group_configs=[NMSGroupConfig(class_names=["car"], nms_radius=0.0, max_size=5)],
        )
        groups = transfusion_head.nms_groups
        assert groups is not None

        # The setUp head default is 1.0, which a falsy check would have substituted.
        self.assertGreater(transfusion_head.nms_min_radius, 0.0)
        self.assertEqual((groups[0].nms_radius, groups[0].max_size), (0.0, 5))

    def test_resolve_nms_groups_rejects_a_class_in_two_groups(self) -> None:
        """
        Test that a class claimed by two NMS groups is rejected, since it would otherwise be
        suppressed against two different radii.
        """
        with self.assertRaises(ValueError):
            self._build_transfusion_head(
                nms_type="circle",
                nms_group_configs=[
                    NMSGroupConfig(class_names=["car", "pedestrian"], nms_radius=1.0, max_size=8),
                    NMSGroupConfig(class_names=["pedestrian"], nms_radius=2.0, max_size=8),
                ],
            )

    def test_rejects_a_bbox_coder_with_other_class_names(self) -> None:
        """Test that a coder resolved against different class names is rejected at construction."""
        with self.assertRaisesRegex(ValueError, "class_names"):
            self._build_transfusion_head(coder_class_names=list(reversed(self.class_names)))

    def test_export_freezes_the_key_positional_encoding_for_the_given_bev_shape(self) -> None:
        """
        Test that exporting with the BEV shape replaces the learned key positional encoding of
        every decoder layer by its precomputed embedding of that grid, without changing the
        outputs, and that exporting without a shape keeps the encoding a runtime computation.
        """
        transfusion_head = self._build_transfusion_head().eval()
        height, width = self.feature_map_size
        bev_features = self._build_bev_features(batch_size=1)

        runtime_head = transfusion_head.prepare_for_export()
        frozen_head = transfusion_head.prepare_for_export(bev_shape=(height, width))

        for decoder_layer in runtime_head.decoder:
            self.assertIsInstance(decoder_layer.key_pos_encoding, LearnedPositionalEncoding)
        for decoder_layer in frozen_head.decoder:
            encoding = decoder_layer.key_pos_encoding
            self.assertIsInstance(encoding, ConstantPositionalEncoding)
            self.assertEqual(encoding.embedding.shape, (1, height * width, self.hidden_channel))
            # The query positions are the proposals, which move with the input, so they stay
            # learned.
            self.assertIsInstance(decoder_layer.query_pos_encoding, LearnedPositionalEncoding)
        self.assertFalse(frozen_head.training)

        with torch.no_grad():
            reference = transfusion_head(bev_features)
            frozen = frozen_head(bev_features)

        self.assertTrue(torch.equal(reference.dense_heatmap, frozen.dense_heatmap))
        self.assertTrue(
            torch.allclose(
                reference.separate_head_outputs.center,
                frozen.separate_head_outputs.center,
                atol=1e-6,
            )
        )
        with self.assertRaisesRegex(ValueError, f"frozen for {height * width} positions"):
            frozen_head(self._build_bev_features(batch_size=1)[..., : height // 2, : width // 2])

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
