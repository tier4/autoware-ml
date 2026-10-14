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

"""Unit tests for the TransFusion detection model.

The model contract (forward, losses, decoding, eval output, input validation and the export
specification) is exercised on any device with stub voxel and middle encoders around the real
BEV backbone, neck and TransFusion head. The real lidar stack, whose sparse middle encoder runs
through spconv CUDA kernels, is exercised end to end on the GPU only.
"""

from __future__ import annotations

from types import MappingProxyType
import unittest

from jaxtyping import Float32, Int32
import torch
import torch.nn as nn

from autoware_ml.dataclasses.batch.detection3d import Detection3DGTBatch
from autoware_ml.dataclasses.batch.sample_batch import ModelGTBatch
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.dataclasses.models.model_outputs import ModelOutputs
from autoware_ml.models.detection3d.backbones.second import SECONDBackbone
from autoware_ml.models.detection3d.heads.transfusions.exportable_multi_head_attention import (
    ExportableMultiheadAttention,
)
from autoware_ml.models.detection3d.heads.transfusions.transfusion_head import (
    ScoreThresholdConfig,
    TransFusionHead,
)
from autoware_ml.models.detection3d.main_modules.transfusion import (
    TransFusionDetectionModel,
    _TransFusionExportWrapper,
    _format_transfusion_export_outputs,
)
from autoware_ml.models.detection3d.necks.second_fpn import SECONDFPN
from autoware_ml.models.detection3d.task_modules.assigners import HungarianAssigner3D
from autoware_ml.models.detection3d.task_modules.bbox_coders import TransFusionBBoxCoder
from autoware_ml.models.detection3d.task_modules.match_costs import (
    BBoxBEVL1Cost,
    ClassificationCost,
    IoU3DCost,
)
from autoware_ml.models.module_base_model import LogDictConfigs
from autoware_ml.ops.voxelization.voxelization import VoxelsData
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor


class _StubVoxelEncoder(nn.Module):
    """Mean-pool the points of each voxel and project them to the middle encoder channels."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(
        self,
        voxels: Float32[torch.Tensor, "num_voxels max_points channels"],
        num_points: Int32[torch.Tensor, " num_voxels"],
        coords: Int32[torch.Tensor, "num_voxels 4"],
    ) -> Float32[torch.Tensor, "num_voxels out_channels"]:
        voxel_mean = voxels.sum(dim=1) / num_points.clamp(min=1).unsqueeze(1).to(voxels.dtype)
        return self.linear(voxel_mean)


class _StubMiddleEncoder(nn.Module):
    """Scatter voxel features onto a dense ``(B, C, H, W)`` canvas at their ``(y, x)`` cells.

    Coordinates arrive in the ``(batch, z, y, x)`` layout the model builds. The stub exposes the
    ``prepare_for_export`` interface of the sparse encoder so the model can export on the CPU.
    """

    def __init__(self, bev_shape: tuple[int, int], exportable: bool = False) -> None:
        super().__init__()
        self._bev_shape = bev_shape
        self.exportable = exportable

    def forward(
        self,
        voxel_features: Float32[torch.Tensor, "num_voxels channels"],
        coords: Int32[torch.Tensor, "num_voxels 4"],
        batch_size: int,
    ) -> Float32[torch.Tensor, "batch_size channels height width"]:
        height, width = self._bev_shape
        canvas = voxel_features.new_zeros(batch_size, voxel_features.shape[1], height, width)
        batch_indices, y, x = coords[:, 0].long(), coords[:, 2].long(), coords[:, 3].long()
        canvas[batch_indices, :, y, x] = voxel_features
        return canvas

    def prepare_for_export(self) -> _StubMiddleEncoder:
        return _StubMiddleEncoder(self._bev_shape, exportable=True)


class _TransFusionDetectionModelTestCase(unittest.TestCase):
    """Shared configuration and builders for the TransFusion detection model test cases.

    Subclasses set the scene geometry and channel layout declared below in their own ``setUp``
    and build the lidar encoders; the head, ground truth and batch builders derive from them.
    """

    point_cloud_range: list[float]
    voxel_size: list[float]
    out_size_factor: int
    bev_shape: tuple[int, int]
    neck_channels: int

    def setUp(self) -> None:
        """Set up the batch layout, class names and head sizes shared by every test case."""
        torch.manual_seed(0)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = 2
        self.class_names = ["car", "pedestrian", "cyclist"]
        self.num_classes = len(self.class_names)
        self.point_channels = 4
        self.num_proposals = 4
        self.num_box_channels = 10
        self.log_dict_configs = LogDictConfigs(
            on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
        )

    def _build_head(self) -> TransFusionHead:
        """Build a small auxiliary TransFusion head on the BEV grid."""
        return TransFusionHead(
            num_proposals=self.num_proposals,
            auxiliary=True,
            in_channels=self.neck_channels,
            hidden_channel=8,
            class_names=self.class_names,
            num_decoder_layers=1,
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

    def _build_model(
        self,
        pts_voxel_encoder: nn.Module,
        pts_middle_encoder: nn.Module,
        pts_backbone: nn.Module,
        pts_neck: nn.Module,
    ) -> TransFusionDetectionModel:
        """Build the detector around the given lidar modules and a fresh head."""
        return TransFusionDetectionModel(
            data_preprocessor=DataPreprocessor(preprocessor_modules=[]),
            pts_voxel_encoder=pts_voxel_encoder,
            pts_middle_encoder=pts_middle_encoder,
            pts_backbone=pts_backbone,
            pts_neck=pts_neck,
            bbox_head=self._build_head(),
            log_dict_configs=self.log_dict_configs,
        ).to(self.device)

    def _build_voxels_data(
        self,
        grid_shape: tuple[int, int, int],
        num_voxels_per_sample: int,
        max_points: int,
    ) -> VoxelsData:
        """Build voxels in distinct ``(x, y, z)`` cells of a ``(X, Y, Z)`` grid per sample.

        Point values are drawn inside the point cloud range with intensities in ``[0, 255]`` so
        that range-normalizing voxel encoders see realistic inputs.
        """
        num_voxels = self.batch_size * num_voxels_per_sample
        width, height, depth = grid_shape
        cells = torch.stack(
            [
                torch.randperm(width * height * depth, device=self.device)[:num_voxels_per_sample]
                for _ in range(self.batch_size)
            ]
        ).flatten()
        point_min = torch.tensor([*self.point_cloud_range[:3], 0.0], device=self.device)
        point_max = torch.tensor([*self.point_cloud_range[3:], 255.0], device=self.device)
        voxels = (
            torch.rand(
                num_voxels, max_points, self.point_channels, dtype=torch.float32, device=self.device
            )
            * (point_max - point_min)
            + point_min
        )
        return VoxelsData(
            voxels=voxels,
            coords=torch.stack(
                [cells % width, (cells // width) % height, cells // (width * height)], dim=1
            ).int(),
            num_points=torch.randint(
                1, max_points + 1, (num_voxels,), dtype=torch.int32, device=self.device
            ),
            batch_indices=torch.arange(
                self.batch_size, dtype=torch.int32, device=self.device
            ).repeat_interleave(num_voxels_per_sample),
            point_voxel_indices=torch.zeros((0,), dtype=torch.int64),
            num_dropped_voxels=torch.zeros((), dtype=torch.int64),
        )

    def _build_detection3d_gt_batch(self) -> Detection3DGTBatch:
        """Build one ground-truth box per sample, all inside the scene, for ``batch_size`` samples."""
        x_range = self.point_cloud_range[3] - self.point_cloud_range[0]
        y_range = self.point_cloud_range[4] - self.point_cloud_range[1]
        gt_bboxes_3d = torch.tensor(
            [
                [[0.125 * x_range, 0.1875 * y_range, 0.2, 4.0, 1.6, 1.5, 0.25, 0.5, -0.1, 0.0]],
                [[0.625 * x_range, 0.375 * y_range, 0.1, 2.0, 1.0, 1.2, -0.5, 0.2, 0.3, 0.0]],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        gt_labels_3d = torch.tensor([[0], [1]], dtype=torch.int32, device=self.device)
        gt_valid_bboxes = torch.tensor([1, 1], dtype=torch.int32, device=self.device)
        gt_bboxes_num_points = torch.tensor([[100], [200]], dtype=torch.int32, device=self.device)
        assert self.batch_size <= gt_bboxes_3d.shape[0]
        return Detection3DGTBatch(
            gt_bboxes_3d=gt_bboxes_3d[: self.batch_size],
            gt_labels_3d=gt_labels_3d[: self.batch_size],
            gt_valid_bboxes=gt_valid_bboxes[: self.batch_size],
            gt_bboxes_num_points=gt_bboxes_num_points[: self.batch_size],
        )

    def _build_batch_inputs(self, voxels_data: VoxelsData | None) -> ModelBatchInputs:
        """Build batch inputs with detection ground truth and the given voxel data."""
        return ModelBatchInputs(
            multi_task_gt_batch=ModelGTBatch(
                point_cloud_gt_batch=None,
                detection3d_gt_batch=self._build_detection3d_gt_batch(),
                image_gt_batch=None,
                segmentation3d_gt_batch=None,
            ),
            voxels_data=voxels_data,
            image_data=None,
            range_view_data=None,
        )

    def _assert_forward_compute_metrics_and_decode_run(
        self, model: TransFusionDetectionModel, batch_inputs: ModelBatchInputs
    ) -> None:
        """
        Assert that the model runs end to end over voxelized inputs, producing head outputs on the
        BEV grid, a finite loss, and one decoded prediction entry per sample.
        """
        multi_task_outputs = model(batch_inputs)
        metrics = model.compute_metrics(batch_inputs, multi_task_outputs)
        multi_task_predictions = model.decode_outputs(batch_inputs, multi_task_outputs)

        assert multi_task_outputs.detection3d_head_outputs is not None
        head_outputs = multi_task_outputs.detection3d_head_outputs.transfusion_head_outputs
        assert head_outputs is not None
        self.assertIsNone(multi_task_outputs.detection3d_head_outputs.center_head_outputs)
        self.assertEqual(
            head_outputs.dense_heatmap.shape, (self.batch_size, self.num_classes, *self.bev_shape)
        )
        self.assertEqual(head_outputs.query_labels.shape, (self.batch_size, self.num_proposals))
        self.assertEqual(
            head_outputs.query_heatmap_score.shape,
            (self.batch_size, self.num_classes, self.num_proposals),
        )
        self.assertTrue(torch.isfinite(head_outputs.dense_heatmap).all())

        self.assertIn("loss", metrics)
        self.assertIn("loss_heatmap", metrics)
        self.assertTrue(torch.isfinite(metrics["loss"]).all())

        predictions = multi_task_predictions.detection3d_predictions
        assert predictions is not None
        self.assertEqual(len(predictions), self.batch_size)
        for sample_predictions in predictions:
            # The head predicts velocity, so decoded boxes carry 9 parameters.
            self.assertEqual(sample_predictions.bboxes_3d.shape[1], 9)
            self.assertLessEqual(sample_predictions.bboxes_3d.shape[0], self.num_proposals)
            self.assertEqual(
                sample_predictions.scores_3d.shape[0], sample_predictions.bboxes_3d.shape[0]
            )
            self.assertEqual(
                sample_predictions.labels_3d.shape[0], sample_predictions.bboxes_3d.shape[0]
            )

    def _assert_export_spec_uses_deployment_io_contract(
        self, model: TransFusionDetectionModel, batch_inputs: ModelBatchInputs
    ) -> None:
        """
        Assert that the export spec exposes the batched voxel tensors as inputs and that the
        wrapper produces the ``cls_score0``, ``bbox_pred0`` and ``dir_cls_pred0`` deployment
        tensors matching the formatted training forward pass.
        """
        voxels_data = batch_inputs.voxels_data
        assert voxels_data is not None
        model.eval()

        spec = model.build_export_spec(batch_inputs)
        # The export wrapper is a freshly built module, so put it in eval mode explicitly.
        spec.module.eval()

        self.assertIsInstance(spec.module, _TransFusionExportWrapper)
        self.assertEqual(spec.module.batch_size, self.batch_size)
        self.assertEqual(spec.input_param_names, ["voxels", "num_points", "coors"])
        self.assertEqual(spec.output_names, ["cls_score0", "bbox_pred0", "dir_cls_pred0"])
        voxels, num_points, coors = spec.args
        torch.testing.assert_close(voxels, voxels_data.voxels)
        torch.testing.assert_close(num_points, voxels_data.num_points)
        torch.testing.assert_close(coors, voxels_data.batch_zyx_coords())

        with torch.no_grad():
            cls_score0, bbox_pred0, dir_cls_pred0 = spec.module(*spec.args)
            outputs = model(batch_inputs).detection3d_head_outputs
        assert outputs is not None and outputs.transfusion_head_outputs is not None

        self.assertEqual(cls_score0.shape, (self.batch_size, self.num_classes, self.num_proposals))
        self.assertEqual(bbox_pred0.shape, (self.batch_size, 8, self.num_proposals))
        self.assertEqual(dir_cls_pred0.shape, (self.batch_size, 2, self.num_proposals))
        self.assertTrue(torch.all((cls_score0 >= 0.0) & (cls_score0 <= 1.0)))
        self.assertTrue(torch.isfinite(bbox_pred0).all())
        self.assertTrue(torch.isfinite(dir_cls_pred0).all())

        expected = _format_transfusion_export_outputs(outputs.transfusion_head_outputs)
        for expected_tensor, exported_tensor in zip(
            expected, (cls_score0, bbox_pred0, dir_cls_pred0)
        ):
            torch.testing.assert_close(exported_tensor, expected_tensor, atol=1e-4, rtol=1e-4)


class TestTransFusionDetectionModel(_TransFusionDetectionModelTestCase):
    """Unit tests for the detector built from stub encoders, which run on any device.

    One BEV cell is one metre, because ``voxel_size[0] * out_size_factor`` is 1.0, so the 16 m
    range gives the 16x16 grid the middle encoder and the head agree on.
    """

    def setUp(self) -> None:
        """Set up the scene geometry, the stub-encoder detector and a voxelized batch."""
        super().setUp()
        self.point_cloud_range = [0.0, 0.0, -2.0, 16.0, 16.0, 2.0]
        self.voxel_size = [1.0, 1.0, 4.0]
        self.out_size_factor = 1
        self.bev_shape = (16, 16)
        self.middle_channels = 16
        self.neck_channels = 32
        self.model = self._build_model(
            pts_voxel_encoder=_StubVoxelEncoder(self.point_channels, self.middle_channels),
            pts_middle_encoder=_StubMiddleEncoder(self.bev_shape),
            pts_backbone=SECONDBackbone(
                in_channels=self.middle_channels,
                out_channels=[16, 32],
                layer_nums=[1, 1],
                layer_strides=[1, 2],
            ),
            pts_neck=SECONDFPN(
                in_channels=[16, 32],
                out_channels=[self.neck_channels // 2, self.neck_channels // 2],
                upsample_strides=[1, 2],
            ),
        )
        self.batch_inputs = self._build_batch_inputs(
            voxels_data=self._build_voxels_data(
                grid_shape=(*self.bev_shape[::-1], 1), num_voxels_per_sample=6, max_points=5
            )
        )

    def test_forward_compute_metrics_and_decode_run(self) -> None:
        """Test that the stub-encoder detector runs end to end and decodes one entry per sample."""
        self._assert_forward_compute_metrics_and_decode_run(self.model, self.batch_inputs)

    def test_build_eval_output_pairs_predictions_with_ground_truth(self) -> None:
        """Test that the eval output carries the valid ground-truth boxes of every sample."""
        multi_task_outputs = self.model(self.batch_inputs)

        eval_output = self.model.build_eval_output(self.batch_inputs, multi_task_outputs)

        self.assertEqual(len(eval_output["gt_boxes"]), self.batch_size)
        self.assertEqual(len(eval_output["gt_labels"]), self.batch_size)
        gt_batch = self.batch_inputs.multi_task_gt_batch.detection3d_gt_batch
        assert gt_batch is not None
        for sample_index in range(self.batch_size):
            torch.testing.assert_close(
                eval_output["gt_boxes"][sample_index], gt_batch.gt_bboxes_3d[sample_index, :1]
            )

    def test_forward_requires_voxels_data(self) -> None:
        """Test that the detector rejects a batch without voxel data."""
        with self.assertRaisesRegex(ValueError, "voxels_data"):
            self.model(self._build_batch_inputs(voxels_data=None))

    def test_metrics_decoding_and_eval_require_head_outputs(self) -> None:
        """Test that every consumer of the head outputs rejects outputs without a detection head."""
        empty_outputs = ModelOutputs(detection3d_head_outputs=None)

        with self.assertRaises(ValueError):
            self.model.compute_metrics(self.batch_inputs, empty_outputs)
        with self.assertRaises(ValueError):
            self.model.decode_outputs(self.batch_inputs, empty_outputs)
        with self.assertRaises(ValueError):
            self.model.build_eval_output(self.batch_inputs, empty_outputs)

    def test_build_export_spec_uses_deployment_io_contract(self) -> None:
        """Test the export IO contract and export-versus-training parity with stub encoders."""
        self._assert_export_spec_uses_deployment_io_contract(self.model, self.batch_inputs)

    def test_build_export_spec_prepares_modules_without_mutating_model(self) -> None:
        """
        Test that the exported wrapper holds the exportable middle encoder and attention layers
        while the training model keeps its original modules.
        """
        spec = self.model.build_export_spec(self.batch_inputs)

        export_middle_encoder = spec.module.pts_middle_encoder
        assert isinstance(export_middle_encoder, _StubMiddleEncoder)
        self.assertTrue(export_middle_encoder.exportable)
        assert isinstance(self.model.pts_middle_encoder, _StubMiddleEncoder)
        self.assertFalse(self.model.pts_middle_encoder.exportable)
        for decoder_layer in spec.module.bbox_head.decoder:
            self.assertIsInstance(decoder_layer.self_attn, ExportableMultiheadAttention)
            self.assertIsInstance(decoder_layer.cross_attn, ExportableMultiheadAttention)
        for decoder_layer in self.model.bbox_head.decoder:
            self.assertIsInstance(decoder_layer.self_attn, nn.MultiheadAttention)
        self.assertIs(spec.module.pts_voxel_encoder, self.model.pts_voxel_encoder)
        self.assertIs(spec.module.pts_backbone, self.model.pts_backbone)
        self.assertIs(spec.module.pts_neck, self.model.pts_neck)

    def test_build_export_spec_requires_voxels_data(self) -> None:
        """Test that export spec construction rejects a batch without voxel data."""
        with self.assertRaisesRegex(ValueError, "voxels_data"):
            self.model.build_export_spec(self._build_batch_inputs(voxels_data=None))

    def test_format_export_outputs_keeps_only_the_last_decoder_layer(self) -> None:
        """
        Test that, with an auxiliary head, only the trailing ``num_proposals`` columns of the
        concatenated layer predictions reach the deployment tensors.
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.batch_inputs).detection3d_head_outputs
        assert outputs is not None and outputs.transfusion_head_outputs is not None
        head_outputs = outputs.transfusion_head_outputs
        separate = head_outputs.separate_head_outputs

        cls_score0, bbox_pred0, dir_cls_pred0 = _format_transfusion_export_outputs(head_outputs)

        torch.testing.assert_close(
            cls_score0,
            separate.heatmap[..., -self.num_proposals :].sigmoid()
            * head_outputs.query_heatmap_score,
        )
        torch.testing.assert_close(bbox_pred0[:, :2], separate.center[..., -self.num_proposals :])
        torch.testing.assert_close(bbox_pred0[:, 2:3], separate.height[..., -self.num_proposals :])
        torch.testing.assert_close(bbox_pred0[:, 3:6], separate.dim[..., -self.num_proposals :])
        assert separate.vel is not None
        torch.testing.assert_close(bbox_pred0[:, 6:8], separate.vel[..., -self.num_proposals :])
        torch.testing.assert_close(dir_cls_pred0, separate.rot[..., -self.num_proposals :])

    def test_format_export_outputs_requires_velocity_branch(self) -> None:
        """Test that a head without a velocity branch cannot be packed for deployment."""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.batch_inputs).detection3d_head_outputs
        assert outputs is not None and outputs.transfusion_head_outputs is not None
        head_outputs = outputs.transfusion_head_outputs
        without_velocity = head_outputs.model_copy(
            update={
                "separate_head_outputs": head_outputs.separate_head_outputs.model_copy(
                    update={"vel": None}
                )
            }
        )

        with self.assertRaisesRegex(ValueError, "velocity"):
            _format_transfusion_export_outputs(without_velocity)


@unittest.skipUnless(
    torch.cuda.is_available(), "The sparse middle encoder runs through spconv CUDA kernels."
)
class TestTransFusionDetectionModelWithSparseEncoder(_TransFusionDetectionModelTestCase):
    """Integration tests for the detector built from the real TransFusion lidar stack.

    Scaled-down mirror of ``tasks/detection3d/transfusion/base.yaml``: an 8 m range with 0.25 m
    voxels gives a 32x32x40 grid, which the sparse encoder's three stride-2 stages reduce to the
    4x4 BEV expected by the head at ``out_size_factor`` 8. The channel wiring follows the config
    (voxel encoder 32 -> sparse 128 * Z2 = 256 -> SECOND [128, 256] -> FPN concat 512 -> head).
    """

    def setUp(self) -> None:
        """Set up the real lidar stack and a random two-sample voxel batch on the GPU."""
        from autoware_ml.models.detection3d.encoders.sparse.sparse_encoder import SparseEncoder
        from autoware_ml.models.detection3d.encoders.voxel import HardSimpleVoxelSinCosEncoder

        super().setUp()
        self.device = torch.device("cuda:0")
        self.point_cloud_range = [0.0, 0.0, -5.0, 8.0, 8.0, 3.0]
        self.voxel_size = [0.25, 0.25, 0.2]
        self.out_size_factor = 8
        self.sparse_shape = (32, 32, 41)  # (Y, X, Z)
        self.bev_shape = (4, 4)
        self.voxel_feature_channels = 32
        self.middle_channels = 256
        self.neck_channels = 512
        self.model = self._build_model(
            pts_voxel_encoder=HardSimpleVoxelSinCosEncoder(
                in_channels=self.point_channels,
                min_norm_values=[*self.point_cloud_range[:3], 0.0],
                max_norm_values=[*self.point_cloud_range[3:], 255.0],
            ),
            pts_middle_encoder=SparseEncoder(
                in_channels=self.voxel_feature_channels,
                sparse_shape=list(self.sparse_shape),
                dense_output_shapes=[*self.bev_shape, 2],
            ),
            pts_backbone=SECONDBackbone(
                in_channels=self.middle_channels,
                out_channels=[128, 256],
                layer_nums=[1, 1],
                layer_strides=[1, 2],
            ),
            pts_neck=SECONDFPN(
                in_channels=[128, 256], out_channels=[256, 256], upsample_strides=[1, 2]
            ),
        )
        height, width, depth = self.sparse_shape
        self.batch_inputs = self._build_batch_inputs(
            voxels_data=self._build_voxels_data(
                grid_shape=(width, height, depth), num_voxels_per_sample=64, max_points=10
            )
        )

    def test_forward_compute_metrics_and_decode_run(self) -> None:
        """Test that the real lidar stack runs end to end and decodes one entry per sample."""
        self._assert_forward_compute_metrics_and_decode_run(self.model, self.batch_inputs)

    def test_build_export_spec_uses_deployment_io_contract(self) -> None:
        """
        Test the export IO contract with the exportable sparse convolutions, and that the export
        path reproduces the native training forward pass.
        """
        self._assert_export_spec_uses_deployment_io_contract(self.model, self.batch_inputs)

    def test_build_export_spec_prepares_modules_without_mutating_model(self) -> None:
        """
        Test that the exported wrapper carries only exportable sparse convolutions and attention
        layers, while the training model keeps its native spconv layers and attention.
        """
        from spconv.pytorch import SparseConv3d as NativeSparseConv3d
        from spconv.pytorch import SubMConv3d as NativeSubMConv3d

        from autoware_ml.ops.spconv.sparse_conv import (
            ExportableSparseConv3d,
            ExportableSubMConv3d,
        )

        spec = self.model.build_export_spec(self.batch_inputs)

        export_middle_encoder = spec.module.pts_middle_encoder
        self.assertIsNot(export_middle_encoder, self.model.pts_middle_encoder)
        self.assertIsInstance(self.model.pts_middle_encoder.conv_input[0], NativeSubMConv3d)
        self.assertIsInstance(export_middle_encoder.conv_input[0], ExportableSubMConv3d)
        self.assertFalse(
            any(
                isinstance(module, (NativeSubMConv3d, NativeSparseConv3d))
                for module in export_middle_encoder.modules()
            )
        )
        self.assertTrue(
            any(
                isinstance(module, ExportableSparseConv3d)
                for module in export_middle_encoder.modules()
            )
        )
        for decoder_layer in spec.module.bbox_head.decoder:
            self.assertIsInstance(decoder_layer.self_attn, ExportableMultiheadAttention)
        for decoder_layer in self.model.bbox_head.decoder:
            self.assertIsInstance(decoder_layer.self_attn, nn.MultiheadAttention)


if __name__ == "__main__":
    unittest.main()
