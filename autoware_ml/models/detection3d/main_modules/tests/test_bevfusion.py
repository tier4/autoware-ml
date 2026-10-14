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

"""Unit tests for the BEVFusion detection model.

The camera branch pools through the CUDA ``bev_pool`` kernel, so the camera-lidar forward and
export paths are only tested on the GPU. The lidar-only detector, the export specifications, the
geometry contract between the branches, and the input validation run on any device.
"""

from __future__ import annotations

from types import MappingProxyType
import unittest

from jaxtyping import Float32, Int32
import torch
import torch.nn as nn

from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.dataclasses.models.model_outputs import ModelOutputs
from autoware_ml.dataclasses.batch.detection3d import Detection3DGTBatch
from autoware_ml.dataclasses.geometry.images import ImageGTBatch
from autoware_ml.dataclasses.batch.sample_batch import ModelGTBatch
from autoware_ml.models.detection3d.backbones.second import SECONDBackbone
from autoware_ml.models.detection3d.heads.transfusions.exportable_multi_head_attention import (
    ExportableMultiheadAttention,
)
from autoware_ml.models.detection3d.heads.transfusions.transfusion_head import (
    ScoreThresholdConfig,
    TransFusionHead,
)
from autoware_ml.models.detection3d.main_modules.bevfusion import (
    BEVFusionDetectionModel,
    _BEVFusionExportWrapper,
    _BEVFusionLidarExportWrapper,
)
from autoware_ml.models.detection3d.main_modules.bevfusions.bevfusion_camera import (
    BEVFusionCamera,
    BEVFusionImageBackboneExportWrapper,
)
from autoware_ml.models.detection3d.main_modules.bevfusions.bevfusion_lidar import BEVFusionLidar
from autoware_ml.models.detection3d.main_modules.bevfusions.fuser import ConvFuser
from autoware_ml.models.detection3d.main_modules.bevfusions.export_wrappers import (
    export_detection_outputs,
)
from autoware_ml.models.detection3d.necks.second_fpn import SECONDFPN
from autoware_ml.models.detection3d.task_modules.assigners import HungarianAssigner3D
from autoware_ml.models.detection3d.task_modules.bbox_coders import TransFusionBBoxCoder
from autoware_ml.models.detection3d.task_modules.match_costs import (
    BBoxBEVL1Cost,
    ClassificationCost,
    IoU3DCost,
)
from autoware_ml.models.detection3d.view_transforms.depth_lss import DepthLSSTransform
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
    ``bev_output_shape`` and ``prepare_for_export`` interface of the sparse encoder so the model
    can run and export on the CPU.
    """

    def __init__(self, bev_shape: tuple[int, int], exportable: bool = False) -> None:
        super().__init__()
        self._bev_shape = bev_shape
        self.exportable = exportable

    @property
    def bev_output_shape(self) -> tuple[int, int]:
        return self._bev_shape

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


class _StubBackbone(nn.Module):
    """Single stride-8 convolution standing in for the image backbone."""

    def __init__(self, feature_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, feature_channels, kernel_size=8, stride=8)

    def forward(
        self, images: Float32[torch.Tensor, "num_images 3 height width"]
    ) -> tuple[Float32[torch.Tensor, "num_images channels feature_height feature_width"]]:
        return (self.conv(images),)


class _StubNeck(nn.Module):
    """Neck that returns the backbone feature maps untouched."""

    def forward(
        self,
        features: tuple[
            Float32[torch.Tensor, "num_images channels feature_height feature_width"], ...
        ],
    ) -> list[Float32[torch.Tensor, "num_images channels feature_height feature_width"]]:
        return list(features)


class _BEVFusionDetectionModelTestCase(unittest.TestCase):
    """Shared configuration and builders for the BEVFusion detection model test cases.

    One BEV cell is one metre, because ``voxel_size[0] * out_size_factor`` is 1.0, so the 16 m
    range gives the 16x16 grid every branch and the head agree on.
    """

    def setUp(self) -> None:
        """Set up the scene geometry, channel layout and head configuration."""
        torch.manual_seed(0)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = 2
        self.class_names = ["car", "pedestrian", "cyclist"]
        self.num_classes = len(self.class_names)
        self.point_cloud_range = [0.0, 0.0, -2.0, 16.0, 16.0, 2.0]
        self.voxel_size = [1.0, 1.0, 4.0]
        self.out_size_factor = 1
        self.bev_shape = (16, 16)
        self.point_channels = 4
        self.middle_channels = 16
        self.neck_channels = 32
        self.image_bev_channels = 8
        self.image_feature_channels = 32
        self.image_size = 64
        self.feature_size = 8
        self.num_cams = 2
        self.num_proposals = 4
        self.num_box_channels = 10
        self.log_dict_configs = LogDictConfigs(
            on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
        )

    def _build_lidar_network(self, with_fuser: bool = False) -> BEVFusionLidar:
        """Build a lidar branch from stub encoders and the real BEV backbone and neck."""
        fuser = (
            ConvFuser(
                in_channels=[self.image_bev_channels, self.middle_channels],
                out_channels=self.middle_channels,
            ).to(self.device)
            if with_fuser
            else None
        )
        return BEVFusionLidar(
            pts_voxel_encoder=_StubVoxelEncoder(self.point_channels, self.middle_channels).to(
                self.device
            ),
            pts_middle_encoder=_StubMiddleEncoder(self.bev_shape).to(self.device),
            pts_backbone=SECONDBackbone(
                in_channels=self.middle_channels,
                out_channels=[16, 32],
                layer_nums=[1, 1],
                layer_strides=[1, 2],
            ).to(self.device),
            pts_neck=SECONDFPN(
                in_channels=[16, 32],
                out_channels=[self.neck_channels // 2, self.neck_channels // 2],
                upsample_strides=[1, 2],
            ).to(self.device),
            fuser=fuser,
        ).to(self.device)

    def _build_camera_network(self, y_range: float = 16.0) -> BEVFusionCamera:
        """Build a camera branch whose BEV grid spans the scene, ``y_range`` metres along y."""
        return BEVFusionCamera(
            img_backbone=_StubBackbone(self.image_feature_channels).to(self.device),
            img_neck=_StubNeck().to(self.device),
            view_transform=DepthLSSTransform(
                in_channels=self.image_feature_channels,
                out_channels=self.image_bev_channels,
                image_size=[self.image_size, self.image_size],
                feature_size=[self.feature_size, self.feature_size],
                xbound=[self.point_cloud_range[0], self.point_cloud_range[3], self.voxel_size[0]],
                ybound=[self.point_cloud_range[1], y_range, self.voxel_size[1]],
                zbound=[self.point_cloud_range[2], self.point_cloud_range[5], self.voxel_size[2]],
                dbound=[1.0, 5.0, 1.0],
            ).to(self.device),
        ).to(self.device)

    def _build_head(self) -> TransFusionHead:
        """Build a small auxiliary TransFusion head on the shared BEV grid."""
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
        ).to(self.device)

    def _build_model(
        self, lidar_network: BEVFusionLidar | None, camera_network: BEVFusionCamera | None
    ) -> BEVFusionDetectionModel:
        """Build the detector around the given branches."""
        return BEVFusionDetectionModel(
            data_preprocessor=DataPreprocessor(preprocessor_modules=[]),
            lidar_network=lidar_network,
            camera_network=camera_network,
            bbox_head=self._build_head(),
            log_dict_configs=self.log_dict_configs,
        ).to(self.device)

    def _build_voxels_data(self) -> VoxelsData:
        """Build voxels in distinct ``(x, y, z)`` cells, half of them in each batch sample."""
        num_voxels_per_sample, max_points = 6, 5
        num_voxels = self.batch_size * num_voxels_per_sample
        height, width = self.bev_shape
        cells = torch.stack(
            [
                torch.randperm(height * width, device=self.device)[:num_voxels_per_sample]
                for _ in range(self.batch_size)
            ]
        ).flatten()
        return VoxelsData(
            voxels=torch.randn(
                num_voxels, max_points, self.point_channels, dtype=torch.float32, device=self.device
            ),
            coords=torch.stack(
                [
                    cells % width,
                    cells // width,
                    torch.zeros(num_voxels, dtype=torch.int64, device=self.device),
                ],
                dim=1,
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
        gt_bboxes_3d = torch.tensor(
            [
                [[2.0, 3.0, 0.2, 4.0, 1.6, 1.5, 0.25, 0.5, -0.1, 0.0]],
                [[10.0, 6.0, 0.1, 2.0, 1.0, 1.2, -0.5, 0.2, 0.3, 0.0]],
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

    def _build_image_gt_batch(self, with_depth_maps: bool = True) -> ImageGTBatch:
        """Build multiview images looking along lidar x with identity image augmentation."""
        focal = self.image_size / 2.0
        camera_intrinsics = torch.tensor(
            [[focal, 0.0, focal], [0.0, focal, focal], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=self.device,
        ).expand(self.batch_size, self.num_cams, 3, 3)
        # Standard camera-to-lidar axis swap: camera z becomes lidar x.
        camera2lidar = torch.tensor(
            [
                [0.0, 0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0, 8.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device=self.device,
        ).expand(self.batch_size, self.num_cams, 4, 4)
        identity = torch.eye(4, device=self.device).expand(self.batch_size, self.num_cams, 4, 4)
        return ImageGTBatch(
            images=torch.rand(
                self.batch_size,
                self.num_cams,
                3,
                self.image_size,
                self.image_size,
                device=self.device,
            ),
            depth_maps=(
                torch.rand(
                    self.batch_size,
                    self.num_cams,
                    1,
                    self.image_size,
                    self.image_size,
                    device=self.device,
                )
                if with_depth_maps
                else None
            ),
            camera_intrinsics=camera_intrinsics.contiguous(),
            image_augmentation_matrices=identity.contiguous(),
            lidar2images=identity.contiguous(),
            lidar2cams=torch.inverse(camera2lidar).contiguous(),
            calibration_statuses=None,
        )

    def _build_batch_inputs(
        self,
        voxels_data: VoxelsData | None,
        image_data: ImageGTBatch | None = None,
    ) -> ModelBatchInputs:
        """Build batch inputs with detection ground truth and the given modality inputs."""
        return ModelBatchInputs(
            multi_task_gt_batch=ModelGTBatch(
                point_cloud_gt_batch=None,
                detection3d_gt_batch=self._build_detection3d_gt_batch(),
                image_gt_batch=image_data,
                segmentation3d_gt_batch=None,
            ),
            voxels_data=voxels_data,
            image_data=image_data,
            range_view_data=None,
        )


class TestBEVFusionDetectionModelLidarOnly(_BEVFusionDetectionModelTestCase):
    """Unit tests for the lidar-only detector, which runs end to end on the CPU."""

    def setUp(self) -> None:
        """Set up a lidar-only detector and a voxelized batch with ground truth."""
        super().setUp()
        self.model = self._build_model(
            lidar_network=self._build_lidar_network(), camera_network=None
        )
        self.batch_inputs = self._build_batch_inputs(voxels_data=self._build_voxels_data())

    def test_forward_compute_metrics_and_decode_run(self) -> None:
        """
        Test that the model runs end to end over voxelized inputs, producing head outputs on the
        shared BEV grid, a finite loss, and one decoded prediction entry per sample.
        """
        multi_task_outputs = self.model(self.batch_inputs)
        metrics = self.model.compute_metrics(self.batch_inputs, multi_task_outputs)
        multi_task_predictions = self.model.decode_outputs(self.batch_inputs, multi_task_outputs)

        assert multi_task_outputs.detection3d_head_outputs is not None
        head_outputs = multi_task_outputs.detection3d_head_outputs.transfusion_head_outputs
        assert head_outputs is not None
        self.assertIsNone(multi_task_outputs.detection3d_head_outputs.center_head_outputs)
        self.assertEqual(
            head_outputs.dense_heatmap.shape, (self.batch_size, self.num_classes, *self.bev_shape)
        )
        self.assertEqual(head_outputs.query_labels.shape, (self.batch_size, self.num_proposals))

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

    def test_forward_invokes_the_branches_through_module_hooks(self) -> None:
        """
        Test that the lidar branch and the head run through ``nn.Module.__call__`` so forward
        hooks fire. Lightning's model summary relies on them to report per-layer FLOPs and sizes;
        calling ``.forward`` directly used to leave the lidar branch at zero FLOPs.
        """
        called: list[str] = []
        handles = [
            self.model.lidar_network.register_forward_hook(
                lambda module, args, output: called.append("lidar_network")
            ),
            self.model.bbox_head.register_forward_hook(
                lambda module, args, output: called.append("bbox_head")
            ),
        ]
        try:
            self.model(self.batch_inputs)
        finally:
            for handle in handles:
                handle.remove()

        self.assertEqual(called, ["lidar_network", "bbox_head"])

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
        """Test that a lidar detector rejects a batch without voxel data."""
        with self.assertRaises(ValueError):
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

    def test_first_sample_voxel_inputs_keeps_first_sample_in_runtime_dtypes(self) -> None:
        """Test that only the voxels of batch sample 0 are exported, as ``float32`` and ``int32``."""
        voxels_data = self.batch_inputs.voxels_data
        assert voxels_data is not None
        first_sample = voxels_data.batch_indices == 0

        voxels, coors, num_points_per_voxel = BEVFusionDetectionModel._first_sample_voxel_inputs(
            self.batch_inputs
        )

        torch.testing.assert_close(voxels, voxels_data.voxels[first_sample])
        self.assertEqual(coors.dtype, torch.int32)
        self.assertTrue(coors.is_contiguous())
        torch.testing.assert_close(coors, voxels_data.coords[first_sample][:, [2, 1, 0]])
        self.assertEqual(num_points_per_voxel.dtype, torch.int32)
        torch.testing.assert_close(num_points_per_voxel, voxels_data.num_points[first_sample])

    def test_first_sample_voxel_inputs_requires_voxels_data(self) -> None:
        """Test that export input extraction rejects a batch without voxel data."""
        with self.assertRaises(ValueError):
            BEVFusionDetectionModel._first_sample_voxel_inputs(
                self._build_batch_inputs(voxels_data=None)
            )

    def test_prepare_export_model_returns_exportable_eval_copy(self) -> None:
        """
        Test that the export model is a separate eval-mode copy with the exportable middle encoder
        and attention layers, leaving the training model untouched.
        """
        self.model.train()

        export_model = self.model._prepare_export_model()

        self.assertIsNot(export_model, self.model)
        self.assertFalse(export_model.training)
        self.assertTrue(self.model.training)
        assert export_model.lidar_network is not None
        assert self.model.lidar_network is not None
        export_middle_encoder = export_model.lidar_network.pts_middle_encoder
        assert isinstance(export_middle_encoder, _StubMiddleEncoder)
        self.assertTrue(export_middle_encoder.exportable)
        original_middle_encoder = self.model.lidar_network.pts_middle_encoder
        assert isinstance(original_middle_encoder, _StubMiddleEncoder)
        self.assertFalse(original_middle_encoder.exportable)
        for decoder_layer in export_model.bbox_head.decoder:
            self.assertIsInstance(decoder_layer.self_attn, ExportableMultiheadAttention)
            self.assertIsInstance(decoder_layer.cross_attn, ExportableMultiheadAttention)
        for decoder_layer in self.model.bbox_head.decoder:
            self.assertIsInstance(decoder_layer.self_attn, nn.MultiheadAttention)

    def test_build_export_specs_exports_single_lidar_main_body(self) -> None:
        """
        Test that a lidar-only model exports one main body whose inputs are the first-sample voxel
        tensors, and that the wrapper produces the runtime detection tensors.
        """
        voxels_data = self.batch_inputs.voxels_data
        assert voxels_data is not None
        num_first_sample_voxels = int((voxels_data.batch_indices == 0).sum())

        export_specs = self.model.build_export_specs(self.batch_inputs)

        self.assertEqual(list(export_specs), ["bevfusion_lidar"])
        spec = export_specs["bevfusion_lidar"]
        self.assertIsInstance(spec.module, _BEVFusionLidarExportWrapper)
        self.assertEqual(spec.input_param_names, ["voxels", "coors", "num_points_per_voxel"])
        voxels, coors, num_points_per_voxel = spec.args
        self.assertEqual(voxels.shape[0], num_first_sample_voxels)
        self.assertEqual(coors.shape, (num_first_sample_voxels, 3))
        self.assertEqual(num_points_per_voxel.shape, (num_first_sample_voxels,))

        with torch.no_grad():
            bbox_pred, score, label_pred = spec.module(*spec.args)

        self.assertEqual(bbox_pred.shape, (self.num_box_channels, self.num_proposals))
        self.assertEqual(score.shape, (self.num_proposals,))
        self.assertEqual(label_pred.shape, (self.num_proposals,))
        self.assertEqual(label_pred.dtype, torch.int64)
        self.assertTrue(torch.isfinite(bbox_pred).all())
        self.assertTrue(torch.all((score >= 0.0) & (score <= 1.0)))


class TestBEVFusionDetectionModelCameraLidar(_BEVFusionDetectionModelTestCase):
    """Unit tests for the camera-lidar detector paths that never reach the pooling kernel."""

    def setUp(self) -> None:
        """Set up a camera-lidar detector and a batch with voxels, images and depth maps."""
        super().setUp()
        self.model = self._build_model(
            lidar_network=self._build_lidar_network(with_fuser=True),
            camera_network=self._build_camera_network(),
        )
        self.batch_inputs = self._build_batch_inputs(
            voxels_data=self._build_voxels_data(), image_data=self._build_image_gt_batch()
        )

    def test_requires_at_least_one_branch(self) -> None:
        """Test that a detector without any BEV branch is rejected."""
        with self.assertRaises(AssertionError):
            self._build_model(lidar_network=None, camera_network=None)

    def test_branches_must_share_the_bev_shape(self) -> None:
        """Test that a camera grid half as deep along y as the lidar grid is rejected."""
        camera_network = self._build_camera_network(y_range=self.point_cloud_range[4] / 2)
        self.assertEqual(
            camera_network.expected_bev_shape, (self.bev_shape[0] // 2, self.bev_shape[1])
        )

        with self.assertRaisesRegex(ValueError, "same BEV shape"):
            self._build_model(
                lidar_network=self._build_lidar_network(with_fuser=True),
                camera_network=camera_network,
            )

    def test_forward_requires_image_data_and_depth_maps(self) -> None:
        """Test that the camera branch is refused inputs without images or without depth maps."""
        voxels_data = self._build_voxels_data()

        with self.assertRaisesRegex(ValueError, "image_data"):
            self.model(self._build_batch_inputs(voxels_data=voxels_data, image_data=None))
        with self.assertRaisesRegex(ValueError, "depth_maps"):
            self.model(
                self._build_batch_inputs(
                    voxels_data=voxels_data,
                    image_data=self._build_image_gt_batch(with_depth_maps=False),
                )
            )

    def test_build_export_specs_exports_image_backbone_and_main_body(self) -> None:
        """
        Test that a camera-lidar model exports the image backbone and the fused main body with the
        runtime tensor layout: raw ``uint8`` images for the backbone, and first-sample voxels plus
        precomputed image features, depth maps and pooling metadata for the main body.
        """
        image_data = self.batch_inputs.image_data
        assert image_data is not None
        assert image_data.depth_maps is not None
        num_frustum_points = self.num_cams * 4 * self.feature_size**2

        export_specs = self.model.build_export_specs(self.batch_inputs)

        self.assertEqual(list(export_specs), ["bevfusion_image_backbone", "bevfusion_camera_lidar"])

        backbone_spec = export_specs["bevfusion_image_backbone"]
        self.assertIsInstance(backbone_spec.module, BEVFusionImageBackboneExportWrapper)
        self.assertEqual(backbone_spec.input_param_names, ["imgs"])
        (imgs,) = backbone_spec.args
        self.assertEqual(imgs.dtype, torch.uint8)
        self.assertEqual(imgs.shape, (self.num_cams, 3, self.image_size, self.image_size))
        torch.testing.assert_close(
            imgs, (image_data.images[0] * 255.0).round().clamp(0.0, 255.0).to(torch.uint8)
        )

        main_body_spec = export_specs["bevfusion_camera_lidar"]
        self.assertIsInstance(main_body_spec.module, _BEVFusionExportWrapper)
        self.assertEqual(
            main_body_spec.input_param_names,
            [
                "voxels",
                "coors",
                "num_points_per_voxel",
                "image_feats",
                "depth_maps",
                "geom_feats",
                "kept",
                "ranks",
                "indices",
            ],
        )
        _, coors, _, image_feats, depth_maps, geom_feats, kept, ranks, indices = main_body_spec.args
        self.assertEqual(coors.shape[1], 3)
        self.assertEqual(
            image_feats.shape,
            (self.num_cams, self.image_feature_channels, self.feature_size, self.feature_size),
        )
        torch.testing.assert_close(depth_maps, image_data.depth_maps[0])
        self.assertEqual(geom_feats.dtype, torch.float32)
        self.assertEqual(geom_feats.shape[1], 4)
        self.assertEqual(kept.dtype, torch.bool)
        self.assertEqual(kept.shape, (num_frustum_points,))
        self.assertEqual(ranks.dtype, torch.int64)
        self.assertEqual(indices.dtype, torch.int64)
        self.assertEqual(ranks.shape, indices.shape)
        self.assertEqual(ranks.shape[0], int(kept.sum()))
        self.assertEqual(geom_feats.shape[0], int(kept.sum()))

    def test_exported_image_backbone_matches_training_feature_extraction(self) -> None:
        """Test that the exported backbone reproduces the training branch's image features."""
        image_data = self.batch_inputs.image_data
        assert image_data is not None
        assert self.model.camera_network is not None
        export_specs = self.model.build_export_specs(self.batch_inputs)
        backbone_spec = export_specs["bevfusion_image_backbone"]
        (imgs,) = backbone_spec.args

        with torch.no_grad():
            exported_features = backbone_spec.module(imgs)
            expected_features = self.model.camera_network.extract_image_features(
                (imgs.float() / 255.0).unsqueeze(0)
            )[0]

        torch.testing.assert_close(exported_features, expected_features)

    def test_build_export_specs_requires_voxels_images_and_depth_maps(self) -> None:
        """Test that export spec construction rejects batches missing any required modality."""
        voxels_data = self._build_voxels_data()

        with self.assertRaisesRegex(ValueError, "voxels_data"):
            self.model.build_export_specs(
                self._build_batch_inputs(voxels_data=None, image_data=self._build_image_gt_batch())
            )
        with self.assertRaisesRegex(ValueError, "image_data"):
            self.model.build_export_specs(
                self._build_batch_inputs(voxels_data=voxels_data, image_data=None)
            )
        with self.assertRaisesRegex(ValueError, "depth_maps"):
            self.model.build_export_specs(
                self._build_batch_inputs(
                    voxels_data=voxels_data,
                    image_data=self._build_image_gt_batch(with_depth_maps=False),
                )
            )


@unittest.skipUnless(
    torch.cuda.is_available(), "The camera branch pools through the CUDA bev_pool kernel."
)
class TestBEVFusionDetectionModelCameraLidarForward(_BEVFusionDetectionModelTestCase):
    """Unit tests for the camera-lidar detector paths that pool image features into BEV."""

    def setUp(self) -> None:
        """Set up a camera-lidar detector and a batch with voxels, images and depth maps."""
        super().setUp()
        self.model = self._build_model(
            lidar_network=self._build_lidar_network(with_fuser=True),
            camera_network=self._build_camera_network(),
        )
        self.batch_inputs = self._build_batch_inputs(
            voxels_data=self._build_voxels_data(), image_data=self._build_image_gt_batch()
        )

    def test_forward_compute_metrics_and_decode_run(self) -> None:
        """
        Test that the fused model runs end to end over voxels, images and depth maps, producing
        head outputs on the shared BEV grid, a finite loss, and one decoded entry per sample.
        """
        multi_task_outputs = self.model(self.batch_inputs)
        metrics = self.model.compute_metrics(self.batch_inputs, multi_task_outputs)
        multi_task_predictions = self.model.decode_outputs(self.batch_inputs, multi_task_outputs)

        assert multi_task_outputs.detection3d_head_outputs is not None
        head_outputs = multi_task_outputs.detection3d_head_outputs.transfusion_head_outputs
        assert head_outputs is not None
        self.assertEqual(
            head_outputs.dense_heatmap.shape, (self.batch_size, self.num_classes, *self.bev_shape)
        )
        self.assertTrue(torch.isfinite(head_outputs.dense_heatmap).all())

        self.assertIn("loss", metrics)
        self.assertTrue(torch.isfinite(metrics["loss"]).all())

        predictions = multi_task_predictions.detection3d_predictions
        assert predictions is not None
        self.assertEqual(len(predictions), self.batch_size)
        for sample_predictions in predictions:
            self.assertEqual(sample_predictions.bboxes_3d.shape[1], 9)
            self.assertLessEqual(sample_predictions.bboxes_3d.shape[0], self.num_proposals)

    def test_forward_depends_on_the_images(self) -> None:
        """Test that the image BEV reaches the head, so different images change the predictions."""
        self.model.eval()
        image_data = self.batch_inputs.image_data
        assert image_data is not None
        other_batch_inputs = self._build_batch_inputs(
            voxels_data=self.batch_inputs.voxels_data,
            image_data=image_data._replace(images=torch.zeros_like(image_data.images)),
        )

        with torch.no_grad():
            outputs = self.model(self.batch_inputs).detection3d_head_outputs
            other_outputs = self.model(other_batch_inputs).detection3d_head_outputs

        assert outputs is not None and outputs.transfusion_head_outputs is not None
        assert other_outputs is not None and other_outputs.transfusion_head_outputs is not None
        self.assertFalse(
            torch.allclose(
                outputs.transfusion_head_outputs.dense_heatmap,
                other_outputs.transfusion_head_outputs.dense_heatmap,
            )
        )

    def test_exported_main_body_returns_runtime_tensors(self) -> None:
        """Test that the exported camera-lidar main body runs on its own export arguments."""
        export_specs = self.model.build_export_specs(self.batch_inputs)
        main_body_spec = export_specs["bevfusion_camera_lidar"]

        with torch.no_grad():
            bbox_pred, score, label_pred = main_body_spec.module(*main_body_spec.args)

        self.assertEqual(bbox_pred.shape, (self.num_box_channels, self.num_proposals))
        self.assertEqual(score.shape, (self.num_proposals,))
        self.assertEqual(label_pred.shape, (self.num_proposals,))
        self.assertEqual(label_pred.dtype, torch.int64)
        self.assertTrue(torch.isfinite(bbox_pred).all())
        self.assertTrue(torch.all((score >= 0.0) & (score <= 1.0)))
        self.assertTrue(torch.all((label_pred >= 0) & (label_pred < self.num_classes)))

    def test_exported_main_body_matches_training_forward_on_a_single_sample(self) -> None:
        """
        Test that the export path, fed the runtime tensors and precomputed pooling metadata, packs
        the same detections as the training forward pass of the same single sample.
        """
        self.batch_size = 1
        batch_inputs = self._build_batch_inputs(
            voxels_data=self._build_voxels_data(), image_data=self._build_image_gt_batch()
        )
        self.model.eval()
        export_specs = self.model.build_export_specs(batch_inputs)
        main_body_spec = export_specs["bevfusion_camera_lidar"]
        # The export wrapper is a freshly built module, so put it in eval mode explicitly.
        main_body_spec.module.eval()

        with torch.no_grad():
            outputs = self.model(batch_inputs).detection3d_head_outputs
            assert outputs is not None and outputs.transfusion_head_outputs is not None
            expected = export_detection_outputs(
                self.model.bbox_head, outputs.transfusion_head_outputs
            )
            exported = main_body_spec.module(*main_body_spec.args)

        for expected_tensor, exported_tensor in zip(expected, exported):
            torch.testing.assert_close(exported_tensor, expected_tensor, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
