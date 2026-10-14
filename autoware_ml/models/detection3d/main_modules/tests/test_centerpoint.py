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

"""Unit tests for CenterPoint detection model."""

import unittest

import torch

from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.dataclasses.batch.detection3d import (
    Detection3DGTBatch,
)
from autoware_ml.dataclasses.batch.sample_batch import ModelGTBatch
from autoware_ml.models.detection3d.main_modules.centerpoint import (
    CenterPointDetectionModel,
    _CenterPointBackboneNeckHeadExportWrapper,
    _CenterPointVoxelEncoderExportWrapper,
)
from autoware_ml.models.detection3d.backbones.second import SECONDBackbone
from autoware_ml.models.detection3d.encoders.pillars.pillar_feature_net import PillarFeatureNet
from autoware_ml.models.detection3d.encoders.pillars.point_pillar_scatter import PointPillarsScatter
from autoware_ml.models.detection3d.heads.centerhead import CenterHead
from autoware_ml.models.detection3d.necks.second_fpn import SECONDFPN
from autoware_ml.models.module_base_model import LogDictConfigs
from autoware_ml.preprocessing.detection3d.point_pillar_preprocessor import PointPillarPreprocessor
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor
from autoware_ml.ops.voxelization.voxelization import VoxelsData


class TestCenterPointDetectionModel(unittest.TestCase):
    """Unit tests for the CenterPointDetectionModel class."""

    def setUp(self) -> None:
        """Set up the common inputs for the tests."""
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = 2
        self.class_names = ["car", "pedestrian", "cyclist"]
        self.num_classes = len(self.class_names)
        torch.manual_seed(0)
        self.data_preprocessor = DataPreprocessor(
            preprocessor_modules=[
                PointPillarPreprocessor(
                    voxel_size=[0.5, 0.5, 4.0],
                    point_cloud_range=[0.0, 0.0, -2.0, 8.0, 8.0, 2.0],
                    max_num_points=16,
                    max_voxels=16,
                    eval_max_voxels=16,
                )
            ]
        )
        self.pillar_feature_net = PillarFeatureNet(
            in_channels=5,
            feat_channels=[32, 32],
            voxel_size=[0.5, 0.5, 4.0],
            point_cloud_range=[0.0, 0.0, -2.0, 8.0, 8.0, 2.0],
        )
        self.middle_encoder = PointPillarsScatter(in_channels=32, output_shape=[16, 16])
        self.backbone = SECONDBackbone(
            in_channels=32,
            out_channels=[64, 128, 256],
            layer_nums=[1, 1, 1],
            layer_strides=[2, 2, 2],
        )
        self.neck = SECONDFPN(
            in_channels=[64, 128, 256],
            out_channels=[128, 128, 128],
            upsample_strides=[1, 2, 4],
        )
        self.bbox_head = self._build_bbox_head(use_velocity=True)
        self.log_dict_configs = LogDictConfigs(
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.centerpoint = self._build_centerpoint(self.bbox_head)

    def _build_bbox_head(self, use_velocity: bool) -> CenterHead:
        """Build the CenterPoint dense head on the 8x8 neck grid, with or without velocity."""
        return CenterHead(
            in_channels=384,
            class_names=self.class_names,
            shared_channels=64,
            point_cloud_range=[0.0, 0.0, -2.0, 8.0, 8.0, 2.0],
            voxel_size=[0.5, 0.5, 4.0],
            out_size_factor=2,
            min_radius=1,
            score_threshold=0.1,
            post_max_size=10,
            nms_min_radius=1.0,
            use_velocity=use_velocity,
        )

    def _build_centerpoint(self, bbox_head: CenterHead) -> CenterPointDetectionModel:
        """Build the detector around the shared encoders, backbone and neck with the given head."""
        return CenterPointDetectionModel(
            data_preprocessor=self.data_preprocessor,
            pts_voxel_encoder=self.pillar_feature_net,
            pts_middle_encoder=self.middle_encoder,
            pts_backbone=self.backbone,
            pts_neck=self.neck,
            bbox_head=bbox_head,
            log_dict_configs=self.log_dict_configs,
        ).to(self.device)

    def test_centerpoint_weights_mean_std(self) -> None:
        """
        Test that the CenterPointDetectionModel weights are initialized
        with means of 0.0 and std < 0.1.
        """
        weights = []
        biases = []
        for name, param in self.centerpoint.named_parameters():
            if "weight" in name:
                weights.append(param.data)
            if "bias" in name:
                biases.append(param.data)

        weights = torch.cat([w.flatten() for w in weights])
        biases = torch.cat([b.flatten() for b in biases])
        weight_mean = weights.mean().item()
        weight_std = weights.std().item()
        bias_mean = biases.mean().item()
        bias_std = biases.std().item()

        expected_weight_mean = 0.0
        # Biases should be almost similar since the bias for the heatmap head is initialized
        # to a negative value and the rest are initialized to zero.
        expected_bias_mean = -0.008480
        expected_bias_std = 0.08971667289733887
        self.assertAlmostEqual(weight_mean, expected_weight_mean, places=2)
        self.assertLess(weight_std, 0.1)
        self.assertAlmostEqual(bias_mean, expected_bias_mean, places=2)
        self.assertAlmostEqual(bias_std, expected_bias_std, places=2)

    def _build_multi_task_batch_inputs(self) -> ModelBatchInputs:
        """
        Build a ModelBatchInputs batch with voxelized lidar inputs and detection ground truth
        that the CenterPointDetectionModel can be run end to end on.
        """
        num_pillars = 12
        voxels_data = VoxelsData(
            voxels=torch.randn((num_pillars, 5, 5), dtype=torch.float32, device=self.device),
            # (x, y, z) voxel coordinates inside the 16x16 scatter canvas
            coords=torch.randint(0, 8, (num_pillars, 3), dtype=torch.int32, device=self.device),
            num_points=torch.randint(1, 5, (num_pillars,), dtype=torch.int32, device=self.device),
            batch_indices=torch.tensor(
                [0] * (num_pillars // 2) + [1] * (num_pillars // 2),
                dtype=torch.int32,
                device=self.device,
            ),
            point_voxel_indices=torch.zeros((0,), dtype=torch.int64),
            num_dropped_voxels=torch.zeros((), dtype=torch.int64),
        )
        # (batch_size, max_num_bboxes, num_Box3DFieldIndex)
        gt_bboxes_3d = torch.tensor(
            [
                [[2.0, 3.0, 0.2, 4.0, 1.6, 1.5, 0.25, 0.5, -0.1, -0.2]],
                [[5.0, 6.0, 0.1, 2.0, 1.0, 1.2, -0.5, 0.2, 0.3, 0.0]],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        detection3d_gt_batch = Detection3DGTBatch(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=torch.tensor([[0], [1]], dtype=torch.int32, device=self.device),
            gt_valid_bboxes=torch.tensor([1, 1], dtype=torch.int32, device=self.device),
            gt_bboxes_num_points=torch.tensor(
                [[100], [200]], dtype=torch.int32, device=self.device
            ),
        )
        return ModelBatchInputs(
            multi_task_gt_batch=ModelGTBatch(
                point_cloud_gt_batch=None,
                detection3d_gt_batch=detection3d_gt_batch,
                image_gt_batch=None,
                segmentation3d_gt_batch=None,
            ),
            voxels_data=voxels_data,
            image_data=None,
            range_view_data=None,
        )

    def test_centerpoint_forward_compute_metrics_and_decode_run(self) -> None:
        """
        Test that the model runs end to end over voxelized inputs, producing head outputs of the
        expected shape, a differentiable loss, and one decoded prediction entry per sample.
        """
        multi_task_batch_inputs = self._build_multi_task_batch_inputs()

        multi_task_outputs = self.centerpoint(multi_task_batch_inputs)
        metrics = self.centerpoint.compute_metrics(multi_task_batch_inputs, multi_task_outputs)
        multi_task_predictions = self.centerpoint.decode_outputs(multi_task_batch_inputs, multi_task_outputs)

        self.assertIsNotNone(multi_task_outputs.detection3d_head_outputs)
        assert multi_task_outputs.detection3d_head_outputs is not None
        center_head_outputs = multi_task_outputs.detection3d_head_outputs.center_head_outputs
        assert center_head_outputs is not None
        # The 16x16 canvas is downsampled by the backbone and fused back to 8x8 by the neck
        self.assertEqual(
            center_head_outputs.heatmap.shape,
            (self.batch_size, self.num_classes, 8, 8),
        )

        self.assertIn("loss", metrics)
        self.assertIn("loss_heatmap", metrics)
        self.assertIn("loss_bbox", metrics)
        self.assertTrue(torch.isfinite(metrics["loss"]).all())

        predictions = multi_task_predictions.detection3d_predictions
        assert predictions is not None
        self.assertEqual(len(predictions), self.batch_size)
        for sample_predictions in predictions:
            # use_velocity=True on the shared head, so decoded boxes carry 9 parameters
            self.assertEqual(sample_predictions.bboxes_3d.shape[1], 9)
            self.assertEqual(
                sample_predictions.scores_3d.shape[0], sample_predictions.bboxes_3d.shape[0]
            )
            self.assertEqual(
                sample_predictions.labels_3d.shape[0], sample_predictions.bboxes_3d.shape[0]
            )

    def test_build_export_specs_splits_pillar_encoder_and_backbone_neck_head(self) -> None:
        """
        Test that deployment export follows the CenterPoint split: decorated pillar features feed
        the PFN module and dense BEV features feed the backbone/neck/head module, with the runtime
        input and output names and the velocity branch appended last.
        """
        self.centerpoint.eval()
        multi_task_batch_inputs = self._build_multi_task_batch_inputs()
        voxels_data = multi_task_batch_inputs.voxels_data
        assert voxels_data is not None
        num_pillars, max_num_points = voxels_data.voxels.shape[:2]
        # [x, y, z, i, t, cluster-offset xyz, voxel-center-offset xyz]
        num_decorated_channels = 11

        specs = self.centerpoint.build_export_specs(multi_task_batch_inputs)

        self.assertEqual(
            list(specs), ["pts_voxel_encoder_centerpoint", "pts_backbone_neck_head_centerpoint"]
        )

        voxel_spec = specs["pts_voxel_encoder_centerpoint"]
        self.assertIsInstance(voxel_spec.module, _CenterPointVoxelEncoderExportWrapper)
        self.assertEqual(voxel_spec.input_param_names, ["input_features"])
        self.assertEqual(voxel_spec.output_names, ["pillar_features"])
        (input_features,) = voxel_spec.args
        self.assertEqual(
            input_features.shape, (num_pillars, max_num_points, num_decorated_channels)
        )
        with torch.no_grad():
            pillar_features = voxel_spec.module(*voxel_spec.args)
        # The singleton point dimension is kept for the runtime pillar-feature ABI.
        self.assertEqual(pillar_features.shape, (num_pillars, 1, 32))

        head_spec = specs["pts_backbone_neck_head_centerpoint"]
        self.assertIsInstance(head_spec.module, _CenterPointBackboneNeckHeadExportWrapper)
        self.assertEqual(head_spec.input_param_names, ["spatial_features"])
        self.assertEqual(head_spec.output_names, ["heatmap", "reg", "height", "dim", "rot", "vel"])
        (spatial_features,) = head_spec.args
        self.assertEqual(spatial_features.shape, (self.batch_size, 32, 16, 16))
        with torch.no_grad():
            outputs = head_spec.module(*head_spec.args)
        self.assertEqual([output.shape[1] for output in outputs], [self.num_classes, 2, 1, 3, 2, 2])
        for output in outputs:
            self.assertEqual(output.shape[0], self.batch_size)
            self.assertEqual(output.shape[2:], (8, 8))

    def test_build_export_specs_follow_head_velocity_configuration(self) -> None:
        """Test that a head built without velocity exports no ``vel`` output."""
        centerpoint = self._build_centerpoint(self._build_bbox_head(use_velocity=False)).eval()

        specs = centerpoint.build_export_specs(self._build_multi_task_batch_inputs())

        head_spec = specs["pts_backbone_neck_head_centerpoint"]
        self.assertEqual(head_spec.output_names, ["heatmap", "reg", "height", "dim", "rot"])
        with torch.no_grad():
            outputs = head_spec.module(*head_spec.args)
        self.assertEqual([output.shape[1] for output in outputs], [self.num_classes, 2, 1, 3, 2])

    def test_build_export_specs_requires_voxels_data(self) -> None:
        """Test that export spec construction rejects a batch without voxel data."""
        multi_task_batch_inputs = self._build_multi_task_batch_inputs()
        multi_task_batch_inputs = ModelBatchInputs(
            multi_task_gt_batch=multi_task_batch_inputs.multi_task_gt_batch,
            voxels_data=None,
            image_data=None,
            range_view_data=None,
        )

        with self.assertRaisesRegex(ValueError, "voxels_data"):
            self.centerpoint.build_export_specs(multi_task_batch_inputs)

    def test_build_export_spec_rejects_single_module_export(self) -> None:
        """Test that the single-module export entry point points at the split export."""
        with self.assertRaisesRegex(RuntimeError, "build_export_specs"):
            self.centerpoint.build_export_spec(self._build_multi_task_batch_inputs())

    def test_build_export_specs_keeps_training_mode(self) -> None:
        """Test that tracing the split modules leaves a training model in training mode."""
        self.centerpoint.train()

        self.centerpoint.build_export_specs(self._build_multi_task_batch_inputs())

        self.assertTrue(self.centerpoint.training)


if __name__ == "__main__":
    unittest.main()
