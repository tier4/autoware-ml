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

"""Unit tests for CenterHead."""

import math
import unittest

from jaxtyping import Float32
import torch

from autoware_ml.models.detection3d.heads.centerhead import CenterHead
from autoware_ml.dataclasses.detection3d.head_outputs import (
    CenterHeadOutputs,
    Detection3DHeadOutputs,
)
from autoware_ml.datamodule.multi_task.dataclasses.detection3d import (
    Detection3DGTBatch,
)


class TestCenterHead(unittest.TestCase):
    """Unit tests for the CenterHead."""

    def setUp(self) -> None:
        """Set up the common classes/inputs for the tests."""
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(0)
        self.center_head = CenterHead(
            in_channels=384,
            class_names=["car", "pedestrian"],
            shared_channels=64,
            point_cloud_range=[0.0, 0.0, -2.0, 8.0, 8.0, 2.0],
            voxel_size=[0.5, 0.5, 4.0],
            out_size_factor=2,
            min_radius=1,
            score_threshold=0.1,
            post_max_size=10,
            nms_min_radius=1.0,
            use_velocity=True,
        ).to(self.device)

        # Dummy inputs
        self.gt_bboxes_3d = torch.tensor(
            [[[2.2, 3.3, 0.2, 4.0, 1.6, 1.5, 0.25, 0.5, -0.1, -0.2]]],
            device=self.device,
            dtype=torch.float32,
        )

        self.gt_labels_3d = torch.tensor([0], dtype=torch.int32, device=self.device)
        self.gt_valid_bboxes = torch.tensor([1], dtype=torch.int32, device=self.device)
        self.gt_bboxes_num_points = torch.tensor([[100]], dtype=torch.int32, device=self.device)
        self.detection3d_gt_batch = Detection3DGTBatch(
            gt_bboxes_3d=self.gt_bboxes_3d,
            gt_labels_3d=self.gt_labels_3d,
            gt_valid_bboxes=self.gt_valid_bboxes,
            gt_bboxes_num_points=self.gt_bboxes_num_points,
        )

    def test_centerhead_weights_mean_std(self) -> None:
        """Test that the CenterHead weights are initialized with means of 0.0 and std < 0.1."""
        weights = []
        biases = []
        for name, param in self.center_head.named_parameters():
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
        expected_bias_std = 0.144860
        self.assertAlmostEqual(weight_mean, expected_weight_mean, places=2)
        self.assertLess(weight_std, 0.1)
        self.assertAlmostEqual(bias_mean, expected_bias_mean, places=2)
        self.assertAlmostEqual(bias_std, expected_bias_std, places=2)

    def test_center_head_zero_forward(self) -> None:
        """
        Test that the CenterHead forward pass with all zero features
        returns outputs of the expected shape.
        """
        # Initialize a dummy feature map with the expected input shape for the CenterHead
        dummy_input_features = torch.zeros((1, 384, 4, 4), device=self.device, dtype=torch.float32)
        outputs = self.center_head(dummy_input_features)

        self.assertEqual(outputs.heatmaps.shape, (1, 2, 4, 4))
        self.assertEqual(outputs.centers.shape, (1, 2, 4, 4))
        self.assertEqual(outputs.dims.shape, (1, 3, 4, 4))
        self.assertEqual(outputs.rots.shape, (1, 2, 4, 4))
        self.assertEqual(outputs.vels.shape, (1, 2, 4, 4))

        # All values are the same since the input features are zeros and biases for heatmap heads
        # are set to -2.19.
        expected_heatmaps = torch.tensor(-2.1900, device=self.device).expand_as(outputs.heatmaps)
        self.assertTrue(torch.allclose(outputs.heatmaps, expected_heatmaps))

        expected_centers = (
            torch.tensor([[0.0541, 0.1058]], device=self.device)
            .view(1, 2, 1, 1)
            .expand_as(outputs.centers)
        )
        self.assertTrue(torch.allclose(outputs.centers, expected_centers, atol=1e-4))

        expected_dims = (
            torch.tensor([[0.0855, -0.0303, 0.0646]], device=self.device)
            .view(1, 3, 1, 1)
            .expand_as(outputs.dims)
        )
        self.assertTrue(torch.allclose(outputs.dims, expected_dims, atol=1e-4))

        expected_rots = (
            torch.tensor([[0.1088, -0.1206]], device=self.device)
            .view(1, 2, 1, 1)
            .expand_as(outputs.rots)
        )
        self.assertTrue(torch.allclose(outputs.rots, expected_rots, atol=1e-4))

        expected_vels = (
            torch.tensor([[0.0809, 0.0154]], device=self.device)
            .view(1, 2, 1, 1)
            .expand_as(outputs.vels)
        )
        self.assertTrue(torch.allclose(outputs.vels, expected_vels, atol=1e-4))

    def _build_center_head_outputs(
        self, batch_size: int, height: int, width: int, use_velocity: bool = True
    ) -> CenterHeadOutputs:
        """Build all-zero CenterHeadOutputs so tests only set the cells they care about."""
        return CenterHeadOutputs(
            heatmaps=torch.zeros((batch_size, 2, height, width), device=self.device),
            centers=torch.zeros((batch_size, 2, height, width), device=self.device),
            heights=torch.zeros((batch_size, 1, height, width), device=self.device),
            dims=torch.zeros((batch_size, 3, height, width), device=self.device),
            rots=torch.zeros((batch_size, 2, height, width), device=self.device),
            vels=(
                torch.zeros((batch_size, 2, height, width), device=self.device)
                if use_velocity
                else None
            ),
        )

    def test_build_targets_populates_heatmap_and_boxes(self) -> None:
        """Test that build_targets populates the heatmap and boxes correctly."""
        targets = self.center_head.get_targets(
            gt_bboxes_3d=self.gt_bboxes_3d,
            gt_labels_3d=self.gt_labels_3d,
            gt_valid_bboxes=self.gt_valid_bboxes,
            feature_map_size=(4, 4),
            device=self.device,
        )

        self.assertEqual(targets.heatmaps.shape, (1, 2, 4, 4))
        self.assertTrue(targets.valid_masks[0, 0].item())
        self.assertEqual(targets.reg_indices[0, 0].item(), 14)
        self.assertEqual(targets.heatmaps[0, 0, 3, 2].item(), 1.0)
        self.assertEqual(targets.reg_targets.shape, (1, 1, 10))
        expected_reg_targets = torch.cat(
            [
                torch.tensor([0.2, 0.3, 0.2], device=self.device),
                torch.tensor([4.0, 1.6, 1.5], device=self.device).log(),
                torch.tensor([math.sin(0.25), math.cos(0.25)], device=self.device),
                torch.tensor([0.5, -0.1], device=self.device),
            ],
            dim=-1,
        )
        self.assertTrue(
            torch.allclose(
                targets.reg_targets,
                expected_reg_targets.view(1, 1, -1),
            )
        )

    def test_decode_outputs_returns_length_width_height_after_unified_dim_order(self) -> None:
        """
        Test that decode_outputs returns the correct length, width, and height after
        applying the unified dimension order.
        """
        # Create a different CenterHead
        center_head = CenterHead(
            in_channels=4,
            class_names=["car", "truck", "bus", "pedestrian", "bicycle"],
            shared_channels=4,
            point_cloud_range=[0.0, 0.0, -2.0, 8.0, 8.0, 2.0],
            voxel_size=[0.5, 0.5, 4.0],
            out_size_factor=2,
            min_radius=1,
            score_threshold=0.1,
            post_max_size=10,
            nms_min_radius=1.0,
            use_velocity=False,
        ).to(self.device)

        # Dummy outputs from CenterHead.forward
        dummy_outputs = self._build_center_head_outputs(
            batch_size=1, height=4, width=4, use_velocity=False
        )
        dummy_outputs.heatmaps[0, :, :, :] = -20.0
        dummy_outputs.heatmaps[0, 0, 3, 2] = 20.0
        dummy_outputs.heights[0, 0, 3, 2] = 0.2
        dummy_outputs.dims[0, :, 3, 2] = torch.tensor([4.0, 1.6, 1.5], device=self.device).log()
        dummy_outputs.rots[0, 1, 3, 2] = 1.0
        dummy_detection3d_outputs = Detection3DHeadOutputs(
            center_head_outputs=dummy_outputs, transfusion_head_outputs=None
        )

        decoded_outputs = center_head.decode_outputs(dummy_detection3d_outputs)
        assert decoded_outputs.detection3d_predictions is not None
        self.assertEqual(decoded_outputs.detection3d_predictions[0].bboxes_3d.shape, (1, 7))
        self.assertTrue(
            torch.allclose(
                decoded_outputs.detection3d_predictions[0].bboxes_3d[0, 3:6],
                torch.tensor([4.0, 1.6, 1.5], device=self.device),
            )
        )

    def test_centerhead_uses_natural_dimension_order(self) -> None:
        """
        Test that feeding the regression targets straight back in as predictions round-trips to
        the original box dimensions, so encoding and decoding agree on the axis order.
        """
        center_head = CenterHead(
            in_channels=4,
            class_names=["car", "pdestrian"],
            shared_channels=4,
            point_cloud_range=[0.0, 0.0, -2.0, 8.0, 8.0, 2.0],
            voxel_size=[0.5, 0.5, 4.0],
            out_size_factor=2,
            min_radius=1,
            score_threshold=0.1,
            post_max_size=10,
            nms_min_radius=1.0,
            use_velocity=False,
        ).to(self.device)

        gt_bboxes_3d = torch.tensor(
            [[[2.0, 3.0, 0.2, 4.0, 1.6, 1.5, 0.25, 0.0, 0.0, 0.0]]],
            dtype=torch.float32,
            device=self.device,
        )
        targets = center_head.get_targets(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=torch.tensor([[0]], dtype=torch.int32, device=self.device),
            gt_valid_bboxes=torch.tensor([1], dtype=torch.int32, device=self.device),
            feature_map_size=(4, 4),
            device=self.device,
        )

        self.assertTrue(
            torch.allclose(
                targets.reg_targets[0, 0, 3:6],
                torch.tensor([4.0, 1.6, 1.5], device=self.device).log(),
            )
        )

        # Replay the regression targets as if the head had predicted them perfectly
        flat_index = int(targets.reg_indices[0, 0].item())
        y_index, x_index = divmod(flat_index, 4)
        reg_target = targets.reg_targets[0, 0]

        dummy_outputs = self._build_center_head_outputs(
            batch_size=1, height=4, width=4, use_velocity=False
        )
        dummy_outputs.heatmaps[0, :, :, :] = -20.0
        dummy_outputs.heatmaps[0, 0, y_index, x_index] = 20.0
        dummy_outputs.centers[0, :, y_index, x_index] = reg_target[0:2]
        dummy_outputs.heights[0, 0, y_index, x_index] = reg_target[2]
        dummy_outputs.dims[0, :, y_index, x_index] = reg_target[3:6]
        dummy_outputs.rots[0, :, y_index, x_index] = reg_target[6:8]

        decoded_outputs = center_head.decode_outputs(
            Detection3DHeadOutputs(center_head_outputs=dummy_outputs, transfusion_head_outputs=None)
        )
        predictions = decoded_outputs.detection3d_predictions
        assert predictions is not None

        self.assertTrue(
            torch.allclose(
                predictions[0].bboxes_3d[0, 3:6],
                torch.tensor([4.0, 1.6, 1.5], device=self.device),
            )
        )
        # The decoded centre returns to the original metric position of the ground truth box
        self.assertTrue(
            torch.allclose(
                predictions[0].bboxes_3d[0, 0:2],
                torch.tensor([2.0, 3.0], device=self.device),
                atol=1e-5,
            )
        )

    def test_loss_function(self) -> None:
        """Test that the loss function computes expected and non-negative losses."""
        # Modify dummy_outputs to have velocity values for testing
        # Dummy outputs from CenterHead.forward
        dummy_outputs = CenterHeadOutputs(
            heatmaps=torch.full((1, 2, 4, 4), -20.0, device=self.device),
            centers=torch.zeros((1, 2, 4, 4), device=self.device),
            heights=torch.zeros((1, 1, 4, 4), device=self.device),
            dims=torch.zeros((1, 3, 4, 4), device=self.device),
            rots=torch.zeros((1, 2, 4, 4), device=self.device),
            vels=torch.zeros((1, 2, 4, 4), device=self.device),
        )
        dummy_outputs.heatmaps[0, 0, 3, 2] = 20.0
        dummy_outputs.heights[0, 0, 3, 2] = 0.2
        dummy_outputs.dims[0, :, 3, 2] = torch.tensor([4.0, 1.6, 1.5], device=self.device).log()
        dummy_outputs.rots[0, 1, 3, 2] = 1.0
        assert dummy_outputs.vels is not None
        dummy_outputs.vels[0, :, 3, 2] = torch.tensor([0.5, -0.1], device=self.device)
        dummy_detection3d_outputs = Detection3DHeadOutputs(
            center_head_outputs=dummy_outputs, transfusion_head_outputs=None
        )

        losses = self.center_head.loss(
            outputs=dummy_detection3d_outputs,
            gt_bboxes_3d=self.gt_bboxes_3d,
            gt_labels_3d=self.gt_labels_3d,
            gt_valid_bboxes=self.gt_valid_bboxes,
        )
        self.assertIn("loss_heatmap", losses)
        self.assertIn("loss_bbox", losses)
        self.assertIn("loss", losses)

        self.assertTrue(losses["loss_heatmap"].item() >= 0.0)
        self.assertTrue(losses["loss_bbox"].item() >= 0.0)
        self.assertTrue(torch.isclose(losses["loss"], losses["loss_heatmap"] + losses["loss_bbox"]))

    def _build_zero_center_head_outputs_for_loss(
        self, batch_size: int = 1, use_velocity: bool = True
    ) -> Detection3DHeadOutputs:
        """
        Build head outputs whose regression maps are all zero so the L1 loss of a target reduces
        to its absolute value, keeping the expectations of the NaN tests readable.
        """
        outputs = self._build_center_head_outputs(
            batch_size=batch_size, height=4, width=4, use_velocity=use_velocity
        )
        outputs.heatmaps[:] = -20.0
        return Detection3DHeadOutputs(center_head_outputs=outputs, transfusion_head_outputs=None)

    def test_get_targets_produces_non_finite_targets_for_degenerate_boxes(self) -> None:
        """
        Test where the non-finite regression targets come from: unknown velocities stay NaN, and
        padded rows are all zeros so their log-dimensions become -inf.
        """
        gt_bboxes_3d = torch.tensor(
            [
                [
                    [2.2, 3.3, 0.2, 4.0, 1.6, 1.5, 0.25, math.nan, math.nan, 0.0],
                    # Padded slot, zero dimensions turn into -inf once log-encoded
                    [0.0] * 10,
                ]
            ],
            device=self.device,
            dtype=torch.float32,
        )
        targets = self.center_head.get_targets(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=torch.tensor([[0, 0]], dtype=torch.int32, device=self.device),
            gt_valid_bboxes=torch.tensor([1], dtype=torch.int32, device=self.device),
            feature_map_size=(4, 4),
            device=self.device,
        )

        # The valid box only carries NaN in its two velocity channels
        self.assertTrue(torch.isnan(targets.reg_targets[0, 0, 8:10]).all())
        self.assertTrue(torch.isfinite(targets.reg_targets[0, 0, 0:8]).all())
        # The padded box carries -inf in its three log-dimension channels
        self.assertTrue(torch.isneginf(targets.reg_targets[0, 1, 3:6]).all())
        # Only the first box counts as valid, so the padded row must never reach the loss
        self.assertTrue(targets.valid_masks[0, 0].item())
        self.assertFalse(targets.valid_masks[0, 1].item())

    def test_loss_ignores_nan_velocity_targets(self) -> None:
        """
        Test that the NaN velocity channels of an otherwise valid box are dropped from the box
        loss instead of poisoning it, and that the remaining channels are still averaged over
        their own count.
        """
        # Velocities are unknown for this box, everything else is a normal target
        gt_bboxes_3d = torch.tensor(
            [[[2.2, 3.3, 0.2, 4.0, 1.6, 1.5, 0.25, math.nan, math.nan, 0.0]]],
            device=self.device,
            dtype=torch.float32,
        )

        losses = self.center_head.loss(
            outputs=self._build_zero_center_head_outputs_for_loss(),
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=self.gt_labels_3d,
            gt_valid_bboxes=self.gt_valid_bboxes,
        )

        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertTrue(torch.isfinite(losses["loss_bbox"]))
        self.assertTrue(torch.isfinite(losses["loss_heatmap"]))

        # Predictions are all zero, so every surviving channel contributes |target|, averaged over
        # the eight non-NaN channels only and scaled by loss_bbox_weight.
        expected_abs_targets = [
            0.2,  # x offset inside the grid cell
            0.3,  # y offset inside the grid cell
            0.2,  # height
            math.log(4.0),
            math.log(1.6),
            math.log(1.5),
            abs(math.sin(0.25)),
            abs(math.cos(0.25)),
        ]
        expected_loss_bbox = 0.25 * sum(expected_abs_targets) / len(expected_abs_targets)
        self.assertAlmostEqual(losses["loss_bbox"].item(), expected_loss_bbox, places=5)

    def test_loss_backward_keeps_gradients_finite_with_nan_velocity_targets(self) -> None:
        """
        Test that a batch holding NaN velocity and padded targets still backpropagates finite
        gradients, since a single NaN in the loss would wipe out every head parameter.
        """
        outputs = self._build_zero_center_head_outputs_for_loss()
        assert outputs.center_head_outputs is not None
        center_head_outputs = outputs.center_head_outputs
        predictions = [
            center_head_outputs.heatmaps,
            center_head_outputs.centers,
            center_head_outputs.heights,
            center_head_outputs.dims,
            center_head_outputs.rots,
            center_head_outputs.vels,
        ]
        for prediction in predictions:
            assert prediction is not None
            prediction.requires_grad_(True)

        losses = self.center_head.loss(
            outputs=outputs,
            gt_bboxes_3d=torch.tensor(
                [
                    [
                        [2.2, 3.3, 0.2, 4.0, 1.6, 1.5, 0.25, math.nan, math.nan, 0.0],
                        [0.0] * 10,
                    ]
                ],
                device=self.device,
                dtype=torch.float32,
            ),
            gt_labels_3d=torch.tensor([[0, 0]], dtype=torch.int32, device=self.device),
            gt_valid_bboxes=torch.tensor([1], dtype=torch.int32, device=self.device),
        )
        losses["loss"].backward()

        for prediction in predictions:
            assert prediction is not None and prediction.grad is not None
            self.assertTrue(torch.isfinite(prediction.grad).all())
        # The velocity channels are the NaN ones, so they must receive no gradient at all
        assert center_head_outputs.vels is not None and center_head_outputs.vels.grad is not None
        self.assertTrue(
            torch.equal(
                center_head_outputs.vels.grad,
                torch.zeros_like(center_head_outputs.vels.grad),
            )
        )

    def test_loss_is_non_finite_when_a_valid_box_has_non_finite_geometry_targets(self) -> None:
        """
        Test that only velocity is allowed to be unknown: a NaN or infinite target on any other
        channel of a valid box is corrupt ground truth and is left to surface as a non-finite
        loss instead of being masked away.
        """
        corrupt_gt_bboxes = {
            # log(0) -> -inf on the length channel
            "zero_length": [2.2, 3.3, 0.2, 0.0, 1.6, 1.5, 0.25, 0.5, -0.1, -0.2],
            # log(negative) -> NaN on the width channel
            "negative_width": [2.2, 3.3, 0.2, 4.0, -1.6, 1.5, 0.25, 0.5, -0.1, -0.2],
            "nan_height": [2.2, 3.3, 0.2, 4.0, 1.6, math.nan, 0.25, 0.5, -0.1, -0.2],
            "nan_z": [2.2, 3.3, math.nan, 4.0, 1.6, 1.5, 0.25, 0.5, -0.1, -0.2],
            # sin/cos of a non-finite yaw -> NaN on both heading channels
            "inf_yaw": [2.2, 3.3, 0.2, 4.0, 1.6, 1.5, math.inf, 0.5, -0.1, -0.2],
        }
        for case_name, gt_bbox in corrupt_gt_bboxes.items():
            with self.subTest(case=case_name):
                losses = self.center_head.loss(
                    outputs=self._build_zero_center_head_outputs_for_loss(),
                    gt_bboxes_3d=torch.tensor([[gt_bbox]], device=self.device, dtype=torch.float32),
                    gt_labels_3d=self.gt_labels_3d,
                    gt_valid_bboxes=self.gt_valid_bboxes,
                )
                # Both the box loss and the total loss come back as NaN or Inf
                self.assertFalse(torch.isfinite(losses["loss_bbox"]))
                self.assertFalse(torch.isfinite(losses["loss"]))

    def test_loss_stays_finite_for_non_finite_geometry_of_invalid_boxes(self) -> None:
        """
        Test that only the boxes that contribute to the loss can make it non-finite, so the zero
        padding that every batch carries, whose dimensions encode to -inf, is not mistaken for
        corrupt ground truth.
        """
        gt_bbox = [2.2, 3.3, 0.2, 4.0, 1.6, 1.5, 0.25, 0.5, -0.1, -0.2]
        corrupt_gt_bbox = [2.2, 3.3, 0.2, 0.0, -1.6, math.nan, math.inf, 0.5, -0.1, -0.2]

        losses = self.center_head.loss(
            outputs=self._build_zero_center_head_outputs_for_loss(),
            # Only the first box is valid, so the corrupt one sits in the padded tail
            gt_bboxes_3d=torch.tensor(
                [[gt_bbox, corrupt_gt_bbox]], device=self.device, dtype=torch.float32
            ),
            gt_labels_3d=torch.tensor([[0, 0]], dtype=torch.int32, device=self.device),
            gt_valid_bboxes=torch.tensor([1], dtype=torch.int32, device=self.device),
        )
        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertTrue(torch.isfinite(losses["loss_bbox"]))

    def test_loss_without_velocity_treats_every_channel_as_geometry(self) -> None:
        """
        Test that a head built without velocity treats all of its channels as geometry: the
        velocity columns of the ground truth are never encoded, so their NaNs are harmless, while
        a corrupt geometry channel still makes the loss non-finite.
        """
        center_head = CenterHead(
            in_channels=384,
            class_names=["car", "pedestrian"],
            shared_channels=64,
            point_cloud_range=[0.0, 0.0, -2.0, 8.0, 8.0, 2.0],
            voxel_size=[0.5, 0.5, 4.0],
            out_size_factor=2,
            min_radius=1,
            score_threshold=0.1,
            post_max_size=10,
            nms_min_radius=1.0,
            use_velocity=False,
        ).to(self.device)

        losses = center_head.loss(
            outputs=self._build_zero_center_head_outputs_for_loss(use_velocity=False),
            gt_bboxes_3d=torch.tensor(
                [[[2.2, 3.3, 0.2, 4.0, 1.6, 1.5, 0.25, math.nan, math.nan, 0.0]]],
                device=self.device,
                dtype=torch.float32,
            ),
            gt_labels_3d=self.gt_labels_3d,
            gt_valid_bboxes=self.gt_valid_bboxes,
        )
        self.assertTrue(torch.isfinite(losses["loss"]))

        # A zero height encodes to -inf and, with no velocity channel to excuse it, poisons the loss
        corrupt_losses = center_head.loss(
            outputs=self._build_zero_center_head_outputs_for_loss(use_velocity=False),
            gt_bboxes_3d=torch.tensor(
                [[[2.2, 3.3, 0.2, 4.0, 1.6, 0.0, 0.25, 0.0, 0.0, 0.0]]],
                device=self.device,
                dtype=torch.float32,
            ),
            gt_labels_3d=self.gt_labels_3d,
            gt_valid_bboxes=self.gt_valid_bboxes,
        )
        self.assertFalse(torch.isfinite(corrupt_losses["loss_bbox"]))
        self.assertFalse(torch.isfinite(corrupt_losses["loss"]))

    def test_decode_regression_outputs_converts_grid_cells_to_physical_boxes(self) -> None:
        """
        Test that _decode_regression_outputs turns the per-cell regression maps into physical
        boxes, applying the grid offset, the exp on dimensions and the atan2 on rotations.
        """
        # voxel_size 0.5 with out_size_factor 2 and a zero point_cloud_range origin means the
        # metric x equals (grid x + predicted offset), which keeps the expectations readable.
        outputs = self._build_center_head_outputs(batch_size=1, height=4, width=4)
        # flattened index 14 -> (y=3, x=2, 3*4+2 = 14), flattened index 5 -> (y=1, x=1, 1*4+1 = 5)
        outputs.centers[0, :, 3, 2] = torch.tensor([0.25, 0.5], device=self.device)
        outputs.centers[0, :, 1, 1] = torch.tensor([-0.5, 0.25], device=self.device)
        outputs.heights[0, 0, 3, 2] = 0.2
        outputs.heights[0, 0, 1, 1] = -1.0
        outputs.dims[0, :, 3, 2] = torch.tensor([4.0, 1.6, 1.5], device=self.device).log()
        outputs.dims[0, :, 1, 1] = torch.tensor([2.0, 3.0, 0.5], device=self.device).log()
        outputs.rots[0, :, 3, 2] = torch.tensor(
            [math.sin(0.25), math.cos(0.25)], device=self.device
        )
        outputs.rots[0, :, 1, 1] = torch.tensor(
            [math.sin(-1.2), math.cos(-1.2)], device=self.device
        )
        assert outputs.vels is not None
        outputs.vels[0, :, 3, 2] = torch.tensor([0.5, -0.1], device=self.device)
        outputs.vels[0, :, 1, 1] = torch.tensor([-2.0, 3.0], device=self.device)

        flatten_indices = torch.tensor([[14, 5]], dtype=torch.int64, device=self.device)
        bboxes_predictions = self.center_head._decode_regression_outputs(
            center_head_outputs=outputs,
            flatten_indices=flatten_indices,
            width=4,
        )

        # (batch_size, num_indices, 9) because the head is built with use_velocity=True
        self.assertEqual(bboxes_predictions.shape, (1, 2, 9))
        expected_bboxes_predictions = torch.tensor(
            [
                [
                    [2.25, 3.5, 0.2, 4.0, 1.6, 1.5, 0.25, 0.5, -0.1],
                    [0.5, 1.25, -1.0, 2.0, 3.0, 0.5, -1.2, -2.0, 3.0],
                ]
            ],
            device=self.device,
        )
        self.assertTrue(torch.allclose(bboxes_predictions, expected_bboxes_predictions, atol=1e-5))

    def test_decode_regression_outputs_gathers_per_sample_feature_maps(self) -> None:
        """
        Test that _decode_regression_outputs gathers along the feature map axis of each sample
        rather than indexing the batch axis, so samples never read each other's predictions.
        """
        outputs = self._build_center_head_outputs(batch_size=2, height=4, width=4)
        # Both samples read the same flattened index but hold different values there.
        outputs.heights[0, 0, 3, 2] = 1.0
        outputs.dims[0, :, 3, 2] = torch.tensor([4.0, 1.6, 1.5], device=self.device).log()
        outputs.heights[1, 0, 3, 2] = -3.0
        outputs.dims[1, :, 3, 2] = torch.tensor([2.0, 8.0, 0.5], device=self.device).log()

        flatten_indices = torch.tensor([[14], [14]], dtype=torch.int64, device=self.device)
        bboxes_predictions = self.center_head._decode_regression_outputs(
            center_head_outputs=outputs,
            flatten_indices=flatten_indices,
            width=4,
        )

        self.assertEqual(bboxes_predictions.shape, (2, 1, 9))
        # Both samples decode the same grid cell, so only the regressed values differ.
        expected_bboxes_predictions = torch.tensor(
            [
                [[2.0, 3.0, 1.0, 4.0, 1.6, 1.5, 0.0, 0.0, 0.0]],
                [[2.0, 3.0, -3.0, 2.0, 8.0, 0.5, 0.0, 0.0, 0.0]],
            ],
            device=self.device,
        )
        self.assertTrue(torch.allclose(bboxes_predictions, expected_bboxes_predictions, atol=1e-5))

    def test_decode_regression_outputs_drops_velocity_when_disabled(self) -> None:
        """
        Test that _decode_regression_outputs returns 7 box parameters when the head is
        configured without velocity.
        """
        center_head = CenterHead(
            in_channels=384,
            class_names=["car", "bus"],
            shared_channels=64,
            point_cloud_range=[0.0, 0.0, -2.0, 8.0, 8.0, 2.0],
            voxel_size=[0.5, 0.5, 4.0],
            out_size_factor=2,
            min_radius=1,
            score_threshold=0.1,
            post_max_size=10,
            nms_min_radius=1.0,
            use_velocity=False,
        ).to(self.device)

        outputs = self._build_center_head_outputs(
            batch_size=1, height=4, width=4, use_velocity=False
        )
        outputs.heights[0, 0, 3, 2] = 0.2
        outputs.dims[0, :, 3, 2] = torch.tensor([4.0, 1.6, 1.5], device=self.device).log()

        flatten_indices = torch.tensor([[14]], dtype=torch.int64, device=self.device)
        bboxes_predictions = center_head._decode_regression_outputs(
            center_head_outputs=outputs,
            flatten_indices=flatten_indices,
            width=4,
        )

        self.assertEqual(bboxes_predictions.shape, (1, 1, 7))
        expected_bboxes_predictions = torch.tensor(
            [[[2.0, 3.0, 0.2, 4.0, 1.6, 1.5, 0.0]]], device=self.device
        )
        self.assertTrue(torch.allclose(bboxes_predictions, expected_bboxes_predictions, atol=1e-5))

    def _build_filter_inputs(
        self, keep_masks: torch.Tensor
    ) -> tuple[
        Float32[torch.Tensor, "batch_size num_boxes box_dim"],
        Float32[torch.Tensor, "batch_size num_boxes"],
        Float32[torch.Tensor, "batch_size num_boxes"],
    ]:
        """
        Build the (batch_size=2, num_classes=2, max_num_bboxes=3) inputs shared by the
        _filter_bbox_predictions tests. Each box row is stamped with its flattened slot so the
        assertions can follow which prediction ends up where.
        """
        scores = torch.tensor(
            [
                [[0.9, 0.5, 0.2], [0.8, 0.4, 0.1]],
                [[0.7, 0.3, 0.05], [0.6, 0.35, 0.02]],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        # class_ids[b, c, k] == c, matching how decode_outputs builds them
        class_ids = (
            torch.arange(2, dtype=torch.int64, device=self.device)
            .view(1, 2, 1)
            .expand_as(keep_masks)
            .contiguous()
        )
        # flattened slot index broadcast across the 9 box parameters
        flatten_bboxes_predictions = (
            torch.arange(6, dtype=torch.float32, device=self.device)
            .view(1, 6, 1)
            .expand(2, 6, 9)
            .contiguous()
        )
        return flatten_bboxes_predictions, scores, class_ids

    def test_filter_bbox_predictions_keeps_survivors_ranked_across_classes(self) -> None:
        """
        Test that _filter_bbox_predictions drops suppressed boxes and ranks the survivors by
        score across classes, carrying the matching class ids and box parameters along.
        """
        # flattened slots are ordered [c0k0, c0k1, c0k2, c1k0, c1k1, c1k2]
        # (batch_size=2, num_classes=2, max_num_bboxes=3)
        keep_masks = torch.tensor(
            [
                [[True, False, True], [True, False, False]],
                [[False, False, False], [True, False, False]],
            ],
            device=self.device,
        )
        flatten_bboxes_predictions, scores, class_ids = self._build_filter_inputs(keep_masks)

        predictions = self.center_head._filter_bbox_predictions(
            flatten_bboxes_predictions=flatten_bboxes_predictions,
            scores=scores,
            class_ids=class_ids,
            keep_masks=keep_masks,
            max_num_bboxes=10,
            batch_size=2,
        ).detection3d_predictions
        assert predictions is not None

        self.assertEqual(len(predictions), 2)

        # Sample 0 keeps slots 0, 2 and 3, ranked 0.9 (class 0), 0.8 (class 1), 0.2 (class 0)
        self.assertTrue(
            torch.allclose(
                predictions[0].scores_3d,
                torch.tensor([0.9, 0.8, 0.2], device=self.device),
            )
        )
        self.assertTrue(
            torch.equal(predictions[0].labels_3d, torch.tensor([0, 1, 0], device=self.device))
        )
        self.assertTrue(
            torch.equal(
                predictions[0].bboxes_3d[:, 0],
                torch.tensor([0.0, 3.0, 2.0], device=self.device),
            )
        )

        # Sample 1 keeps a single box, so its tensors are shorter than sample 0's
        self.assertTrue(
            torch.allclose(predictions[1].scores_3d, torch.tensor([0.6], device=self.device))
        )
        self.assertTrue(
            torch.equal(predictions[1].labels_3d, torch.tensor([1], device=self.device))
        )
        self.assertTrue(
            torch.equal(predictions[1].bboxes_3d[:, 0], torch.tensor([3.0], device=self.device))
        )

    def test_filter_bbox_predictions_caps_each_sample_at_max_num_bboxes(self) -> None:
        """
        Test that _filter_bbox_predictions truncates each sample to max_num_bboxes, keeping the
        highest scoring survivors, and that the cap counts across classes rather than per class.
        """
        # (batch_size=2, num_classes=2, max_num_bboxes=3)
        keep_masks = torch.tensor(
            [
                [[True, False, True], [True, False, False]],
                [[False, False, False], [True, False, False]],
            ],
            device=self.device,
        )
        flatten_bboxes_predictions, scores, class_ids = self._build_filter_inputs(keep_masks)

        predictions = self.center_head._filter_bbox_predictions(
            flatten_bboxes_predictions=flatten_bboxes_predictions,
            scores=scores,
            class_ids=class_ids,
            keep_masks=keep_masks,
            max_num_bboxes=2,
            batch_size=2,
        ).detection3d_predictions
        assert predictions is not None

        # Sample 0 has three survivors but only the top two fit under the cap
        self.assertEqual(predictions[0].scores_3d.shape, (2,))
        self.assertTrue(
            torch.allclose(predictions[0].scores_3d, torch.tensor([0.9, 0.8], device=self.device))
        )
        # Sample 1 has fewer survivors than the cap, so the padded slots are dropped
        self.assertEqual(predictions[1].scores_3d.shape, (1,))
        self.assertTrue(
            torch.allclose(predictions[1].scores_3d, torch.tensor([0.6], device=self.device))
        )

    def test_filter_bbox_predictions_returns_empty_entry_per_sample_when_all_suppressed(
        self,
    ) -> None:
        """
        Test that _filter_bbox_predictions still returns one entry per sample when NMS
        suppresses everything, so the predictions stay aligned with the batch.
        """
        keep_masks = torch.zeros((2, 2, 3), dtype=torch.bool, device=self.device)
        flatten_bboxes_predictions, scores, class_ids = self._build_filter_inputs(keep_masks)

        predictions = self.center_head._filter_bbox_predictions(
            flatten_bboxes_predictions=flatten_bboxes_predictions,
            scores=scores,
            class_ids=class_ids,
            keep_masks=keep_masks,
            max_num_bboxes=10,
            batch_size=2,
        ).detection3d_predictions
        assert predictions is not None

        self.assertEqual(len(predictions), 2)
        for sample_predictions in predictions:
            self.assertEqual(sample_predictions.scores_3d.shape, (0,))
            self.assertEqual(sample_predictions.labels_3d.shape, (0,))
            self.assertEqual(sample_predictions.bboxes_3d.shape, (0, 9))
            # The suppressed slots are sunk to -inf internally and must not leak out
            self.assertFalse(bool(torch.isinf(sample_predictions.scores_3d).any()))


if __name__ == "__main__":
    unittest.main()
