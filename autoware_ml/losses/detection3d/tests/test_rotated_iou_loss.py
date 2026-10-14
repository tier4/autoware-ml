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

"""Unit tests for the rotated BEV IoU loss."""

from __future__ import annotations

import unittest

import torch

from autoware_ml.losses.detection3d.rotated_iou_loss import Rotated2DIouLoss
from autoware_ml.models.detection3d.task_modules.bbox_coders import TransFusionBBoxCoder
from autoware_ml.ops.diff_iou_rotated.diff_iou_rotated import box2corners


class TestRotated2DIouLoss(unittest.TestCase):
    """Check the corner decoding and the IoU loss of ``Rotated2DIouLoss``."""

    def setUp(self) -> None:
        """Build the loss, a matching coder, and gravity-center gt boxes of two classes."""
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.class_names = ["car", "pedestrian"]
        # One grid cell is one metre: voxel_size[0] * out_size_factor == 1.0.
        self.point_cloud_range = [0.0, 0.0, -2.0, 16.0, 16.0, 2.0]
        self.voxel_size = [1.0, 1.0, 4.0]
        self.out_size_factor = 1
        self.coder = TransFusionBBoxCoder(
            pc_range=self.point_cloud_range,
            out_size_factor=self.out_size_factor,
            voxel_size=self.voxel_size,
            score_threshold_groups=None,
            post_center_range=None,
            code_size=10,
        )
        # (cx, cy, cz, length, width, height, yaw, vx, vy): a car and a pedestrian, both rotated.
        self.boxes = torch.tensor(
            [
                [
                    [2.0, 2.0, 0.0, 4.0, 1.6, 1.5, 0.25, 0.5, -0.1],
                    [8.0, 8.0, 0.0, 1.0, 0.6, 1.7, 1.0, 0.0, 0.0],
                    [12.0, 4.0, 0.0, 3.0, 1.8, 1.5, 1.0, -0.2, 0.3],
                ]
            ],
            device=self.device,
        )
        self.labels = torch.tensor([[0, 1, 0]], device=self.device)
        self.encoded = self.coder.encode(self.boxes)

    def _build_loss(
        self, labels_to_ignore_rotation: list[str] | None = None, loss_weight: float = 1.0
    ) -> Rotated2DIouLoss:
        """Build the loss on the fixture geometry."""
        return Rotated2DIouLoss(
            class_names=self.class_names,
            point_cloud_range=self.point_cloud_range,
            voxel_size=self.voxel_size,
            out_size_factor=self.out_size_factor,
            labels_to_ignore_rotation=labels_to_ignore_rotation,
            loss_weight=loss_weight,
        )

    def _reference_corners(self, boxes: torch.Tensor) -> torch.Tensor:
        """Corners of (x, y, length, width, yaw) boxes from the IoU op's own converter."""
        return box2corners(boxes[..., [0, 1, 3, 4, 6]])

    def test_resolves_ignored_labels_to_ids(self) -> None:
        """Class names to compare axis-aligned resolve to their label ids."""
        loss = self._build_loss(labels_to_ignore_rotation=["pedestrian"])

        self.assertEqual(loss.labels_id_to_ignore_rotation, [1])
        self.assertEqual(self._build_loss().labels_id_to_ignore_rotation, [])

    def test_rejects_unknown_ignored_label(self) -> None:
        """A class name outside class_names is a configuration error."""
        with self.assertRaises(ValueError):
            self._build_loss(labels_to_ignore_rotation=["cone"])

    def test_convert_to_bev_corners_matches_the_reference_corner_layout(self) -> None:
        """Encoded gt boxes decode into the corners box2corners builds from the metric boxes."""
        corners, dims = self._build_loss().convert_to_bev_corners(
            self.encoded, self.labels, is_gt=True
        )

        self.assertEqual(corners.shape, (1, 3, 4, 2))
        torch.testing.assert_close(
            corners, self._reference_corners(self.boxes), rtol=1e-5, atol=1e-5
        )
        torch.testing.assert_close(dims, self.boxes[..., 3:5], rtol=1e-5, atol=1e-5)

    def test_convert_to_bev_corners_normalizes_prediction_rotation(self) -> None:
        """A prediction's un-normalized (sin, cos) pair decodes like the unit one."""
        loss = self._build_loss()
        scaled = self.encoded.clone()
        scaled[..., 6:8] *= 3.0

        corners_gt, _ = loss.convert_to_bev_corners(self.encoded, self.labels, is_gt=True)
        corners_pred, _ = loss.convert_to_bev_corners(scaled, self.labels, is_gt=False)

        torch.testing.assert_close(corners_pred, corners_gt, rtol=1e-4, atol=1e-4)

    def test_convert_to_bev_corners_ignores_rotation_for_configured_classes(self) -> None:
        """Ignored classes are laid out axis-aligned while the others keep their yaw."""
        loss = self._build_loss(labels_to_ignore_rotation=["pedestrian"])

        corners, _ = loss.convert_to_bev_corners(self.encoded, self.labels, is_gt=True)

        rotated = self._reference_corners(self.boxes)
        axis_aligned_boxes = self.boxes.clone()
        axis_aligned_boxes[..., 6] = 0.0
        aligned = self._reference_corners(axis_aligned_boxes)
        torch.testing.assert_close(corners[:, [0, 2]], rotated[:, [0, 2]], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(corners[:, 1], aligned[:, 1], rtol=1e-5, atol=1e-5)

    @unittest.skipUnless(torch.cuda.is_available(), "the rotated IoU op requires CUDA")
    def test_perfect_predictions_have_zero_loss(self) -> None:
        """Predicting the encoded target exactly gives an IoU of one and no loss."""
        loss = self._build_loss()
        weights = torch.ones(self.labels.shape, device=self.device)

        pair_losses = loss(self.encoded.clone(), self.encoded, self.labels, weights)

        self.assertEqual(pair_losses.shape, (1, 3))
        torch.testing.assert_close(pair_losses, torch.zeros_like(pair_losses), rtol=0.0, atol=1e-4)

    @unittest.skipUnless(torch.cuda.is_available(), "the rotated IoU op requires CUDA")
    def test_loss_matches_one_minus_iou_for_a_known_shift(self) -> None:
        """An axis-aligned 2x2 box shifted by one cell has IoU 1/3, so the loss is 2/3."""
        loss = self._build_loss(loss_weight=1.0)
        target = self.coder.encode(
            torch.tensor([[[8.0, 8.0, 0.0, 2.0, 2.0, 1.5, 0.0, 0.0, 0.0]]], device=self.device)
        )
        prediction = target.clone()
        prediction[..., 0] += 1.0  # one grid cell is one metre
        labels = torch.tensor([[0]], device=self.device)
        weights = torch.ones(labels.shape, device=self.device)

        pair_losses = loss(prediction, target, labels, weights)

        torch.testing.assert_close(
            pair_losses, torch.tensor([[2.0 / 3.0]], device=self.device), rtol=0.0, atol=1e-4
        )

    @unittest.skipUnless(torch.cuda.is_available(), "the rotated IoU op requires CUDA")
    def test_loss_weight_and_pair_weights_scale_the_loss(self) -> None:
        """The loss weight scales every pair and a zero pair weight drops that pair."""
        target = self.encoded
        prediction = target.clone()
        prediction[..., 0] += 0.5
        weights = torch.tensor([[1.0, 0.0, 1.0]], device=self.device)

        unit = self._build_loss(loss_weight=1.0)(prediction, target, self.labels, weights)
        doubled = self._build_loss(loss_weight=2.0)(prediction, target, self.labels, weights)

        self.assertEqual(float(unit[0, 1]), 0.0)
        self.assertGreater(float(unit[0, 0]), 0.0)
        torch.testing.assert_close(doubled, 2.0 * unit, rtol=1e-6, atol=0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "the rotated IoU op requires CUDA")
    def test_ignored_rotation_makes_the_loss_insensitive_to_yaw(self) -> None:
        """For an ignored class, a prediction that only differs in yaw has zero loss."""
        loss = self._build_loss(labels_to_ignore_rotation=["pedestrian"])
        target = self.encoded[:, 1:2]
        prediction = target.clone()
        prediction[..., 6] = 0.0  # sin(yaw)
        prediction[..., 7] = 1.0  # cos(yaw)
        labels = self.labels[:, 1:2]
        weights = torch.ones(labels.shape, device=self.device)

        pair_losses = loss(prediction, target, labels, weights)

        torch.testing.assert_close(pair_losses, torch.zeros_like(pair_losses), rtol=0.0, atol=1e-4)

    @unittest.skipUnless(torch.cuda.is_available(), "the rotated IoU op requires CUDA")
    def test_loss_is_differentiable_with_respect_to_the_prediction(self) -> None:
        """Gradients reach the encoded prediction and point back towards the target."""
        loss = self._build_loss()
        target = self.encoded
        prediction = target.clone()
        prediction[..., 0] += 0.5
        prediction.requires_grad_(True)
        weights = torch.ones(self.labels.shape, device=self.device)

        loss(prediction, target, self.labels, weights).sum().backward()

        assert prediction.grad is not None
        self.assertTrue(torch.isfinite(prediction.grad).all())
        # Moving the shifted centers back towards the targets lowers the loss.
        self.assertTrue((prediction.grad[..., 0] > 0).all())


if __name__ == "__main__":
    unittest.main()
