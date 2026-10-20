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
from autoware_ml.ops.diff_iou_rotated.diff_iou_rotated import diff_iou_rotated_2d


class TestRotated2DIouLoss(unittest.TestCase):
    """Check the BEV footprint selection and the IoU loss of ``Rotated2DIouLoss``."""

    def setUp(self) -> None:
        """Build gravity-center metric boxes of two classes."""
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.class_names = ["car", "pedestrian"]
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
        self.weights = torch.ones(self.labels.shape, device=self.device)

    def _build_loss(
        self, labels_to_ignore_rotation: list[str] | None = None, loss_weight: float = 1.0
    ) -> Rotated2DIouLoss:
        """Build the loss on the fixture classes."""
        return Rotated2DIouLoss(
            class_names=self.class_names,
            labels_to_ignore_rotation=labels_to_ignore_rotation,
            loss_weight=loss_weight,
        )

    def _shifted_prediction(self, shift_x: float) -> torch.Tensor:
        """Copy the fixture boxes with their BEV center moved along x."""
        prediction = self.boxes.clone()
        prediction[..., 0] += shift_x
        return prediction

    def test_resolves_ignored_labels_to_ids(self) -> None:
        """Class names to compare axis-aligned resolve to their label ids."""
        loss = self._build_loss(labels_to_ignore_rotation=["pedestrian"])

        self.assertEqual(loss.labels_id_to_ignore_rotation, [1])
        self.assertEqual(self._build_loss().labels_id_to_ignore_rotation, [])

    def test_rejects_unknown_ignored_label(self) -> None:
        """A class name outside class_names is a configuration error."""
        with self.assertRaises(ValueError):
            self._build_loss(labels_to_ignore_rotation=["cone"])

    def test_bev_boxes_pick_the_footprint_fields(self) -> None:
        """The BEV boxes are (x, y, length, width, yaw) of the metric boxes."""
        bev = self._build_loss()._bev_boxes(self.boxes, self.labels)

        self.assertEqual(bev.shape, (1, 3, 5))
        torch.testing.assert_close(bev, self.boxes[..., [0, 1, 3, 4, 6]])

    def test_bev_boxes_drop_the_yaw_of_ignored_classes(self) -> None:
        """Ignored classes are laid out axis-aligned while the others keep their yaw."""
        loss = self._build_loss(labels_to_ignore_rotation=["pedestrian"])

        bev = loss._bev_boxes(self.boxes, self.labels)

        torch.testing.assert_close(bev[:, [0, 2], 4], self.boxes[:, [0, 2], 6])
        self.assertEqual(float(bev[0, 1, 4]), 0.0)
        torch.testing.assert_close(bev[..., :4], self.boxes[..., [0, 1, 3, 4]])

    @unittest.skipUnless(torch.cuda.is_available(), "the rotated IoU op requires CUDA")
    def test_perfect_predictions_have_zero_loss(self) -> None:
        """Predicting the target exactly gives an IoU of one and no loss."""
        pair_losses = self._build_loss()(self.boxes.clone(), self.boxes, self.labels, self.weights)

        self.assertEqual(pair_losses.shape, (1, 3))
        torch.testing.assert_close(pair_losses, torch.zeros_like(pair_losses), rtol=0.0, atol=1e-4)

    @unittest.skipUnless(torch.cuda.is_available(), "the rotated IoU op requires CUDA")
    def test_loss_matches_one_minus_iou_for_a_known_shift(self) -> None:
        """An axis-aligned 2x2 box shifted by one metre has IoU 1/3, so the loss is 2/3."""
        target = torch.tensor([[[8.0, 8.0, 0.0, 2.0, 2.0, 1.5, 0.0, 0.0, 0.0]]], device=self.device)
        prediction = target.clone()
        prediction[..., 0] += 1.0
        labels = torch.tensor([[0]], device=self.device)
        weights = torch.ones(labels.shape, device=self.device)

        pair_losses = self._build_loss()(prediction, target, labels, weights)

        torch.testing.assert_close(
            pair_losses, torch.tensor([[2.0 / 3.0]], device=self.device), rtol=0.0, atol=1e-4
        )

    @unittest.skipUnless(torch.cuda.is_available(), "the rotated IoU op requires CUDA")
    def test_loss_is_one_minus_the_rotated_iou_op(self) -> None:
        """The loss of rotated pairs is exactly one minus ``diff_iou_rotated_2d``."""
        prediction = self._shifted_prediction(0.5)

        pair_losses = self._build_loss()(prediction, self.boxes, self.labels, self.weights)

        ious = diff_iou_rotated_2d(
            prediction[..., [0, 1, 3, 4, 6]], self.boxes[..., [0, 1, 3, 4, 6]]
        )
        torch.testing.assert_close(pair_losses, 1.0 - ious)

    @unittest.skipUnless(torch.cuda.is_available(), "the rotated IoU op requires CUDA")
    def test_loss_weight_and_pair_weights_scale_the_loss(self) -> None:
        """The loss weight scales every pair and a zero pair weight drops that pair."""
        prediction = self._shifted_prediction(0.5)
        weights = torch.tensor([[1.0, 0.0, 1.0]], device=self.device)

        unit = self._build_loss(loss_weight=1.0)(prediction, self.boxes, self.labels, weights)
        doubled = self._build_loss(loss_weight=2.0)(prediction, self.boxes, self.labels, weights)

        self.assertEqual(float(unit[0, 1]), 0.0)
        self.assertGreater(float(unit[0, 0]), 0.0)
        torch.testing.assert_close(doubled, unit.add(unit), rtol=1e-6, atol=0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "the rotated IoU op requires CUDA")
    def test_ignored_rotation_makes_the_loss_insensitive_to_yaw(self) -> None:
        """For an ignored class, a prediction that only differs in yaw has zero loss."""
        loss = self._build_loss(labels_to_ignore_rotation=["pedestrian"])
        target = self.boxes[:, 1:2]
        prediction = target.clone()
        prediction[..., 6] = 0.0
        labels = self.labels[:, 1:2]
        weights = torch.ones(labels.shape, device=self.device)

        pair_losses = loss(prediction, target, labels, weights)

        torch.testing.assert_close(pair_losses, torch.zeros_like(pair_losses), rtol=0.0, atol=1e-4)

    @unittest.skipUnless(torch.cuda.is_available(), "the rotated IoU op requires CUDA")
    def test_loss_is_differentiable_with_respect_to_the_prediction(self) -> None:
        """Gradients reach the prediction and point back towards the target."""
        prediction = self._shifted_prediction(0.5).requires_grad_(True)

        self._build_loss()(prediction, self.boxes, self.labels, self.weights).sum().backward()

        assert prediction.grad is not None
        self.assertTrue(torch.isfinite(prediction.grad).all())
        # Moving the shifted centers back towards the targets lowers the loss.
        self.assertTrue((prediction.grad[..., 0] > 0).all())

    @unittest.skipUnless(torch.cuda.is_available(), "the rotated IoU op requires CUDA")
    def test_half_precision_inputs_under_autocast_give_the_float32_loss(self) -> None:
        """Half boxes under fp16 autocast give a finite float32 loss equal to the fp32 one."""
        loss = self._build_loss(labels_to_ignore_rotation=["pedestrian"])
        prediction = self._shifted_prediction(0.5)
        expected = loss(prediction, self.boxes, self.labels, self.weights)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            actual = loss(prediction.half(), self.boxes.half(), self.labels, self.weights.half())

        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
