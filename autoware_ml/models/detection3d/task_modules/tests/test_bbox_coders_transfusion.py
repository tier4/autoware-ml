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

"""Unit tests for the TransFusion box coder.

Ground-truth boxes in this framework are gravity center only (see ``Box3DCenterCoordinateType``),
so ``encode`` and ``decode_boxes`` must agree on what the height channel means: the value that
``encode`` writes for a gravity-center box has to come back out of ``decode_boxes`` as the same
gravity-center z, with no half-height offset in either direction.
"""

from __future__ import annotations

import unittest

import torch

from autoware_ml.models.detection3d.task_modules.bbox_coders import (
    ScoreThresholdConfig,
    TransFusionBBoxCoder,
)
from autoware_ml.types.geometry import Box3DFieldIndex


class TestTransFusionBBoxCoderHeight(unittest.TestCase):
    """Check that ``encode`` and ``decode_boxes`` share the gravity-center height convention."""

    def setUp(self) -> None:
        """Build a coder and a batch of gravity-center boxes with distinct heights."""
        self.pc_range = [-10.0, -20.0, -2.0, 10.0, 20.0, 4.0]
        self.voxel_size = [0.5, 0.25, 0.2]
        self.out_size_factor = 2
        self.post_center_range = [-1.0, -1.0, -5.0, 10.0, 10.0, 5.0]
        self.coder = self._build_coder(code_size=10)
        # (cx, cy, cz, length, width, height, yaw, vx, vy) in gravity-center convention. Heights
        # differ per box so a half-height offset cannot cancel out across the batch.
        self.boxes = torch.tensor(
            [
                [
                    [2.0, 4.0, 1.0, 4.0, 2.0, 1.5, 0.25, 0.1, -0.2],
                    [-3.0, 0.5, -0.75, 0.8, 0.8, 1.7, -2.9, 0.0, 0.0],
                ],
                [
                    [6.0, -8.0, 0.2, 10.0, 2.5, 3.6, 1.2, 3.0, 0.5],
                    [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                ],
            ],
            dtype=torch.float32,
        )

    def _build_coder(self, code_size: int) -> TransFusionBBoxCoder:
        """Build a coder with the shared scene geometry and the given code size."""
        return TransFusionBBoxCoder(
            pc_range=self.pc_range,
            out_size_factor=self.out_size_factor,
            voxel_size=self.voxel_size,
            class_names=["car", "pedestrian"],
            score_threshold_configs=None,
            post_center_range=self.post_center_range,
            code_size=code_size,
        )

    @staticmethod
    def _split_encoded(encoded: torch.Tensor) -> dict[str, torch.Tensor | None]:
        """Slice ``(batch, boxes, code)`` targets into the ``(batch, channels, boxes)`` layout.

        This is the layout the prediction heads emit and ``decode_boxes`` consumes.
        """
        channels_first = encoded.permute(0, 2, 1)
        return {
            "centers": channels_first[:, 0:2, :],
            "heights": channels_first[:, 2:3, :],
            "dims": channels_first[:, 3:6, :],
            "rots": channels_first[:, 6:8, :],
            "vels": channels_first[:, 8:10, :] if encoded.shape[-1] == 10 else None,
        }

    def _round_trip(self, coder: TransFusionBBoxCoder) -> torch.Tensor:
        """Encode the test boxes and decode them back into metric boxes."""
        return coder.decode_boxes(**self._split_encoded(coder.encode(self.boxes)))

    def test_encode_keeps_gravity_center_height(self) -> None:
        """The height target of a gravity-center box is its z, not z shifted by half its height."""
        encoded = self.coder.encode(self.boxes)

        torch.testing.assert_close(encoded[..., 2], self.boxes[..., Box3DFieldIndex.Z])

    def test_decode_boxes_emits_height_channel_as_gravity_center_z(self) -> None:
        """``decode_boxes`` passes the regressed height straight through as the box z."""
        parts = self._split_encoded(self.coder.encode(self.boxes))

        decoded = self.coder.decode_boxes(**parts)

        heights = parts["heights"]
        assert heights is not None
        torch.testing.assert_close(decoded[..., Box3DFieldIndex.Z], heights[:, 0, :])

    def test_encode_decode_boxes_round_trip_preserves_height(self) -> None:
        """A gravity-center box survives ``encode`` followed by ``decode_boxes`` unchanged in z."""
        decoded = self._round_trip(self.coder)

        torch.testing.assert_close(
            decoded[..., Box3DFieldIndex.Z],
            self.boxes[..., Box3DFieldIndex.Z],
            rtol=1e-5,
            atol=1e-5,
        )

    def test_encode_decode_boxes_round_trip_preserves_full_geometry(self) -> None:
        """Every channel, not only z, comes back after ``encode`` then ``decode_boxes``."""
        decoded = self._round_trip(self.coder)

        self.assertEqual(decoded.shape, self.boxes.shape)
        # Yaw is stored as (sin, cos) so it round-trips through atan2 with float noise only.
        torch.testing.assert_close(decoded, self.boxes, rtol=1e-5, atol=1e-5)

    def test_round_trip_height_is_independent_of_velocity_channels(self) -> None:
        """The height convention does not change when the coder has no velocity channels."""
        coder = self._build_coder(code_size=8)
        parts = self._split_encoded(coder.encode(self.boxes))
        self.assertIsNone(parts["vels"])

        decoded = coder.decode_boxes(**parts)

        self.assertEqual(decoded.shape, (*self.boxes.shape[:2], 7))
        torch.testing.assert_close(
            decoded[..., Box3DFieldIndex.Z],
            self.boxes[..., Box3DFieldIndex.Z],
            rtol=1e-5,
            atol=1e-5,
        )


class TestTransFusionBBoxCoderScoreThresholds(unittest.TestCase):
    """Check that the score threshold groups are resolved once and applied per class."""

    def setUp(self) -> None:
        """Share the scene geometry and the class names across the tests."""
        self.class_names = ["car", "pedestrian", "bicycle"]
        self.bbox_scores = torch.tensor([[0.1, 0.4, 0.1, 0.4, 0.0, 0.05]])
        self.bbox_classes = torch.tensor([[0, 0, 1, 1, 2, 2]], dtype=torch.int64)

    def _build_coder(
        self, score_threshold_configs: list[ScoreThresholdConfig] | None
    ) -> TransFusionBBoxCoder:
        """Build a coder with the given score threshold groups."""
        return TransFusionBBoxCoder(
            pc_range=[-10.0, -20.0, -2.0, 10.0, 20.0, 4.0],
            out_size_factor=2,
            voxel_size=[0.5, 0.25, 0.2],
            class_names=self.class_names,
            score_threshold_configs=score_threshold_configs,
            post_center_range=None,
            code_size=10,
        )

    def test_resolves_one_threshold_per_class_at_construction(self) -> None:
        """Test that the groups are flattened into class order once, unnamed classes at zero."""
        coder = self._build_coder(
            [
                ScoreThresholdConfig(class_names=["pedestrian"], score_threshold=0.2),
                ScoreThresholdConfig(class_names=["car"], score_threshold=0.3),
            ]
        )

        self.assertEqual(coder.class_score_thresholds, (0.3, 0.2, 0.0))

    def test_no_groups_keeps_every_proposal(self) -> None:
        """Test that None disables the score filter instead of thresholding at zero."""
        coder = self._build_coder(None)

        keep_masks = coder.filter_bboxes_score(
            bbox_scores=self.bbox_scores, bbox_classes=self.bbox_classes
        )

        self.assertIsNone(coder.class_score_thresholds)
        self.assertEqual(keep_masks.tolist(), [[True] * 6])

    def test_filter_bboxes_score_uses_the_group_of_each_class(self) -> None:
        """Test that each class is thresholded by its own group, unnamed classes by zero."""
        coder = self._build_coder(
            [
                ScoreThresholdConfig(class_names=["car"], score_threshold=0.3),
                ScoreThresholdConfig(class_names=["pedestrian"], score_threshold=0.0),
            ]
        )

        keep_masks = coder.filter_bboxes_score(
            bbox_scores=self.bbox_scores, bbox_classes=self.bbox_classes
        )

        # Car needs 0.3, pedestrian and the unnamed bicycle only a positive score.
        self.assertEqual(keep_masks.tolist(), [[False, True, True, True, False, True]])

    def test_rejects_a_class_in_two_groups(self) -> None:
        """Test that a class claimed by two groups is rejected as ambiguous, at construction."""
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            self._build_coder(
                [
                    ScoreThresholdConfig(class_names=["car"], score_threshold=0.1),
                    ScoreThresholdConfig(class_names=["car", "pedestrian"], score_threshold=0.5),
                ]
            )

    def test_rejects_an_unknown_class_name(self) -> None:
        """Test that a group naming a class outside class_names is rejected at construction."""
        with self.assertRaisesRegex(ValueError, "truck"):
            self._build_coder([ScoreThresholdConfig(class_names=["truck"], score_threshold=0.1)])


if __name__ == "__main__":
    unittest.main()
