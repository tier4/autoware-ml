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

"""Unit tests for the TransFusion box coder's height convention.

Ground-truth boxes in this framework are gravity center only (see ``Box3DCenterCoordinateType``),
so ``encode`` and ``decode_boxes`` must agree on what the height channel means: the value that
``encode`` writes for a gravity-center box has to come back out of ``decode_boxes`` as the same
gravity-center z, with no half-height offset in either direction.
"""

from __future__ import annotations

import unittest

import torch

from autoware_ml.models.detection3d.task_modules.bbox_coders import TransFusionBBoxCoder
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
            score_threshold_groups=None,
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


if __name__ == "__main__":
    unittest.main()
