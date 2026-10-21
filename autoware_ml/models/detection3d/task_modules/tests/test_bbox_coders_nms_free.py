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

"""Unit tests for the NMS-free 3D box coder used by query-based detectors."""

from __future__ import annotations

import unittest

import torch

from autoware_ml.models.detection3d.task_modules.bbox_coders import (
    NMSFreeBBoxCoder3D,
    denormalize_boxes3d,
    normalize_boxes3d,
)


class TestBoxes3DNormalization(unittest.TestCase):
    """Check the metric-to-normalized box encoding shared by query-based coders."""

    def setUp(self) -> None:
        """Build two metric boxes as (cx, cy, cz, dx, dy, dz, yaw, vx, vy)."""
        self.boxes = torch.tensor(
            [
                [1.0, 2.0, -0.5, 4.0, 2.0, 1.5, 0.3, 1.0, -2.0],
                [-3.0, 0.5, 0.25, 0.8, 0.8, 1.7, -2.9, 0.0, 0.0],
            ]
        )

    def test_normalize_denormalize_round_trip(self) -> None:
        """Denormalizing a normalized box restores the metric box."""
        restored = denormalize_boxes3d(normalize_boxes3d(self.boxes))

        torch.testing.assert_close(restored, self.boxes, rtol=1e-5, atol=1e-6)

    def test_normalize_uses_log_sizes_and_split_yaw(self) -> None:
        """Sizes are stored in log space and yaw as a (sin, cos) pair."""
        encoded = normalize_boxes3d(self.boxes)

        self.assertEqual(encoded.shape, (2, 10))
        torch.testing.assert_close(encoded[:, 3:6], self.boxes[:, 3:6].log())
        torch.testing.assert_close(encoded[:, 6], self.boxes[:, 6].sin())
        torch.testing.assert_close(encoded[:, 7], self.boxes[:, 6].cos())


class TestNMSFreeBBoxCoder3D(unittest.TestCase):
    """Check the top-k decoding, filtering and batching of ``NMSFreeBBoxCoder3D``."""

    def setUp(self) -> None:
        """Build the scene range and the normalized version of two metric boxes."""
        self.pc_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
        # (cx, cy, cz, dx, dy, dz, yaw, vx, vy)
        self.boxes = torch.tensor(
            [
                [1.0, 2.0, -0.5, 4.0, 2.0, 1.5, 0.3, 1.0, -2.0],
                [-3.0, 0.5, 0.25, 0.8, 0.8, 1.7, -2.9, 0.0, 0.0],
            ]
        )
        self.encoded = normalize_boxes3d(self.boxes)

    def test_decode_maps_flat_topk_index_to_class_and_box(self) -> None:
        """A flat top-k index resolves to the right (query, class) pair."""
        coder = NMSFreeBBoxCoder3D(pc_range=self.pc_range, max_num=2)
        # Two queries, three classes; the strongest scores are query1/class2 then
        # query0/class1, so labels and boxes must follow that pairing.
        logits = torch.tensor([[-9.0, 1.0, -9.0], [-9.0, -9.0, 5.0]])

        predictions = coder.decode(logits.unsqueeze(0), self.encoded.unsqueeze(0))[0]

        self.assertEqual(predictions["labels"].tolist(), [2, 1])
        torch.testing.assert_close(predictions["scores"], logits.sigmoid().flatten().topk(2).values)
        torch.testing.assert_close(
            predictions["bboxes"],
            denormalize_boxes3d(self.encoded[[1, 0]]),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_decode_applies_score_threshold(self) -> None:
        """Predictions below the score threshold are dropped."""
        logits = torch.tensor([[2.0, -9.0, -9.0], [-9.0, -9.0, -1.0]])
        coder = NMSFreeBBoxCoder3D(pc_range=self.pc_range, max_num=6, score_threshold=0.5)

        thresholded = coder.decode(logits.unsqueeze(0), self.encoded.unsqueeze(0))[0]

        self.assertEqual(thresholded["labels"].numel(), 1)
        self.assertTrue((thresholded["scores"] >= 0.5).all())

    def test_decode_applies_post_center_range(self) -> None:
        """Boxes whose center leaves the post-center range are dropped with their score and label."""
        logits = torch.tensor([[2.0, -9.0, -9.0], [-9.0, -9.0, -1.0]])
        # Box 0 sits at x=1, box 1 at x=-3; a range that keeps only positive x
        # must drop box 1 together with its score and label.
        coder = NMSFreeBBoxCoder3D(
            pc_range=self.pc_range,
            post_center_range=[0.0, -50.0, -5.0, 50.0, 50.0, 3.0],
            max_num=6,
        )

        ranged = coder.decode(logits.unsqueeze(0), self.encoded.unsqueeze(0))[0]

        self.assertEqual(ranged["bboxes"].shape[0], ranged["scores"].numel())
        self.assertEqual(ranged["bboxes"].shape[0], ranged["labels"].numel())
        self.assertTrue((ranged["bboxes"][:, 0] >= 0.0).all())

    def test_decode_returns_one_entry_per_sample(self) -> None:
        """Decoding a batch yields one prediction dictionary per sample."""
        coder = NMSFreeBBoxCoder3D(pc_range=self.pc_range, max_num=1)
        encoded = self.encoded.unsqueeze(0).repeat(3, 1, 1)
        logits = torch.zeros(3, 2, 3)

        predictions = coder.decode(logits, encoded)

        self.assertEqual(len(predictions), 3)
        for prediction in predictions:
            self.assertEqual(prediction["bboxes"].shape[0], 1)


if __name__ == "__main__":
    unittest.main()
