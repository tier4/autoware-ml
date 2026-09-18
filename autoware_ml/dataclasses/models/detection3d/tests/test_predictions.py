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

"""Unit tests for ``Detection3DSamplePredictions``."""

from __future__ import annotations

import unittest

from pydantic import ValidationError
import torch

from autoware_ml.dataclasses.models.detection3d.predictions import Detection3DSamplePredictions


class TestDetection3DSamplePredictions(unittest.TestCase):
    """Unit tests for the per-sample decoded 3D detections."""

    def test_accepts_boxes_with_and_without_velocity(self) -> None:
        """Test that both the 7- and 9-parameter box layouts are accepted."""
        for num_bbox_params in (7, 9):
            with self.subTest(num_bbox_params=num_bbox_params):
                predictions = Detection3DSamplePredictions(
                    bboxes_3d=torch.zeros(3, num_bbox_params),
                    scores_3d=torch.zeros(3),
                    labels_3d=torch.zeros(3, dtype=torch.int64),
                )

                self.assertEqual(tuple(predictions.bboxes_3d.shape), (3, num_bbox_params))

    def test_accepts_empty_predictions(self) -> None:
        """Test that a sample with no boxes is representable."""
        predictions = Detection3DSamplePredictions(
            bboxes_3d=torch.zeros(0, 9),
            scores_3d=torch.zeros(0),
            labels_3d=torch.zeros(0, dtype=torch.int64),
        )

        self.assertEqual(predictions.scores_3d.shape[0], 0)

    def test_rejects_wrong_dtypes(self) -> None:
        """Test that boxes and scores must be float32 and labels int64."""
        valid = {
            "bboxes_3d": torch.zeros(2, 9),
            "scores_3d": torch.zeros(2),
            "labels_3d": torch.zeros(2, dtype=torch.int64),
        }
        for field, bad_value in (
            ("bboxes_3d", torch.zeros(2, 9, dtype=torch.float64)),
            ("scores_3d", torch.zeros(2, dtype=torch.float16)),
            ("labels_3d", torch.zeros(2, dtype=torch.int32)),
        ):
            with self.subTest(field=field):
                with self.assertRaises(ValidationError):
                    Detection3DSamplePredictions(**{**valid, field: bad_value})

    def test_rejects_wrong_ranks(self) -> None:
        """Test that boxes are 2-D and scores and labels are 1-D."""
        valid = {
            "bboxes_3d": torch.zeros(2, 9),
            "scores_3d": torch.zeros(2),
            "labels_3d": torch.zeros(2, dtype=torch.int64),
        }
        for field, bad_value in (
            ("bboxes_3d", torch.zeros(9)),
            ("scores_3d", torch.zeros(2, 1)),
            ("labels_3d", torch.zeros(2, 1, dtype=torch.int64)),
        ):
            with self.subTest(field=field):
                with self.assertRaises(ValidationError):
                    Detection3DSamplePredictions(**{**valid, field: bad_value})

    def test_is_frozen(self) -> None:
        """Test that decoded predictions cannot be mutated after construction."""
        predictions = Detection3DSamplePredictions(
            bboxes_3d=torch.zeros(1, 9),
            scores_3d=torch.zeros(1),
            labels_3d=torch.zeros(1, dtype=torch.int64),
        )

        with self.assertRaises(ValidationError):
            predictions.scores_3d = torch.ones(1)  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
