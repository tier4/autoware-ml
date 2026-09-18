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

"""Unit tests for ``ModelPredictions`` and ``ModelOutputs``."""

from __future__ import annotations

import unittest

from pydantic import ValidationError
import torch

from autoware_ml.dataclasses.models.detection3d.head_outputs import (
    CenterHeadOutputs,
    Detection3DHeadOutputs,
)
from autoware_ml.dataclasses.models.detection3d.predictions import Detection3DSamplePredictions
from autoware_ml.dataclasses.models.model_outputs import ModelOutputs
from autoware_ml.dataclasses.models.model_predictions import ModelPredictions


class TestModelPredictions(unittest.TestCase):
    """Unit tests for the decoded multi-task predictions container."""

    def _build_sample_predictions(self, num_boxes: int) -> Detection3DSamplePredictions:
        """Build ``num_boxes`` decoded boxes with velocity, distinct per box."""
        return Detection3DSamplePredictions(
            # Box i holds the value i in every parameter, so boxes stay distinguishable.
            bboxes_3d=torch.arange(num_boxes, dtype=torch.float32)
            .unsqueeze(1)
            .expand(num_boxes, 9)
            .contiguous(),
            scores_3d=torch.linspace(1.0, 0.0, num_boxes, dtype=torch.float32),
            labels_3d=torch.arange(num_boxes, dtype=torch.int64),
        )

    def test_to_list_emits_one_dict_per_sample_in_batch_order(self) -> None:
        """Test that the legacy list layout keeps sample order and hands over the same tensors."""
        samples = [self._build_sample_predictions(2), self._build_sample_predictions(1)]
        predictions = ModelPredictions(detection3d_predictions=samples)

        predictions_list = predictions.to_list()

        self.assertEqual(len(predictions_list), 2)
        for sample, entry in zip(samples, predictions_list):
            self.assertEqual(set(entry), {"bboxes_3d", "scores_3d", "labels_3d"})
            self.assertIs(entry["bboxes_3d"], sample.bboxes_3d)
            self.assertIs(entry["scores_3d"], sample.scores_3d)
            self.assertIs(entry["labels_3d"], sample.labels_3d)

    def test_to_list_without_detection_predictions_is_empty(self) -> None:
        """Test that a container without 3D detections converts to an empty list."""
        self.assertEqual(ModelPredictions(detection3d_predictions=None).to_list(), [])

    def test_to_list_keeps_empty_samples(self) -> None:
        """Test that a sample with no surviving boxes still occupies its slot."""
        predictions = ModelPredictions(
            detection3d_predictions=[
                self._build_sample_predictions(0),
                self._build_sample_predictions(3),
            ]
        )

        predictions_list = predictions.to_list()

        self.assertEqual(len(predictions_list), 2)
        self.assertEqual(tuple(predictions_list[0]["bboxes_3d"].shape), (0, 9))
        self.assertEqual(tuple(predictions_list[1]["bboxes_3d"].shape), (3, 9))

    def test_rejects_plain_dicts_as_sample_predictions(self) -> None:
        """Test that the legacy dict layout is not accepted as input, only produced as output."""
        with self.assertRaises(ValidationError):
            ModelPredictions(
                detection3d_predictions=[{"bboxes_3d": torch.zeros(1, 9)}]  # type: ignore[list-item]
            )

    def test_is_frozen(self) -> None:
        """Test that the container cannot be mutated after construction."""
        predictions = ModelPredictions(detection3d_predictions=None)

        with self.assertRaises(ValidationError):
            predictions.detection3d_predictions = []  # type: ignore[misc]


class TestModelOutputs(unittest.TestCase):
    """Unit tests for the raw multi-task head outputs container."""

    def _build_center_head_outputs(self) -> CenterHeadOutputs:
        """Build all-zero CenterHead outputs on a 4x4 grid with two classes."""
        return CenterHeadOutputs(
            heatmaps=torch.zeros(1, 2, 4, 4),
            centers=torch.zeros(1, 2, 4, 4),
            heights=torch.zeros(1, 1, 4, 4),
            dims=torch.zeros(1, 3, 4, 4),
            rots=torch.zeros(1, 2, 4, 4),
            vels=None,
        )

    def test_holds_detection3d_head_outputs(self) -> None:
        """Test that the detection3d head outputs are carried through untouched."""
        head_outputs = Detection3DHeadOutputs(
            center_head_outputs=self._build_center_head_outputs(), transfusion_head_outputs=None
        )

        outputs = ModelOutputs(detection3d_head_outputs=head_outputs)

        self.assertIs(outputs.detection3d_head_outputs, head_outputs)

    def test_allows_no_detection3d_head_outputs(self) -> None:
        """Test that a model without a 3D detection head produces an empty container."""
        self.assertIsNone(ModelOutputs(detection3d_head_outputs=None).detection3d_head_outputs)

    def test_rejects_raw_head_outputs(self) -> None:
        """Test that a head-specific output object must be wrapped in ``Detection3DHeadOutputs``."""
        with self.assertRaises(ValidationError):
            ModelOutputs(
                detection3d_head_outputs=self._build_center_head_outputs()  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
