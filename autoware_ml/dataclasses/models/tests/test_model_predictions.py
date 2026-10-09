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
from autoware_ml.dataclasses.models.calibration_status.head_outputs import (
    CalibrationStatusHeadOutputs,
)
from autoware_ml.dataclasses.models.model_outputs import ModelOutputs
from autoware_ml.dataclasses.models.model_predictions import ModelPredictions
from autoware_ml.dataclasses.models.segmentation3d.head_outputs import Segmentation3DHeadOutputs
from autoware_ml.dataclasses.models.segmentation3d.predictions import Segmentation3DPredictions


class TestModelPredictions(unittest.TestCase):
    """Unit tests for the decoded predictions container."""

    def test_rejects_plain_dicts_as_sample_predictions(self) -> None:
        """Test that plain dicts are not accepted as sample predictions."""
        with self.assertRaises(ValidationError):
            ModelPredictions(
                detection3d_predictions=[{"bboxes_3d": torch.zeros(1, 9)}]  # type: ignore[list-item]
            )

    def test_single_task_model_leaves_the_other_tasks_unset(self) -> None:
        """Test that a segmentation model fills its predictions only."""
        segmentation = Segmentation3DPredictions(
            pred_labels=torch.zeros(3, dtype=torch.int64), pred_probs=torch.ones(3, 2)
        )

        predictions = ModelPredictions(segmentation3d_predictions=segmentation)

        self.assertIs(predictions.segmentation3d_predictions, segmentation)
        self.assertIsNone(predictions.detection3d_predictions)
        self.assertIsNone(predictions.calibration_status_predictions)

    def test_segmentation_labels_and_probabilities_describe_the_same_points(self) -> None:
        """Test that labels and probabilities of different point counts are rejected."""
        with self.assertRaises(ValidationError):
            Segmentation3DPredictions(
                pred_labels=torch.zeros(3, dtype=torch.int64), pred_probs=torch.ones(4, 2)
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
            heatmap=torch.zeros(1, 2, 4, 4),
            reg=torch.zeros(1, 2, 4, 4),
            height=torch.zeros(1, 1, 4, 4),
            dim=torch.zeros(1, 3, 4, 4),
            rot=torch.zeros(1, 2, 4, 4),
            vel=None,
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


class TestModelOutputAccessors(unittest.TestCase):
    """The task accessors return the outputs a model filled and reject the others."""

    def test_accessors_read_the_filled_task_and_reject_the_rest(self) -> None:
        segmentation = Segmentation3DHeadOutputs(logits=torch.zeros(4, 3))
        calibration = CalibrationStatusHeadOutputs(logits=torch.zeros(2, 2))

        segmentation_outputs = ModelOutputs(segmentation3d_head_outputs=segmentation)
        calibration_outputs = ModelOutputs(calibration_status_head_outputs=calibration)

        self.assertIs(segmentation_outputs.segmentation3d(), segmentation)
        self.assertEqual(segmentation_outputs.segmentation3d().auxiliary_features, ())
        self.assertIs(calibration_outputs.calibration_status(), calibration)
        with self.assertRaises(ValueError):
            segmentation_outputs.detection3d()
        with self.assertRaises(ValueError):
            calibration_outputs.segmentation3d()
        with self.assertRaises(ValueError):
            segmentation_outputs.calibration_status()

    def test_detection_accessors_follow_the_head_family(self) -> None:
        center = CenterHeadOutputs(
            heatmap=torch.zeros(1, 2, 4, 4),
            reg=torch.zeros(1, 2, 4, 4),
            height=torch.zeros(1, 1, 4, 4),
            dim=torch.zeros(1, 3, 4, 4),
            rot=torch.zeros(1, 2, 4, 4),
            vel=None,
        )
        outputs = Detection3DHeadOutputs(center_head_outputs=center, transfusion_head_outputs=None)

        self.assertIs(outputs.center_head(), center)
        with self.assertRaises(ValueError):
            outputs.transfusion_head()


if __name__ == "__main__":
    unittest.main()
