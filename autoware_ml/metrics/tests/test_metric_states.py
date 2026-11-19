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

"""Unit tests for the typed states the suites hand to the metrics."""

import unittest

import numpy as np
import torch
from pydantic import ValidationError

from autoware_ml.metrics.detection3d.structures import Detection3DSample, MatchCurve
from autoware_ml.metrics.segmentation3d.confusion import ConfusionState
from autoware_ml.metrics.segmentation3d.point_cloud import FramePoints


class TestDetection3DSample(unittest.TestCase):
    """Unit tests for the shape and dtype contract of one detection frame."""

    def setUp(self) -> None:
        """Set up the fields of a frame with two predictions and one ground truth box."""
        self.fields = {
            "pred_boxes": torch.zeros((2, 9), dtype=torch.float32),
            "pred_scores": torch.tensor([0.9, 0.4], dtype=torch.float32),
            "pred_labels": torch.tensor([0, 1], dtype=torch.long),
            "gt_boxes": torch.zeros((1, 9), dtype=torch.float32),
            "gt_labels": torch.tensor([0], dtype=torch.long),
        }

    def test_valid_frame_keeps_the_tensors_it_was_given(self) -> None:
        """
        Input: two predictions and one ground truth box, no time-to-collision.
        Expected: the sample holds the very tensors it was built from, the time-to-collision
        stays unset and the frame counts as covered.
        Check: identity of the box tensors and the defaults of the optional fields.
        """
        sample = Detection3DSample.model_validate(self.fields)
        self.assertIs(sample.pred_boxes, self.fields["pred_boxes"])
        self.assertIs(sample.gt_boxes, self.fields["gt_boxes"])
        self.assertIsNone(sample.gt_ttc)
        self.assertIsNone(sample.pred_ttc)
        self.assertTrue(sample.ttc_covered)

    def test_frame_without_boxes_is_valid(self) -> None:
        """
        Input: a frame with no prediction and no ground truth box.
        Expected: empty tensors of the right rank describe the frame.
        Check: the sample builds and counts zero boxes on both sides.
        """
        sample = Detection3DSample(
            pred_boxes=torch.zeros((0, 7), dtype=torch.float32),
            pred_scores=torch.zeros(0, dtype=torch.float32),
            pred_labels=torch.zeros(0, dtype=torch.long),
            gt_boxes=torch.zeros((0, 7), dtype=torch.float32),
            gt_labels=torch.zeros(0, dtype=torch.long),
        )
        self.assertEqual(sample.pred_boxes.shape[0], 0)
        self.assertEqual(sample.gt_boxes.shape[0], 0)

    def test_wrong_dtype_or_rank_is_rejected(self) -> None:
        """
        Input: the valid frame with one field replaced by a tensor of the wrong dtype or rank.
        Expected: float64 boxes, int32 labels, integer scores and a flat box tensor all fail.
        Check: pydantic raises for every replaced field.
        """
        wrong_fields = {
            "pred_boxes": torch.zeros((2, 9), dtype=torch.float64),
            "pred_labels": torch.tensor([0, 1], dtype=torch.int32),
            "pred_scores": torch.tensor([1, 0], dtype=torch.long),
            "gt_boxes": torch.zeros(9, dtype=torch.float32),
        }
        for name, value in wrong_fields.items():
            with self.subTest(field=name):
                fields = dict(self.fields)
                fields[name] = value
                with self.assertRaises(ValidationError):
                    Detection3DSample.model_validate(fields)

    def test_per_box_tensor_of_another_length_is_rejected(self) -> None:
        """
        Input: the valid frame with one per-box tensor describing a different number of boxes.
        Expected: scores, labels and time-to-collision must describe the boxes of their side.
        Check: pydantic raises and the message names the offending field.
        """
        wrong_fields = {
            "pred_scores": torch.zeros(3, dtype=torch.float32),
            "pred_labels": torch.zeros(1, dtype=torch.long),
            "gt_labels": torch.zeros(2, dtype=torch.long),
            "pred_ttc": torch.zeros(1, dtype=torch.float32),
            "gt_ttc": torch.zeros(2, dtype=torch.float32),
        }
        for name, value in wrong_fields.items():
            with self.subTest(field=name):
                fields = dict(self.fields)
                fields[name] = value
                with self.assertRaisesRegex(ValidationError, name):
                    Detection3DSample.model_validate(fields)

    def test_boxes_with_fewer_than_seven_parameters_are_rejected(self) -> None:
        """
        Input: the valid frame with six parameters per predicted or ground truth box.
        Expected: the matching reads the yaw at column six, so narrower boxes fail at
        construction instead of inside the matching.
        Check: pydantic raises and the message names the side.
        """
        for name, rows in (("pred_boxes", 2), ("gt_boxes", 1)):
            with self.subTest(field=name):
                fields = dict(self.fields)
                fields[name] = torch.zeros((rows, 6), dtype=torch.float32)
                with self.assertRaisesRegex(ValidationError, name):
                    Detection3DSample.model_validate(fields)

    def test_sample_is_frozen(self) -> None:
        """
        Input: a built sample.
        Expected: a field cannot be reassigned after construction.
        Check: pydantic raises on the assignment.
        """
        sample = Detection3DSample.model_validate(self.fields)
        with self.assertRaises(ValidationError):
            sample.ttc_covered = False  # type: ignore[misc]


class TestMatchCurve(unittest.TestCase):
    """Unit tests for the per-prediction contract of a match curve."""

    def setUp(self) -> None:
        """Set up the fields of a curve with one match followed by one false positive."""
        nan = float("nan")
        self.fields = {
            "total_gt": 1,
            "scores": np.array([0.9, 0.4], dtype=np.float64),
            "true_positive": np.array([True, False]),
            "false_positive": np.array([False, True]),
            "heading_score": np.array([0.5, 0.0], dtype=np.float64),
            "translation_error": np.array([0.1, nan], dtype=np.float64),
            "orientation_error": np.array([0.2, nan], dtype=np.float64),
            "scale_error": np.array([0.3, nan], dtype=np.float64),
            "velocity_error": np.array([0.4, nan], dtype=np.float64),
            "attribute_error": np.array([1.0, nan], dtype=np.float64),
            "corner_error": np.array([0.5, nan], dtype=np.float64),
            "nearest_surface_error": np.array([-0.1, nan], dtype=np.float64),
        }

    def test_counts_and_cumulative_curves(self) -> None:
        """
        Input: a curve with one match followed by one false positive.
        Expected: two predictions, one match, and cumulative counts in score order.
        Check: the counts and the three cumulative arrays.
        """
        curve = MatchCurve.model_validate(self.fields)
        self.assertEqual(curve.num_predictions, 2)
        self.assertEqual(curve.num_match, 1)
        self.assertEqual(curve.cumulative_tp.tolist(), [1, 1])
        self.assertEqual(curve.cumulative_fp.tolist(), [0, 1])
        self.assertEqual(curve.cumulative_heading_tp.tolist(), [0.5, 0.5])

    def test_array_of_another_length_is_rejected(self) -> None:
        """
        Input: the valid curve with one array holding three entries instead of two.
        Expected: every array describes the same predictions, so the odd one fails.
        Check: pydantic raises and the message names the offending field.
        """
        wrong_fields = {
            "true_positive": np.array([True, False, False]),
            "translation_error": np.zeros(3, dtype=np.float64),
            "nearest_surface_error": np.zeros(3, dtype=np.float64),
        }
        for name, value in wrong_fields.items():
            with self.subTest(field=name):
                fields = dict(self.fields)
                fields[name] = value
                with self.assertRaisesRegex(ValidationError, name):
                    MatchCurve.model_validate(fields)

    def test_wrong_dtype_is_rejected(self) -> None:
        """
        Input: the valid curve with float32 scores, and with an integer match flag.
        Expected: the channels are float64 and the flags boolean.
        Check: pydantic raises for both.
        """
        wrong_fields = {
            "scores": np.array([0.9, 0.4], dtype=np.float32),
            "true_positive": np.array([1, 0], dtype=np.int64),
        }
        for name, value in wrong_fields.items():
            with self.subTest(field=name):
                fields = dict(self.fields)
                fields[name] = value
                with self.assertRaises(ValidationError):
                    MatchCurve.model_validate(fields)


class TestConfusionState(unittest.TestCase):
    """Unit tests for the matrix contract and the cached marginals of a confusion state."""

    def setUp(self) -> None:
        """Set up the fields of a two class matrix: rows are ground truth, columns predictions."""
        self.fields = {
            "confusion": torch.tensor([[5, 1], [2, 7]], dtype=torch.long),
            "class_names": ("road", "obstacle"),
            "num_classes": 2,
            "num_frames": 3,
        }

    def test_marginals_read_rows_as_ground_truth(self) -> None:
        """
        Input: the matrix [[5, 1], [2, 7]].
        Expected: the diagonal is the true positives, the row sums the ground truth counts and
        the column sums the predicted counts, cached on the frozen state.
        Check: every marginal, and that a second read returns the cached tensor.
        """
        state = ConfusionState.model_validate(self.fields)
        self.assertEqual(state.true_positive.tolist(), [5.0, 7.0])
        self.assertEqual(state.actual.tolist(), [6.0, 9.0])
        self.assertEqual(state.predicted.tolist(), [7.0, 8.0])
        self.assertEqual(float(state.total), 15.0)
        self.assertEqual(state.has_support.tolist(), [True, True])
        self.assertEqual(state.frequency.tolist(), [0.4, 0.6])
        self.assertIs(state.actual, state.actual)

    def test_matrix_that_is_not_square_or_not_integer_is_rejected(self) -> None:
        """
        Input: a 2 by 3 matrix, a flat vector and a float matrix.
        Expected: the state holds a square matrix of integer point counts.
        Check: pydantic raises for each.
        """
        wrong_matrices = {
            "not_square": torch.zeros((2, 3), dtype=torch.long),
            "flat": torch.zeros(4, dtype=torch.long),
            "float": torch.zeros((2, 2), dtype=torch.float32),
        }
        for name, confusion in wrong_matrices.items():
            with self.subTest(matrix=name):
                fields = dict(self.fields)
                fields["confusion"] = confusion
                with self.assertRaises(ValidationError):
                    ConfusionState.model_validate(fields)

    def test_num_classes_must_match_the_matrix(self) -> None:
        """
        Input: a two class matrix declared as three classes.
        Expected: the metrics loop over num_classes, so a mismatch fails at construction.
        Check: pydantic raises and the message names num_classes.
        """
        fields = dict(self.fields)
        fields["num_classes"] = 3
        with self.assertRaisesRegex(ValidationError, "num_classes"):
            ConfusionState.model_validate(fields)


class TestFramePoints(unittest.TestCase):
    """Unit tests for the per-point contract of one cached segmentation frame."""

    def setUp(self) -> None:
        """Set up the fields of a frame with three points and one ground truth box."""
        self.fields = {
            "coord": np.zeros((3, 3), dtype=np.float32),
            "pred": np.array([0, 1, 1], dtype=np.int16),
            "target": np.array([0, 1, -1], dtype=np.int16),
            "confidence": np.array([0.9, 0.8, 0.7], dtype=np.float32),
            "entropy": np.array([0.1, 0.2, 0.3], dtype=np.float32),
            "gt_boxes": np.zeros((1, 7), dtype=np.float32),
            "gt_box_labels": np.array([0], dtype=np.int64),
        }

    def test_read_only_arrays_are_shared_not_copied(self) -> None:
        """
        Input: the valid frame with every array frozen, as the suite hands the whole-scene slice.
        Expected: the frame holds the very arrays of the cache, still read only.
        Check: identity of the coordinates and that a write into them raises.
        """
        for array in self.fields.values():
            array.flags.writeable = False
        frame = FramePoints.model_validate(self.fields)
        self.assertIs(frame.coord, self.fields["coord"])
        with self.assertRaises(ValueError):
            frame.coord[0, 0] = 1.0

    def test_per_point_array_of_another_length_is_rejected(self) -> None:
        """
        Input: the valid frame with one per-point array holding two entries for three points.
        Expected: every per-point array describes the same points.
        Check: pydantic raises and the message names the offending field.
        """
        wrong_fields = {
            "pred": np.zeros(2, dtype=np.int16),
            "target": np.zeros(2, dtype=np.int16),
            "confidence": np.zeros(2, dtype=np.float32),
            "entropy": np.zeros(2, dtype=np.float32),
        }
        for name, value in wrong_fields.items():
            with self.subTest(field=name):
                fields = dict(self.fields)
                fields[name] = value
                with self.assertRaisesRegex(ValidationError, name):
                    FramePoints.model_validate(fields)

    def test_boxes_and_their_labels_must_agree(self) -> None:
        """
        Input: one ground truth box with two labels, and one box with six parameters.
        Expected: the labels describe the boxes, and a box carries at least seven parameters.
        Check: pydantic raises and the message names the offending field.
        """
        wrong_fields = {
            "gt_box_labels": np.zeros(2, dtype=np.int64),
            "gt_boxes": np.zeros((1, 6), dtype=np.float32),
        }
        for name, value in wrong_fields.items():
            with self.subTest(field=name):
                fields = dict(self.fields)
                fields[name] = value
                with self.assertRaisesRegex(ValidationError, name):
                    FramePoints.model_validate(fields)

    def test_wrong_dtype_or_rank_is_rejected(self) -> None:
        """
        Input: the valid frame with float labels, integer coordinates and flat coordinates.
        Expected: labels are integers and coordinates a float array with one row per point.
        Check: pydantic raises for each.
        """
        wrong_fields = {
            "pred": np.zeros(3, dtype=np.float32),
            "coord": np.zeros((3, 3), dtype=np.int64),
            "entropy": np.zeros((3, 1), dtype=np.float32),
        }
        for name, value in wrong_fields.items():
            with self.subTest(field=name):
                fields = dict(self.fields)
                fields[name] = value
                with self.assertRaises(ValidationError):
                    FramePoints.model_validate(fields)
