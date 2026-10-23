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

"""Tests for the partial annotation masks."""

import unittest

import torch

from autoware_ml.models.detection3d.task_modules.partial_annotations import PartialAnnotations


class TestPartialAnnotations(unittest.TestCase):
    """Unit tests for PartialAnnotations."""

    def setUp(self) -> None:
        self.class_names = ["car", "pedestrian", "traffic_cone", "barrier"]
        self.partial_annotations = PartialAnnotations(
            class_names=self.class_names, partial_ignore_labels=["traffic_cone", "barrier"]
        )

    def test_resolves_labels_to_sorted_unique_class_ids(self) -> None:
        """Test that labels resolve to sorted, de-duplicated class ids."""
        partial_annotations = PartialAnnotations(
            class_names=self.class_names,
            partial_ignore_labels=["barrier", "traffic_cone", "barrier"],
        )

        self.assertEqual(list(partial_annotations.partial_ignore_class_ids), [2, 3])
        self.assertEqual(partial_annotations.num_classes, 4)

    def test_rejects_unknown_labels(self) -> None:
        """Test that a label outside class_names is rejected rather than silently ignored."""
        with self.assertRaisesRegex(ValueError, "helicopter"):
            PartialAnnotations(class_names=self.class_names, partial_ignore_labels=["helicopter"])

    def test_rejects_no_labels(self) -> None:
        """Test that an empty label list is rejected, since it would never ignore anything."""
        with self.assertRaises(ValueError):
            PartialAnnotations(class_names=self.class_names, partial_ignore_labels=[])

    def test_builds_masks_dropping_the_ignored_classes_of_unannotated_samples(self) -> None:
        """
        Test that the mask is True everywhere except the ignored classes of the samples flagged
        False, and that it lands on the flag's device.
        """
        status = torch.tensor([True, False, True])

        masks = self.partial_annotations.build_annotated_class_masks(status)

        expected = torch.tensor(
            [
                [True, True, True, True],
                [True, True, False, False],
                [True, True, True, True],
            ]
        )
        self.assertEqual(masks.dtype, torch.bool)
        self.assertEqual(masks.device, status.device)
        self.assertTrue(torch.equal(masks, expected))

    def test_raises_without_the_annotation_flag(self) -> None:
        """
        Test that a batch carrying no flag is rejected when partial ignoring is configured, since
        supervising every class would silently teach the head that the ignored ones are absent.
        """
        with self.assertRaisesRegex(ValueError, "gt_traffic_cone_barrier_bbox_status"):
            self.partial_annotations.build_annotated_class_masks(None)

    def test_rejects_a_flag_that_is_not_one_per_sample(self) -> None:
        """Test that a flag not shaped (batch_size,) is rejected."""
        with self.assertRaises(ValueError):
            self.partial_annotations.build_annotated_class_masks(torch.ones(2, 2, dtype=torch.bool))


if __name__ == "__main__":
    unittest.main()
