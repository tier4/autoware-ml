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

"""Unit tests for the filter interface over points and boxes."""

import unittest

import numpy as np

from autoware_ml.metrics.base import IDENTITY, IdentityFilter, MetricSuite
from autoware_ml.metrics.filters import CollisionFilter, CorridorFilter, RegionFilter
from autoware_ml.metrics.tests.conftest import EGO


class ForgetfulFilter(IdentityFilter):
    """A filter that declares the frame metadata and never calls require_frame_meta."""

    name = "forgetful"
    needs_frame_meta = True


class TestMetricFilterInterface(unittest.TestCase):
    """Unit tests for keep_points, keep_boxes and the frame metadata requirement."""

    def setUp(self) -> None:
        """Set up two points and two boxes shared by the tests."""
        self.points = np.array([[10.0, 0.0, 0.0], [10.0, 5.0, 0.0]])
        self.boxes = np.array(
            [
                [10.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0],
                [10.0, 5.0, 0.0, 4.0, 2.0, 1.5, 0.0],
            ]
        )

    def test_identity_filter_keeps_every_point_and_box(self) -> None:
        """
        Input: two points and two boxes, no frame metadata.
        Expected: the identity filter keeps every element and needs no metadata.
        Check: both masks are all true and needs_frame_meta is false.
        """
        self.assertFalse(IDENTITY.needs_frame_meta)
        self.assertEqual(IDENTITY.keep_points(self.points, None).tolist(), [True, True])
        self.assertEqual(IDENTITY.keep_boxes(self.boxes, None).tolist(), [True, True])
        self.assertTrue(IDENTITY.available(None))

    def test_corridor_filter_works_without_frame_metadata(self) -> None:
        """
        Input: a 3 m corridor, one point and one box on the centerline, one of each 5 m aside.
        Expected: the strip lives in the ego frame, so no frame metadata is read.
        Check: only the centerline point and box are kept with frame_meta None.
        """
        corridor_filter = CorridorFilter(width_m=3.0)

        self.assertFalse(corridor_filter.needs_frame_meta)
        self.assertEqual(corridor_filter.keep_points(self.points, None).tolist(), [True, False])
        self.assertEqual(corridor_filter.keep_boxes(self.boxes, None).tolist(), [True, False])
        self.assertTrue(corridor_filter.available(None))

    def test_map_backed_filters_raise_without_frame_metadata(self) -> None:
        """
        Input: a region filter and a collision filter, a frame without metadata.
        Expected: both read the ego pose and the map path, so a missing metadata fails loud.
        Check: keep_points, keep_boxes and available raise a ValueError naming the filter.
        """
        for map_filter in (RegionFilter(["road"]), CollisionFilter(vehicle=EGO)):
            with self.subTest(filter=map_filter.name):
                self.assertTrue(map_filter.needs_frame_meta)
                with self.assertRaisesRegex(ValueError, map_filter.name):
                    map_filter.keep_points(self.points, None)
                with self.assertRaisesRegex(ValueError, map_filter.name):
                    map_filter.keep_boxes(self.boxes, None)
                with self.assertRaisesRegex(ValueError, map_filter.name):
                    map_filter.available(None)

    def test_suite_raises_for_filter_that_skips_the_metadata_check(self) -> None:
        """
        Input: a filter with needs_frame_meta set whose available never reads the metadata.
        Expected: the suite checks the metadata itself before asking the filter.
        Check: a frame without metadata raises a ValueError naming the filter,
            while a filter that needs none is reported available.
        """
        forgetful_filter = ForgetfulFilter()

        self.assertTrue(forgetful_filter.available(None))
        with self.assertRaisesRegex(ValueError, "forgetful"):
            MetricSuite._filter_available(forgetful_filter, None)
        self.assertTrue(MetricSuite._filter_available(IDENTITY, None))

    def test_empty_inputs_still_raise_without_frame_metadata(self) -> None:
        """
        Input: no points and no boxes, a region filter and a collision filter, a frame without metadata.
        Expected: missing metadata fails loud whatever the input size.
        Check: keep_points and keep_boxes raise a ValueError naming the filter.
        """
        for map_filter in (RegionFilter(["road"]), CollisionFilter(vehicle=EGO)):
            with self.subTest(filter=map_filter.name):
                with self.assertRaisesRegex(ValueError, map_filter.name):
                    map_filter.keep_points(np.zeros((0, 3)), None)
                with self.assertRaisesRegex(ValueError, map_filter.name):
                    map_filter.keep_boxes(np.zeros((0, 7)), None)


if __name__ == "__main__":
    unittest.main()
