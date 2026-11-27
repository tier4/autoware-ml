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
from unittest import mock

import numpy as np
from shapely.geometry import Polygon
import torch

from autoware_ml.metrics.base import IDENTITY, IdentityFilter, MetricSuite
from autoware_ml.metrics.filters import CollisionFilter, CorridorFilter, RegionFilter
from autoware_ml.metrics.geometry.lanelet import LaneletMap
from autoware_ml.metrics.segmentation3d.frame_count import FrameCount
from autoware_ml.metrics.segmentation3d.point_cloud import Segmentation3DPointCloudMetricSuite
from autoware_ml.metrics.segmentation3d.suite import Segmentation3DConfusionMatrixMetricSuite
from autoware_ml.metrics.tests.conftest import EGO
from autoware_ml.metrics.tests.eval_output_fixtures import (
    frame_meta,
    segmentation_eval_output,
    segmentation_frame,
)


class RecordingFilter(IdentityFilter):
    """A filter that keeps everything and remembers the points it was handed."""

    name = "recording"

    def __init__(self) -> None:
        """Start with nothing seen."""
        self.seen: list = []

    def keep_points(self, points, frame_meta):
        """Remember the points, then keep them all."""
        self.seen.append(points)
        return super().keep_points(points, frame_meta)


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


class TestKeepPointsOnTensors(unittest.TestCase):
    """Unit tests for keep_points given the points as a tensor."""

    def setUp(self) -> None:
        """Set up scattered points, an ego pose far from the map origin and a road around ego."""
        generator = np.random.default_rng(0)
        self.points = np.column_stack(
            [
                generator.uniform(-20.0, 80.0, 4000),
                generator.uniform(-30.0, 30.0, 4000),
                generator.uniform(-2.0, 2.0, 4000),
            ]
        ).astype(np.float32)
        # Map coordinates of this size are where float32 would lose the millimeters.
        yaw = 0.3
        self.pose = np.eye(4)
        self.pose[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
        self.pose[:2, 3] = [81234.5, 43210.25]
        road = np.array([[0.0, -10.0], [50.0, -10.0], [50.0, 10.0], [0.0, 10.0]])
        road_map = LaneletMap({"road": [Polygon(road @ self.pose[:2, :2].T + self.pose[:2, 3])]})
        loader = mock.patch.object(LaneletMap, "from_osm", return_value=road_map)
        loader.start()
        self.addCleanup(loader.stop)
        self.meta = frame_meta("road.osm", ego2global=torch.from_numpy(self.pose))
        self.filters = (
            IDENTITY,
            CorridorFilter(width_m=3.3),
            RegionFilter(["road"]),
            RegionFilter(["road"], margin=-1.0),
            RegionFilter(["road"], margin=1.0),
            CollisionFilter(vehicle=EGO),
        )

    def test_masks_a_tensor_as_it_masks_an_array(self) -> None:
        """Test that every filter masks points given as a tensor exactly as given as an array.

        Input: 4000 scattered points, once as an array and once as a tensor, every filter.
        Expected: the kind of the points decides the kind of the mask, never its values.
        Check: an array gives an array, a tensor a boolean tensor on its device, with equal
            values, and every filter but the identity keeps some points and drops others.
        """
        tensor = torch.from_numpy(self.points)
        for metric_filter in self.filters:
            with self.subTest(filter=metric_filter.name or "identity"):
                from_array = metric_filter.keep_points(self.points, self.meta)
                from_tensor = metric_filter.keep_points(tensor, self.meta)

                self.assertIsInstance(from_array, np.ndarray)
                self.assertIsInstance(from_tensor, torch.Tensor)
                self.assertEqual(from_tensor.dtype, torch.bool)
                self.assertEqual(from_tensor.device, tensor.device)
                self.assertEqual(from_tensor.tolist(), from_array.tolist())
                if metric_filter is not IDENTITY:
                    self.assertTrue(0 < int(from_array.sum()) < self.points.shape[0])

    def test_tensor_without_points_gives_an_empty_tensor_mask(self) -> None:
        """Test that every filter answers a tensor of no points with an empty tensor mask.

        Input: a tensor of no points, every filter.
        Expected: a frame without points is answered in kind.
        Check: every filter returns an empty boolean tensor.
        """
        for metric_filter in self.filters:
            with self.subTest(filter=metric_filter.name or "identity"):
                mask = metric_filter.keep_points(torch.zeros((0, 3)), self.meta)

                self.assertIsInstance(mask, torch.Tensor)
                self.assertEqual(mask.dtype, torch.bool)
                self.assertEqual(tuple(mask.shape), (0,))

    @unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
    def test_masks_a_cuda_tensor_on_its_device(self) -> None:
        """Test that every filter masks a CUDA tensor on its device, as it masks the array.

        Input: the points and the ego pose on a CUDA device, as a test run holds them.
        Expected: the mask stays on that device and equals the mask of the array.
        Check: every filter returns a tensor on the points' device with the array's values.
        """
        tensor = torch.from_numpy(self.points).cuda()
        meta = frame_meta("road.osm", ego2global=torch.from_numpy(self.pose).cuda())
        for metric_filter in self.filters:
            with self.subTest(filter=metric_filter.name or "identity"):
                mask = metric_filter.keep_points(tensor, meta)

                self.assertEqual(mask.device, tensor.device)
                self.assertEqual(
                    mask.tolist(), metric_filter.keep_points(self.points, self.meta).tolist()
                )

    def test_segmentation_suites_hand_the_points_over_as_a_tensor(self) -> None:
        """Test that both segmentation suites hand a filter the frame's points as a tensor.

        Input: one frame of two points, a recording filter, each segmentation suite.
        Expected: the suite gives the filter the frame's points without converting them.
        Check: the filter saw one (2, 3) tensor on the device of the frame's points.
        """
        coord = torch.tensor([[10.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
        suites = (Segmentation3DConfusionMatrixMetricSuite, Segmentation3DPointCloudMetricSuite)
        for suite_class in suites:
            with self.subTest(suite=suite_class.__name__):
                recording = RecordingFilter()
                suite = suite_class(
                    components=[FrameCount(stages=["test"], filter=recording)],
                    num_classes=2,
                    ranges=(),
                )
                suite.update(
                    segmentation_eval_output(
                        [
                            segmentation_frame(
                                pred=torch.tensor([0, 1]),
                                target=torch.tensor([0, 0]),
                                coord=coord,
                                scores=torch.eye(2),
                            )
                        ]
                    )
                )

                self.assertEqual(len(recording.seen), 1)
                self.assertIsInstance(recording.seen[0], torch.Tensor)
                self.assertEqual(tuple(recording.seen[0].shape), (2, 3))
                self.assertEqual(recording.seen[0].device, coord.device)


if __name__ == "__main__":
    unittest.main()
