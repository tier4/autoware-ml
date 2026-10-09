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

"""Unit tests for the nearest point selection of the range view interpolation."""

from __future__ import annotations

import unittest

import torch

from autoware_ml.dataclasses.batch.segmentation3d import Segmentation3DGTBatch
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.models.tests.batch_inputs_fixtures import (
    build_batch_inputs,
    build_point_cloud_batch,
)
from autoware_ml.preprocessing.segmentation3d.range_view import (
    FrustumMix,
    InstanceCopy,
    RangeInterpolation,
    nearest_point_per_pixel,
)


def reference_nearest(pixels: torch.Tensor, depths: torch.Tensor) -> dict[int, int]:
    """Nearest point of every pixel, found one point at a time."""
    best: dict[int, int] = {}
    for index, (pixel, depth) in enumerate(zip(pixels.tolist(), depths.tolist())):
        if pixel not in best or depth < depths[best[pixel]]:
            best[pixel] = index
    return best


class TestNearestPointPerPixel(unittest.TestCase):
    """Every pixel is represented by its nearest point, on every device."""

    def devices(self) -> list[torch.device]:
        """CPU, and the GPU when there is one."""
        return [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else [])

    def test_matches_a_point_by_point_reference_under_heavy_collisions(self) -> None:
        """
        Input: 20000 points falling into 50 pixels at random depths.
        Expected: the selected point of each pixel is the one a point by point search finds.
        Check: pixel to point mapping against the reference, on CPU and GPU.
        """
        generator = torch.Generator().manual_seed(0)
        pixels = torch.randint(0, 50, (20000,), generator=generator)
        depths = torch.rand(20000, generator=generator) * 100.0
        expected = reference_nearest(pixels, depths)
        for device in self.devices():
            with self.subTest(device=str(device)):
                nearest = nearest_point_per_pixel(pixels.to(device), depths.to(device)).cpu()
                self.assertEqual({int(pixels[i]): int(i) for i in nearest}, expected)

    def test_interpolation_reads_the_nearest_point_of_each_neighbor(self) -> None:
        """
        Input: pixels 3 and 5 of one row, each hit by a near and a far point with different labels,
        and an empty pixel between them.
        Expected: the empty pixel takes the midpoint of the two near points and their label.
        Check: the interpolated point and label.
        """
        layer = RangeInterpolation(height=1, width=8, fov_up=10.0, fov_down=-10.0, ignore_index=-1)
        near_left = [10.0, 10.0, 0.0, 0.5]
        far_left = [20.0, 20.0, 0.0, 0.9]
        near_right = [10.0, -10.0, 0.0, 0.5]
        far_right = [20.0, -20.0, 0.0, 0.9]
        points = torch.tensor([far_left, near_left, far_right, near_right])
        labels = torch.tensor([7, 1, 7, 1])

        new_points, new_labels = layer.interpolate(points, labels)

        self.assertEqual(new_points.shape[0], 1)
        torch.testing.assert_close(new_points[0], torch.tensor([10.0, 0.0, 0.0, 0.5]))
        assert new_labels is not None
        self.assertEqual(new_labels.tolist(), [1])


class TestRangeViewBatchLayers(unittest.TestCase):
    """The batch layers rebuild the typed point and label batches they change."""

    def _inputs(self) -> ModelBatchInputs:
        left = [10.0, 10.0, 0.0, 0.5]
        right = [10.0, -10.0, 0.0, 0.5]
        points = torch.tensor([left, right])
        point_cloud = build_point_cloud_batch([points, points])
        labels = Segmentation3DGTBatch(
            gt_semantic_masks=torch.tensor([1, 1, 2, 2]), batch_indices=point_cloud.batch_indices
        )
        return build_batch_inputs(point_cloud=point_cloud, segmentation=labels)

    def test_interpolation_keeps_points_and_labels_aligned_per_sample(self) -> None:
        """
        Input: two samples, each with two points around one empty range image pixel.
        Expected: every sample gains one interpolated point and its label.
        Check: the sample of every point and label and the appended labels.
        """
        layer = RangeInterpolation(height=1, width=8, fov_up=10.0, fov_down=-10.0, ignore_index=-1)

        outputs = layer(self._inputs(), is_training=False).multi_task_gt_batch

        assert outputs.point_cloud_gt_batch is not None
        assert outputs.segmentation3d_gt_batch is not None
        self.assertEqual(outputs.point_cloud_gt_batch.batch_indices.tolist(), [0, 0, 0, 1, 1, 1])
        self.assertEqual(outputs.segmentation3d_gt_batch.batch_indices.tolist(), [0, 0, 0, 1, 1, 1])
        self.assertEqual(
            outputs.segmentation3d_gt_batch.gt_semantic_masks.tolist(), [1, 1, 1, 2, 2, 2]
        )

    def test_mixing_layers_leave_an_evaluation_batch_untouched(self) -> None:
        inputs = self._inputs()
        mix = FrustumMix(height=1, width=8, fov_up=10.0, fov_down=-10.0, num_areas=[2])

        self.assertIs(mix(inputs, is_training=False), inputs)
        self.assertIs(InstanceCopy(instance_classes=[1])(inputs, is_training=False), inputs)


if __name__ == "__main__":
    unittest.main()
