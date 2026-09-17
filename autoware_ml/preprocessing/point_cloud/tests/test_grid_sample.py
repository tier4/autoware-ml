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

"""Unit tests for the runtime grid subsampling of a batched point cloud."""

from __future__ import annotations

from collections.abc import Sequence
import unittest

import numpy as np
import torch

from autoware_ml.preprocessing.point_cloud.grid_sample import GridSamplePreprocessor

POINT_CLOUD_RANGE = (-10.0, -10.0, -10.0, 10.0, 10.0, 10.0)


def build_inputs(coords: Sequence[Sequence[float]], batch_indices: Sequence[int]) -> dict:
    """Build the model inputs the grid subsampling reads."""
    coord = torch.tensor(coords, dtype=torch.float32)
    return {
        "coord": coord,
        "feat": coord,
        "batch_indices": torch.tensor(batch_indices, dtype=torch.int32),
        "offset": torch.cumsum(torch.bincount(torch.tensor(batch_indices)), dim=0),
        "sample_count": len(set(batch_indices)),
    }


class TestGridSamplePreprocessor(unittest.TestCase):
    """One representative point per occupied voxel of every sample."""

    def setUp(self) -> None:
        """Voxelize at one meter, so a coordinate names its own voxel."""
        self.preprocessor = GridSamplePreprocessor(
            grid_size=1.0, point_cloud_range=POINT_CLOUD_RANGE
        )

    def test_keeps_one_point_per_voxel(self) -> None:
        # Two points fall into the voxel at the origin, one into the voxel next to it
        inputs = build_inputs([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [1.5, 0.0, 0.0]], [0, 0, 0])

        outputs = self.preprocessor(inputs, is_training=False)

        self.assertEqual(outputs["coord"].shape[0], 2)
        self.assertEqual(outputs["grid_coord"].tolist(), [[10, 10, 10], [11, 10, 10]])

    def test_separates_the_same_voxel_of_two_samples(self) -> None:
        inputs = build_inputs([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]], [0, 1])

        outputs = self.preprocessor(inputs, is_training=False)

        self.assertEqual(outputs["coord"].shape[0], 2)
        self.assertEqual(outputs["batch_indices"].tolist(), [0, 1])
        self.assertEqual(outputs["offset"].tolist(), [1, 2])

    def test_inverse_scatters_the_voxels_back_onto_the_points(self) -> None:
        inputs = build_inputs([[0.1, 0.0, 0.0], [1.5, 0.0, 0.0], [0.2, 0.0, 0.0]], [0, 0, 0])

        outputs = self.preprocessor(inputs, is_training=False)

        inverse = outputs["inverse"]
        scattered = outputs["coord"][inverse]
        self.assertEqual(inverse[0].item(), inverse[2].item())
        self.assertNotEqual(inverse[0].item(), inverse[1].item())
        self.assertEqual(scattered.shape[0], 3)

    def test_evaluation_picks_the_first_point_of_a_voxel(self) -> None:
        inputs = build_inputs([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]], [0, 0])

        outputs = self.preprocessor(inputs, is_training=False)

        torch.testing.assert_close(outputs["coord"], torch.tensor([[0.1, 0.0, 0.0]]))

    def test_training_picks_a_point_of_the_voxel(self) -> None:
        candidates = [[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]
        allowed = set(torch.tensor(candidates, dtype=torch.float32)[:, 0].tolist())

        picked = {
            float(
                self.preprocessor(build_inputs(candidates, [0, 0]), is_training=True)["coord"][0, 0]
            )
            for _ in range(64)
        }

        self.assertTrue(picked <= allowed)
        self.assertEqual(
            self.preprocessor(build_inputs(candidates, [0, 0]), is_training=True)["coord"].shape[0],
            1,
        )

    def test_segment_follows_the_representatives(self) -> None:
        inputs = build_inputs([[0.1, 0.0, 0.0], [1.5, 0.0, 0.0]], [0, 0])
        inputs["segment"] = torch.tensor([3, 7], dtype=torch.int64)

        outputs = self.preprocessor(inputs, is_training=False)

        self.assertEqual(outputs["segment"].tolist(), [3, 7])
        self.assertEqual(outputs["origin_segment"].tolist(), [3, 7])


@unittest.skipUnless(torch.cuda.is_available(), "exact division on the device requires CUDA")
class TestGridSampleVoxelCoordsOnDevice(unittest.TestCase):
    """The device voxel coordinates match an exact division up to the last cell."""

    def test_points_below_the_maximum_stay_inside_the_grid(self) -> None:
        grid_size = 0.12
        point_cloud_range = (-122.88, -122.88, -3.0, 122.88, 122.88, 5.0)
        preprocessor = GridSamplePreprocessor(
            grid_size=grid_size, point_cloud_range=point_cloud_range
        )
        minimum = np.array(point_cloud_range[:3], dtype=np.float32)
        maximum = np.array(point_cloud_range[3:], dtype=np.float32)
        # Dense coordinates plus the largest values below the maximum of every axis
        dense = np.random.default_rng(0).uniform(minimum, maximum, size=(1_000_000, 3))
        below_maximum = np.nextafter(maximum, minimum)[None, :].repeat(4, axis=0)
        coord = np.concatenate([dense.astype(np.float32), below_maximum])

        grid_coord = preprocessor.voxel_coords(torch.from_numpy(coord).cuda()).cpu().numpy()

        divisor = np.float32(grid_size)
        expected = np.floor(coord / divisor) - np.floor(minimum / divisor)
        np.testing.assert_array_equal(grid_coord, expected.astype(np.int64))
        # The BEV axes span a whole number of voxels, the last one below the maximum is 2047
        self.assertEqual(grid_coord[:, :2].max(axis=0).tolist(), [2047, 2047])
