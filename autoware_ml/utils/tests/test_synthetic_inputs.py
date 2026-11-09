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

"""Tests for the synthetic input helpers."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from autoware_ml.models.segmentation3d.ptv3_base import (
    PTv3BaseModel,
    build_serialized_pooling_metadata,
)
from autoware_ml.utils.deploy import ExportSpec
from autoware_ml.utils.point_cloud.structures import serialize_point_cloud_batch
from autoware_ml.utils.synthetic_inputs import (
    VoxelLayoutSpec,
    mean_neighbors,
    voxel_layout,
    write_export_inputs,
)

_NEIGHBORS = (8.24, 9.15, 9.44, 9.93, 10.60)


class TestVoxelLayoutSpec(unittest.TestCase):
    """Unit tests for VoxelLayoutSpec."""

    def test_rejects_stages_that_do_not_line_up(self) -> None:
        """
        Input: two voxel counts with two pooling strides and two neighbor targets.
        Expected: no spec, as two stages pool once.
        Check: a ValueError naming the counts, targets and strides.
        """
        with self.assertRaisesRegex(ValueError, "2 voxel counts and 2 neighbor targets for 2"):
            VoxelLayoutSpec([320, 48], [2, 2], torch.tensor([8, 8, 5]), _NEIGHBORS[:2])


class TestVoxelLayout(unittest.TestCase):
    """Unit tests for voxel_layout."""

    def test_pools_to_the_requested_counts_and_neighbors(self) -> None:
        """
        Input: five stages of 8000 to 256 voxels on a 512 x 512 x 64 grid, pooled by 2.
        Expected: the model's own serialized pooling yields every requested count, each stage
        with the requested mean neighbors.
        Check: the voxel count of every stage and its mean neighbors within 0.1.
        """
        counts = [8000, 4000, 1600, 640, 256]
        strides = [2, 2, 2, 2]
        grid_shape = torch.tensor([512, 512, 64])
        generator = torch.Generator().manual_seed(0)
        grid = voxel_layout(VoxelLayoutSpec(counts, strides, grid_shape, _NEIGHBORS), generator)

        point, _ = serialize_point_cloud_batch(
            {
                "coord": grid.float(),
                "feat": grid.float(),
                "grid_coord": grid,
                "offset": torch.tensor([len(grid)]),
            },
            PTv3BaseModel.EXPORT_ORDER,
            torch.tensor(int(grid_shape.max()).bit_length()),
        )
        metadata = build_serialized_pooling_metadata(
            point["grid_coord"],
            point["serialized_code"],
            point["serialized_order"],
            strides,
            [512] * len(counts),
        )
        stages = [grid, *(meta.grid_coord for meta in metadata)]
        self.assertEqual([len(stage) for stage in stages], counts)
        for stage, target in zip(stages, _NEIGHBORS):
            self.assertAlmostEqual(mean_neighbors(stage), target, delta=0.1)

    def test_rejects_counts_that_cannot_pool(self) -> None:
        """
        Input: 100 fine voxels under 10 coarse ones on a 64^3 grid, while 10 parents have only
        80 children at stride 2, and 10 coarse voxels on a 4^3 grid, whose coarse grid has 8.
        Expected: neither layout exists.
        Check: both raise a ValueError saying the counts cannot pool.
        """
        for counts, size in [([100, 10], 64), ([40, 10], 4)]:
            with self.subTest(counts=counts, size=size):
                with self.assertRaisesRegex(ValueError, "cannot pool"):
                    voxel_layout(
                        VoxelLayoutSpec(counts, [2], torch.tensor([size] * 3), _NEIGHBORS[:2]),
                        torch.Generator(),
                    )

    def test_fills_a_grid_whose_size_is_not_a_multiple_of_the_stride(self) -> None:
        """
        Input: 320 voxels pooled by 2 to 48 on an 8 x 8 x 5 grid, i.e. every cell of the grid
        and every cell of its 4 x 4 x 3 coarse grid.
        Expected: the coarse cells on the odd z edge keep only their children inside the grid.
        Check: 320 distinct voxels whose largest coordinates are the grid's last cells.
        """
        grid_shape = torch.tensor([8, 8, 5])
        spec = VoxelLayoutSpec([320, 48], [2], grid_shape, _NEIGHBORS[:2])
        grid = voxel_layout(spec, torch.Generator())

        self.assertEqual(len(torch.unique(grid, dim=0)), 320)
        self.assertEqual(grid.max(dim=0).values.tolist(), [7, 7, 4])


class TestWriteExportInputs(unittest.TestCase):
    """Unit tests for write_export_inputs."""

    def test_writes_raw_tensors_and_trtexec_shapes(self) -> None:
        """
        Input: one module spec with a float32 [2, 3] and an int64 [2] input.
        Expected: one raw .bin per input and shapes.txt in the trtexec --shapes format.
        Check: shapes.txt reads ``feat:2x3,index:2`` and each .bin reads back its tensor.
        """
        feat = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        index = torch.tensor([1, 2])
        specs = {
            "encoder": ExportSpec(
                module=torch.nn.Identity(),
                args=(feat, index),
                input_names=["feat", "index"],
                output_names=["out"],
            )
        }

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            write_export_inputs(specs, out)

            self.assertEqual((out / "encoder" / "shapes.txt").read_text(), "feat:2x3,index:2\n")
            self.assertEqual(
                np.fromfile(out / "encoder" / "feat.bin", dtype=np.float32).tolist(),
                feat.flatten().tolist(),
            )
            self.assertEqual(
                np.fromfile(out / "encoder" / "index.bin", dtype=np.int64).tolist(), [1, 2]
            )


if __name__ == "__main__":
    unittest.main()
