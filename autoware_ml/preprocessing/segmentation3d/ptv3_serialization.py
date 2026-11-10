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

"""Export-side preprocessing of PTv3: the serialized voxels its exported graphs read."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.models.segmentation3d.ptv3_base import (
    ptv3_export_geometry,
    serialize_voxels,
    split_voxels_by_sample,
)
from autoware_ml.preprocessing.data_preprocessor_modules import DataPreprocessorModule


class PTv3Serialization(DataPreprocessorModule):
    """Serialize the voxels of every sample and precompute their pooling metadata.

    What the deployed runtime computes before it runs the exported graphs. Runs after the
    voxelizer and sets ``serialized_voxels``, one ``PTv3ExportInputs`` per sample, which the
    export and the deployed graphs read. Training skips it: the trained encoder serializes the
    voxels itself, with its own orders.

    Args:
        grid_size: Edge of the cubic voxels in meters.
        point_cloud_range: ``[x_min, y_min, z_min, x_max, y_max, z_max]`` in meters.
        strides: One pooling stride per encoder stage.
        patch_sizes: Attention window per level, the input level first.
        orders: Serialization orders of the deployed graphs.
    """

    def __init__(
        self,
        grid_size: float,
        point_cloud_range: Sequence[float],
        strides: Sequence[int],
        patch_sizes: Sequence[int],
        orders: Sequence[str] = ("z", "z-trans"),
    ) -> None:
        super().__init__()
        self.grid_size = float(grid_size)
        self.point_cloud_range = tuple(float(value) for value in point_cloud_range)
        self.strides = tuple(int(stride) for stride in strides)
        self.patch_sizes = tuple(int(size) for size in patch_sizes)
        self.orders = tuple(orders)

    def __call__(self, batch: ModelBatchInputs, *, is_training: bool) -> ModelBatchInputs:
        """Set the serialized voxels of every sample, outside training.

        Raises:
            ValueError: If no voxelizer ran before.
        """
        if is_training:
            return batch
        if batch.voxels_data is None:
            raise ValueError("PTv3Serialization needs the voxels of a voxelizer before it.")
        device = batch.voxels_data.voxels.device
        sparse_shape, serialization_depth = ptv3_export_geometry(
            self.grid_size, self.point_cloud_range, device
        )
        with torch.no_grad():
            samples = tuple(
                serialize_voxels(
                    voxels,
                    num_points,
                    grid_coord,
                    sparse_shape,
                    serialization_depth,
                    self.orders,
                    self.strides,
                    self.patch_sizes,
                )
                for voxels, num_points, grid_coord in split_voxels_by_sample(
                    batch.voxels_data, batch.batch_size()
                )
            )
        return batch.replace(serialized_voxels=samples)
