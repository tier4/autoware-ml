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

"""Export-side preprocessing of pillar encoders: the decorated pillars their graphs read."""

from __future__ import annotations

from collections.abc import Sequence

from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.models.detection3d.encoders.pillars.pillar_feature_net import decorate_pillars
from autoware_ml.preprocessing.data_preprocessor_modules import DataPreprocessorModule


class PillarDecoration(DataPreprocessorModule):
    """Decorate the pillars of the voxelizer, as the deployed runtime does before the encoder.

    Runs after the voxelizer and sets ``decorated_pillars``, which both the model and its
    export read, so the exported encoder and the trained one see the same features.

    Args:
        voxel_size: Pillar size ``[dx, dy, dz]`` in meters.
        point_cloud_range: ``[x_min, y_min, z_min, x_max, y_max, z_max]`` in meters.
        with_distance: Whether to append point distance from the origin.
        with_cluster_center: Whether to append the offset to the mean of the pillar's points.
        with_voxel_center: Whether to append the offset to the pillar center.
    """

    def __init__(
        self,
        voxel_size: Sequence[float],
        point_cloud_range: Sequence[float],
        with_distance: bool = False,
        with_cluster_center: bool = True,
        with_voxel_center: bool = True,
    ) -> None:
        super().__init__()
        self.voxel_size = tuple(float(size) for size in voxel_size)
        self.voxel_origin = tuple(
            size / 2 + float(minimum) for size, minimum in zip(self.voxel_size, point_cloud_range)
        )
        self.with_distance = with_distance
        self.with_cluster_center = with_cluster_center
        self.with_voxel_center = with_voxel_center

    def __call__(self, batch: ModelBatchInputs, *, is_training: bool) -> ModelBatchInputs:
        """Set the decorated pillars of the batch.

        Raises:
            ValueError: If no voxelizer ran before.
        """
        del is_training
        if batch.voxels_data is None:
            raise ValueError("PillarDecoration needs the voxels of a voxelizer before it.")
        return batch.replace(
            decorated_pillars=decorate_pillars(
                batch.voxels_data,
                voxel_size=self.voxel_size,
                voxel_origin=self.voxel_origin,
                with_distance=self.with_distance,
                with_cluster_center=self.with_cluster_center,
                with_voxel_center=self.with_voxel_center,
            )
        )
