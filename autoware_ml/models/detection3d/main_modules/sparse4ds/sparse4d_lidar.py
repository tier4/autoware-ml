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

"""Lidar branch of the Sparse4D detector."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn

from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.models.detection3d.heads.sparse4d import SparseStage
from autoware_ml.models.detection3d.ptv3 import PTv3Det3DFeaturePyramid
from autoware_ml.models.segmentation3d.encoders.ptv3 import PointTransformerV3Encoder
from autoware_ml.models.segmentation3d.encoders.voxel import VoxelFeatureEncoder
from autoware_ml.models.segmentation3d.ptv3_base import (
    PTv3BaseModel,
    PTv3ExportContext,
    build_ptv3_export_context,
    split_block_parameters,
    validate_serialization_geometry,
)
from autoware_ml.utils.point_cloud.batching import batch_to_offset
from autoware_ml.utils.point_cloud.structures import Point, bit_length_tensor


class Sparse4DLidar(nn.Module):
    """Encode voxelized lidar points into the sparse stages the Sparse4D head samples.

    The branch owns the whole lidar path: the voxel encoder, the PTv3 encoder and the feature
    pyramid that fuses the deepest encoder stage top-down into the finer stages. That leaves the
    detector to run the head, so a camera branch can join with its own path.

    The branch fulfils the encoder owner interface of the PTv3 export helpers: the encoder, the
    voxel encoder, the export order, the encoder inputs and the export geometry, see
    :func:`autoware_ml.models.segmentation3d.ptv3_base.build_ptv3_export_context`.
    """

    EXPORT_ORDER = PTv3BaseModel.EXPORT_ORDER

    def __init__(
        self,
        encoder: PointTransformerV3Encoder,
        voxel_encoder: VoxelFeatureEncoder,
        det_fusion: PTv3Det3DFeaturePyramid,
        grid_size: float,
        point_cloud_range: Sequence[float],
        freeze_encoder: bool = False,
    ) -> None:
        """Initialize the lidar branch.

        Args:
            encoder: PTv3 encoder module.
            voxel_encoder: Voxel feature encoder feeding the embedding stem.
            det_fusion: Top-down fusion of the deepest encoder stage into the finer stages,
                whose sparse voxels the head samples.
            grid_size: Edge of the cubic voxels the encoder serializes, in meters.
            point_cloud_range: Range ``[x_min, y_min, z_min, x_max, y_max, z_max]`` of the
                voxel grid.
            freeze_encoder: When ``True``, the encoder is kept in eval mode with its parameters
                frozen.

        Raises:
            ValueError: If the voxel encoder and the encoder disagree on the point features, or
                the geometry cannot cover the pooling hierarchy of the encoder.
        """
        super().__init__()
        if encoder.in_channels != voxel_encoder.out_channels:
            raise ValueError(
                f"The encoder reads {encoder.in_channels} point features but the voxel encoder "
                f"describes a voxel with {voxel_encoder.out_channels}."
            )
        self.encoder = encoder
        self.voxel_encoder = voxel_encoder
        self.det_fusion = det_fusion
        self.grid_size = float(grid_size)
        self.point_cloud_range = tuple(float(value) for value in point_cloud_range)
        validate_serialization_geometry(encoder, self.grid_size, self.point_cloud_range)
        self.freeze_encoder = bool(freeze_encoder)
        if self.freeze_encoder:
            self.encoder.requires_grad_(False)
            self.encoder.eval()

    def train(self, mode: bool = True) -> Sparse4DLidar:
        """Keep the frozen encoder in eval mode during training.

        Args:
            mode: When ``True``, set the branch to training mode; otherwise to evaluation mode.

        Returns:
            This branch.
        """
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    @property
    def num_levels(self) -> int:
        """Number of stages the branch returns, which the head samples."""
        return self.det_fusion.num_levels

    def build_optimizer_groups(self) -> Mapping[str, Sequence[torch.nn.Parameter]]:
        """Group the pretrained encoder parameters apart from the newly initialized pyramid.

        Returns:
            The default and the block parameters of the encoder, and the pyramid parameters.
        """
        encoder_default_params, encoder_block_params = split_block_parameters(self.encoder)
        return {
            "encoder_default": encoder_default_params,
            "encoder_block": encoder_block_params,
            "det_fusion": [
                parameter for parameter in self.det_fusion.parameters() if parameter.requires_grad
            ],
        }

    def forward(
        self,
        voxels: torch.Tensor,
        num_points: torch.Tensor,
        voxel_coords: torch.Tensor,
        num_dropped_voxels: torch.Tensor,
        time_lag_column: int,
    ) -> list[SparseStage]:
        """Encode the voxels of a batch and fuse the stages the head samples.

        Args:
            voxels: Padded voxel points ``(num_voxels, max_points, channels)``.
            num_points: Valid point count per voxel ``(num_voxels,)``.
            voxel_coords: Voxel coordinates ``(num_voxels, 4)`` laid out as ``(batch, x, y, z)``
                with the voxels grouped by batch index.
            num_dropped_voxels: Occupied voxels the voxelizer discarded.
            time_lag_column: Point column holding the time lag.

        Returns:
            Features, voxel coordinates and batch offsets of every level, from the finest to the
            coarsest.
        """
        return self.det_fusion(
            self.encode(voxels, num_points, voxel_coords, num_dropped_voxels, time_lag_column)
        )

    def encode(
        self,
        voxels: torch.Tensor,
        num_points: torch.Tensor,
        voxel_coords: torch.Tensor,
        num_dropped_voxels: torch.Tensor,
        time_lag_column: int,
    ) -> Point:
        """Run the voxel encoder and the PTv3 encoder on preprocessed voxels.

        Args:
            voxels: Padded voxel points ``(num_voxels, max_points, channels)``.
            num_points: Valid point count per voxel ``(num_voxels,)``.
            voxel_coords: Voxel coordinates ``(num_voxels, 4)`` with a leading batch column.
            num_dropped_voxels: Occupied voxels the voxelizer discarded.
            time_lag_column: Point column holding the time lag.

        Returns:
            Deepest encoder point with the full pooling chain attached.

        Raises:
            ValueError: If the points do not follow the layout the voxel encoder reads.
            RuntimeError: If the voxelizer dropped occupied voxels.
        """
        self.voxel_encoder.check_points(voxels, time_lag_column)
        if int(num_dropped_voxels) > 0:
            raise RuntimeError(
                f"The voxelizer dropped {int(num_dropped_voxels)} occupied voxels. Increase "
                "'max_voxels' and 'eval_max_voxels' of the data preprocessing so every voxel "
                "is kept."
            )
        return self.encoder(self.build_encoder_inputs(voxels, num_points, voxel_coords))

    def build_encoder_inputs(
        self, voxels: torch.Tensor, num_points: torch.Tensor, voxel_coords: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Turn preprocessed voxels into the PTv3 encoder input dictionary.

        Args:
            voxels: Padded voxel points ``(num_voxels, max_points, channels)``.
            num_points: Valid point count per voxel ``(num_voxels,)``.
            voxel_coords: Voxel coordinates ``(num_voxels, 4)`` laid out as ``(batch, x, y, z)``
                with the voxels grouped by batch index.

        Returns:
            Dictionary with ``coord``, ``feat``, ``grid_coord`` and ``offset``.
        """
        feat = self.voxel_encoder(voxels, num_points)
        return {
            "coord": feat[:, :3],
            "feat": feat,
            "grid_coord": voxel_coords[:, 1:],
            "offset": batch_to_offset(voxel_coords[:, 0].long()),
        }

    def build_export_context(self, batch_inputs: ModelBatchInputs) -> PTv3ExportContext:
        """Serialize an example batch and run the export-ready encoder once.

        Args:
            batch_inputs: Example preprocessed batch with the voxels of one sample.

        Returns:
            The shared front half of the split exports: the encoder graph and its stage
            features.
        """
        return build_ptv3_export_context(self, batch_inputs)

    def _compute_export_geometry(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the sparse shape and the serialization depth baked into the exports.

        Args:
            device: Device of the voxels of the example batch.

        Returns:
            ``(sparse_shape, serialization_depth)`` as long tensors on ``device``.
        """
        point_cloud_range = torch.tensor(self.point_cloud_range, dtype=torch.float32, device=device)
        axis_extents = (point_cloud_range[3:] - point_cloud_range[:3]) / self.grid_size
        serialization_depth = bit_length_tensor(torch.max(axis_extents))
        sparse_shape = torch.round(axis_extents).to(dtype=torch.long)
        return sparse_shape, serialization_depth

    def _prepare_encoder_export(self) -> PointTransformerV3Encoder:
        """Return an export-ready copy of the encoder with the export order."""
        return self.encoder.prepare_for_export(self.EXPORT_ORDER)
