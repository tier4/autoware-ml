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

"""The PTv3 encoder as the sparse middle encoder of a detection lidar branch."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from jaxtyping import Float32, Int
import torch
import torch.nn as nn

from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.models.segmentation3d.encoders.ptv3 import PointTransformerV3Encoder
from autoware_ml.models.segmentation3d.encoders.voxel import VoxelFeatureEncoder
from autoware_ml.models.segmentation3d.ptv3_base import (
    PTv3BaseModel,
    PTv3ExportContext,
    run_ptv3_export_encoder,
    serialize_ptv3_export_inputs,
    split_block_parameters,
    validate_serialization_geometry,
)
from autoware_ml.utils.point_cloud.batching import batch_to_offset
from autoware_ml.utils.point_cloud.structures import Point, bit_length_tensor


class PTv3MiddleEncoder(nn.Module):
    """Run the PTv3 encoder on voxel features and return its deepest stage with the pooling chain.

    The middle encoder owns what the PTv3 task models keep on themselves: the serialization geometry
    the encoder is checked against and the exports bake in, the optionally frozen encoder and
    the optimizer groups of its pretrained blocks, see
    :class:`autoware_ml.models.segmentation3d.ptv3_base.PTv3BaseModel`.
    """

    EXPORT_ORDER = PTv3BaseModel.EXPORT_ORDER

    def __init__(
        self,
        encoder: PointTransformerV3Encoder,
        grid_size: float,
        point_cloud_range: Sequence[float],
        freeze_encoder: bool = False,
    ) -> None:
        """Initialize the middle encoder.

        Args:
            encoder: PTv3 encoder module.
            grid_size: Edge of the cubic voxels the encoder serializes, in meters.
            point_cloud_range: Range ``[x_min, y_min, z_min, x_max, y_max, z_max]`` of the
                voxel grid.
            freeze_encoder: When ``True``, the encoder is kept in eval mode with its parameters
                frozen.

        Raises:
            ValueError: If the geometry cannot cover the pooling hierarchy of the encoder.
        """
        super().__init__()
        self.encoder = encoder
        self.grid_size = float(grid_size)
        self.point_cloud_range = tuple(float(value) for value in point_cloud_range)
        validate_serialization_geometry(encoder, self.grid_size, self.point_cloud_range)
        self.freeze_encoder = bool(freeze_encoder)
        if self.freeze_encoder:
            self.encoder.requires_grad_(False)
            self.encoder.eval()

    def train(self, mode: bool = True) -> PTv3MiddleEncoder:
        """Keep the frozen encoder in eval mode during training.

        Args:
            mode: When ``True``, set the middle encoder to training mode; otherwise to
                evaluation mode.

        Returns:
            This middle encoder.
        """
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def build_optimizer_groups(self) -> Mapping[str, Sequence[torch.nn.Parameter]]:
        """Group the attention blocks of the pretrained encoder apart from its other parameters.

        Returns:
            ``encoder_default`` and ``encoder_block``, see
            :func:`autoware_ml.models.segmentation3d.ptv3_base.split_block_parameters`.
        """
        encoder_default_params, encoder_block_params = split_block_parameters(self.encoder)
        return {"encoder_default": encoder_default_params, "encoder_block": encoder_block_params}

    def forward(
        self,
        voxel_features: Float32[torch.Tensor, "num_voxels channels"],
        coords: Int[torch.Tensor, "num_voxels 4"],
    ) -> Point:
        """Run the PTv3 encoder on the voxel features of a batch.

        Args:
            voxel_features: One feature vector per voxel, its first three features the position.
            coords: Voxel coordinates ``(num_voxels, 4)`` laid out as ``(batch, x, y, z)`` with
                the voxels grouped by batch index.

        Returns:
            Deepest encoder point with the full pooling chain attached.
        """
        return self.encoder(self.build_encoder_inputs(voxel_features, coords))

    @staticmethod
    def build_encoder_inputs(
        voxel_features: Float32[torch.Tensor, "num_voxels channels"],
        coords: Int[torch.Tensor, "num_voxels 4"],
    ) -> dict[str, torch.Tensor]:
        """Turn the voxel features into the PTv3 encoder input dictionary.

        Args:
            voxel_features: One feature vector per voxel, its first three features the position.
            coords: Voxel coordinates ``(num_voxels, 4)`` laid out as ``(batch, x, y, z)`` with
                the voxels grouped by batch index.

        Returns:
            Dictionary with ``coord``, ``feat``, ``grid_coord`` and ``offset``.
        """
        return {
            "coord": voxel_features[:, :3],
            "feat": voxel_features,
            "grid_coord": coords[:, 1:],
            "offset": batch_to_offset(coords[:, 0].long()),
        }

    def build_export_context(
        self, voxel_encoder: VoxelFeatureEncoder, batch_inputs: ModelBatchInputs
    ) -> PTv3ExportContext:
        """Serialize an example batch and run the export-ready encoder once.

        Args:
            voxel_encoder: Voxel feature encoder feeding the encoder, part of the exported graph.
            batch_inputs: Example preprocessed batch with the voxels of one sample.

        Returns:
            The shared front half of the split exports: the encoder graph and its stage
            features.

        Raises:
            ValueError: If the batch carries no voxels.
        """
        voxels_data = batch_inputs.voxels_data
        if voxels_data is None:
            raise ValueError("PTv3 needs the voxels of a voxelizer in the model inputs.")
        sparse_shape, serialization_depth = self._compute_export_geometry(voxels_data.voxels.device)
        inputs = serialize_ptv3_export_inputs(
            voxels_data,
            self.build_encoder_inputs(
                voxel_encoder(voxels_data.voxels, voxels_data.num_points),
                voxels_data.batch_xyz_coords(),
            ),
            export_order=self.EXPORT_ORDER,
            sparse_shape=sparse_shape,
            serialization_depth=serialization_depth,
            strides=self.encoder.stride,
            patch_sizes=list(self.encoder.enc_patch_sizes),
        )
        return run_ptv3_export_encoder(
            inputs,
            encoder=self.encoder.prepare_for_export(self.EXPORT_ORDER),
            voxel_encoder=voxel_encoder,
            strides=self.encoder.stride,
        )

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
