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

"""Export wrapper of the lidar Sparse4D graph.

The graph takes the voxels of one frame and the instance bank state of the last frame, and
returns the frame outputs with the bank state of the next frame.
"""

from __future__ import annotations

from collections.abc import Sequence

from jaxtyping import Float32, Int32
import torch
import torch.nn as nn

from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.models.detection3d.heads.sparse4d import (
    TEMPORAL_QUERY_INDEX_OUTPUT_NAME,
    TEMPORAL_STATE_INPUT_NAMES,
    TEMPORAL_STATE_OUTPUT_NAMES,
    SparseQueryHead,
)
from autoware_ml.models.detection3d.main_modules.sparse4ds.sparse4d_lidar import Sparse4DLidar
from autoware_ml.utils.deploy import ExportSpec

SPARSE4D_EXPORT_OUTPUT_NAMES = (
    "classification",
    "prediction",
    "quality",
    *TEMPORAL_STATE_OUTPUT_NAMES,
    TEMPORAL_QUERY_INDEX_OUTPUT_NAME,
)
SPARSE4D_LIDAR_EXPORT_INPUT_NAMES = (
    "voxels",
    "coors",
    "num_points_per_voxel",
    *TEMPORAL_STATE_INPUT_NAMES,
)


class Sparse4DLidarExportWrapper(nn.Module):
    """Export-only lidar Sparse4D main body with the bank state at the graph boundary.

    The wrapper exposes the single-sample voxel interface of the BEVFusion lidar export followed
    by the instance bank state of the last frame, and returns the last decoder layer of the
    frame with the state of the next one.
    """

    def __init__(
        self,
        lidar_network: Sparse4DLidar,
        bbox_head: SparseQueryHead,
        output_names: Sequence[str] = SPARSE4D_EXPORT_OUTPUT_NAMES,
    ) -> None:
        """Initialize the export wrapper.

        Args:
            lidar_network: Export-ready lidar branch.
            bbox_head: Export-ready temporal query head.
            output_names: Frame outputs the graph returns, in order.
        """
        super().__init__()
        self.lidar_network = lidar_network
        self.bbox_head = bbox_head
        self.output_names = tuple(output_names)

    def forward(
        self,
        voxels: Float32[torch.Tensor, "num_voxels max_num_points num_point_features"],
        coors: Int32[torch.Tensor, "num_voxels 3"],
        num_points_per_voxel: Int32[torch.Tensor, " num_voxels"],
        temp_instance_feature: torch.Tensor,
        temp_anchor: torch.Tensor,
        temp_confidence: torch.Tensor,
        temp_valid: torch.Tensor,
        time_interval: torch.Tensor,
        prev2current: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Decode one frame with the bank state of the last one.

        Args:
            voxels: Voxel features.
            coors: Voxel coordinates in ``(z, y, x)`` order without batch column. Must be in
                this order to align with autoware.
            num_points_per_voxel: Number of points in each voxel.
            temp_instance_feature: Cached instance features ``(1, K, C)``.
            temp_anchor: Cached anchors ``(1, K, D)`` in the previous frame.
            temp_confidence: Cached instance confidences ``(1, K)``.
            temp_valid: Continuity gate of the slots ``(1,)`` in ``{0, 1}``.
            time_interval: Frame gap ``(1,)`` in seconds.
            prev2current: Transform from the previous into the current frame ``(1, 4, 4)``.

        Returns:
            The configured outputs, see :meth:`SparseQueryHead.forward_temporal_export`.
        """
        # The runtime hands one sample, the branch reads ``(batch, z, y, x)`` coordinates
        coords = torch.cat([coors.new_zeros((coors.shape[0], 1)), coors], dim=1)
        bev_map = self.lidar_network(
            voxels=voxels, coords=coords, num_points=num_points_per_voxel, batch_size=1
        )
        outputs = self.bbox_head.forward_temporal_export(
            bev_map,
            temp_instance_feature,
            temp_anchor,
            temp_confidence,
            temp_valid,
            time_interval,
            prev2current,
        )
        return tuple(outputs[name] for name in self.output_names)


def first_sample_voxel_inputs(
    batch_inputs: ModelBatchInputs,
) -> tuple[
    Float32[torch.Tensor, "num_voxels max_num_points num_point_features"],
    Int32[torch.Tensor, "num_voxels 3"],
    Int32[torch.Tensor, " num_voxels"],
]:
    """Extract the single-sample voxel export inputs in the runtime layout.

    The exported graph is a single-sample graph, so only the voxels of the first batch sample
    are kept, with their coordinates in the runtime ``(z, y, x)`` layout without a batch column.

    Args:
        batch_inputs: Batched model inputs used to derive the export tensors.

    Returns:
        Tuple of voxels, coordinates, and per-voxel point counts of the first sample.

    Raises:
        ValueError: If the batch carries no voxel data.
    """
    voxels_data = batch_inputs.voxels_data
    if voxels_data is None:
        raise ValueError(
            "ModelBatchInputs must contain voxels_data to build Sparse4D export specs."
        )
    first_sample = voxels_data.batch_indices == 0
    voxels = voxels_data.voxels[first_sample].float()
    # Autoware feeds the voxel coordinates in ``(z, y, x)`` order
    coors = voxels_data.coords[first_sample][:, [2, 1, 0]].int().contiguous()
    num_points_per_voxel = voxels_data.num_points[first_sample].int()
    return voxels, coors, num_points_per_voxel


def build_sparse4d_lidar_export_spec(
    lidar_network: Sparse4DLidar, bbox_head: SparseQueryHead, batch_inputs: ModelBatchInputs
) -> ExportSpec:
    """Build the export spec of the lidar Sparse4D graph.

    Args:
        lidar_network: Export-ready lidar branch.
        bbox_head: Export-ready temporal query head.
        batch_inputs: Batched model inputs used to derive the export tensors.

    Returns:
        Export spec whose graph takes the voxels of one frame with the bank state of the last
        frame and returns the frame outputs with the state of the next one.
    """
    voxels, coors, num_points_per_voxel = first_sample_voxel_inputs(batch_inputs)
    module = Sparse4DLidarExportWrapper(lidar_network, bbox_head).eval()
    return ExportSpec(
        module=module,
        args=(
            voxels,
            coors,
            num_points_per_voxel,
            *bbox_head.temporal_state_export_inputs(voxels.device),
        ),
        input_names=list(SPARSE4D_LIDAR_EXPORT_INPUT_NAMES),
        output_names=list(module.output_names),
    )
