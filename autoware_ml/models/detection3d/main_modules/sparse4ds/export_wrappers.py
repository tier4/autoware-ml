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

"""Export wrappers of the Sparse4D detection graph.

The detection graph takes the encoder stages the pyramid fuses and the instance bank state of
the last frame, and returns the frame outputs with the bank state of the next frame.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn

from autoware_ml.models.detection3d.heads.sparse4d import (
    TEMPORAL_QUERY_INDEX_OUTPUT_NAME,
    TEMPORAL_STATE_INPUT_NAMES,
    TEMPORAL_STATE_OUTPUT_NAMES,
    SparseQueryHead,
    SparseStage,
)
from autoware_ml.models.detection3d.ptv3 import (
    PTv3Det3DFeaturePyramid,
    det3d_head_export_args,
    det3d_head_export_dynamic_axes,
    det3d_head_export_input_names,
    rebuild_point_chain,
)
from autoware_ml.models.segmentation3d.ptv3_base import PTv3BaseModel, PTv3ExportContext
from autoware_ml.utils.deploy import ExportSpec

SPARSE4D_EXPORT_OUTPUT_NAMES = (
    "classification",
    "prediction",
    "quality",
    *TEMPORAL_STATE_OUTPUT_NAMES,
    TEMPORAL_QUERY_INDEX_OUTPUT_NAME,
)


def fuse_export_stages(
    det_fusion: PTv3Det3DFeaturePyramid, inputs: Sequence[torch.Tensor]
) -> tuple[list[SparseStage], tuple[torch.Tensor, ...]]:
    """Fuse the encoder stages a head graph takes ahead of its other inputs.

    Args:
        det_fusion: Detection feature pyramid.
        inputs: Stage features from the finest to the deepest stage, the pooling clusters of every
            stage but the deepest, the voxel coordinates of every stage the head reads (the
            deepest one too when the pyramid keeps it), then the other graph inputs.

    Returns:
        The fused stages and the other graph inputs.
    """
    num_stages = det_fusion.num_stages
    grid_end = 3 * num_stages - 2 + int(det_fusion.keep_deepest)
    feats = inputs[:num_stages]
    clusters = inputs[num_stages : 2 * num_stages - 1]
    grid_coords = inputs[2 * num_stages - 1 : grid_end]
    return det_fusion(rebuild_point_chain(feats, clusters, grid_coords)), tuple(inputs[grid_end:])


class Sparse4DHeadExportWrapper(nn.Module):
    """Export-only temporal Sparse4D head consuming the encoder stages the pyramid fuses.

    The graph inputs are the stage features from the finest to the deepest stage, the pooling
    clusters of every stage but the deepest, the voxel coordinates of every stage the head reads
    (the deepest one too when the pyramid keeps it), the bank state, then the inputs a subclass
    turns into the head context.
    """

    def __init__(
        self,
        det_fusion: PTv3Det3DFeaturePyramid,
        bbox_head: SparseQueryHead,
        output_names: Sequence[str] = SPARSE4D_EXPORT_OUTPUT_NAMES,
    ) -> None:
        """Initialize the export-only head module.

        Args:
            det_fusion: Detection feature pyramid (copied internally).
            bbox_head: Export-ready temporal query head.
            output_names: Frame outputs the graph returns, in order.
        """
        super().__init__()
        self.det_fusion = deepcopy(det_fusion).eval()
        self.bbox_head = bbox_head
        self.output_names = tuple(output_names)

    def forward(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Decode one frame and return the configured outputs."""
        outputs = self.frame_outputs(*inputs)
        return tuple(outputs[name] for name in self.output_names)

    def frame_outputs(self, *inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Fuse the encoder stages and decode one frame with the bank state of the last one.

        Args:
            *inputs: Stage features, pooling clusters, voxel coordinates, the bank state and the
                head context inputs.

        Returns:
            The outputs of :meth:`Sparse4DHead.forward_temporal_export`.
        """
        stages, rest = fuse_export_stages(self.det_fusion, inputs)
        state_end = len(TEMPORAL_STATE_INPUT_NAMES)
        return self.bbox_head.forward_temporal_export(
            stages, *rest[:state_end], context=self.head_context(rest[state_end:])
        )

    def head_context(self, inputs: Sequence[torch.Tensor]) -> Any:
        """Head context from the graph inputs after the bank state, none for the lidar head.

        Raises:
            ValueError: If the lidar head graph gets inputs after the bank state.
        """
        if inputs:
            raise ValueError("The lidar head graph takes no inputs after the bank state.")
        return None


def build_sparse4d_head_export_spec(
    context: PTv3ExportContext,
    module: Sparse4DHeadExportWrapper,
    head_context_args: Sequence[torch.Tensor] = (),
    head_context_names: Sequence[str] = (),
) -> ExportSpec:
    """Build the temporal Sparse4D head export spec from the stages the pyramid fuses.

    Args:
        context: Shared export context of the encoder.
        module: Export wrapper of a temporal query head.
        head_context_args: Example graph inputs after the bank state that the module turns into
            the head context.
        head_context_names: Graph names of ``head_context_args``.

    Returns:
        Export spec whose graph takes the bank state of the last frame and returns the next one
        with the query every state slot was selected from.
    """
    det_fusion = module.det_fusion
    finest_stage = context.stage_count - det_fusion.num_stages
    deepest_grid_coord = det_fusion.keep_deepest
    stage_args = det3d_head_export_args(context, finest_stage, deepest_grid_coord)
    return ExportSpec(
        module=module.eval(),
        args=(
            *stage_args,
            *module.bbox_head.temporal_state_export_inputs(stage_args[0].device),
            *head_context_args,
        ),
        input_names=[
            *det3d_head_export_input_names(context.stage_count, finest_stage, deepest_grid_coord),
            *TEMPORAL_STATE_INPUT_NAMES,
            *head_context_names,
        ],
        output_names=list(module.output_names),
        dynamic_axes=det3d_head_export_dynamic_axes(
            context.stage_count, finest_stage, deepest_grid_coord
        ),
        supported_stages=PTv3BaseModel.EXPORT_SUPPORTED_STAGES,
    )
