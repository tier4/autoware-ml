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

from jaxtyping import Float32
import torch
import torch.nn.functional as F

from autoware_ml.dataclasses.models.detection3d.head_outputs import (
    TransFusionHeadOutputs,
)
from autoware_ml.models.detection3d.heads.transfusions.transfusion_head import TransFusionHead


# TODO(Kok Seang): Move this to a more appropriate location, e.g. in the head module.
def export_detection_outputs(
    head: TransFusionHead, outputs: TransFusionHeadOutputs
) -> tuple[
    Float32[torch.Tensor, "10 num_proposals"],
    Float32[torch.Tensor, " num_proposals"],
    Float32[torch.Tensor, " num_proposals"],
]:
    """Pack raw head outputs into the runtime detection interface.

    The runtime consumes the raw regression channels and decodes them with
    its own parameters, so no metric-space decoding happens in the graph.

    Args:
        head: TransFusion detection head producing the output dictionary.
        outputs: Raw prediction tensors from the head forward pass.

    Returns:
        Tuple of ``bbox_pred`` with the concatenated regression channels of
        shape ``(10, num_proposals)``, ``score`` of shape ``(num_proposals,)``,
        and ``label_pred`` of shape ``(num_proposals,)``.
    """
    num_proposals = head.num_proposals
    if outputs.separate_head_outputs is None:
        raise ValueError("BEVFusion export requires separate head outputs.")

    separate_head_outputs = outputs.separate_head_outputs
    if separate_head_outputs.vel is None:
        raise ValueError("BEVFusion export requires a velocity branch in the detection head.")

    # The per-proposal class heatmap of the prediction heads, not the dense BEV heatmap. An
    # auxiliary head concatenates every decoder layer along the proposal axis, so only the trailing
    # ``num_proposals`` columns belonging to the last layer are exported.
    query_labels = outputs.query_labels
    heatmap = separate_head_outputs.heatmap[..., -num_proposals:].sigmoid()
    one_hot = (
        F.one_hot(query_labels, num_classes=head.num_classes).permute(0, 2, 1).to(heatmap.dtype)
    )
    score = (heatmap * outputs.query_heatmap_score * one_hot)[0].max(dim=0).values

    bbox_pred = torch.cat(
        [
            separate_head_outputs.center[0, :, -num_proposals:],
            separate_head_outputs.height[0, :, -num_proposals:],
            separate_head_outputs.dim[0, :, -num_proposals:],
            separate_head_outputs.rot[0, :, -num_proposals:],
            separate_head_outputs.vel[0, :, -num_proposals:],
        ],
        dim=0,
    )
    return bbox_pred, score, query_labels[0]
