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

"""The rule of a physical 3D box shape, shared by the box filters and the anchor recipes."""

from jaxtyping import Bool, Float
import torch
from torch import Tensor

from autoware_ml.types.geometry import Box3DFieldIndex


def physical_shape_mask(
    bbox_params: Float[Tensor, "num_bboxes box_dim"],
) -> Bool[Tensor, " num_bboxes"]:
    """Mark the boxes whose center, size and yaw are finite and whose sizes are positive.

    The box size targets are log-encoded, so a zero or negative extent is not representable.

    Args:
        bbox_params: Box parameters, the center, the size and the yaw first.

    Returns:
        Whether every box has a physical shape.
    """
    shape = bbox_params[:, : Box3DFieldIndex.YAW + 1]
    sizes = bbox_params[:, Box3DFieldIndex.LENGTH : Box3DFieldIndex.HEIGHT + 1]
    return torch.isfinite(shape).all(dim=1) & (sizes > 0.0).all(dim=1)
