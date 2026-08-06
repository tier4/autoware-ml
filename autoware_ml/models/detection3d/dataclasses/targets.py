"""
Module to save encoded targets for a 3D detection head.
"""

from jaxtyping import Float32, Int64, Bool
from pydantic import BaseModel, ConfigDict

import torch


class CenterHeadTargets(BaseModel):
    """
    Dataclass to encode bbox and save targets for a CenterHead-based detection3d head.

    Attributes:
        heatmaps: Heatmap targets for the CenterHead-based detection3d dense heatmap head.
        reg_targets: Regression targets for the CenterHead-based detection3d regression head.
        reg_indices: Indices of the regression targets in the heatmap.
        valid_masks: Mask to indicate valid regression targets.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    heatmaps: Float32[torch.Tensor, "batch_size num_classes height width"]
    # 8 (center_x, center_y, center_z, length, width, height, sin(heading), cos(heading)) if not velocity else
    # 10 (center_x, center_y, center_z, length, width, height, heading, velocity_x, velocity_y)
    reg_targets: Float32[torch.Tensor, "batch_size max_num_boxes num_reg_targets"]
    reg_indices: Int64[torch.Tensor, "batch_size max_num_boxes"]
    valid_masks: Bool[torch.Tensor, "batch_size max_num_boxes"]
