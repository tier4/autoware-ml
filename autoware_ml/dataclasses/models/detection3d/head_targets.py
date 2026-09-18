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


class TransFusionHeadTargets(BaseModel):
    """Store assignment targets for one TransFusion training batch.

    Attributes:
        labels: Target class labels for all decoder queries.
        label_weights: Per-query, per-class classification weights. Zero where a negative query
            must not be pushed away from a class its sample is not annotated for.
        bbox_targets: Encoded box regression targets.
        bbox_weights: Per-query box regression weights.
        num_pos: Number of matched positive queries.
        matched_iou: Mean IoU of matched positive queries.
        dense_heatmaps: Dense heatmap target used for query initialization.
        class_weights: Per-sample class weights, zero for the classes a sample is not annotated
            for. They scale the dense heatmap loss of every cell of that class.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    labels: Int64[torch.Tensor, "batch_size num_proposals"]
    label_weights: Float32[torch.Tensor, "batch_size num_proposals num_classes"]
    bbox_targets: Float32[torch.Tensor, "batch_size num_proposals code_size"]
    bbox_weights: Float32[torch.Tensor, "batch_size num_proposals code_size"]
    num_pos: int
    matched_iou: float
    dense_heatmaps: Float32[torch.Tensor, "batch_size num_classes height width"]
    class_weights: Float32[torch.Tensor, "batch_size num_classes"]
