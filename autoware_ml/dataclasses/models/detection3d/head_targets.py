"""
Module to save encoded targets for a 3D detection head.
"""

from __future__ import annotations

from jaxtyping import Bool, Float, Int64
from pydantic import BaseModel, ConfigDict, model_validator

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

    heatmaps: Float[torch.Tensor, "batch_size num_classes height width"]
    # 8 (center_x, center_y, center_z, length, width, height, sin(heading), cos(heading)) if not velocity else
    # 10 (center_x, center_y, center_z, length, width, height, sin(heading), cos(heading), velocity_x, velocity_y)
    reg_targets: Float[torch.Tensor, "batch_size max_num_boxes num_reg_targets"]
    reg_indices: Int64[torch.Tensor, "batch_size max_num_boxes"]
    valid_masks: Bool[torch.Tensor, "batch_size max_num_boxes"]

    @model_validator(mode="after")
    def _check_batch_and_box_budget(self) -> CenterHeadTargets:
        """Check that the heatmap and the box targets share a batch and a box budget."""
        batch_size = self.heatmaps.shape[0]
        max_num_boxes = self.reg_targets.shape[1]
        if self.reg_targets.shape[0] != batch_size:
            raise ValueError(
                f"reg_targets must have batch_size={batch_size}, got {tuple(self.reg_targets.shape)}."
            )
        for name, tensor in (("reg_indices", self.reg_indices), ("valid_masks", self.valid_masks)):
            if tensor.shape != (batch_size, max_num_boxes):
                raise ValueError(
                    f"{name} must have shape ({batch_size}, {max_num_boxes}), "
                    f"got {tuple(tensor.shape)}."
                )
        return self


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
    label_weights: Float[torch.Tensor, "batch_size num_proposals num_classes"]
    bbox_targets: Float[torch.Tensor, "batch_size num_proposals code_size"]
    bbox_weights: Float[torch.Tensor, "batch_size num_proposals code_size"]
    num_pos: int
    matched_iou: float
    dense_heatmaps: Float[torch.Tensor, "batch_size num_classes height width"]
    class_weights: Float[torch.Tensor, "batch_size num_classes"]

    @model_validator(mode="after")
    def _check_shared_dims(self) -> TransFusionHeadTargets:
        """Check that all targets share a batch, proposal count, class count and code size."""
        batch_size, num_proposals = self.labels.shape
        num_classes = self.class_weights.shape[1]
        code_size = self.bbox_targets.shape[2]
        if self.class_weights.shape[0] != batch_size:
            raise ValueError(
                f"class_weights must have batch_size={batch_size}, got {tuple(self.class_weights.shape)}."
            )
        if self.label_weights.shape != (batch_size, num_proposals, num_classes):
            raise ValueError(
                f"label_weights must have shape ({batch_size}, {num_proposals}, {num_classes}), "
                f"got {tuple(self.label_weights.shape)}."
            )
        for name, tensor in (
            ("bbox_targets", self.bbox_targets),
            ("bbox_weights", self.bbox_weights),
        ):
            if tensor.shape != (batch_size, num_proposals, code_size):
                raise ValueError(
                    f"{name} must have shape ({batch_size}, {num_proposals}, {code_size}), "
                    f"got {tuple(tensor.shape)}."
                )
        if self.dense_heatmaps.shape[:2] != (batch_size, num_classes):
            raise ValueError(
                f"dense_heatmaps must have batch_size={batch_size} and num_classes={num_classes}, "
                f"got {tuple(self.dense_heatmaps.shape)}."
            )
        return self
