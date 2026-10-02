"""
Modules to save decoded predictions from a detection3d head.
"""

from __future__ import annotations

from jaxtyping import Float32, Int64
from pydantic import BaseModel, ConfigDict, model_validator

import torch


class Detection3DSamplePredictions(BaseModel):
    """
    Dataclass to save decoded predictions from a 3D detection model for a sample.

    Attributes:
      bboxes_3d: Predicted 3D bounding boxes, 7 (center_x, center_y, center_z, length, width,
        height, heading) parameters per box, or 9 when velocity (velocity_x, velocity_y) is
        predicted as well.
      scores_3d: Confidence score for each predicted box.
      labels_3d: Class index for each predicted box.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    # 7 (center_x, center_y, center_z, length, width, height, heading) if not velocity else
    # 9 (center_x, center_y, center_z, length, width, height, heading, velocity_x, velocity_y)
    bboxes_3d: Float32[torch.Tensor, "num_boxes num_bbox_params"]
    scores_3d: Float32[torch.Tensor, " num_boxes"]
    labels_3d: Int64[torch.Tensor, " num_boxes"]

    @model_validator(mode="after")
    def _check_num_boxes(self) -> Detection3DSamplePredictions:
        """Check that boxes, scores and labels describe the same number of boxes."""
        num_boxes = self.bboxes_3d.shape[0]
        if self.scores_3d.shape[0] != num_boxes or self.labels_3d.shape[0] != num_boxes:
            raise ValueError(
                "bboxes_3d, scores_3d and labels_3d must have the same number of boxes, got "
                f"{num_boxes}, {self.scores_3d.shape[0]} and {self.labels_3d.shape[0]}."
            )
        return self
