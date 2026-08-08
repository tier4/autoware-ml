"""
Modules to save decoded predictions from a detection3d head.
"""

from jaxtyping import Float32, Int64
from pydantic import BaseModel, ConfigDict

import torch


class Detection3DSamplePredictions(BaseModel):
    """
    Dataclass to save decoded predictions from a 3D detection model for a sample.

    Attributes:
      model_name: Name of the model.
      dataset_name: Name of the dataset.
      max_sweeps: Maximum number of sweeps to include.
      sample_steps: Number of steps to sample.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    # 7 (center_x, center_y, center_z, length, width, height, heading) if not velocity else
    # 9 (center_x, center_y, center_z, length, width, height, heading, velocity_x, velocity_y)
    bboxes_3d: Float32[torch.Tensor, "num_boxes num_bbox_params"]
    scores_3d: Float32[torch.Tensor, " num_boxes"]
    labels_3d: Int64[torch.Tensor, " num_boxes"]
