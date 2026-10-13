"""
Modules to save raw outputs from a segmentation3d head.
"""

from __future__ import annotations

from jaxtyping import Float
from pydantic import BaseModel, ConfigDict, InstanceOf

import torch


class Segmentation3DHeadOutputs(BaseModel):
    """
    Dataclass to save the raw outputs of a 3D segmentation head.

    Attributes:
      logits: Class logits of every point or voxel the head predicts for.
      auxiliary_features: Feature maps the auxiliary heads of the model read, empty for a model
        without auxiliary heads.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    logits: Float[torch.Tensor, "num_points num_classes"]
    auxiliary_features: tuple[InstanceOf[torch.Tensor], ...] = ()
