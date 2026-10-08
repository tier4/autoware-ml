"""
Modules to save raw outputs from a calibration status head.
"""

from __future__ import annotations

from jaxtyping import Float
from pydantic import BaseModel, ConfigDict

import torch


class CalibrationStatusHeadOutputs(BaseModel):
    """
    Dataclass to save the raw outputs of a calibration status classification head.

    Attributes:
      logits: Status logits of every image the head classifies.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    logits: Float[torch.Tensor, "num_images num_classes"]
