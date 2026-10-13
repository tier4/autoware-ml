"""
Modules to save decoded predictions from a calibration status model.
"""

from __future__ import annotations

from jaxtyping import Float
from pydantic import BaseModel, ConfigDict

import torch


class CalibrationStatusPredictions(BaseModel):
    """
    Dataclass to save the decoded predictions of a calibration status model.

    Attributes:
      probabilities: Status probabilities of every image.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    probabilities: Float[torch.Tensor, "num_images num_classes"]
