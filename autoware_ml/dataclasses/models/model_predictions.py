"""
Modules to save decoded predictions from multi-task models.
"""

from typing import Sequence

from pydantic import BaseModel, ConfigDict

from autoware_ml.dataclasses.models.detection3d.predictions import Detection3DSamplePredictions


class ModelPredictions(BaseModel):
    """
    Dataclass to save decoded predictions from multi-task models.

    Attributes:
      detection3d_predictions: Decoded predictions from a 3D detection task.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    # Decoded predictions across samples.
    detection3d_predictions: Sequence[Detection3DSamplePredictions] | None

    # TODO(Kok Seang): Add predictions for other tasks in the future.
