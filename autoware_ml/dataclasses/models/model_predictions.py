"""
Modules to save decoded predictions from multi-task models.
"""

from __future__ import annotations

from typing import Sequence

from pydantic import BaseModel, ConfigDict

from autoware_ml.dataclasses.models.calibration_status.predictions import (
    CalibrationStatusPredictions,
)
from autoware_ml.dataclasses.models.detection3d.predictions import Detection3DSamplePredictions
from autoware_ml.dataclasses.models.segmentation3d.predictions import Segmentation3DPredictions


class ModelPredictions(BaseModel):
    """
    Dataclass to save decoded predictions from multi-task models.

    A model fills the predictions of the tasks it predicts and leaves the others unset.

    Attributes:
      detection3d_predictions: Decoded predictions from a 3D detection task.
      segmentation3d_predictions: Decoded predictions from a 3D segmentation task.
      calibration_status_predictions: Decoded predictions from a calibration status task.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    # Decoded predictions across samples.
    detection3d_predictions: Sequence[Detection3DSamplePredictions] | None = None
    segmentation3d_predictions: Segmentation3DPredictions | None = None
    calibration_status_predictions: CalibrationStatusPredictions | None = None
