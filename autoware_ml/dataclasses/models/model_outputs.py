"""
Modules to save raw outputs from multi-task models.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from autoware_ml.dataclasses.models.calibration_status.head_outputs import (
    CalibrationStatusHeadOutputs,
)
from autoware_ml.dataclasses.models.detection3d.head_outputs import Detection3DHeadOutputs
from autoware_ml.dataclasses.models.segmentation3d.head_outputs import Segmentation3DHeadOutputs


class ModelOutputs(BaseModel):
    """
    Dataclass to save raw outputs from multi-task models.

    A model fills the outputs of the tasks it predicts and leaves the others unset.

    Attributes:
        detection3d_head_outputs: Raw outputs from a 3D detection head.
        segmentation3d_head_outputs: Raw outputs from a 3D segmentation head.
        calibration_status_head_outputs: Raw outputs from a calibration status head.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    detection3d_head_outputs: Detection3DHeadOutputs | None = None
    segmentation3d_head_outputs: Segmentation3DHeadOutputs | None = None
    calibration_status_head_outputs: CalibrationStatusHeadOutputs | None = None

    def detection3d(self) -> Detection3DHeadOutputs:
        """
        Read the 3D detection outputs.

        Raises:
            ValueError: If the model predicts no 3D detection.
        """
        if self.detection3d_head_outputs is None:
            raise ValueError("The model outputs carry no 3D detection head outputs.")
        return self.detection3d_head_outputs

    def segmentation3d(self) -> Segmentation3DHeadOutputs:
        """
        Read the 3D segmentation outputs.

        Raises:
            ValueError: If the model predicts no 3D segmentation.
        """
        if self.segmentation3d_head_outputs is None:
            raise ValueError("The model outputs carry no 3D segmentation head outputs.")
        return self.segmentation3d_head_outputs

    def calibration_status(self) -> CalibrationStatusHeadOutputs:
        """
        Read the calibration status outputs.

        Raises:
            ValueError: If the model predicts no calibration status.
        """
        if self.calibration_status_head_outputs is None:
            raise ValueError("The model outputs carry no calibration status head outputs.")
        return self.calibration_status_head_outputs
