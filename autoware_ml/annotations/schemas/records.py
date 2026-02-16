from typing import Iterable

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict


class AnnotationTableRecord(BaseModel):
    """
    Data class to save a record for each column in the annotation table.
    :param scenario_id: Scenario id.
    :param sample_id: Sample id.
    :param location: Location of the vehicle.
    :param vehicle_type: Type of the vehicle.
    :param bbox_3d: List of 3D bounding boxes with center_x, center_y, center_z, length, width, height, yaw, velocity_x, velocity_y.
    :param bbox_2d: List of 2D bounding boxes with center_x, center_y, width, height.
    """
    
    # Set model config to frozen
    model_config = ConfigDict(frozen=True, strict=True)

    scenario_id: str
    sample_id: str
    location: str
    vehicle_type: str
    bbox_3d: Iterable[npt.NDArray[np.float64]]
    bbox_2d: Iterable[npt.NDArray[np.float64]]
