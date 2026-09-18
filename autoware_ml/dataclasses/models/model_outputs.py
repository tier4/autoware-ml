"""
Modules to save raw outputs from multi-task models.
"""

from pydantic import BaseModel, ConfigDict
from autoware_ml.dataclasses.models.detection3d.head_outputs import Detection3DHeadOutputs


class ModelOutputs(BaseModel):
    """
    Dataclass to save raw outputs from multi-task models.

    Attributes:
        detection3d_head_outputs: Raw outputs from a 3D detection head.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    detection3d_head_outputs: Detection3DHeadOutputs | None

    # TODO (Kok Seang): Add outputs for other tasks in the future.
