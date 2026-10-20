from enum import StrEnum


class TaskType(StrEnum):
    """Enum for different types of tasks."""

    CLASSIFICATION2D = "classification2d"
    DETECTION3D = "detection3d"
    SEGMENTATION3D = "segmentation3d"
