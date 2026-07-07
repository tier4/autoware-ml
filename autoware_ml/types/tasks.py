from enum import StrEnum


class TaskType(StrEnum):
    """Enum for different types of tasks."""

    CLASSIFICATION2D = "Classification2D"
    DETECTION3D = "Detection3D"
    SEGMENTATION3D = "Segmentation3D"
