"""Point-cloud cropping and centering transforms."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from autoware_ml.transforms.base import BaseTransform


class CropBoxOuter(BaseTransform):
    """Remove points that are outside a 3D bounding box."""

    _required_keys = ["points"]

    def __init__(self, crop_box: list[float]):
        """Initialize the outer-distance crop transform.

        Args:
            crop_box: Box bounds ``[x_min, y_min, z_min, x_max, y_max, z_max]``.
        """
        super().__init__()
        if len(crop_box) != 6:
            raise ValueError(f"crop_box must have 6 elements, got {len(crop_box)}")
        self.crop_box = np.asarray(crop_box, dtype=np.float32)

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Keep only points inside the configured box."""
        points: npt.NDArray[np.float32] = input_dict["points"]
        x_min, y_min, z_min, x_max, y_max, z_max = self.crop_box
        mask = (
            (points[:, 0] >= x_min)
            & (points[:, 0] <= x_max)
            & (points[:, 1] >= y_min)
            & (points[:, 1] <= y_max)
            & (points[:, 2] >= z_min)
            & (points[:, 2] <= z_max)
        )
        input_dict["points"] = points[mask]
        return input_dict


class CropBoxInner(BaseTransform):
    """Remove points that are inside a 3D bounding box."""

    _required_keys = ["points"]

    def __init__(self, crop_box: list[float]):
        """Initialize the inner-distance crop transform.

        Args:
            crop_box: Box bounds ``[x_min, y_min, z_min, x_max, y_max, z_max]``.
        """
        super().__init__()
        if len(crop_box) != 6:
            raise ValueError(f"crop_box must have 6 elements, got {len(crop_box)}")
        self.crop_box = np.asarray(crop_box, dtype=np.float32)

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Keep only points outside the configured box."""
        points: npt.NDArray[np.float32] = input_dict["points"]
        x_min, y_min, z_min, x_max, y_max, z_max = self.crop_box
        mask = (
            (points[:, 0] < x_min)
            | (points[:, 0] > x_max)
            | (points[:, 1] < y_min)
            | (points[:, 1] > y_max)
            | (points[:, 2] < z_min)
            | (points[:, 2] > z_max)
        )
        input_dict["points"] = points[mask]
        return input_dict
