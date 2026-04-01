# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Point-cloud cropping and centering transforms."""

from __future__ import annotations

from collections.abc import Sequence
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


class PointsRangeFilter(BaseTransform):
    """Keep only points within a configured point-cloud range."""

    _required_keys = ["points"]

    def __init__(self, point_cloud_range: list[float]) -> None:
        """Initialize the points range filter.

        Args:
            point_cloud_range: Bounds ``[x_min, y_min, z_min, x_max, y_max, z_max]``.
        """
        self.point_cloud_range = np.asarray(point_cloud_range, dtype=np.float32)

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Filter points to the configured spatial range."""
        points: npt.NDArray[np.float32] = input_dict["points"]
        lower = self.point_cloud_range[:3]
        upper = self.point_cloud_range[3:]
        mask = ((points[:, :3] >= lower) & (points[:, :3] <= upper)).all(axis=1)
        for key, value in list(input_dict.items()):
            if (
                isinstance(value, np.ndarray)
                and value.ndim > 0
                and value.shape[0] == points.shape[0]
            ):
                input_dict[key] = value[mask]
        return input_dict


class PointClip(BaseTransform):
    """Clamp point coordinates to a configured spatial range."""

    _required_keys = ["coord"]

    def __init__(self, point_cloud_range: Sequence[float]) -> None:
        """Initialize the point clipping transform.

        Args:
            point_cloud_range: Point cloud bounds used for clipping.
        """
        self.point_cloud_range = np.asarray(point_cloud_range, dtype=np.float32)

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Clamp point coordinates to the configured range.

        Args:
            input_dict: Sample dictionary updated in place.

        Returns:
            Updated sample dictionary.
        """
        input_dict["coord"] = np.clip(
            input_dict["coord"],
            a_min=self.point_cloud_range[:3],
            a_max=self.point_cloud_range[3:],
        )
        return input_dict


class CenterShift(BaseTransform):
    """Center point coordinates by subtracting their spatial midpoint."""

    _required_keys = ["coord"]

    def __init__(self, apply_z: bool = True) -> None:
        """Initialize the centering transform.

        Args:
            apply_z: Whether to center the z coordinate as well.
        """
        self.apply_z = apply_z

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Center the point cloud around the origin.

        Args:
            input_dict: Sample dictionary updated in place.

        Returns:
            Updated sample dictionary.
        """
        coord = input_dict["coord"]
        center = (coord.min(axis=0) + coord.max(axis=0)) / 2
        if not self.apply_z:
            center[2] = 0.0
        input_dict["coord"] = coord - center
        return input_dict


class SphereCrop(BaseTransform):
    """Keep the points closest to a selected crop center."""

    _required_keys = ["coord"]

    def __init__(self, point_max: int, mode: str = "random") -> None:
        """Initialize the sphere crop transform.

        Args:
            point_max: Maximum number of points kept after cropping.
            mode: Crop-center strategy, either ``random`` or ``center``.
        """
        self.point_max = point_max
        self.mode = mode

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Keep the nearest points around the selected crop center.

        Args:
            input_dict: Sample dictionary updated in place.

        Returns:
            Updated sample dictionary.
        """
        if input_dict["coord"].shape[0] <= self.point_max:
            return input_dict
        if self.mode == "random":
            center = input_dict["coord"][np.random.randint(0, input_dict["coord"].shape[0])]
        elif self.mode == "center":
            center = input_dict["coord"][input_dict["coord"].shape[0] // 2]
        else:
            raise ValueError("SphereCrop mode must be 'random' or 'center'.")
        point_count = input_dict["coord"].shape[0]
        distances = np.linalg.norm(input_dict["coord"] - center, axis=1)
        keep = np.sort(np.argsort(distances)[: self.point_max])
        for key, value in list(input_dict.items()):
            if isinstance(value, np.ndarray) and value.shape[0] == point_count:
                input_dict[key] = value[keep]
        return input_dict
