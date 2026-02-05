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

from typing import Any, Dict, List

import numpy as np
import numpy.typing as npt

from autoware_ml.transforms.base import BaseTransform


class CropBoxOuter(BaseTransform):
    """Remove points that are OUTSIDE a 3D bounding box (Keep Inside).

    Required keys:
        - points: (N, C) float32 point cloud array where first 3 columns are [x, y, z].

    Optional keys:
        - None

    Generated keys:
        - points: Modified in-place with filtered points (only inside the box).

    Args:
        crop_box: List of 6 floats defining the box [x_min, y_min, z_min, x_max, y_max, z_max].
    """

    _required_keys = ["points"]

    def __init__(self, crop_box: List[float]):
        super().__init__()
        if len(crop_box) != 6:
            raise ValueError(f"crop_box must have 6 elements, got {len(crop_box)}")
        self.crop_box = np.array(crop_box, dtype=np.float32)

    def transform(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Remove points outside the box.

        Args:
            input_dict: Dictionary with 'points' key containing (N, C) float32 array.

        Returns:
            Dictionary with points inside the box.
        """
        points: npt.NDArray[np.float32] = input_dict["points"]

        x_min, y_min, z_min, x_max, y_max, z_max = self.crop_box

        # Keep points INSIDE the box
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
    """Remove points that are INSIDE a 3D bounding box (Keep Outside).

    Typically used to remove ego vehicle points from the point cloud.

    Required keys:
        - points: (N, C) float32 point cloud array where first 3 columns are [x, y, z].

    Optional keys:
        - None

    Generated keys:
        - points: Modified in-place with filtered points (only outside the box).

    Args:
        crop_box: List of 6 floats defining the box [x_min, y_min, z_min, x_max, y_max, z_max].
    """

    _required_keys = ["points"]

    def __init__(self, crop_box: List[float]):
        super().__init__()
        if len(crop_box) != 6:
            raise ValueError(f"crop_box must have 6 elements, got {len(crop_box)}")
        self.crop_box = np.array(crop_box, dtype=np.float32)

    def transform(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Remove points inside the box.

        Args:
            input_dict: Dictionary with 'points' key containing (N, C) float32 array.

        Returns:
            Dictionary with points outside the box.
        """
        points: npt.NDArray[np.float32] = input_dict["points"]

        x_min, y_min, z_min, x_max, y_max, z_max = self.crop_box

        # Keep points OUTSIDE the box
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
