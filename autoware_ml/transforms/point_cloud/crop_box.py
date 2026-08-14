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

"""Remove points inside fixed volumes, such as the ego vehicle's own body."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from autoware_ml.transforms.base import BaseTransform
from autoware_ml.transforms.point_cloud.assets import resolve_asset_path
from autoware_ml.transforms.point_cloud.ego_motion import pre_correction_points


class CropBoxFilter(BaseTransform):
    """Remove points falling inside any of a set of axis-aligned boxes.

    Autoware crops fixed volumes -- typically the vehicle body and the swept volume of the steered
    wheels -- out of every LiDAR scan before concatenation, as a set of negative crop boxes.
    T4Dataset carries un-cropped concatenated clouds, so the step has to be reapplied to match the
    inference-time point distribution. How much it removes depends entirely on the boxes and the
    sensor layout; for one vehicle it reached 18% of a single LiDAR's points.

    The boxes are inputs, not knowledge this transform holds: deriving them from vehicle dimensions
    is platform-specific, so it is left to whoever produces the configuration. Bundled box sets live
    beside the other per-vehicle assets under ``autoware_ml/configs/assets/<platform>/``.

    Points are expected in the frame the boxes are expressed in -- for vehicle crop boxes that is
    the ego/``base_link`` frame, which is how T4Dataset stores them. On the vehicle the crop mask is
    computed *before* ego-motion correction, so run
    :class:`~autoware_ml.transforms.point_cloud.ego_motion.InvertEgoMotionCorrection` first and this
    transform will decide on the pre-correction coordinates it attaches. Without it the boxes are
    applied to the corrected coordinates directly, which is measurably worse: reconstructing a
    recorded concatenated cloud from an unfiltered T4Dataset, voxel IoU at 0.12 m drops from 0.877
    to 0.702.
    """

    _required_keys = ["points"]

    def __init__(self, *, crop_boxes: Sequence[Sequence[float]]) -> None:
        """Initialize the CropBoxFilter transform.

        Args:
            crop_boxes: Boxes to remove, each ``[x_min, y_min, z_min, x_max, y_max, z_max]``,
                expressed in the same frame as ``points``.
        """
        self.crop_boxes = np.asarray(crop_boxes, dtype=np.float32).reshape(-1, 6)
        if self.crop_boxes.size == 0:
            raise ValueError("crop_boxes must contain at least one box")
        lower, upper = self.crop_boxes[:, :3], self.crop_boxes[:, 3:]
        if np.any(lower > upper):
            raise ValueError(
                "each crop box must be [x_min, y_min, z_min, x_max, y_max, z_max] with "
                f"min <= max; got {self.crop_boxes.tolist()}"
            )

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Drop points inside any configured box, keeping aligned per-point arrays consistent."""
        points = np.asarray(input_dict["points"], dtype=np.float32)
        # The vehicle crops the raw sweep, so decide on pre-correction coordinates when the sample
        # carries them. Surviving points keep their original corrected coordinates either way.
        decision_xyz = pre_correction_points(input_dict, points)
        if decision_xyz is None:
            decision_xyz = points[:, :3]
        inside = np.zeros(points.shape[0], dtype=bool)
        for x_min, y_min, z_min, x_max, y_max, z_max in self.crop_boxes:
            inside |= (
                (decision_xyz[:, 0] >= x_min)
                & (decision_xyz[:, 0] <= x_max)
                & (decision_xyz[:, 1] >= y_min)
                & (decision_xyz[:, 1] <= y_max)
                & (decision_xyz[:, 2] >= z_min)
                & (decision_xyz[:, 2] <= z_max)
            )

        keep_mask = ~inside
        for key, value in list(input_dict.items()):
            if (
                isinstance(value, np.ndarray)
                and value.ndim > 0
                and value.shape[0] == points.shape[0]
            ):
                input_dict[key] = value[keep_mask]
        return input_dict


def load_crop_boxes(path: str) -> list[list[float]]:
    """Read a crop-box list from a YAML asset.

    The file holds a ``crop_boxes`` list of ``[x_min, y_min, z_min, x_max, y_max, z_max]`` entries.
    A relative path resolves under the bundled asset root, so
    ``load_crop_boxes("<platform>/crop_boxes.param.yaml")`` finds a bundled set. How a given
    platform's boxes were derived is recorded in the asset file itself.

    Args:
        path: Path to the YAML file, absolute or relative to the bundled asset root.

    Returns:
        The boxes, ready to pass to :class:`CropBoxFilter`.
    """
    resolved = resolve_asset_path(path)
    document = yaml.safe_load(Path(resolved).read_text())
    boxes = (document or {}).get("crop_boxes")
    if not boxes:
        raise ValueError(f"No 'crop_boxes' entry in {resolved}")
    return [[float(value) for value in box] for box in boxes]
