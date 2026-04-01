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

"""Range-view transforms for point-cloud segmentation."""

from __future__ import annotations

from typing import Any

import numpy as np

from autoware_ml.transforms.base import BaseTransform
from autoware_ml.transforms.segmentation3d.utils import project_range


class RangeInterpolation(BaseTransform):
    """Fill empty range-image pixels with horizontal interpolation."""

    _required_keys = ["points"]

    def __init__(
        self,
        height: int,
        width: int,
        fov_up: float,
        fov_down: float,
        ignore_index: int,
    ) -> None:
        self.height = height
        self.width = width
        self.fov_up_rad = np.deg2rad(fov_up)
        self.fov_down_rad = np.deg2rad(fov_down)
        self.ignore_index = ignore_index

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        points = input_dict["points"]
        proj_y, proj_x = project_range(
            points, self.height, self.width, self.fov_up_rad, self.fov_down_rad
        )
        input_dict["num_points"] = points.shape[0]

        depth = np.linalg.norm(points[:, :3], ord=2, axis=1)
        order = np.argsort(depth)[::-1]

        proj_image = np.full(
            (self.height, self.width, points.shape[1]), fill_value=-1.0, dtype=np.float32
        )
        proj_mask = np.zeros((self.height, self.width), dtype=bool)
        proj_image[proj_y[order], proj_x[order]] = points[order]
        proj_mask[proj_y[order], proj_x[order]] = True

        proj_labels = None
        if "pts_semantic_mask" in input_dict:
            labels = input_dict["pts_semantic_mask"]
            proj_labels = np.full(
                (self.height, self.width), fill_value=self.ignore_index, dtype=np.int64
            )
            proj_labels[proj_y[order], proj_x[order]] = labels[order]

        # Vectorized interpolation: find all empty pixels that have both
        # a filled left and right neighbor in the same row.  For a 128×4096
        # grid the pixel count is ~500 K, making the element-wise Python loop
        # significantly slower than the numpy approach below.
        inner = proj_mask[:, 1:-1]  # shape (H, W-2)
        can_interp = ~inner & proj_mask[:, :-2] & proj_mask[:, 2:]
        interp_rows, interp_cols_inner = np.where(can_interp)
        interp_cols = interp_cols_inner + 1  # shift back to full-width indices

        if interp_rows.size > 0:
            new_points = 0.5 * (
                proj_image[interp_rows, interp_cols - 1] + proj_image[interp_rows, interp_cols + 1]
            )
            input_dict["points"] = np.concatenate([points, new_points.astype(np.float32)], axis=0)

            if proj_labels is not None:
                left_labels = proj_labels[interp_rows, interp_cols - 1]
                right_labels = proj_labels[interp_rows, interp_cols + 1]
                same = left_labels == right_labels
                new_labels = np.where(same, left_labels, self.ignore_index).astype(np.int64)
                input_dict["pts_semantic_mask"] = np.concatenate(
                    [input_dict["pts_semantic_mask"], new_labels], axis=0
                )

        return input_dict
