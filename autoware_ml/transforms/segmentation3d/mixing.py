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

"""Sample mixing transforms for point-cloud segmentation."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from autoware_ml.transforms.base import BaseTransform, TransformsCompose
from autoware_ml.transforms.segmentation3d.utils import project_range


class FrustumMix(BaseTransform):
    """Mix two point clouds along frustum-aligned stripes."""

    _required_keys = ["points", "pts_semantic_mask"]

    def __init__(
        self,
        height: int,
        width: int,
        fov_up: float,
        fov_down: float,
        num_areas: list[int],
        pre_transform: TransformsCompose | None = None,
        prob: float = 1.0,
    ) -> None:
        self.height = height
        self.width = width
        self.fov_up_rad = np.deg2rad(fov_up)
        self.fov_down_rad = np.deg2rad(fov_down)
        self.num_areas = num_areas
        self.pre_transform = pre_transform
        self.p = prob

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        mix_sample = self._prepare_mix_sample()
        mix_points = mix_sample["points"]
        mix_labels = mix_sample["pts_semantic_mask"]

        if np.random.rand() < 0.5:
            input_dict["points"], input_dict["pts_semantic_mask"] = self._mix_vertical(
                input_dict["points"], input_dict["pts_semantic_mask"], mix_points, mix_labels
            )
        else:
            input_dict["points"], input_dict["pts_semantic_mask"] = self._mix_horizontal(
                input_dict["points"], input_dict["pts_semantic_mask"], mix_points, mix_labels
            )
        return input_dict

    def _prepare_mix_sample(self) -> dict[str, Any]:
        if self.context is None:
            raise RuntimeError("FrustumMix requires pipeline context to sample secondary inputs")
        return self.context.sample_secondary(pre_transform=self.pre_transform)

    def _mix_vertical(
        self,
        points: npt.NDArray[np.float32],
        labels: npt.NDArray[np.int64],
        mix_points: npt.NDArray[np.float32],
        mix_labels: npt.NDArray[np.int64],
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        proj_y, _ = project_range(
            points, self.height, self.width, self.fov_up_rad, self.fov_down_rad
        )
        mix_proj_y, _ = project_range(
            mix_points, self.height, self.width, self.fov_up_rad, self.fov_down_rad
        )

        num_areas = int(np.random.choice(self.num_areas))
        row_bins = np.linspace(0, self.height, num_areas + 1, dtype=np.int64)
        mixed_points: list[npt.NDArray[np.float32]] = []
        mixed_labels: list[npt.NDArray[np.int64]] = []

        for area_index in range(num_areas):
            start_row = row_bins[area_index]
            end_row = row_bins[area_index + 1]
            if area_index % 2 == 0:
                mask = (proj_y >= start_row) & (proj_y < end_row)
                mixed_points.append(points[mask])
                mixed_labels.append(labels[mask])
            else:
                mask = (mix_proj_y >= start_row) & (mix_proj_y < end_row)
                mixed_points.append(mix_points[mask])
                mixed_labels.append(mix_labels[mask])

        return np.concatenate(mixed_points, axis=0), np.concatenate(mixed_labels, axis=0)

    def _mix_horizontal(
        self,
        points: npt.NDArray[np.float32],
        labels: npt.NDArray[np.int64],
        mix_points: npt.NDArray[np.float32],
        mix_labels: npt.NDArray[np.int64],
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        _, proj_x = project_range(
            points, self.height, self.width, self.fov_up_rad, self.fov_down_rad
        )
        _, mix_proj_x = project_range(
            mix_points, self.height, self.width, self.fov_up_rad, self.fov_down_rad
        )

        start_col = np.random.randint(0, self.width // 2)
        end_col = start_col + self.width // 2
        keep_mask = (proj_x < start_col) | (proj_x >= end_col)
        mix_mask = (mix_proj_x >= start_col) & (mix_proj_x < end_col)

        out_points = np.concatenate([points[keep_mask], mix_points[mix_mask]], axis=0)
        out_labels = np.concatenate([labels[keep_mask], mix_labels[mix_mask]], axis=0)
        return out_points, out_labels


class InstanceCopy(BaseTransform):
    """Copy selected semantic instances from a second point cloud."""

    _required_keys = ["points", "pts_semantic_mask"]

    def __init__(
        self,
        instance_classes: list[int],
        pre_transform: TransformsCompose | None = None,
        prob: float = 1.0,
    ) -> None:
        self.instance_classes = instance_classes
        self.pre_transform = pre_transform
        self.p = prob

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        mix_sample = self._prepare_mix_sample()
        mix_points = mix_sample["points"]
        mix_labels = mix_sample["pts_semantic_mask"]

        point_parts = [input_dict["points"]]
        label_parts = [input_dict["pts_semantic_mask"]]
        for class_id in self.instance_classes:
            class_mask = mix_labels == class_id
            point_parts.append(mix_points[class_mask])
            label_parts.append(mix_labels[class_mask])

        input_dict["points"] = np.concatenate(point_parts, axis=0)
        input_dict["pts_semantic_mask"] = np.concatenate(label_parts, axis=0)
        return input_dict

    def _prepare_mix_sample(self) -> dict[str, Any]:
        if self.context is None:
            raise RuntimeError("InstanceCopy requires pipeline context to sample secondary inputs")
        return self.context.sample_secondary(pre_transform=self.pre_transform)
