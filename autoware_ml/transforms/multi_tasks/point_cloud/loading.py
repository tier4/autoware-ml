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

"""Point-cloud loading transforms to support MultiTaskGTSample."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from autoware_ml.transforms.multi_tasks.base import MultiTaskBaseTransform
from autoware_ml.datamodule.multi_tasks.dataclasses.multi_task_samples import MultiTaskGTSample


class LoadPointsFromFile(MultiTaskBaseTransform):
    """Load point clouds from a lidar file path stored in sample metadata."""

    _required_keys = ["lidar_point_cloud_samples"]

    def __init__(self, load_dim: int = 5, use_dim: Sequence[int] | int = (0, 1, 2, 3)) -> None:
        """Initialize the point-cloud loader.

        Args:
            load_dim: Number of features stored per point in the source file.
            use_dim: Selected feature dimensions preserved in the loaded tensor.
        """
        super().__init__(probability=None)
        self.load_dim = load_dim
        self.use_dim = use_dim

    def transform(self, multi_task_gt_sample: MultiTaskGTSample) -> MultiTaskGTSample:
        """Load point data from the configured lidar file.

        Args:
            multi_task_gt_sample: MultiTaskGTSample instance containing ``lidar_path``.

        Returns:
            Updated MultiTaskGTSample instance with a loaded ``points`` array.
        """
        # Load the first index of the point cloud file, and reshape it to (N, load_dim)
        if not multi_task_gt_sample.lidar_point_cloud_samples:
            raise ValueError("No lidar point cloud samples found in the MultiTaskGTSample.")

        current_lidar_point_path = multi_task_gt_sample.lidar_point_cloud_samples[
            0
        ].point_cloud_path
        points = np.fromfile(current_lidar_point_path, dtype=np.float32).reshape(-1, self.load_dim)

        if isinstance(self.use_dim, int):
            points = points[:, : self.use_dim]
        else:
            points = points[:, self.use_dim]

        return MultiTaskGTSample(
            lidar_point_cloud_samples=multi_task_gt_sample.lidar_point_cloud_samples,
            detection3d_gt_sample=multi_task_gt_sample.detection3d_gt_sample,
            point_cloud_features=points,
            segmentation3d_gt_sample=multi_task_gt_sample.segmentation3d_gt_sample,
        )
