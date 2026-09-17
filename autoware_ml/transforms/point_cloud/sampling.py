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

"""Point cloud subsampling transforms."""

from __future__ import annotations

import torch

from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.geometry.points.base_points import BasePoints
from autoware_ml.transforms.base import BaseTransform


class RandomDropout(BaseTransform):
    """Randomly remove a fraction of the points, keeping their order."""

    _required_keys = ["point_cloud_data"]

    def __init__(self, dropout_ratio: float, probability: float | None = None) -> None:
        """Initialize the RandomDropout transform.

        Args:
            dropout_ratio: Fraction of points removed when the dropout is applied.
            probability: Probability of applying the transform, None to always apply it.
        """
        super().__init__(probability=probability)
        if not 0.0 <= dropout_ratio < 1.0:
            raise ValueError(f"dropout_ratio must be in [0.0, 1.0), got {dropout_ratio}.")
        self.dropout_ratio = dropout_ratio

    def transform(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        """Drop a random subset of the points and of their semantic labels."""
        # This is checked in the _validate_required_keys()
        point_cloud_data: BasePoints = (
            model_gt_sample.point_cloud_data  # type: ignore[reportOptionalMemberAccess]
        )
        num_points = len(point_cloud_data)
        if not num_points:
            return model_gt_sample

        num_kept_points = max(1, int(num_points * (1.0 - self.dropout_ratio)))
        keep_mask = torch.zeros(num_points, dtype=torch.bool, device=point_cloud_data.device)
        keep_mask[torch.randperm(num_points, device=point_cloud_data.device)[:num_kept_points]] = (
            True
        )

        return model_gt_sample.keep_points(keep_mask)
