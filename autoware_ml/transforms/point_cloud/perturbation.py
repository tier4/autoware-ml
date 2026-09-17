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

"""Point cloud perturbation transforms."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.geometry.points.base_points import BasePoints
from autoware_ml.transforms.base import BaseTransform
from autoware_ml.types.geometry import PointFeatureName


class RandomJitter(BaseTransform):
    """Perturb point coordinates with clipped Gaussian noise."""

    _required_keys = ["point_cloud_data"]

    def __init__(self, sigma: float, clip: float, probability: float | None = None) -> None:
        """Initialize the RandomJitter transform.

        Args:
            sigma: Standard deviation of the Gaussian noise.
            clip: Maximum absolute jitter applied per coordinate.
            probability: Probability of applying the transform, None to always apply it.
        """
        super().__init__(probability=probability)
        self.sigma = sigma
        self.clip = clip

    def transform(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        """Perturb point coordinates with Gaussian noise."""
        # This is checked in the _validate_required_keys()
        point_cloud_data: BasePoints = (
            model_gt_sample.point_cloud_data  # type: ignore[reportOptionalMemberAccess]
        )

        noise = torch.clamp(
            self.sigma * torch.randn_like(point_cloud_data.coords), -self.clip, self.clip
        )
        point_cloud_data.coords = point_cloud_data.coords + noise
        return model_gt_sample


class RandomStrengthJitter(BaseTransform):
    """Perturb the normalized intensity with a random gamma, scale, and shift.

    Applies ``clip(intensity ** gamma * scale + shift, 0, 1)`` with parameters drawn uniformly
    per sample, emulating reflectivity calibration differences between sensors.
    """

    _required_keys = ["point_cloud_data"]

    def __init__(
        self,
        gamma_range: Sequence[float],
        scale_range: Sequence[float],
        shift_range: Sequence[float],
        probability: float | None = None,
    ) -> None:
        """Initialize the RandomStrengthJitter transform.

        Args:
            gamma_range: Min and max exponent applied to the normalized intensity.
            scale_range: Min and max multiplicative factor.
            shift_range: Min and max additive offset.
            probability: Probability of applying the transform, None to always apply it.
        """
        super().__init__(probability=probability)
        for name, bounds in (
            ("gamma_range", gamma_range),
            ("scale_range", scale_range),
            ("shift_range", shift_range),
        ):
            if len(bounds) != 2 or bounds[0] > bounds[1]:
                raise ValueError(f"{name} must be an ascending [min, max] pair, got {bounds}.")
        if gamma_range[0] <= 0.0:
            raise ValueError(f"gamma_range values must be positive, got {gamma_range}.")
        self.gamma_range = tuple(gamma_range)
        self.scale_range = tuple(scale_range)
        self.shift_range = tuple(shift_range)

    def sample_uniform(self, bounds: Sequence[float]) -> float:
        """Draw one value from the given bounds.

        Args:
            bounds: Ascending min and max pair.

        Returns:
            float: Value drawn uniformly between the bounds.
        """
        return float(torch.empty(()).uniform_(bounds[0], bounds[1]).item())

    def transform(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        """Apply the sampled gamma, scale, and shift to the intensity of every point."""
        # This is checked in the _validate_required_keys()
        point_cloud_data: BasePoints = (
            model_gt_sample.point_cloud_data  # type: ignore[reportOptionalMemberAccess]
        )

        gamma = self.sample_uniform(self.gamma_range)
        scale = self.sample_uniform(self.scale_range)
        shift = self.sample_uniform(self.shift_range)
        intensity = point_cloud_data.feature(PointFeatureName.INTENSITY)
        point_cloud_data.set_feature(
            PointFeatureName.INTENSITY,
            torch.clamp(torch.pow(intensity, gamma) * scale + shift, 0.0, 1.0),
        )
        return model_gt_sample
