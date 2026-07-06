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

"""Point-cloud perturbation transforms."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from autoware_ml.transforms.base import BaseTransform


class RandomJitter(BaseTransform):
    """Perturb point coordinates with clipped Gaussian noise."""

    _required_keys = ["coord"]

    def __init__(self, *, p: float | None = None, sigma: float, clip: float) -> None:
        """Initialize the RandomJitter transform.

        Args:
            p: Probability of applying the transform (``None`` means always apply).
            sigma: Standard deviation of the Gaussian noise.
            clip: Maximum absolute jitter applied per coordinate.
        """
        self.p = p
        self.sigma = sigma
        self.clip = clip

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Perturb point coordinates with Gaussian noise.

        Args:
            input_dict: Sample dictionary updated in place.

        Returns:
            Updated sample dictionary.
        """
        noise = np.clip(
            self.sigma * np.random.randn(input_dict["coord"].shape[0], 3),
            -self.clip,
            self.clip,
        ).astype(np.float32)
        input_dict["coord"] = input_dict["coord"] + noise
        return input_dict


class RandomShift(BaseTransform):
    """Translate point coordinates by a sampled per-axis offset."""

    _required_keys = ["coord"]

    def __init__(self, *, p: float | None = None, shift: Sequence[float]) -> None:
        """Initialize the RandomShift transform.

        Args:
            p: Probability of applying the transform (``None`` means always apply).
            shift: Maximum absolute translation per axis.
        """
        self.p = p
        self.shift = np.asarray(shift, dtype=np.float32)

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Shift coordinates by a random translation.

        Args:
            input_dict: Sample dictionary updated in place.

        Returns:
            Updated sample dictionary.
        """
        translation = np.random.uniform(-self.shift, self.shift).astype(np.float32)
        input_dict["coord"] = input_dict["coord"] + translation
        return input_dict
