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

"""Point-cloud geometric transforms."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from autoware_ml.transforms.base import BaseTransform


class RandomRotate(BaseTransform):
    """Randomly rotate point coordinates around the z axis."""

    _required_keys = ["coord"]

    def __init__(
        self,
        angle: Sequence[float],
        axis: str = "z",
        center: Sequence[float] | None = None,
        p: float = 0.5,
    ) -> None:
        """Initialize the random rotation transform.

        Args:
            angle: Minimum and maximum rotation angles in turns.
            axis: Rotation axis. Only ``z`` is supported.
            center: Optional rotation center.
            p: Probability of applying the transform.
        """
        self.angle = angle
        self.axis = axis
        self.center = np.asarray(center, dtype=np.float32) if center is not None else None
        self.p = p

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Rotate point coordinates around the configured axis.

        Args:
            input_dict: Sample dictionary updated in place.

        Returns:
            Updated sample dictionary.
        """
        angle = np.random.uniform(self.angle[0], self.angle[1]) * np.pi
        cos, sin = np.cos(angle), np.sin(angle)
        if self.axis != "z":
            raise ValueError("Only z-axis rotation is supported in the point-cloud pipeline.")
        rotation = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]], dtype=np.float32)
        coord = input_dict["coord"]
        center = self.center if self.center is not None else coord.mean(axis=0)
        input_dict["coord"] = (coord - center) @ rotation.T + center
        if "normal" in input_dict:
            input_dict["normal"] = input_dict["normal"] @ rotation.T
        return input_dict


class RandomRotateTargetAngle(BaseTransform):
    """Rotate point coordinates by one sampled discrete target angle."""

    _required_keys = ["coord"]

    def __init__(
        self,
        angle: Sequence[float],
        axis: str = "z",
        center: Sequence[float] | None = None,
        p: float = 0.5,
    ) -> None:
        """Initialize the target-angle rotation transform.

        Args:
            angle: Candidate rotation angles in turns.
            axis: Rotation axis. Only ``z`` is supported.
            center: Optional rotation center.
            p: Probability of applying the transform.
        """
        self.angle = list(angle)
        self.axis = axis
        self.center = np.asarray(center, dtype=np.float32) if center is not None else None
        self.p = p

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Rotate the sample by one selected target angle.

        Args:
            input_dict: Sample dictionary updated in place.

        Returns:
            Updated sample dictionary.
        """
        angle = float(np.random.choice(self.angle)) * np.pi
        if self.axis != "z":
            raise ValueError("Only z-axis rotation is supported in the point-cloud pipeline.")
        cos, sin = np.cos(angle), np.sin(angle)
        rotation = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]], dtype=np.float32)
        coord = input_dict["coord"]
        center = self.center if self.center is not None else coord.mean(axis=0)
        input_dict["coord"] = (coord - center) @ rotation.T + center
        if "normal" in input_dict:
            input_dict["normal"] = input_dict["normal"] @ rotation.T
        return input_dict


class RandomScale(BaseTransform):
    """Randomly scale point coordinates by a sampled global factor."""

    _required_keys = ["coord"]

    def __init__(self, scale: Sequence[float]) -> None:
        """Initialize the random scaling transform.

        Args:
            scale: Minimum and maximum random scaling factors.
        """
        self.scale = scale

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Scale point coordinates by a random factor.

        Args:
            input_dict: Sample dictionary updated in place.

        Returns:
            Updated sample dictionary.
        """
        scale = np.random.uniform(self.scale[0], self.scale[1], 1).astype(np.float32)
        input_dict["coord"] = input_dict["coord"] * scale
        return input_dict


class RandomFlip(BaseTransform):
    """Randomly flip point coordinates across BEV axes."""

    _required_keys = ["coord"]

    def __init__(self, p: float = 0.5) -> None:
        """Initialize the random flip transform.

        Args:
            p: Probability of applying each axis flip.
        """
        self.p = p

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Flip point coordinates across x and y axes.

        Args:
            input_dict: Sample dictionary updated in place.

        Returns:
            Updated sample dictionary.
        """
        if np.random.rand() < self.p:
            input_dict["coord"][:, 0] = -input_dict["coord"][:, 0]
        if np.random.rand() < self.p:
            input_dict["coord"][:, 1] = -input_dict["coord"][:, 1]
        return input_dict
