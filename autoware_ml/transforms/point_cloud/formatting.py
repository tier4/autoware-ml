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

"""Formatting transforms shared across point-cloud pipelines."""

from typing import Any

import numpy as np

from autoware_ml.transforms.base import BaseTransform


class PreparePointCloudInput(BaseTransform):
    """Split the ``points`` array into per-point feature fields.

    Required keys:
        points: Point cloud array laid out as ``(x, y, z, intensity)`` with the
            per-point time lag at ``time_lag_dim`` when the pipeline carries one.

    Generated keys:
        coord: XYZ coordinates ``(N, 3)``, float32.
        strength: Normalised intensity ``(N, 1)``, float32 in ``[0, 1]``.
        time_lag: Seconds elapsed since each point was captured ``(N, 1)``,
            float32, ``0`` for the current frame. Only when ``time_lag_dim``
            is set.
    """

    _required_keys = ["points"]

    def __init__(self, *, time_lag_dim: int | None) -> None:
        """Initialize the PreparePointCloudInput transform.

        Args:
            time_lag_dim: Column of ``points`` holding the per-point time lag,
                or ``None`` when the cloud carries no time lag.
        """
        if time_lag_dim is not None and time_lag_dim < 4:
            raise ValueError(
                f"time_lag_dim must be at least 4, the first four columns hold (x, y, z, "
                f"intensity), got {time_lag_dim}."
            )
        self.time_lag_dim = time_lag_dim

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Split raw point features into coordinate, intensity and time fields.

        Args:
            input_dict: Sample dictionary containing ``points``.

        Returns:
            Updated sample dictionary with the per-point feature fields.
        """
        points = input_dict.pop("points")
        expected_width = 4 if self.time_lag_dim is None else self.time_lag_dim + 1
        if points.ndim != 2 or points.shape[1] != expected_width:
            raise ValueError(
                f"PreparePointCloudInput requires points with {expected_width} features for "
                f"time_lag_dim {self.time_lag_dim}, got shape {points.shape}."
            )
        input_dict["coord"] = points[:, :3].astype(np.float32)
        input_dict["strength"] = (points[:, 3:4] / 255.0).astype(np.float32)
        if self.time_lag_dim is not None:
            time_lag_dim = self.time_lag_dim
            input_dict["time_lag"] = points[:, time_lag_dim : time_lag_dim + 1].astype(np.float32)
        return input_dict
