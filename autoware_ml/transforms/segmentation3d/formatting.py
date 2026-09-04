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

"""Formatting transforms for point-cloud segmentation pipelines."""

from typing import Any

import numpy as np

from autoware_ml.transforms.base import BaseTransform


class PreparePointSegInput(BaseTransform):
    """Build the ``segment`` field from the current-frame semantic mask.

    Must be preceded by ``PreparePointCloudInput``. Labels exist only for the
    current frame, which the point loader places as the leading block of the
    cloud. Points appended from earlier sweeps receive ``ignore_index`` so they
    contribute to the geometry but never to the loss or the metrics.

    Required keys:
        coord: Point coordinates ``(N, 3)``.
        time_lag: Per-point time lag ``(N, 1)``, ``0`` for the current frame.
            Only when ``time_lag_dim`` is set.
        num_current_points: Size of the leading current-frame block.

    Optional keys:
        pts_semantic_mask: Semantic label per current-frame point. Defaults to
            ``ignore_index`` everywhere when absent.

    Generated keys:
        segment: Per-point semantic labels ``(N,)``, int64.
    """

    _optional_keys = ["pts_semantic_mask"]

    def __init__(self, *, ignore_index: int, time_lag_dim: int | None) -> None:
        """Initialize the PreparePointSegInput transform.

        Args:
            ignore_index: Label assigned to points without supervision.
            time_lag_dim: The pipeline's time-lag column declaration, or
                ``None`` when the cloud carries no time lag. The lag is read
                from the split ``time_lag`` field here, only the declaration
                that one exists matters.
        """
        self.ignore_index = int(ignore_index)
        self.time_lag_dim = time_lag_dim
        self._required_keys = ["coord", "num_current_points"]
        if time_lag_dim is not None:
            self._required_keys.append("time_lag")

    def apply_defaults(self, input_dict: dict[str, Any]) -> None:
        """Populate missing semantic labels with the ignore label."""
        input_dict.setdefault(
            "pts_semantic_mask",
            np.full(int(input_dict["num_current_points"]), self.ignore_index, dtype=np.int64),
        )

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert the current-frame mask into the padded ``segment`` key."""
        mask = input_dict["pts_semantic_mask"]
        num_current = int(input_dict.pop("num_current_points"))
        num_points = input_dict["coord"].shape[0]
        if mask.shape[0] != num_current:
            raise ValueError(
                "PreparePointSegInput requires one semantic label per current-frame point: "
                f"got {mask.shape[0]} labels for {num_current} points."
            )
        if self.time_lag_dim is None:
            if "time_lag" in input_dict:
                raise ValueError(
                    "PreparePointSegInput has time_lag_dim None but the sample carries a "
                    "'time_lag' field. Declare the time-lag column so sweep points can be told "
                    "apart from current-frame points."
                )
            if num_current != num_points:
                raise ValueError(
                    "PreparePointSegInput has time_lag_dim None, so every point must belong to "
                    f"the current frame: got {num_current} current points for {num_points} "
                    "points."
                )
        else:
            time_lag = input_dict["time_lag"][:, 0]
            if np.any(time_lag[:num_current] != 0.0) or np.any(time_lag[num_current:] == 0.0):
                raise ValueError(
                    "PreparePointSegInput requires the current frame (time_lag 0) to be exactly "
                    f"the leading block of {num_current} points."
                )
        segment = np.full(num_points, self.ignore_index, dtype=np.int64)
        segment[:num_current] = mask
        return {"segment": segment}
