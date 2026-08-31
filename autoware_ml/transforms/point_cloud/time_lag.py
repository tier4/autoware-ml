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


"""Per-point time lag of a densified point cloud.

Point loaders place the current frame first and stamp every point with the seconds elapsed since it
was captured: ``0`` for the current frame, positive for points appended from earlier sweeps. Any
transform whose decision must concern the current frame alone selects its points with the mask built
here.

The lag lives in one of two layouts, mirroring the two shapes a sample takes across the pipelines:
packed, as one column of ``points``, and split, as its own ``time_lag`` field once
``PreparePointCloudInput`` has separated the point features. Which column holds it depends on the
raw corpus layout, so consumers declare the column instead of guessing it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

TIME_LAG_KEY = "time_lag"


def current_frame_mask(
    input_dict: Mapping[str, Any], time_lag_dim: int | None
) -> npt.NDArray[np.bool_] | None:
    """Return the mask selecting the points captured in the current frame.

    Args:
        input_dict: Sample holding the point cloud, either split into fields (``time_lag``) or
            packed (``points``).
        time_lag_dim: Column of ``points`` holding the time lag, or ``None`` when the pipeline
            declares that its cloud carries no time lag at all.

    Returns:
        Boolean mask of shape ``(num_points,)``, or ``None`` when ``time_lag_dim`` is ``None``:
        every point then belongs to the current frame and no selection is needed.

    Raises:
        ValueError: If ``time_lag_dim`` is ``None`` although the sample carries a time lag, or if a
            lag is declared but the sample provides neither layout to read it from.
    """
    if time_lag_dim is None:
        if TIME_LAG_KEY in input_dict:
            raise ValueError(
                f"time_lag_dim is None but the sample carries a '{TIME_LAG_KEY}' field. Declare "
                "the time-lag column so current-frame points can be told apart from sweep points."
            )
        return None

    if TIME_LAG_KEY in input_dict:
        return np.asarray(input_dict[TIME_LAG_KEY]).reshape(-1) == 0

    if "points" not in input_dict:
        raise ValueError(
            f"A time lag was declared (time_lag_dim={time_lag_dim}) but the sample carries neither "
            f"'{TIME_LAG_KEY}' nor 'points' to read it from."
        )
    points = np.asarray(input_dict["points"])
    if not 0 <= time_lag_dim < points.shape[1]:
        raise ValueError(
            f"time_lag_dim={time_lag_dim} is outside the {points.shape[1]} point features."
        )
    return points[:, time_lag_dim] == 0
