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
    """Split the ``points`` array into ``coord`` and ``strength`` fields.

    Required keys:
        points: Raw point cloud array of shape ``(N, D)`` where D >= 4.
                Columns 0-2 are XYZ coordinates; column 3 is intensity.

    Generated keys:
        coord: XYZ coordinates ``(N, 3)``, float32.
        strength: Normalised intensity ``(N, 1)``, float32 in ``[0, 1]``.
    """

    _required_keys = ["points"]

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        points = input_dict.pop("points")
        input_dict["coord"] = points[:, :3].astype(np.float32)
        input_dict["strength"] = (points[:, 3:4] / 255.0).astype(np.float32)
        return input_dict
