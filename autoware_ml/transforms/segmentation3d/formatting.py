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
    """Convert point-cloud segmentation samples into model-ready point fields."""

    _required_keys = ["points"]
    _optional_keys = ["pts_semantic_mask"]

    def apply_defaults(self, input_dict: dict[str, Any]) -> None:
        """Populate missing semantic labels with the ignore label."""
        input_dict.setdefault(
            "pts_semantic_mask", np.full(input_dict["points"].shape[0], -1, dtype=np.int64)
        )

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert dataset fields into standardized point-segmentation keys."""
        points = input_dict["points"]
        return {
            "coord": points[:, :3].astype(np.float32),
            "strength": (points[:, 3:4] / 255.0).astype(np.float32),
            "segment": input_dict["pts_semantic_mask"].astype(np.int64),
        }
