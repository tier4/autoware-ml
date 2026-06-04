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
    """Add the ``segment`` field from semantic mask annotations.

    Must be preceded by ``PreparePointCloudInput`` which produces ``coord``.

    Required keys:
        coord: Point coordinates ``(N, 3)``, used only to determine N for the
               default mask when ``pts_semantic_mask`` is absent.

    Optional keys:
        pts_semantic_mask: Per-point semantic label array of shape ``(N,)``.
                           Defaults to all ``-1`` (ignore label) when absent.

    Generated keys:
        segment: Per-point semantic labels ``(N,)``, int64.
    """

    _required_keys = ["coord"]
    _optional_keys = ["pts_semantic_mask"]

    def apply_defaults(self, input_dict: dict[str, Any]) -> None:
        """Populate missing semantic labels with the ignore label."""
        input_dict.setdefault(
            "pts_semantic_mask", np.full(input_dict["coord"].shape[0], -1, dtype=np.int64)
        )

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert semantic mask into the standardized ``segment`` key."""
        return {
            "segment": input_dict["pts_semantic_mask"].astype(np.int64),
        }
