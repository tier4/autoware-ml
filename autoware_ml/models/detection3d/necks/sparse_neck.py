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

"""Interface of the necks that turn the backbone stages into the sparse levels a head samples."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn

from autoware_ml.models.detection3d.heads.sparse4d import SparseStage
from autoware_ml.utils.point_cloud.structures import Point


class SparseNeck(nn.Module, ABC):
    """A neck the Sparse4D lidar branch runs on the stages of its backbone.

    The detector checks ``num_levels`` against the levels its head samples, so the count is
    part of the contract rather than an attribute a neck happens to have.
    """

    @property
    @abstractmethod
    def num_levels(self) -> int:
        """Return the number of levels the neck returns."""

    @abstractmethod
    def forward(self, point: Point) -> list[SparseStage]:
        """Turn the backbone stages into the levels a head samples.

        Args:
            point: Deepest backbone stage with the pooling chain of the finer stages attached.

        Returns:
            Features, voxel coordinates and batch offsets of every level, from the finest to the
            coarsest.
        """
