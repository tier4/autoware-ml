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

"""Interface of the sparse backbones that turn voxel features into a hierarchy of sparse stages."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

from jaxtyping import Float32, Int
import torch
import torch.nn as nn

from autoware_ml.utils.point_cloud.structures import Point


class SparseBackbone(nn.Module, ABC):
    """A backbone the Sparse4D lidar branch runs between its voxel encoder and its neck.

    The branch checks the voxel features against ``in_channels``, feeds them with their
    coordinates and hands the deepest stage to the neck. It also reads the optimizer groups,
    since a pretrained backbone trains under its own learning rates, so all three are part of
    the contract rather than attributes a backbone happens to have.
    """

    @property
    @abstractmethod
    def in_channels(self) -> int:
        """Return the number of features the backbone reads per voxel."""

    @abstractmethod
    def build_optimizer_groups(self) -> Mapping[str, Sequence[torch.nn.Parameter]]:
        """Group the trainable parameters by the learning rate they train under.

        Returns:
            The trainable parameters keyed by optimizer group name.
        """

    @abstractmethod
    def forward(
        self,
        voxel_features: Float32[torch.Tensor, "num_voxels channels"],
        coords: Int[torch.Tensor, "num_voxels 4"],
    ) -> Point:
        """Encode the voxel features of a batch.

        Args:
            voxel_features: One feature vector per voxel.
            coords: Voxel coordinates ``(num_voxels, 4)`` laid out as ``(batch, x, y, z)`` with
                the voxels grouped by batch index.

        Returns:
            Deepest stage with the pooling chain of the finer stages attached.
        """
