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

"""Sparse tensor container shared by the spconv and rulebook convolutions."""

from __future__ import annotations

from typing import Protocol

import torch

from autoware_ml.ops.sparse_conv.rulebook.availability import IS_RULEBOOK_AVAILABLE
from autoware_ml.ops.spconv.availability import IS_SPCONV_AVAILABLE


class SparseTensorLike(Protocol):
    """Interface of ``spconv.SparseConvTensor`` and ``rulebook_torch.SparseTensor``.

    Attributes:
      features: Per voxel features ``[N, C]``.
      indices: Integer ``[N, 4]`` tensor of batch index and grid coordinate.
      spatial_shape: Dense grid extent per axis.
      batch_size: Number of samples in the batch.
    """

    features: torch.Tensor
    indices: torch.Tensor
    spatial_shape: list[int]
    batch_size: int

    def replace_feature(self, feature: torch.Tensor) -> SparseTensorLike:
        """Return the same sparse tensor with other features.

        Args:
            feature: Per voxel features ``[N, C']`` replacing the current ones.

        Returns:
            Sparse tensor sharing indices, shape and index cache with this one.
        """
        ...


def sparse_conv_tensor(
    features: torch.Tensor,
    indices: torch.Tensor,
    spatial_shape: list[int],
    batch_size: int,
) -> SparseTensorLike:
    """Build the sparse tensor container the installed backends consume.

    spconv's ``SparseConvTensor`` is used whenever spconv is installed, because
    rulebook layers accept it as well and the export wrappers require it. Without
    spconv the rulebook container, which carries the same four fields plus the
    ``indice_dict`` cache, is used instead.

    Args:
        features: Per voxel features ``[N, C]``.
        indices: Integer ``[N, 4]`` tensor of batch index and grid coordinate.
        spatial_shape: Dense grid extent per axis.
        batch_size: Number of samples in the batch.

    Returns:
        The sparse tensor container.

    Raises:
        ModuleNotFoundError: Raised when neither backend is installed.
    """
    if IS_SPCONV_AVAILABLE:
        import spconv.pytorch as spconv

        return spconv.SparseConvTensor(
            features=features,
            indices=indices,
            spatial_shape=spatial_shape,
            batch_size=batch_size,
        )
    if IS_RULEBOOK_AVAILABLE:
        import rulebook_torch

        return rulebook_torch.SparseTensor(
            features, indices, spatial_shape=spatial_shape, batch_size=batch_size
        )
    raise ModuleNotFoundError(
        "Building a sparse tensor requires 'spconv' or 'rulebook_torch'; neither is installed."
    )
