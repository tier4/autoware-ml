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

"""Submanifold sparse convolution on spconv."""

from __future__ import annotations

import spconv.pytorch as spconv
import torch
from spconv.pytorch.core import SparseConvTensor

__all__ = ["SpconvSubMConv3d", "with_features_as"]


def with_features_as(x: SparseConvTensor, dtype: torch.dtype) -> SparseConvTensor:
    """Return ``x`` with its features in ``dtype``, ``x`` itself when they already are."""
    if x.features.dtype == dtype:
        return x
    return x.replace_feature(x.features.to(dtype))


class SpconvSubMConv3d(spconv.SubMConv3d):
    """spconv's submanifold convolution, run outside autocast in the dtype of its weights.

    spconv's kernels need features and weights in one dtype, and autocast does
    not cast for them.
    """

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        """Convolve ``x`` on spconv in the dtype of the weights.

        Args:
            x: Sparse tensor holding the input features.

        Returns:
            Sparse tensor holding the convolved features.
        """
        with torch.autocast("cuda", enabled=False):
            return super().forward(with_features_as(x, self.weight.dtype))
