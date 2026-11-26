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

"""Stateless tensor functions shared by the models of more than one task."""

from __future__ import annotations

from jaxtyping import Float
import torch


def inverse_sigmoid(
    x: Float[torch.Tensor, "*shape"], eps: float = 1e-4
) -> Float[torch.Tensor, "*shape"]:
    """Apply the inverse sigmoid transform with clamping for stability.

    Args:
        x: Values to transform; they are clamped to [0, 1] first.
        eps: Floor of ``x`` and of ``1 - x`` before the logarithm, which bounds the result.

    Returns:
        ``log(x / (1 - x))`` in the dtype of ``x``, computed in float32.
    """
    dtype = x.dtype
    x = x.float().clamp(min=0.0, max=1.0)
    x1 = x.clamp(min=eps)
    x2 = (1.0 - x).clamp(min=eps)
    return torch.log(x1 / x2).to(dtype=dtype)
