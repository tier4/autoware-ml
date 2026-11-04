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

"""Deformable aggregation of dense BEV features at the (x, y) of keypoints."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from jaxtyping import Float

from autoware_ml.dataclasses.models.detection3d.bev_level import BEVLevel


def bev_deformable_aggregation(
    key_points: Float[torch.Tensor, "batch_size num_queries num_points 3"],
    weights: Float[torch.Tensor, "batch_size num_queries num_samples num_groups"],
    levels: Sequence[BEVLevel],
) -> Float[torch.Tensor, "batch_size num_queries channels"]:
    """Sum the BEV samples of every level and keypoint with their weights per feature group.

    Exports as one ONNX ``GridSample`` per level and plain arithmetic.

    Args:
        key_points: Metric keypoints ``(B, N, P, 3)``.
        weights: Softmaxed sampling weights ``(B, N, L * P, G)``, level major.
        levels: Sampled BEV levels, finest first.

    Returns:
        Aggregated features ``(B, N, C)``.
    """
    batch_size, num_anchor, num_samples, num_groups = weights.shape
    channels = levels[0].feat.shape[1]
    sampled = torch.cat([level.sample(key_points) for level in levels], dim=2).reshape(
        batch_size, num_anchor, num_samples, num_groups, channels // num_groups
    )
    return (sampled * weights[..., None]).sum(dim=2).reshape(batch_size, num_anchor, channels)
