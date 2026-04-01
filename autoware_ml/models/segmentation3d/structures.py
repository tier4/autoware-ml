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

"""Typed feature carriers shared by segmentation models."""

from __future__ import annotations

from typing import TypedDict

import torch


class FRNetInputs(TypedDict):
    """Required FRNet tensors and batch metadata passed into the encoder stack."""

    points: torch.Tensor
    coors: torch.Tensor
    voxel_coors: torch.Tensor
    inverse_map: torch.Tensor
    sample_count: int


class FRNetFeatureDict(FRNetInputs, total=False):
    """Intermediate FRNet features produced by the encoder and backbone."""

    point_feats: list[torch.Tensor]
    voxel_feats: torch.Tensor | list[torch.Tensor]
    point_feats_backbone: list[torch.Tensor]


class FRNetDecodedOutputs(FRNetFeatureDict):
    """Decoded FRNet features with final point-wise logits."""

    point_logits: torch.Tensor
