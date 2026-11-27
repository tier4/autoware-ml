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

"""Reusable temporal-query helpers for camera-only detection heads.

This module groups the geometry, positional-encoding, and memory helpers used
by PETR-style streaming detectors. The utilities are backend-agnostic and can
be reused by future camera-query models beyond StreamPETR.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import NamedTuple

from jaxtyping import Float, Int
import torch
import torch.nn as nn


class StreamStep(NamedTuple):
    """Continuity of every batch lane with its previous frame.

    Attributes:
        prev_exists: 1.0 where the memory of a lane carries over and 0.0 where it is reset.
        gap: Time of every lane since its previous frame in seconds, 0 without one.
    """

    prev_exists: Float[torch.Tensor, " batch_size"]
    gap: Float[torch.Tensor, " batch_size"]


class StreamContinuity:
    """Tell which batch lanes continue the stream of their previous frame.

    A lane continues when its frame belongs to the scene of the lane's previous frame and
    comes later in time, at least ``min_stream_gap`` and at most ``max_stream_gap`` seconds
    later when they are set. A lane without a previous frame, one that moved to another scene,
    one whose frame is not later than the previous one (the streaming sampler ends a shorter
    lane with a padding slot that repeats its last frame) and one whose gap leaves the bounds
    start a new stream, so a temporal model resets its memory there. Every temporal head that
    carries a memory across batches makes this same check, and the frames are remembered for
    the next call.

    Args:
        max_stream_gap: Longest gap in seconds a stream survives, ``None`` for any gap.
        min_stream_gap: Shortest gap in seconds a stream survives, ``None`` for any gap above
            zero.
    """

    def __init__(
        self, max_stream_gap: float | None = None, min_stream_gap: float | None = None
    ) -> None:
        if max_stream_gap is not None and max_stream_gap <= 0:
            raise ValueError(f"max_stream_gap must be positive, got {max_stream_gap}.")
        if min_stream_gap is not None and min_stream_gap <= 0:
            raise ValueError(f"min_stream_gap must be positive, got {min_stream_gap}.")
        bounded = min_stream_gap is not None and max_stream_gap is not None
        if bounded and min_stream_gap >= max_stream_gap:
            raise ValueError(
                f"min_stream_gap {min_stream_gap} must be below max_stream_gap {max_stream_gap}."
            )
        self.max_stream_gap = max_stream_gap
        self.min_stream_gap = min_stream_gap
        self.reset()

    def reset(self) -> None:
        """Forget the previous frame of every lane."""
        self.scene_tokens: list[str] | None = None
        self.timestamps: Float[torch.Tensor, " batch_size"] | None = None

    def __call__(
        self, scene_tokens: Sequence[str], timestamp: Float[torch.Tensor, " batch_size"]
    ) -> StreamStep:
        """Compare the frames of the lanes with the lanes' previous frames.

        Args:
            scene_tokens: Scene of the frame of every lane.
            timestamp: Sample time of the frame of every lane in seconds, float64.

        Returns:
            The continuity of every lane and its gap to the previous frame.
        """
        previous_tokens = self.scene_tokens
        if previous_tokens is None or len(previous_tokens) != len(scene_tokens):
            prev_exists = torch.zeros(len(scene_tokens), device=timestamp.device)
            gap = torch.zeros_like(timestamp)
        else:
            same_scene = torch.tensor(
                [current == previous for current, previous in zip(scene_tokens, previous_tokens)],
                device=timestamp.device,
            )
            gap = timestamp - self.timestamps
            if self.min_stream_gap is None:
                continues = same_scene & (gap > 0)
            else:
                continues = same_scene & (gap >= self.min_stream_gap)
            if self.max_stream_gap is not None:
                continues &= gap <= self.max_stream_gap
            prev_exists = continues.float()
        self.scene_tokens = list(scene_tokens)
        self.timestamps = timestamp.detach().clone()
        return StreamStep(prev_exists, gap)


def memory_refresh(
    memory: Float[torch.Tensor, "batch_size num_slots *dims"],
    prev_exists: Float[torch.Tensor, " batch_size"],
) -> Float[torch.Tensor, "batch_size num_slots *dims"]:
    """Reset memory entries when the previous frame does not belong to the stream."""
    view_shape = [1] * memory.ndim
    prev_exists = prev_exists.view(-1, *view_shape[1:])
    return memory * prev_exists


def topk_gather(
    features: Float[torch.Tensor, "batch_size num_slots *dims"],
    topk_indices: Int[torch.Tensor, "batch_size k 1"] | None,
) -> Float[torch.Tensor, "batch_size k *dims"]:
    """Gather the same top-k proposal indices across a feature tensor."""
    if topk_indices is None:
        return features

    feature_shape = list(features.shape)
    view_shape = [1] * len(feature_shape)
    view_shape[:2] = list(topk_indices.shape[:2])
    gather_index = topk_indices.view(*view_shape).repeat(1, 1, *feature_shape[2:])
    return torch.gather(features, 1, gather_index)


def transform_reference_points(
    reference_points: Float[torch.Tensor, "batch_size num_slots 3"],
    ego_pose: Float[torch.Tensor, "batch_size 4 4"],
) -> Float[torch.Tensor, "batch_size num_slots 3"]:
    """Transform 3D reference points between ego frames.

    The bank keeps its reference points in the map frame, so like :func:`compose_poses` this is a
    sum of elementwise products rather than a matmul, which would follow the global float32 matmul
    precision.
    """
    homogeneous = torch.cat([reference_points, torch.ones_like(reference_points[..., :1])], dim=-1)
    return (ego_pose.unsqueeze(1) * homogeneous.unsqueeze(-2)).sum(-1)[..., :3]


def compose_poses(
    lhs: Float[torch.Tensor, "*batch 4 4"], rhs: Float[torch.Tensor, "*batch 4 4"]
) -> Float[torch.Tensor, "*batch 4 4"]:
    """Compose batches of 4x4 poses, ``lhs @ rhs``, as a sum of elementwise products.

    A float32 ``@`` follows the global matmul precision, which the CLI runtime sets to
    ``"medium"``: TF32 on CUDA, which rounds a translation 65 km from the map origin by tens of
    meters. Elementwise products and their sum stay in IEEE float32 under every precision
    setting; they export as Mul and ReduceSum.

    Args:
        lhs: Left poses ``(..., 4, 4)``.
        rhs: Right poses ``(..., 4, 4)``, broadcast against ``lhs``.

    Returns:
        The composed poses ``(..., 4, 4)``.
    """
    return (lhs.unsqueeze(-1) * rhs.unsqueeze(-3)).sum(-2)


def _sincos_posemb(
    component: Float[torch.Tensor, "*batch"], num_pos_feats: int, temperature: int
) -> Float[torch.Tensor, "*batch num_pos_feats"]:
    """Embed one scaled position component with interleaved sine/cosine pairs."""
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=component.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)
    scaled = component[..., None] / dim_t
    return torch.stack((scaled[..., 0::2].sin(), scaled[..., 1::2].cos()), dim=-1).flatten(-2)


def pos2posemb3d(
    positions: Float[torch.Tensor, "*batch 3"],
    num_pos_feats: int = 128,
    temperature: int = 10000,
) -> Float[torch.Tensor, "*batch 3*num_pos_feats"]:
    """Convert 3D positions in ``[0, 1]`` space into sine/cosine embeddings.

    The output concatenates the (y, x, z) component embeddings in that order,
    matching the reference implementation the pretrained weights expect.
    """
    positions = positions * (2.0 * math.pi)
    pos_x = _sincos_posemb(positions[..., 0], num_pos_feats, temperature)
    pos_y = _sincos_posemb(positions[..., 1], num_pos_feats, temperature)
    pos_z = _sincos_posemb(positions[..., 2], num_pos_feats, temperature)
    return torch.cat((pos_y, pos_x, pos_z), dim=-1)


def pos2posemb2d(
    positions: Float[torch.Tensor, "*batch 2"],
    num_pos_feats: int = 128,
    temperature: int = 10000,
) -> Float[torch.Tensor, "*batch channels"]:
    """Convert 2D positions into sine/cosine embeddings of the (y, x) components in that order."""
    positions = positions * (2.0 * math.pi)
    pos_x = _sincos_posemb(positions[..., 0], num_pos_feats, temperature)
    pos_y = _sincos_posemb(positions[..., 1], num_pos_feats, temperature)
    return torch.cat((pos_y, pos_x), dim=-1)


def pos2posemb1d(
    positions: Float[torch.Tensor, "*batch 1"],
    num_pos_feats: int = 256,
    temperature: int = 10000,
) -> Float[torch.Tensor, "*batch num_pos_feats"]:
    """Convert 1D temporal positions into sine/cosine embeddings."""
    positions = positions * (2.0 * math.pi)
    return _sincos_posemb(positions[..., 0], num_pos_feats, temperature)


def nerf_positional_encoding(
    tensor: Float[torch.Tensor, "*batch num_features"],
    num_encoding_functions: int = 6,
    include_input: bool = False,
    log_sampling: bool = True,
) -> Float[torch.Tensor, "*batch num_encoded"]:
    """Apply NeRF-style positional encoding to pose and time features.

    Every feature becomes ``2 * num_encoding_functions`` sines and cosines, after the feature
    itself when ``include_input`` is set.
    """
    encoding = [tensor] if include_input else []
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            1.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    for frequency in frequency_bands:
        encoding.append(torch.sin(tensor * frequency))
        encoding.append(torch.cos(tensor * frequency))
    return encoding[0] if len(encoding) == 1 else torch.cat(encoding, dim=-1)


class SELayerLinear(nn.Module):
    """Linear squeeze-excitation used to modulate positional embeddings."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.reduce = nn.Linear(channels, channels)
        self.expand = nn.Linear(channels, channels)
        self.activation = nn.ReLU()
        self.gate = nn.Sigmoid()

    def forward(
        self,
        x: Float[torch.Tensor, "*batch channels"],
        gating: Float[torch.Tensor, "*batch channels"],
    ) -> Float[torch.Tensor, "*batch channels"]:
        """Apply channel-wise gating from the auxiliary ``gating`` tensor."""
        gating = self.reduce(gating)
        gating = self.activation(gating)
        gating = self.expand(gating)
        return x * self.gate(gating)


class ModulatedLayerNorm(nn.Module):
    """LayerNorm whose scale and bias are predicted from an auxiliary code."""

    def __init__(self, code_dim: int, feature_dim: int = 256) -> None:
        super().__init__()
        self.reduce = nn.Sequential(nn.Linear(code_dim, feature_dim), nn.ReLU())
        self.gamma = nn.Linear(feature_dim, feature_dim)
        self.beta = nn.Linear(feature_dim, feature_dim)
        self.layer_norm = nn.LayerNorm(feature_dim, elementwise_affine=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the modulation to identity."""
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.bias)

    def forward(
        self,
        x: Float[torch.Tensor, "*batch feature_dim"],
        code: Float[torch.Tensor, "*batch code_dim"],
    ) -> Float[torch.Tensor, "*batch feature_dim"]:
        """Apply feature-wise affine modulation predicted from ``code``."""
        x = self.layer_norm(x)
        code = self.reduce(code)
        return self.gamma(code) * x + self.beta(code)
