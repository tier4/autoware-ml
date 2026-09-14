"""Reusable temporal-query helpers for camera-only detection heads.

This module groups the geometry, positional-encoding, and memory helpers used
by PETR-style streaming detectors. The utilities are backend-agnostic and can
be reused by future camera-query models beyond StreamPETR.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Apply the inverse sigmoid transform with clamping for stability."""
    dtype = x.dtype
    x = x.float().clamp(min=0.0, max=1.0)
    x1 = x.clamp(min=eps)
    x2 = (1.0 - x).clamp(min=eps)
    return torch.log(x1 / x2).to(dtype=dtype)


def memory_refresh(memory: torch.Tensor, prev_exists: torch.Tensor) -> torch.Tensor:
    """Reset memory entries when the previous frame does not belong to the stream."""
    view_shape = [1] * memory.ndim
    prev_exists = prev_exists.view(-1, *view_shape[1:])
    return memory * prev_exists


def topk_gather(features: torch.Tensor, topk_indices: torch.Tensor | None) -> torch.Tensor:
    """Gather the same top-k proposal indices across a feature tensor."""
    if topk_indices is None:
        return features

    feature_shape = list(features.shape)
    view_shape = [1] * len(feature_shape)
    view_shape[:2] = list(topk_indices.shape[:2])
    gather_index = topk_indices.view(*view_shape).repeat(1, 1, *feature_shape[2:])
    return torch.gather(features, 1, gather_index)


def transform_reference_points(
    reference_points: torch.Tensor,
    ego_pose: torch.Tensor,
) -> torch.Tensor:
    """Transform 3D reference points between ego frames."""
    homogeneous = torch.cat([reference_points, torch.ones_like(reference_points[..., :1])], dim=-1)
    return (ego_pose.unsqueeze(1) @ homogeneous.unsqueeze(-1)).squeeze(-1)[..., :3]


def _sincos_posemb(component: torch.Tensor, num_pos_feats: int, temperature: int) -> torch.Tensor:
    """Embed one scaled position component with interleaved sine/cosine pairs."""
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=component.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)
    scaled = component[..., None] / dim_t
    return torch.stack((scaled[..., 0::2].sin(), scaled[..., 1::2].cos()), dim=-1).flatten(-2)


def pos2posemb3d(
    positions: torch.Tensor,
    num_pos_feats: int = 128,
    temperature: int = 10000,
) -> torch.Tensor:
    """Convert 3D positions in ``[0, 1]`` space into sine/cosine embeddings.

    The output concatenates the (y, x, z) component embeddings in that order,
    matching the reference implementation the pretrained weights expect.
    """
    positions = positions * (2.0 * math.pi)
    pos_x = _sincos_posemb(positions[..., 0], num_pos_feats, temperature)
    pos_y = _sincos_posemb(positions[..., 1], num_pos_feats, temperature)
    pos_z = _sincos_posemb(positions[..., 2], num_pos_feats, temperature)
    return torch.cat((pos_y, pos_x, pos_z), dim=-1)


def pos2posemb1d(
    positions: torch.Tensor,
    num_pos_feats: int = 256,
    temperature: int = 10000,
) -> torch.Tensor:
    """Convert 1D temporal positions into sine/cosine embeddings."""
    positions = positions * (2.0 * math.pi)
    return _sincos_posemb(positions[..., 0], num_pos_feats, temperature)


def nerf_positional_encoding(
    tensor: torch.Tensor,
    num_encoding_functions: int = 6,
    include_input: bool = False,
    log_sampling: bool = True,
) -> torch.Tensor:
    """Apply NeRF-style positional encoding to pose and time features."""
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

    def forward(self, x: torch.Tensor, gating: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        """Apply feature-wise affine modulation predicted from ``code``."""
        x = self.layer_norm(x)
        code = self.reduce(code)
        return self.gamma(code) * x + self.beta(code)
