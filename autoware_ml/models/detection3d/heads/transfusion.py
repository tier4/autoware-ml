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

"""TransFusion detection head components.

This module contains the query decoder, target generation, and loss logic used
by the native TransFusion lidar detector.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from autoware_ml.losses.detection3d.focal import SigmoidFocalLoss
from autoware_ml.losses.detection3d.gaussian_focal import GaussianFocalLoss
from autoware_ml.models.common.layers.conv import ConvModule
from autoware_ml.models.detection3d.task_modules.assigners import HungarianAssigner3D
from autoware_ml.models.detection3d.task_modules.bbox_coders import TransFusionBBoxCoder
from autoware_ml.models.detection3d.task_modules.heatmap import (
    circle_nms,
    draw_heatmap_gaussian,
    draw_heatmap_gaussian_oriented,
    gaussian_radius,
)


class LearnedPositionalEncoding(nn.Module):
    """Learn positional embeddings from 2D BEV coordinates.

    The module maps BEV cell coordinates into query or key embeddings used by
    the TransFusion decoder.
    """

    def __init__(self, input_channels: int, embed_dims: int) -> None:
        """Initialize the positional encoding module.

        Args:
            input_channels: Number of input coordinate channels.
            embed_dims: Output embedding dimension.
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_channels, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
        )

    def forward(self, position: torch.Tensor) -> torch.Tensor:
        """Encode BEV positions into query embeddings.

        Args:
            position: BEV coordinate tensor.

        Returns:
            Learned positional embeddings.
        """
        return self.proj(position)


class TransFusionDecoderLayer(nn.Module):
    """Refine TransFusion proposals with self- and cross-attention.

    Each decoder layer updates query features using BEV proposal positions and
    shared BEV feature maps.
    """

    def __init__(
        self, embed_dims: int, num_heads: int, feedforward_channels: int, dropout: float = 0.1
    ) -> None:
        """Initialize one TransFusion decoder layer.

        Args:
            embed_dims: Query and key embedding dimension.
            num_heads: Number of attention heads.
            feedforward_channels: Hidden dimension of the feed-forward block.
            dropout: Dropout probability used throughout the decoder.
        """
        super().__init__()
        self.query_pos_encoding = LearnedPositionalEncoding(2, embed_dims)
        self.key_pos_encoding = LearnedPositionalEncoding(2, embed_dims)
        self.self_attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, feedforward_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feedforward_channels, embed_dims),
        )
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, query_pos: torch.Tensor, key_pos: torch.Tensor
    ) -> torch.Tensor:
        """Refine query embeddings with self- and cross-attention.

        Args:
            query: Query feature tensor.
            key: Key/value feature tensor.
            query_pos: BEV coordinates for the queries.
            key_pos: BEV coordinates for the keys.

        Returns:
            Refined query tensor.
        """
        query_tokens = query.transpose(1, 2)
        key_tokens = key.transpose(1, 2)
        query_tokens = query_tokens + self.query_pos_encoding(query_pos)
        key_tokens = key_tokens + self.key_pos_encoding(key_pos)

        attended, _ = self.self_attn(query_tokens, query_tokens, query_tokens)
        query_tokens = self.norm1(query_tokens + self.dropout(attended))

        attended, _ = self.cross_attn(query_tokens, key_tokens, key_tokens)
        query_tokens = self.norm2(query_tokens + self.dropout(attended))

        ffn_output = self.ffn(query_tokens)
        query_tokens = self.norm3(query_tokens + self.dropout(ffn_output))
        return query_tokens.transpose(1, 2)


class ExportableMultiheadAttention(nn.Module):
    """ONNX/TensorRT-friendly equivalent of ``nn.MultiheadAttention``.

    The runtime math and weights are unchanged, but the exported graph avoids
    PyTorch's fused MultiheadAttention decomposition that TensorRT produces NaNs
    for in the TransHead decoder.
    """

    def __init__(self, attention: nn.MultiheadAttention) -> None:
        """Copy trained weights from a batch-first MultiheadAttention module."""
        super().__init__()
        if not attention.batch_first:
            raise ValueError("TransFusion export expects batch_first=True attention.")
        if attention.in_proj_weight is None or attention.in_proj_bias is None:
            raise ValueError("TransFusion export expects packed QKV projection weights.")

        self.embed_dim = attention.embed_dim
        self.num_heads = attention.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.dropout = attention.dropout
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self._copy_weights(attention)
        self.to(device=attention.in_proj_weight.device, dtype=attention.in_proj_weight.dtype)

    def _copy_weights(self, attention: nn.MultiheadAttention) -> None:
        """Copy packed PyTorch MHA weights into explicit projection layers."""
        q_weight, k_weight, v_weight = attention.in_proj_weight.chunk(3, dim=0)
        q_bias, k_bias, v_bias = attention.in_proj_bias.chunk(3, dim=0)
        self.q_proj.weight.data.copy_(q_weight)
        self.k_proj.weight.data.copy_(k_weight)
        self.v_proj.weight.data.copy_(v_weight)
        self.q_proj.bias.data.copy_(q_bias)
        self.k_proj.bias.data.copy_(k_bias)
        self.v_proj.bias.data.copy_(v_bias)
        self.out_proj.weight.data.copy_(attention.out_proj.weight)
        self.out_proj.bias.data.copy_(attention.out_proj.bias)

    def _project(self, projection: nn.Linear, tokens: torch.Tensor) -> torch.Tensor:
        """Project and reshape tokens to ``(batch, heads, sequence, channels)``."""
        batch_size, sequence_length, _ = tokens.shape
        projected = projection(tokens)
        projected = projected.view(batch_size, sequence_length, self.num_heads, self.head_dim)
        return projected.transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *_args: Any,
        **_kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        """Run explicit scaled dot-product attention with a PyTorch-compatible return."""
        q = self._project(self.q_proj, query)
        k = self._project(self.k_proj, key)
        v = self._project(self.v_proj, value)

        scale = self.head_dim**0.5
        attention = torch.matmul(q.float() / scale, k.float().transpose(-2, -1))
        attention = attention - attention.max(dim=-1, keepdim=True).values
        attention = attention.softmax(dim=-1)
        if self.training and self.dropout > 0.0:
            attention = F.dropout(attention, p=self.dropout)
        attended = torch.matmul(attention.to(v.dtype), v)
        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .view(query.shape[0], query.shape[1], self.embed_dim)
        )
        return self.out_proj(attended), None


class SeparateHead1D(nn.Module):
    """Apply per-query prediction branches for TransFusion outputs.

    Each branch is implemented as a lightweight 1D convolutional stack over the
    query dimension.
    """

    def __init__(self, in_channels: int, heads: dict[str, tuple[int, int]]) -> None:
        """Initialize the per-query prediction heads.

        Args:
            in_channels: Input feature dimension.
            heads: Mapping from head name to ``(out_channels, num_convs)``.
        """
        super().__init__()
        self.heads = nn.ModuleDict()
        for name, (out_channels, num_convs) in heads.items():
            layers: list[nn.Module] = []
            current_channels = in_channels
            for _ in range(max(num_convs - 1, 0)):
                layers.append(nn.Conv1d(current_channels, in_channels, kernel_size=1, bias=False))
                layers.append(nn.BatchNorm1d(in_channels, eps=1e-3, momentum=0.01))
                layers.append(nn.ReLU(inplace=True))
                current_channels = in_channels
            layers.append(nn.Conv1d(current_channels, out_channels, kernel_size=1))
            self.heads[name] = nn.Sequential(*layers)

        if "heatmap" in self.heads:
            nn.init.constant_(self.heads["heatmap"][-1].bias, -2.19)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Apply all prediction branches to the query features.

        Args:
            x: Per-query feature tensor.

        Returns:
            Dictionary of prediction tensors keyed by branch name.
        """
        return {name: head(x) for name, head in self.heads.items()}


@dataclass
class TransFusionTargets:
    """Store assignment targets for one TransFusion training batch.

    Attributes:
        labels: Target class labels for all decoder queries.
        label_weights: Per-query classification weights.
        bbox_targets: Encoded box regression targets.
        bbox_weights: Per-query box regression weights.
        num_pos: Number of matched positive queries.
        matched_iou: Mean IoU of matched positive queries.
        heatmap: Dense heatmap target used for query initialization.
    """

    labels: torch.Tensor
    label_weights: torch.Tensor
    bbox_targets: torch.Tensor
    bbox_weights: torch.Tensor
    num_pos: int
    matched_iou: float
    heatmap: torch.Tensor


class TransFusionHead(nn.Module):
    """Implement the native TransFusion lidar detection head.

    The head predicts a dense BEV heatmap, selects top proposals, refines them
    with decoder layers, and computes assignment-based training targets.
    """

    def __init__(
        self,
        num_proposals: int,
        auxiliary: bool,
        in_channels: int,
        hidden_channel: int,
        num_classes: int,
        num_decoder_layers: int,
        num_heads: int,
        feedforward_channels: int,
        common_heads: dict[str, tuple[int, int]],
        bbox_coder: TransFusionBBoxCoder,
        assigner: HungarianAssigner3D,
        point_cloud_range: list[float],
        voxel_size: list[float],
        out_size_factor: int,
        code_weights: list[float],
        min_radius: int,
        gaussian_overlap: float,
        score_threshold: float,
        post_max_size: int,
        nms_min_radius: float,
        class_names: Sequence[str] | None = None,
        heatmap_target: str = "round",
        dense_heatmap_pooling_classes: Sequence[str | int] | None = None,
        nms_type: str | None = None,
        nms_groups: Sequence[dict[str, Any]] | None = None,
        loss_cls_weight: float = 1.0,
        loss_bbox_weight: float = 0.25,
        loss_heatmap_weight: float = 1.0,
        heatmap_init_bias: float = -2.19,
        nms_kernel_size: int = 3,
        use_velocity: bool = True,
    ) -> None:
        """Initialize the TransFusion detection head.

        Args:
            num_proposals: Number of proposals selected from the dense heatmap.
            auxiliary: Whether to keep predictions from all decoder layers.
            in_channels: Input BEV feature dimension.
            hidden_channel: Internal decoder feature dimension.
            num_classes: Number of foreground classes.
            num_decoder_layers: Number of decoder refinement layers.
            num_heads: Number of attention heads per decoder layer.
            feedforward_channels: Hidden dimension of decoder feed-forward blocks.
            common_heads: Prediction-branch specification shared by decoder layers.
            bbox_coder: Box coder used for encoding and decoding predictions.
            assigner: Proposal assigner used during training.
            point_cloud_range: Detector point-cloud range.
            voxel_size: Voxel size used by the detector.
            out_size_factor: BEV downsampling factor between point and feature space.
            code_weights: Weights applied to each regression channel.
            min_radius: Minimum heatmap Gaussian radius.
            gaussian_overlap: Required overlap for Gaussian radius computation.
                Only used by the ``"round"`` heatmap target.
            score_threshold: Prediction score threshold used during decoding.
            post_max_size: Maximum number of predictions kept after NMS.
            nms_min_radius: Minimum center distance used by circle NMS.
            class_names: Optional ordered class names used to resolve config-friendly
                class lists.
            heatmap_target: Shape of the dense heatmap supervision. ``"round"``
                (default) draws a circular Gaussian sized by ``gaussian_radius``.
                ``"oriented"`` draws an elliptical Gaussian stretched along the box
                length and rotated by the box yaw, so elongated objects such as a
                tractor and trailer rig get one connected positive region instead of
                a small blob in the low-density gap at the coupling.
            dense_heatmap_pooling_classes: Optional class names or class indices
                that should use local max pooling before proposal selection.
            nms_type: Optional NMS type applied during prediction. Supported values
                are ``None`` and ``"circle"``.
            nms_groups: Optional grouped NMS configuration. Each entry must provide
                ``class_names`` or ``class_ids`` and may override ``nms_radius``
                and ``post_max_size``. A group with ``nms_radius`` of ``0`` keeps
                its highest-scoring predictions up to ``post_max_size``.
            loss_cls_weight: Weight applied to the classification loss.
            loss_bbox_weight: Weight applied to the box regression loss.
            loss_heatmap_weight: Weight applied to the dense heatmap loss.
            heatmap_init_bias: Initial bias used by the dense heatmap branch.
            nms_kernel_size: Kernel size used for local-maximum suppression.
            use_velocity: Whether the head predicts object velocity.
        """
        super().__init__()
        self.num_proposals = num_proposals
        self.auxiliary = auxiliary
        self.num_classes = num_classes
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.out_size_factor = out_size_factor
        self.code_weights = code_weights
        self.min_radius = min_radius
        self.gaussian_overlap = gaussian_overlap
        if heatmap_target not in {"round", "oriented"}:
            raise ValueError(f"Unsupported TransFusion heatmap_target: {heatmap_target!r}")
        self.heatmap_target = heatmap_target
        self.score_threshold = score_threshold
        self.post_max_size = post_max_size
        self.nms_min_radius = nms_min_radius
        self.nms_kernel_size = nms_kernel_size
        self.class_names = tuple(class_names) if class_names is not None else None
        self.bbox_coder = bbox_coder
        self.assigner = assigner
        self.loss_cls_weight = loss_cls_weight
        self.loss_bbox_weight = loss_bbox_weight
        self.loss_heatmap_weight = loss_heatmap_weight
        self.heatmap_init_bias = heatmap_init_bias
        self.use_velocity = use_velocity
        if nms_type not in {None, "circle"}:
            raise ValueError(f"Unsupported TransFusion NMS type: {nms_type!r}")
        self.nms_type = nms_type
        self.dense_heatmap_pooling_class_ids = self._resolve_class_ids(
            dense_heatmap_pooling_classes
        )
        self.nms_groups = self._resolve_nms_groups(nms_groups)

        self.shared_conv = nn.Conv2d(
            in_channels, hidden_channel, kernel_size=3, padding=1, bias=False
        )
        self.shared_norm = nn.BatchNorm2d(hidden_channel, eps=1e-3, momentum=0.01)
        self.shared_act = nn.ReLU(inplace=True)

        self.heatmap_head = nn.Sequential(
            ConvModule(hidden_channel, hidden_channel),
            nn.Conv2d(hidden_channel, num_classes, kernel_size=3, padding=1),
        )
        nn.init.constant_(self.heatmap_head[-1].bias, heatmap_init_bias)
        self.class_encoding = nn.Conv1d(num_classes, hidden_channel, kernel_size=1)

        self.decoder = nn.ModuleList(
            [
                TransFusionDecoderLayer(hidden_channel, num_heads, feedforward_channels)
                for _ in range(num_decoder_layers)
            ]
        )
        prediction_heads = dict(common_heads)
        prediction_heads["heatmap"] = (num_classes, 2)
        if not use_velocity and "vel" in prediction_heads:
            prediction_heads.pop("vel")
        self.prediction_heads = nn.ModuleList(
            [SeparateHead1D(hidden_channel, prediction_heads) for _ in range(num_decoder_layers)]
        )

        self.loss_heatmap = GaussianFocalLoss()
        self.loss_cls = SigmoidFocalLoss()
        self.loss_bbox = nn.L1Loss(reduction="none")

    def _resolve_class_ids(self, classes: Sequence[str | int] | None) -> list[int] | None:
        """Resolve class names or indices into validated class ids."""
        if classes is None:
            return None

        resolved: list[int] = []
        for class_ref in classes:
            if isinstance(class_ref, int):
                class_id = class_ref
            elif isinstance(class_ref, str):
                if self.class_names is None:
                    raise ValueError(
                        "Class-name based TransFusion configuration requires class_names to be set."
                    )
                try:
                    class_id = self.class_names.index(class_ref)
                except ValueError as exc:
                    raise ValueError(f"Unknown TransFusion class name: {class_ref!r}") from exc
            else:
                raise TypeError(f"Unsupported class reference type: {type(class_ref).__name__}")

            if not 0 <= class_id < self.num_classes:
                raise ValueError(f"TransFusion class index out of range: {class_id}")
            resolved.append(class_id)
        return sorted(set(resolved))

    def _resolve_nms_groups(
        self, nms_groups: Sequence[dict[str, Any]] | None
    ) -> list[dict[str, Any]] | None:
        """Resolve grouped NMS configuration into class-id form."""
        if nms_groups is None:
            return None

        resolved_groups: list[dict[str, Any]] = []
        for group in nms_groups:
            if "nms_threshold" in group:
                raise ValueError("TransFusion NMS groups use 'nms_radius', not 'nms_threshold'.")
            class_spec = group.get("class_ids")
            if class_spec is None:
                class_spec = group.get("class_names")
            if class_spec is None:
                raise ValueError("Each TransFusion NMS group must define class_ids or class_names.")
            resolved_groups.append(
                {
                    "class_ids": self._resolve_class_ids(class_spec) or [],
                    "nms_radius": float(group.get("nms_radius", self.nms_min_radius)),
                    "post_max_size": int(group.get("post_max_size", self.post_max_size)),
                }
            )
        return resolved_groups

    def _suppress_dense_heatmap(self, heatmap: torch.Tensor) -> torch.Tensor:
        """Suppress non-maximal dense heatmap activations before proposal sampling."""
        if self.dense_heatmap_pooling_class_ids is None:
            pooled = F.max_pool2d(
                heatmap,
                kernel_size=self.nms_kernel_size,
                stride=1,
                padding=self.nms_kernel_size // 2,
            )
            return heatmap * (pooled == heatmap)

        local_max = heatmap.clone()
        if not self.dense_heatmap_pooling_class_ids:
            return heatmap

        padding = self.nms_kernel_size // 2
        selected_heatmap = heatmap[:, self.dense_heatmap_pooling_class_ids, :, :]
        pooled = F.max_pool2d(
            selected_heatmap,
            kernel_size=self.nms_kernel_size,
            stride=1,
            padding=0,
        )
        if padding == 0:
            local_max[:, self.dense_heatmap_pooling_class_ids, :, :] = pooled
        else:
            local_max[
                :,
                self.dense_heatmap_pooling_class_ids,
                padding:-padding,
                padding:-padding,
            ] = pooled
        return heatmap * (local_max == heatmap)

    def _circle_nms_groups(self) -> list[dict[str, Any]]:
        """Build grouped circle-NMS rules for prediction."""
        if self.nms_groups is not None:
            return list(self.nms_groups)
        return [
            {
                "class_ids": [class_id],
                "nms_radius": self.nms_min_radius,
                "post_max_size": self.post_max_size,
            }
            for class_id in range(self.num_classes)
        ]

    def _apply_circle_nms(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Apply grouped circle NMS and return kept indices."""
        keep_mask = torch.zeros(scores.shape[0], dtype=torch.bool, device=scores.device)
        covered_mask = torch.zeros(scores.shape[0], dtype=torch.bool, device=scores.device)

        for group in self._circle_nms_groups():
            group_mask = torch.zeros(scores.shape[0], dtype=torch.bool, device=scores.device)
            for class_id in group["class_ids"]:
                group_mask |= labels == class_id
            if not group_mask.any():
                continue
            covered_mask |= group_mask
            group_indices = group_mask.nonzero(as_tuple=False).squeeze(1)
            group_post_max_size = group["post_max_size"]
            if group["nms_radius"] <= 0:
                # No suppression: keep the group's highest-scoring predictions.
                keep = scores[group_mask].argsort(descending=True)[:group_post_max_size]
            else:
                keep = circle_nms(
                    boxes[group_mask],
                    scores[group_mask],
                    group["nms_radius"],
                    group_post_max_size,
                )
            keep_mask[group_indices[keep]] = True

        keep_mask |= ~covered_mask
        return keep_mask.nonzero(as_tuple=False).squeeze(1)

    def _create_2d_grid(self, width: int, height: int, device: torch.device) -> torch.Tensor:
        """Create BEV cell centers for positional encoding.

        Args:
            width: Feature-map width.
            height: Feature-map height.
            device: Device used for the created tensor.

        Returns:
            Flattened BEV cell-center coordinates.
        """
        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, device=device, dtype=torch.float32),
            torch.arange(width, device=device, dtype=torch.float32),
            indexing="ij",
        )
        grid = torch.stack([grid_x + 0.5, grid_y + 0.5], dim=-1)
        return grid.view(1, -1, 2)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict TransFusion heatmap, queries, and box parameters.

        Args:
            x: BEV feature tensor.

        Returns:
            Dictionary of dense heatmap and per-query predictions.
        """
        batch_size, _, height, width = x.shape
        features = self.shared_act(self.shared_norm(self.shared_conv(x)))
        flat_features = features.flatten(2)
        bev_pos = self._create_2d_grid(width, height, x.device).repeat(batch_size, 1, 1)

        dense_heatmap = self.heatmap_head(features)
        heatmap = self._suppress_dense_heatmap(dense_heatmap.detach().sigmoid())
        flat_heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)

        proposal_count = min(self.num_proposals, flat_heatmap.shape[1] * flat_heatmap.shape[2])
        _, top_indices = torch.topk(
            flat_heatmap.view(batch_size, -1),
            k=proposal_count,
            dim=-1,
        )
        top_classes = top_indices // flat_heatmap.shape[-1]
        top_positions = top_indices % flat_heatmap.shape[-1]

        query_feat = flat_features.gather(
            2, top_positions[:, None, :].expand(-1, flat_features.shape[1], -1)
        )
        class_one_hot = (
            F.one_hot(top_classes, num_classes=self.num_classes).permute(0, 2, 1).float()
        )
        query_feat = query_feat + self.class_encoding(class_one_hot)
        query_pos = bev_pos.gather(1, top_positions[..., None].expand(-1, -1, bev_pos.shape[-1]))

        predictions: list[dict[str, torch.Tensor]] = []
        for decoder_layer, prediction_head in zip(self.decoder, self.prediction_heads):
            query_feat = decoder_layer(query_feat, flat_features, query_pos, bev_pos)
            prediction = prediction_head(query_feat)
            prediction["center"] = prediction["center"] + query_pos.permute(0, 2, 1)
            predictions.append(prediction)
            query_pos = prediction["center"].detach().permute(0, 2, 1)

        if self.auxiliary:
            outputs = {}
            for key in predictions[0]:
                outputs[key] = torch.cat([prediction[key] for prediction in predictions], dim=-1)
        else:
            outputs = predictions[-1]
        outputs["dense_heatmap"] = dense_heatmap
        outputs["query_heatmap_score"] = flat_heatmap.gather(
            2, top_positions[:, None, :].expand(-1, flat_heatmap.shape[1], -1)
        )
        outputs["query_labels"] = top_classes
        return outputs

    def predict(self, outputs: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        """Decode predictions into metric-space boxes.

        Args:
            outputs: Raw prediction tensors produced by the head.

        Returns:
            List of decoded prediction dictionaries, one per batch element.
        """
        batch_score = outputs["heatmap"][..., -self.num_proposals :].sigmoid()
        query_labels = outputs.get("query_labels")
        if query_labels is None:
            raise ValueError("TransFusion prediction requires query_labels from forward().")
        one_hot = (
            F.one_hot(query_labels, num_classes=self.num_classes)
            .permute(0, 2, 1)
            .to(batch_score.dtype)
        )
        batch_score = batch_score * outputs["query_heatmap_score"] * one_hot
        batch_center = outputs["center"][..., -self.num_proposals :]
        batch_height = outputs["height"][..., -self.num_proposals :]
        batch_dim = outputs["dim"][..., -self.num_proposals :]
        batch_rot = outputs["rot"][..., -self.num_proposals :]
        batch_vel = outputs.get("vel")
        if batch_vel is not None:
            batch_vel = batch_vel[..., -self.num_proposals :]

        decoded = self.bbox_coder.decode(
            batch_score,
            batch_rot,
            batch_dim,
            batch_center,
            batch_height,
            batch_vel,
            filter_predictions=True,
        )

        results = []
        for prediction in decoded:
            boxes = prediction["bboxes"]
            scores = prediction["scores"]
            labels = prediction["labels"]
            if boxes.numel() == 0:
                results.append({"bboxes_3d": boxes, "scores_3d": scores, "labels_3d": labels})
                continue
            if self.nms_type is None:
                kept_indices = torch.arange(scores.shape[0], device=scores.device)
            elif self.nms_type == "circle":
                kept_indices = self._apply_circle_nms(boxes, scores, labels)
            else:
                raise RuntimeError(
                    f"Unsupported TransFusion NMS type at runtime: {self.nms_type!r}"
                )
            results.append(
                {
                    "bboxes_3d": boxes[kept_indices],
                    "scores_3d": scores[kept_indices],
                    "labels_3d": labels[kept_indices],
                }
            )
        return results

    def _build_heatmap_targets(
        self,
        gt_boxes: list[torch.Tensor],
        gt_labels: list[torch.Tensor],
        feature_shape: tuple[int, int],
        device: torch.device,
    ) -> torch.Tensor:
        """Build dense heatmap targets for query initialization.

        Args:
            gt_boxes: Ground-truth boxes for each batch element.
            gt_labels: Ground-truth labels for each batch element.
            feature_shape: Heatmap height and width.
            device: Device used for the generated target tensor.

        Returns:
            Dense training heatmap targets.
        """
        batch_size = len(gt_boxes)
        height, width = feature_shape
        heatmap = torch.zeros((batch_size, self.num_classes, height, width), device=device)
        voxel_size = torch.tensor(self.voxel_size, device=device)
        point_cloud_range = torch.tensor(self.point_cloud_range, device=device)

        for batch_index, (sample_boxes, sample_labels) in enumerate(zip(gt_boxes, gt_labels)):
            sample_boxes = sample_boxes.to(device=device, dtype=torch.float32)
            sample_labels = sample_labels.to(device=device, dtype=torch.long)
            for box, label in zip(sample_boxes, sample_labels):
                center_x = (box[0] - point_cloud_range[0]) / voxel_size[0] / self.out_size_factor
                center_y = (box[1] - point_cloud_range[1]) / voxel_size[1] / self.out_size_factor
                if not (0 <= center_x < width and 0 <= center_y < height):
                    continue
                length_cells = box[3] / voxel_size[0] / self.out_size_factor
                width_cells = box[4] / voxel_size[1] / self.out_size_factor
                if self.heatmap_target == "oriented":
                    draw_heatmap_gaussian_oriented(
                        heatmap[batch_index, int(label.item())],
                        (int(center_x), int(center_y)),
                        length_cells.item(),
                        width_cells.item(),
                        float(box[6].item()),
                        min_sigma=self.min_radius / 3.0,
                    )
                else:
                    radius = max(
                        self.min_radius,
                        gaussian_radius(
                            (width_cells.item(), length_cells.item()), self.gaussian_overlap
                        ),
                    )
                    draw_heatmap_gaussian(
                        heatmap[batch_index, int(label.item())],
                        (int(center_x), int(center_y)),
                        radius,
                    )
        return heatmap

    def get_targets(
        self,
        gt_boxes: list[torch.Tensor],
        gt_labels: list[torch.Tensor],
        outputs: dict[str, torch.Tensor],
    ) -> TransFusionTargets:
        """Build TransFusion training targets.

        Args:
            gt_boxes: Ground-truth boxes for each batch element.
            gt_labels: Ground-truth labels for each batch element.
            outputs: Raw prediction tensors produced by the head.

        Returns:
            Structured training targets for classification, boxes, and heatmaps.
        """
        batch_size = len(gt_boxes)
        num_layers = outputs["center"].shape[-1] // self.num_proposals
        all_labels = []
        all_label_weights = []
        all_bbox_targets = []
        all_bbox_weights = []
        num_pos = 0
        matched_ious = 0.0

        score = outputs["heatmap"].detach()
        center = outputs["center"].detach()
        height = outputs["height"].detach()
        dim = outputs["dim"].detach()
        rot = outputs["rot"].detach()
        vel = outputs.get("vel")
        if vel is not None:
            vel = vel.detach()

        for batch_index in range(batch_size):
            batch_labels = []
            batch_label_weights = []
            batch_bbox_targets = []
            batch_bbox_weights = []
            gt_boxes_tensor = gt_boxes[batch_index].to(score.device, dtype=torch.float32)
            gt_labels_tensor = gt_labels[batch_index].to(score.device, dtype=torch.long)
            for layer_index in range(num_layers):
                start = layer_index * self.num_proposals
                end = (layer_index + 1) * self.num_proposals
                decoded = self.bbox_coder.decode(
                    score[batch_index : batch_index + 1, :, start:end],
                    rot[batch_index : batch_index + 1, :, start:end],
                    dim[batch_index : batch_index + 1, :, start:end],
                    center[batch_index : batch_index + 1, :, start:end],
                    height[batch_index : batch_index + 1, :, start:end],
                    vel[batch_index : batch_index + 1, :, start:end] if vel is not None else None,
                    filter_predictions=False,
                )[0]["bboxes"]
                assign_result = self.assigner.assign(
                    bboxes=decoded,
                    gt_bboxes=gt_boxes_tensor[:, :7],
                    gt_labels=gt_labels_tensor,
                    cls_pred=score[batch_index, :, start:end],
                    point_cloud_range=self.point_cloud_range,
                )
                labels = decoded.new_full((self.num_proposals,), self.num_classes, dtype=torch.long)
                label_weights = decoded.new_ones((self.num_proposals,))
                bbox_targets = decoded.new_zeros((self.num_proposals, self.bbox_coder.code_size))
                bbox_weights = decoded.new_zeros((self.num_proposals, self.bbox_coder.code_size))

                pos_mask = assign_result.gt_inds > 0
                if pos_mask.any():
                    pos_gt_inds = assign_result.gt_inds[pos_mask] - 1
                    labels[pos_mask] = gt_labels_tensor[pos_gt_inds]
                    bbox_targets[pos_mask] = self.bbox_coder.encode(gt_boxes_tensor[pos_gt_inds])
                    bbox_weights[pos_mask] = 1.0
                    num_pos += int(pos_mask.sum().item())
                    if assign_result.max_overlaps is not None:
                        matched_ious += float(assign_result.max_overlaps[pos_mask].sum().item())

                batch_labels.append(labels)
                batch_label_weights.append(label_weights)
                batch_bbox_targets.append(bbox_targets)
                batch_bbox_weights.append(bbox_weights)

            all_labels.append(torch.cat(batch_labels, dim=0))
            all_label_weights.append(torch.cat(batch_label_weights, dim=0))
            all_bbox_targets.append(torch.cat(batch_bbox_targets, dim=0))
            all_bbox_weights.append(torch.cat(batch_bbox_weights, dim=0))

        dense_heatmap = self._build_heatmap_targets(
            gt_boxes,
            gt_labels,
            outputs["dense_heatmap"].shape[-2:],
            outputs["dense_heatmap"].device,
        )
        return TransFusionTargets(
            labels=torch.stack(all_labels, dim=0),
            label_weights=torch.stack(all_label_weights, dim=0),
            bbox_targets=torch.stack(all_bbox_targets, dim=0),
            bbox_weights=torch.stack(all_bbox_weights, dim=0),
            num_pos=num_pos,
            matched_iou=matched_ious / max(num_pos, 1),
            heatmap=dense_heatmap,
        )

    def loss(
        self,
        outputs: dict[str, torch.Tensor],
        gt_boxes: list[torch.Tensor],
        gt_labels: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute TransFusion losses.

        Args:
            outputs: Raw prediction tensors produced by the head.
            gt_boxes: Ground-truth boxes for each batch element.
            gt_labels: Ground-truth labels for each batch element.

        Returns:
            Loss dictionary consumed by the training loop.
        """
        targets = self.get_targets(gt_boxes, gt_labels, outputs)
        loss_dict: dict[str, torch.Tensor] = {}
        loss_heatmap = self.loss_heatmap(outputs["dense_heatmap"], targets.heatmap)
        loss_dict["loss_heatmap"] = self.loss_heatmap_weight * loss_heatmap

        num_layers = outputs["center"].shape[-1] // self.num_proposals
        for layer_index in range(num_layers):
            start = layer_index * self.num_proposals
            end = (layer_index + 1) * self.num_proposals
            prefix = "layer_-1" if layer_index == num_layers - 1 else f"layer_{layer_index}"

            layer_logits = (
                outputs["heatmap"][..., start:end].permute(0, 2, 1).reshape(-1, self.num_classes)
            )
            layer_labels = targets.labels[:, start:end].reshape(-1)
            cls_targets = layer_logits.new_zeros((layer_labels.shape[0], self.num_classes))
            valid_mask = layer_labels < self.num_classes
            cls_targets[valid_mask, layer_labels[valid_mask]] = 1.0
            layer_label_weights = targets.label_weights[:, start:end].reshape(-1)
            loss_cls = self.loss_cls(
                layer_logits, cls_targets, layer_label_weights, avg_factor=max(targets.num_pos, 1)
            )

            preds = torch.cat(
                [
                    outputs["center"][..., start:end],
                    outputs["height"][..., start:end],
                    outputs["dim"][..., start:end],
                    outputs["rot"][..., start:end],
                    outputs["vel"][..., start:end]
                    if "vel" in outputs
                    else outputs["center"].new_zeros(
                        outputs["center"].shape[0], 0, self.num_proposals
                    ),
                ],
                dim=1,
            ).permute(0, 2, 1)
            layer_bbox_targets = targets.bbox_targets[:, start:end, :]
            layer_bbox_weights = targets.bbox_weights[:, start:end, :] * preds.new_tensor(
                self.code_weights
            )
            loss_bbox = self.loss_bbox(preds, layer_bbox_targets)
            loss_bbox = (loss_bbox * layer_bbox_weights).sum() / max(targets.num_pos, 1)

            loss_dict[f"{prefix}_loss_cls"] = self.loss_cls_weight * loss_cls
            loss_dict[f"{prefix}_loss_bbox"] = self.loss_bbox_weight * loss_bbox

        loss_dict["matched_ious"] = outputs["dense_heatmap"].new_tensor(targets.matched_iou)
        loss_dict["loss"] = sum(value for key, value in loss_dict.items() if "loss" in key)
        return loss_dict

    def prepare_for_export(self) -> "TransFusionHead":
        """Return an export-ready copy with attention replaced by exportable equivalents.

        Returns:
            Deep copy of the head with MultiheadAttention layers replaced by
            ExportableMultiheadAttention in all decoder layers.
        """
        head = deepcopy(self).eval()
        if not hasattr(head, "decoder"):
            return head
        for decoder_layer in head.decoder:
            if isinstance(decoder_layer.self_attn, nn.MultiheadAttention):
                decoder_layer.self_attn = ExportableMultiheadAttention(decoder_layer.self_attn)
            if isinstance(decoder_layer.cross_attn, nn.MultiheadAttention):
                decoder_layer.cross_attn = ExportableMultiheadAttention(decoder_layer.cross_attn)
        return head
