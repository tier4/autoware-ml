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

"""PTv3 segmentation decoder head.

The head owns the PTv3 decoder: it unpools the deepest encoder stage back to
full resolution through the encoder skip chain and classifies each point.
Keeping the decoder inside the segmentation head (instead of a shared
encoder) lets seg-only finetuning train the full decoder while a frozen
encoder guarantees the detection branch is untouched.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn

from autoware_ml.losses.segmentation3d.lovasz import LovaszLoss
from autoware_ml.models.segmentation3d.encoders.ptv3 import (
    Block,
    PointSequential,
    SerializedUnpooling,
    deepcopy_without_flash,
    prepare_point_module_for_export,
    set_block_serialization_order,
)
from autoware_ml.utils.point_cloud.structures import Point


class PTv3SegDecoderHead(nn.Module):
    """Decode PTv3 encoder stages into point-wise segmentation logits.

    The head mirrors the original PTv3 decoder (serialized unpooling with
    skip fusion, plus optional attention blocks) and appends a linear
    classifier. It also owns the segmentation losses, so task models delegate
    to :meth:`loss` the same way detection models delegate to
    ``bbox_head.loss``.

    Note: :class:`SerializedUnpooling` pops the ``pooling_parent`` chain and
    overwrites parent features in place, so any consumer of raw encoder
    stages (e.g. a detection BEV neck) must read them before this head runs.
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: int,
        order: Sequence[str],
        enc_channels: Sequence[int],
        dec_depths: Sequence[int],
        dec_channels: Sequence[int],
        dec_num_head: Sequence[int],
        dec_patch_size: Sequence[int],
        mlp_ratio: float,
        qkv_bias: bool,
        qk_scale: float | None,
        attn_drop: float,
        proj_drop: float,
        drop_path: float,
        pre_norm: bool,
        enable_rpe: bool,
        enable_flash: bool,
        upcast_attention: bool,
        upcast_softmax: bool,
        lovasz_weight: float = 1.0,
    ) -> None:
        """Initialize the PTv3 segmentation decoder head.

        Args:
            num_classes: Number of semantic classes.
            ignore_index: Label value ignored by the losses.
            order: Serialization orders used by decoder blocks.
            enc_channels: Encoder channel widths per stage (skip dimensions).
            dec_depths: Number of blocks per decoder stage.
            dec_channels: Decoder channel widths per stage.
            dec_num_head: Attention head counts per decoder stage.
            dec_patch_size: Attention patch sizes per decoder stage.
            mlp_ratio: Hidden-layer expansion ratio for each block MLP.
            qkv_bias: Whether to use learnable bias in QKV projections.
            qk_scale: Optional manual attention scale.
            attn_drop: Dropout applied to attention weights.
            proj_drop: Dropout applied after output projections.
            drop_path: Stochastic-depth probability.
            pre_norm: Whether to apply pre-normalization.
            enable_rpe: Whether to use relative positional encoding.
            enable_flash: Whether to use flash attention.
            upcast_attention: Whether to upcast Q/K before attention.
            upcast_softmax: Whether to upcast logits before softmax.
            lovasz_weight: Weight applied to the Lovasz loss term.
        """
        super().__init__()
        self.order = list(order)
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.dec_depths = list(dec_depths)
        stage_count = len(enc_channels)

        dec_drop_path = [value.item() for value in torch.linspace(0, drop_path, sum(dec_depths))]
        self.dec = PointSequential()
        decoder_channels = list(dec_channels) + [enc_channels[-1]]
        for stage_index in reversed(range(stage_count - 1)):
            decoder = PointSequential()
            decoder.add(
                SerializedUnpooling(
                    decoder_channels[stage_index + 1],
                    enc_channels[stage_index],
                    decoder_channels[stage_index],
                ),
                name="up",
            )
            stage_drop = dec_drop_path[
                sum(dec_depths[:stage_index]) : sum(dec_depths[: stage_index + 1])
            ]
            stage_drop.reverse()
            for block_index in range(dec_depths[stage_index]):
                decoder.add(
                    Block(
                        channels=decoder_channels[stage_index],
                        num_heads=dec_num_head[stage_index],
                        patch_size=dec_patch_size[stage_index],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=stage_drop[block_index],
                        pre_norm=pre_norm,
                        order_index=block_index % len(self.order),
                        # The decoder must not share indice caches with the
                        # encoder: a frozen (eval) encoder caches pairs without
                        # backward metadata, and spconv reuses them by key.
                        cpe_indice_key=f"dec_stage{stage_index}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                    ),
                    name=f"block{block_index}",
                )
            self.dec.add(decoder, name=f"dec{stage_index}")

        self.classifier = nn.Linear(decoder_channels[0], self.num_classes)
        # Sum-reduction with an explicit valid-count divisor equals mean-over-valid,
        # but degrades to a clean zero (instead of 0/0 = nan) when a batch carries
        # no segmentation supervision at all.
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction="sum")
        self.lovasz = LovaszLoss(ignore_index=self.ignore_index, loss_weight=lovasz_weight)

    def forward(self, point: Point) -> torch.Tensor:
        """Decode the deepest encoder stage into point-wise logits.

        Args:
            point: Deepest encoder point with its pooling chain attached.

        Returns:
            Segmentation logits of shape ``(num_points, num_classes)`` at the
            finest (input) hierarchy level.
        """
        decoded = self.dec(point)
        return self.classifier(decoded.feat)

    def loss(self, seg_logits: torch.Tensor, segment: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute segmentation losses against point-level targets.

        Args:
            seg_logits: Point-wise segmentation logits.
            segment: Point-level segmentation targets.

        Returns:
            Dictionary with ``loss_ce``, ``loss_lovasz``, and their sum ``loss``.
        """
        valid_count = (segment != self.ignore_index).sum().clamp(min=1)
        loss_ce = self.cross_entropy(seg_logits, segment) / valid_count
        loss_lovasz = self.lovasz(seg_logits, segment)
        return {
            "loss_ce": loss_ce,
            "loss_lovasz": loss_lovasz,
            "loss": loss_ce + loss_lovasz,
        }

    def set_serialization_order(self, order: Sequence[str]) -> None:
        """Update serialization order and reassign block order indices.

        Args:
            order: Serialization orders used by decoder blocks.
        """
        self.order = list(order)
        set_block_serialization_order(self.dec, len(self.order))

    def prepare_for_export(self, order: Sequence[str]) -> PTv3SegDecoderHead:
        """Return an isolated head copy configured for ONNX export.

        Args:
            order: Serialization orders used by the export graph.

        Returns:
            Export-ready decoder head copy.
        """
        export_head = deepcopy_without_flash(self)
        export_head.set_serialization_order(order)
        prepare_point_module_for_export(export_head)
        return export_head


def segmentation_eval_output(
    seg_logits: torch.Tensor, batch: Mapping[str, Any]
) -> dict[str, torch.Tensor]:
    """Scatter point-level predictions back to original points for the metric.

    Args:
        seg_logits: Point-wise segmentation logits at the sampled-point level.
        batch: Batch dictionary with ``inverse`` (sampled-to-original point
            map), ``origin_segment``, and ``origin_coord``.

    Returns:
        Original-point predictions and targets keyed for the segmentation
        metric.
    """
    return {
        "seg_pred_labels": seg_logits.argmax(dim=1)[batch["inverse"].long()],
        "seg_target_labels": batch["origin_segment"].long(),
        "seg_coord": batch["origin_coord"],
    }


def segmentation_predict_outputs(
    seg_logits: torch.Tensor, batch: Mapping[str, Any]
) -> dict[str, torch.Tensor]:
    """Format segmentation predictions at the original-point level.

    Args:
        seg_logits: Point-wise segmentation logits at the sampled-point level.
        batch: Batch dictionary with ``inverse`` (sampled-to-original point
            map).

    Returns:
        Dictionary with ``pred_labels`` and per-class ``pred_probs`` at the
        original-point level.
    """
    point_probs = torch.softmax(seg_logits, dim=1)[batch["inverse"].long()]
    return {"pred_labels": point_probs.argmax(dim=1), "pred_probs": point_probs}
