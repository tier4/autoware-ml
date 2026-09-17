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

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from autoware_ml.dataclasses.batch.segmentation3d import Segmentation3DGTBatch
from autoware_ml.dataclasses.geometry.point_clouds import PointCloudGTBatch
from autoware_ml.dataclasses.geometry.voxels import VoxelsData
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.dataclasses.models.segmentation3d.predictions import Segmentation3DPredictions
from autoware_ml.losses.segmentation3d.lovasz import LovaszLoss
from autoware_ml.metrics.segmentation3d.eval_output import segmentation_frames_eval_output
from autoware_ml.models.segmentation3d.encoders.ptv3 import (
    Block,
    PointSequential,
    SerializedUnpooling,
    deepcopy_without_flash,
    expand_stage_flags,
    prepare_point_module_for_export,
    set_block_serialization_order,
)
from autoware_ml.models.segmentation3d.encoders.voxel import TIME_LAG_COLUMN
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
        mixed_voxel_weight: float,
        lovasz_weight: float = 1.0,
        dec_conv: Sequence[bool] | bool = True,
        dec_attn: Sequence[bool] | bool = True,
        dec_rope_base: Sequence[float | None] | float | None = None,
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
            mixed_voxel_weight: Largest extra weight a voxel gains in the cross entropy
                when its points carry different classes.
            lovasz_weight: Weight applied to the Lovasz loss term.
            dec_conv: Per-stage flag for the submanifold-convolution positional
                encoding in decoder blocks, or one flag for every stage.
            dec_attn: Per-stage flag for attention and its MLP in decoder
                blocks, or one flag for every stage.
            dec_rope_base: Rotary-embedding frequency base, either one value for
                every stage or one per stage. ``None`` disables RoPE.
        """
        super().__init__()
        self.order = list(order)
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.dec_depths = list(dec_depths)
        stage_count = len(enc_channels)
        decoder_stage_count = stage_count - 1
        self.dec_conv = expand_stage_flags(dec_conv, decoder_stage_count, True, "dec_conv")
        self.dec_attn = expand_stage_flags(dec_attn, decoder_stage_count, True, "dec_attn")
        self.dec_rope_base = expand_stage_flags(
            dec_rope_base, decoder_stage_count, None, "dec_rope_base"
        )

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
                        enable_conv=self.dec_conv[stage_index],
                        enable_attn=self.dec_attn[stage_index],
                        rope_base=self.dec_rope_base[stage_index],
                    ),
                    name=f"block{block_index}",
                )
            self.dec.add(decoder, name=f"dec{stage_index}")

        self.classifier = nn.Linear(decoder_channels[0], self.num_classes)
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction="none")
        self.lovasz = LovaszLoss(ignore_index=self.ignore_index, loss_weight=lovasz_weight)
        self.mixed_voxel_weight = float(mixed_voxel_weight)

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

    def loss(
        self, seg_logits: torch.Tensor, segment: torch.Tensor, weights: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Compute segmentation losses against the targets of every row.

        Cross entropy scores each row on its own, so it takes the weights. Lovasz optimizes
        the intersection over union of a whole set of rows and has no per row term to weigh,
        so it takes the targets alone.

        Args:
            seg_logits: Row-wise segmentation logits.
            segment: Target of every row.
            weights: Weight of every row, zero where the row carries no supervision.

        Returns:
            Dictionary with ``loss_ce``, ``loss_lovasz``, and their sum ``loss``.
        """
        # Dividing the weighted sum by the summed weights gives the weighted mean over the
        # supervised rows, and zero instead of nan when a batch has no supervision.
        loss_ce = (self.cross_entropy(seg_logits, segment) * weights).sum() / weights.sum().clamp(
            min=1e-6
        )
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


def current_frame_mask(points: torch.Tensor) -> torch.Tensor:
    """Return the mask of points captured in the current frame.

    Loaders give the current frame points a time lag of exactly ``0`` and every
    sweep point a nonzero one.

    Args:
        points: Point array of shape ``(num_points, num_features)``.

    Returns:
        Boolean mask of shape ``(num_points,)``.
    """
    return points[:, TIME_LAG_COLUMN] == 0


def assigned_point_mask(point_voxel_indices: torch.Tensor) -> torch.Tensor:
    """Return the mask of points that fall inside the voxel grid.

    Points with no voxel lie outside the grid, which happens at the upper bound when
    the range is not a multiple of the voxel size. They are left out of the loss and
    the scoring.
    """
    return point_voxel_indices >= 0


def gather_point_logits(
    seg_logits: torch.Tensor, point_voxel_indices: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Gather the voxel logits of the selected points through their voxel row.

    Args:
        seg_logits: Voxel-level logits of shape ``(num_voxels, num_classes)``.
        point_voxel_indices: Voxel row of every point.
        mask: Points to gather, all of which must have a voxel.

    Returns:
        Point-level logits of shape ``(mask.sum(), num_classes)``.
    """
    return seg_logits[point_voxel_indices[mask]]


def voxelized_points(
    batch_inputs: ModelBatchInputs,
) -> tuple[PointCloudGTBatch, VoxelsData, Segmentation3DGTBatch | None]:
    """Read the points of the batch, their voxels and, when labelled, their labels.

    Args:
        batch_inputs: Model inputs of the batch.

    Returns:
        The point cloud, the voxels a voxelizer built from it and the point labels, None for
        a batch without labels.

    Raises:
        ValueError: If the batch carries no point cloud or no voxels.
    """
    point_batch = batch_inputs.multi_task_gt_batch.point_cloud_gt_batch
    if point_batch is None or batch_inputs.voxels_data is None:
        raise ValueError("PTv3 segmentation needs the points of the batch and their voxels.")
    return (
        point_batch,
        batch_inputs.voxels_data,
        batch_inputs.multi_task_gt_batch.segmentation3d_gt_batch,
    )


def _point_labels(label_batch: Segmentation3DGTBatch | None) -> torch.Tensor:
    """Read the point labels of a labelled batch.

    Raises:
        ValueError: If the batch carries no labels.
    """
    if label_batch is None:
        raise ValueError("PTv3 segmentation needs the semantic labels of the batch.")
    return label_batch.gt_semantic_masks


def voxel_supervision(
    point_voxel_indices: torch.Tensor,
    segment: torch.Tensor,
    *,
    num_voxels: int,
    num_classes: int,
    ignore_index: int,
    sample: bool,
    mixed_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce the points of every voxel to the label and the weight supervising it.

    A voxel predicts one class for all of its points, so it gets one label. When sampling,
    the label is drawn from the label distribution of the voxel, so over many steps every
    class is supervised in proportion to its points. Otherwise the majority label is used.
    A voxel whose points are all ignored stays ignored.

    A voxel whose points disagree gets a larger weight, which grows with the share of points
    its label leaves out. A single class voxel has weight 1.

    Args:
        point_voxel_indices: Voxel row of every point.
        segment: Segmentation label of every point.
        num_voxels: Number of voxels of the batch.
        num_classes: Number of classes the head predicts.
        ignore_index: Label of the points that carry no supervision.
        sample: Whether to draw the label instead of taking the majority.
        mixed_weight: Weight a voxel gains when its points disagree completely.

    Returns:
        Label and weight of every voxel, both of shape ``(num_voxels,)``.
    """
    labelled = assigned_point_mask(point_voxel_indices) & (segment != ignore_index)
    counts = torch.zeros((num_voxels, num_classes), dtype=torch.float32, device=segment.device)
    # Only labelled points index the table because the ignore index can lie outside the classes.
    counts.index_put_(
        (point_voxel_indices[labelled], segment[labelled]),
        torch.ones_like(segment[labelled], dtype=torch.float32),
        accumulate=True,
    )
    totals = counts.sum(dim=1)
    supervised = totals > 0
    labels = torch.full((num_voxels,), ignore_index, dtype=torch.long, device=segment.device)
    if sample:
        drawn = torch.multinomial(counts + (~supervised).unsqueeze(1).to(counts.dtype), 1)
        labels = torch.where(supervised, drawn.squeeze(1), labels)
    else:
        labels = torch.where(supervised, counts.argmax(dim=1), labels)
    purity = counts.max(dim=1).values / totals.clamp(min=1.0)
    weights = torch.where(supervised, 1.0 + mixed_weight * (1.0 - purity), torch.zeros_like(purity))
    return labels, weights


def segmentation_point_loss(
    head: PTv3SegDecoderHead, seg_logits: torch.Tensor, batch_inputs: ModelBatchInputs
) -> dict[str, torch.Tensor]:
    """Compute the segmentation losses with one term per voxel.

    Every voxel adds one cross entropy term, so dense areas near the sensor do not dominate
    the loss. Voxels whose points disagree weigh more.

    Args:
        head: Segmentation head owning the losses.
        seg_logits: Voxel-level segmentation logits.
        batch_inputs: Model inputs with the voxels and the point labels.

    Returns:
        Dictionary with the segmentation losses.
    """
    _, voxels_data, label_batch = voxelized_points(batch_inputs)
    labels, weights = voxel_supervision(
        voxels_data.point_voxel_indices,
        _point_labels(label_batch),
        num_voxels=seg_logits.shape[0],
        num_classes=seg_logits.shape[1],
        ignore_index=head.ignore_index,
        sample=head.training,
        mixed_weight=head.mixed_voxel_weight,
    )
    return head.loss(seg_logits, labels, weights)


def segmentation_eval_output(
    seg_logits: torch.Tensor, batch_inputs: ModelBatchInputs
) -> dict[str, Any]:
    """Gather the voxel predictions of the current frame points, per frame.

    Produces the ``seg_frames`` entries both segmentation suites read: one entry
    per frame with the coordinates, predicted and target labels and per class
    softmax scores of the current frame points inside the voxel grid. Sweep points
    only shape the voxel features. They are found by their time lag and left out.

    Args:
        seg_logits: Voxel-level segmentation logits.
        batch_inputs: Model inputs with the points, the voxels and the point labels. Per
            frame metadata (ego pose, scene token) is passed through when the dataset supplies
            it.

    Returns:
        ``{"seg_frames": [...]}`` keyed for the segmentation suites.
    """
    point_batch, voxels_data, label_batch = voxelized_points(batch_inputs)
    points = point_batch.points
    point_voxel_indices = voxels_data.point_voxel_indices
    scored = current_frame_mask(points) & assigned_point_mask(point_voxel_indices)
    point_logits = gather_point_logits(seg_logits, point_voxel_indices, scored)
    return segmentation_frames_eval_output(
        coord=points[scored, :3],
        pred_labels=point_logits.argmax(dim=1),
        target_labels=_point_labels(label_batch).long()[scored],
        scores=torch.softmax(point_logits, dim=1),
        frame_ids=point_batch.batch_indices.long()[scored],
        num_frames=point_batch.batch_size,
        batch_inputs=batch_inputs,
    )


def segmentation_predict_outputs(
    seg_logits: torch.Tensor, batch_inputs: ModelBatchInputs
) -> Segmentation3DPredictions:
    """Format segmentation predictions for the current-frame points.

    Args:
        seg_logits: Voxel-level segmentation logits.
        batch_inputs: Model inputs with the points and their voxels.

    Returns:
        The predicted label and the class probabilities of every current-frame point inside
        the voxel grid, in batch order.
    """
    point_batch, voxels_data, _ = voxelized_points(batch_inputs)
    point_voxel_indices = voxels_data.point_voxel_indices
    scored = current_frame_mask(point_batch.points) & assigned_point_mask(point_voxel_indices)
    point_probs = torch.softmax(gather_point_logits(seg_logits, point_voxel_indices, scored), dim=1)
    return Segmentation3DPredictions(pred_labels=point_probs.argmax(dim=1), pred_probs=point_probs)
