# Copyright (c) 2023 lyuwenyu. All Rights Reserved.
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

"""Contrastive denoising queries of RT-DETR, ported from the official code (lyuwenyu/RT-DETR):
src/zoo/rtdetr/denoising.py.

In training, noised copies of the ground truth boxes are fed to the decoder next to the
matching queries. Positives are moved by less than half the box size and must reconstruct
their box; negatives are moved by between half and the full box size and must predict no
object. The random draws happen in the same order as in the official code, so a seeded run
reproduces its queries.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from jaxtyping import Bool, Float, Int64
import torch
import torch.nn as nn
from torch import Tensor

from autoware_ml.geometry.bbox_2d.base_bbox2d import BaseBBoxes2D
from autoware_ml.models.common.functional import inverse_sigmoid

# The clamp of RT-DETR's inverse sigmoid, tighter than the default of inverse_sigmoid.
INVERSE_SIGMOID_EPS = 1e-5


@dataclass(frozen=True)
class DenoisingQueries:
    """Denoising queries, placed before the matching queries in the decoder.

    The queries are split into ``num_groups`` groups. Each group is one noised copy of all the
    ground truth boxes: first one positive per box, then one negative per box. Images with
    fewer boxes than the largest count of the batch are padded.

    Attributes:
        content: Label embedding of each query.
        boxes_unact: Noised ``(cx, cy, w, h)`` box of each query, before the sigmoid.
        attn_mask: Decoder self-attention mask over the denoising queries and then the
            matching ones; True blocks attention. The matching queries cannot see the
            denoising ones, and a group cannot see the other groups.
        positive_indices: For each image, the query index of each of its positives, ordered
            by group, then by box.
        num_groups: Number of groups.
    """

    content: Float[Tensor, "batch_size num_denoising embed_dim"]
    boxes_unact: Float[Tensor, "batch_size num_denoising 4"]
    attn_mask: Bool[Tensor, "num_total_queries num_total_queries"]
    positive_indices: tuple[Int64[Tensor, " num_positives"], ...]
    num_groups: int

    @property
    def num_queries(self) -> int:
        """Number of denoising queries per image."""
        return self.boxes_unact.shape[1]


def build_contrastive_denoising_queries(
    gt_labels: Sequence[Int64[Tensor, " num_boxes"]],
    gt_boxes: Sequence[Float[Tensor, "num_boxes 4"]],
    num_classes: int,
    num_matching_queries: int,
    class_embed: nn.Embedding,
    num_denoising: int = 100,
    label_noise_ratio: float = 0.5,
    box_noise_scale: float = 1.0,
) -> DenoisingQueries:
    """Build the denoising queries from the ground truth.

    The number of groups is ``num_denoising`` divided by the largest box count of the batch,
    and at least one. A batch without any box gets one dummy pair instead. It is never a
    positive, so it has no box to reconstruct and is trained as background, but it keeps the
    label embedding in use, so distributed ranks with and without boxes update the same
    parameters (lyuwenyu/RT-DETR#684).

    Args:
        gt_labels: Class index of every box of every image.
        gt_boxes: ``(cx, cy, w, h)`` of every box of every image, normalized by the image size.
        num_classes: Number of classes; label ``num_classes`` is the padding embedding.
        num_matching_queries: Number of matching queries that follow the denoising ones.
        class_embed: Label embedding with ``num_classes + 1`` entries.
        num_denoising: Target number of denoising queries per image.
        label_noise_ratio: The probability that a query gets a random class is half of this
            value.
        box_noise_scale: Size of the box noise, in units of half the box size.

    Returns:
        The denoising queries.

    Raises:
        ValueError: If ``num_denoising`` is not positive, or the images of the labels and the
            boxes differ.
    """
    if num_denoising <= 0:
        raise ValueError(f"num_denoising must be positive, got {num_denoising}.")
    if len(gt_labels) != len(gt_boxes):
        raise ValueError(
            f"gt_labels and gt_boxes must cover the same images, got {len(gt_labels)} and "
            f"{len(gt_boxes)}."
        )

    num_gts = [len(labels) for labels in gt_labels]
    device = gt_labels[0].device
    max_gt_num = max(num_gts)
    is_empty_batch = max_gt_num == 0
    if is_empty_batch:
        max_gt_num = 1
        num_groups = 1
    else:
        num_groups = max(num_denoising // max_gt_num, 1)

    batch_size = len(num_gts)
    query_class = torch.full(
        [batch_size, max_gt_num], num_classes, dtype=torch.int32, device=device
    )
    query_box = torch.zeros([batch_size, max_gt_num, 4], device=device)
    pad_gt_mask = torch.zeros([batch_size, max_gt_num], dtype=torch.bool, device=device)
    if is_empty_batch:
        # A small box at the image center, as in the official code.
        query_box[..., :2] = 0.5
        query_box[..., 2:] = 0.1

    for i, num_gt in enumerate(num_gts):
        if num_gt > 0:
            query_class[i, :num_gt] = gt_labels[i]
            query_box[i, :num_gt] = gt_boxes[i]
            pad_gt_mask[i, :num_gt] = True

    query_class = query_class.tile([1, 2 * num_groups])
    query_box = query_box.tile([1, 2 * num_groups, 1])
    pad_gt_mask = pad_gt_mask.tile([1, 2 * num_groups])
    negative_gt_mask = torch.zeros([batch_size, max_gt_num * 2, 1], device=device)
    negative_gt_mask[:, max_gt_num:] = 1
    negative_gt_mask = negative_gt_mask.tile([1, num_groups, 1])
    positive_gt_mask = (1 - negative_gt_mask).squeeze(-1) * pad_gt_mask
    positive_indices = torch.nonzero(positive_gt_mask)[:, 1]
    positive_indices = torch.split(positive_indices, [n * num_groups for n in num_gts])
    num_queries = max_gt_num * 2 * num_groups

    if label_noise_ratio > 0:
        mask = torch.rand_like(query_class, dtype=torch.float) < (label_noise_ratio * 0.5)
        new_label = torch.randint_like(mask, 0, num_classes, dtype=query_class.dtype)
        query_class = torch.where(mask & pad_gt_mask, new_label, query_class)

    if box_noise_scale > 0:
        # Each corner moves by up to half the box size for a positive, and by half to the
        # full box size for a negative.
        known_box = BaseBBoxes2D.from_cxcywh(query_box.reshape(-1, 4)).bbox_params
        known_box = known_box.reshape(query_box.shape)
        diff = torch.tile(query_box[..., 2:] * 0.5, [1, 1, 2]) * box_noise_scale
        rand_sign = torch.randint_like(query_box, 0, 2) * 2.0 - 1.0
        rand_part = torch.rand_like(query_box)
        rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (1 - negative_gt_mask)
        known_box = torch.clip(known_box + rand_sign * rand_part * diff, min=0.0, max=1.0)
        query_box = BaseBBoxes2D(known_box.reshape(-1, 4)).cxcywh.reshape(query_box.shape)

    num_total = num_queries + num_matching_queries
    attn_mask = torch.zeros([num_total, num_total], dtype=torch.bool, device=device)
    attn_mask[num_queries:, :num_queries] = True
    group_size = max_gt_num * 2
    for i in range(num_groups):
        rows = slice(group_size * i, group_size * (i + 1))
        attn_mask[rows, group_size * (i + 1) : num_queries] = True
        attn_mask[rows, : group_size * i] = True

    return DenoisingQueries(
        content=class_embed(query_class),
        boxes_unact=inverse_sigmoid(query_box, eps=INVERSE_SIGMOID_EPS),
        attn_mask=attn_mask,
        positive_indices=tuple(positive_indices),
        num_groups=num_groups,
    )
