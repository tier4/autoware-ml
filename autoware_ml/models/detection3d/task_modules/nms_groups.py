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

"""Grouped circle NMS shared by the detection heads: the classes of a group suppress each other."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

from jaxtyping import Bool, Float32, Int64
import torch

from autoware_ml.models.detection3d.task_modules.heatmap import batch_circle_nms


class NMSGroupConfig(NamedTuple):
    """
    Config for each NMS group.
    Args:
        class_names: Names of classes for this group.
        nms_radius: Optional NMS radius threshold for this group.
            If not provided, the default nms_radius will be used.
        max_size: Optional maximum number of predictions to keep for this group.
            If not provided, the default max_size will be used.
    """

    class_names: Sequence[str]
    nms_radius: float | None
    max_size: int | None


class NMSGroup(NamedTuple):
    """
    Resolved config for each NMS group.
    Args:
        class_ids: Class indices for this group.
        nms_radius: NMS radius threshold for this group.
        max_size: Maximum number of predictions to keep for this group.
    """

    class_ids: Sequence[int]
    nms_radius: float
    max_size: int


def resolve_nms_groups(
    nms_group_configs: Sequence[NMSGroupConfig] | None,
    class_names: Sequence[str],
    default_radius: float,
    default_max_size: int,
) -> list[NMSGroup]:
    """
    Resolve grouped NMS configuration into class-id form.

    Without an explicit configuration every class becomes its own group with the defaults, so
    classes never suppress each other and the groups behave like plain per-class circle NMS.

    Args:
        nms_group_configs: Groups naming their classes, None for one group per class.
        class_names: Class names of the head in label order.
        default_radius: Radius of a group that sets none.
        default_max_size: Cap of a group that sets none.

    Returns:
        The groups with their class ids, radius and cap.

    Raises:
        ValueError: If a group names an unknown class, no class, or a class another group claims.
    """
    if nms_group_configs is None:
        return [
            NMSGroup(class_ids=[class_id], nms_radius=default_radius, max_size=default_max_size)
            for class_id in range(len(class_names))
        ]

    resolved_groups: list[NMSGroup] = []
    # A class may appear in at most one group, otherwise it would be suppressed against two
    # different radii. Maps each claimed class id to the group that claimed it, so the error
    # can name both groups.
    claimed_classes: dict[int, int] = {}
    for group_index, nms_group_config in enumerate(nms_group_configs):
        if not nms_group_config.class_names:
            raise ValueError("NMS group must define class_names.")
        unknown = set(nms_group_config.class_names) - set(class_names)
        if unknown:
            raise ValueError(f"NMS group {group_index} names unknown classes {sorted(unknown)}.")
        class_ids = sorted({list(class_names).index(name) for name in nms_group_config.class_names})

        for class_id in class_ids:
            if class_id in claimed_classes:
                raise ValueError(
                    f"Class id {class_id} appears in NMS groups {claimed_classes[class_id]} and "
                    f"{group_index}, so it would be suppressed against two different radii."
                )
            claimed_classes[class_id] = group_index

        # `is None` on purpose: a radius of 0 is a valid setting that disables the suppression
        # for the group (its boxes are only capped by score), so it must not fall back to the
        # default.
        resolved_groups.append(
            NMSGroup(
                class_ids=class_ids,
                nms_radius=(
                    default_radius
                    if nms_group_config.nms_radius is None
                    else nms_group_config.nms_radius
                ),
                max_size=(
                    default_max_size
                    if nms_group_config.max_size is None
                    else nms_group_config.max_size
                ),
            )
        )
    return resolved_groups


def filter_nms_groups(
    bbox_centers: Float32[torch.Tensor, "batch_size num_boxes 2"],
    bbox_scores: Float32[torch.Tensor, "batch_size num_boxes"],
    bbox_classes: Int64[torch.Tensor, "batch_size num_boxes"],
    keep_masks: Bool[torch.Tensor, "batch_size num_boxes"],
    nms_groups: Sequence[NMSGroup],
    num_classes: int,
) -> Bool[torch.Tensor, "batch_size num_boxes"]:
    """
    Refine a keep mask with grouped center-distance NMS.

    Each NMS group becomes one channel of :func:`batch_circle_nms`, so all classes in a group
    land in the same channel and therefore compete with each other, while classes in different
    groups never suppress one another. No box is moved: a box simply appears as invalid in the
    channels of the groups that do not claim its class, which batch_circle_nms already handles,
    so the returned mask stays aligned with the input box axis.

    ``keep_masks`` is consumed as well as refined: a box already rejected by an earlier filter
    takes no part in the suppression, so it can never remove a survivor.

    A class no group claims is not subject to NMS at all and keeps its incoming mask value.
    Groups are disjoint, which resolve_nms_groups enforces, so no class is ever weighed against
    two different radii.

    Args:
        bbox_centers: BEV box centers as (center_x, center_y).
        bbox_scores: Confidence score for each box.
        bbox_classes: Class id of each box, a valid index in [0, num_classes).
        keep_masks: Keep mask from the earlier filters.
        nms_groups: Disjoint groups from resolve_nms_groups.
        num_classes: Number of classes of the head.

    Returns:
        The keep mask with grouped NMS applied.

    Raises:
        ValueError: If a group has no class ids or ids out of range.
    """
    batch_size, num_boxes = bbox_scores.shape
    num_groups = len(nms_groups)
    device = bbox_scores.device

    # (num_groups, num_classes) membership table: True where a group claims that class.
    group_class_masks = torch.zeros((num_groups, num_classes), dtype=torch.bool, device=device)
    for group_index, group in enumerate(nms_groups):
        class_ids = list(group.class_ids)
        if not class_ids:
            raise ValueError(f"NMS group {group_index} has no class ids.")
        if min(class_ids) < 0 or max(class_ids) >= num_classes:
            raise ValueError(
                f"NMS group {group_index} class ids {class_ids} are out of range for "
                f"{num_classes} classes."
            )
        group_class_masks[group_index, class_ids] = True

    # A box takes part in a group's channel only when that group claims its class and it survived
    # the earlier filters.
    # (num_groups, num_classes)[:, (batch_size, num_boxes)] & (batch_size, num_boxes)
    # -> (num_groups, batch_size, num_boxes) -> (batch_size, num_groups, num_boxes)
    group_masks = (group_class_masks[:, bbox_classes] & keep_masks).permute(1, 0, 2)

    # Every channel sees the same boxes. Only the validity mask differs, so these are views even
    # they expand the same boxes across groups.
    group_centers = bbox_centers.unsqueeze(1).expand(
        batch_size, num_groups, num_boxes, bbox_centers.shape[-1]
    )
    group_scores = bbox_scores.unsqueeze(1).expand(batch_size, num_groups, num_boxes)

    # One radius and one cap per channel, so all groups suppress in a single call. A group with a
    # radius of 0 disables the suppression in its channel: batch_circle_nms then only keeps its
    # highest-scoring boxes up to the cap.
    group_keep_masks = batch_circle_nms(
        bboxes_centers=group_centers,
        scores=group_scores,
        min_radii=[float(group.nms_radius) for group in nms_groups],
        valid_bboxes_masks=group_masks,
        post_max_sizes=[int(group.max_size) for group in nms_groups],
    )

    # Groups are disjoint, so a box enters at most one channel and is kept exactly when that
    # channel kept it. batch_circle_nms never keeps an invalid entry, so a box is already False
    # in every channel that does not claim its class.
    nms_keep_masks = group_keep_masks.any(dim=1)

    # Classes no group claims skip NMS and keep whatever the earlier filters decided.
    class_covered_masks = group_class_masks.any(dim=0)[bbox_classes]
    return nms_keep_masks | (keep_masks & ~class_covered_masks)
