"""Behaviour-taxonomy class folding, shared by the detection and segmentation suites.

A ``class_groups`` spec maps a grouped class name to the trained class names it
contains, e.g. ``{"grouped_vehicle": ["car", "truck", "bus"]}``. It is a *full*
taxonomy: every trained class belongs to exactly one group and singleton groups
are allowed (``grouped_pedestrian: [pedestrian]``), so the whole label space is
remapped onto behaviour-equivalent classes and intra-group confusion counts as
correct: confusing a bus for a truck is a true positive for ``grouped_vehicle``.

A member that is not (yet) a trained class, e.g. ``ghost_point`` while it still
maps to the ignore index, is a forward-compatibility slot: it is skipped until
trained, as long as its group keeps at least one trained member.

There is no "additive" mode: grouping always replaces. A suite is run twice, once
without ``class_groups`` (per class) and once with it (grouped), so both
taxonomies are reported side by side.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch


def resolve_class_groups(
    class_names: tuple[str, ...] | None,
    class_groups: Mapping[str, Sequence[str]],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return the trained-index to grouped-index LUT and the ordered grouped names.

    The groups must partition every trained class exactly once (singletons
    allowed), and a member absent from ``class_names`` is skipped as a
    forward-compatibility slot as long as its group keeps at least one trained
    member. Grouped classes are ordered by definition order.

    Args:
        class_names: Trained class names in index order, or ``None``.
        class_groups: Grouped name to member names mapping.

    Returns:
        The LUT array and the grouped names tuple.
    """
    if class_names is None:
        raise ValueError("class_groups requires class_names so member names can be resolved.")
    name_to_index = {name: index for index, name in enumerate(class_names)}
    lut = np.full(len(class_names), -1, dtype=np.int64)
    grouped_names: list[str] = []
    claimed: dict[int, str] = {}
    for group_name, members in class_groups.items():
        if group_name in name_to_index:
            raise ValueError(f"Grouped class name {group_name!r} collides with a trained class.")
        indices = []
        for member in members:
            if member not in name_to_index:
                continue  # forward-compatibility slot for a not-yet-trained class
            index = name_to_index[member]
            if index in claimed:
                raise ValueError(
                    f"Class {member!r} appears in both {claimed[index]!r} and {group_name!r}."
                )
            claimed[index] = group_name
            indices.append(index)
        if not indices:
            raise ValueError(
                f"Grouped class {group_name!r} has no trained member, at least one of "
                f"{list(members)} must be a trained class."
            )
        position = len(grouped_names)
        grouped_names.append(group_name)
        for index in indices:
            lut[index] = position
    unassigned = [class_names[i] for i in range(len(class_names)) if lut[i] < 0]
    if unassigned:
        raise ValueError(
            f"class_groups must cover every trained class exactly once, unassigned: {unassigned}."
        )
    return lut, tuple(grouped_names)


def fold_labels(labels: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Relabel integer class indices through the LUT, out-of-range (ignore) untouched.

    Args:
        labels: Integer class indices of any shape.
        lut: Trained-index to grouped-index lookup table.

    Returns:
        Relabeled array of the same shape.
    """
    folded = labels.copy()
    in_range = (labels >= 0) & (labels < lut.shape[0])
    folded[in_range] = lut[labels[in_range]]
    return folded


def fold_labels_tensor(labels: torch.Tensor, lut: np.ndarray) -> torch.Tensor:
    """:func:`fold_labels` for tensors, computed on the labels' device.

    Args:
        labels: Integer class indices of any shape.
        lut: Trained-index to grouped-index lookup table.

    Returns:
        Relabeled tensor of the same shape, on the input device.
    """
    lut_tensor = torch.as_tensor(lut, dtype=labels.dtype, device=labels.device)
    folded = labels.clone()
    in_range = (labels >= 0) & (labels < lut_tensor.shape[0])
    folded[in_range] = lut_tensor[labels[in_range]]
    return folded


def fold_confusion(confusion: torch.Tensor, lut: np.ndarray, new_num_classes: int) -> torch.Tensor:
    """Fold a confusion matrix's rows and columns through the grouped-class LUT.

    Args:
        confusion: Trained-space ``(C, C)`` confusion matrix.
        lut: Trained-index to grouped-index lookup table.
        new_num_classes: Number of grouped classes.

    Returns:
        Grouped ``(new_num_classes, new_num_classes)`` confusion matrix.
    """
    index = torch.as_tensor(lut, dtype=torch.long, device=confusion.device)
    rows = torch.zeros(
        (new_num_classes, confusion.shape[1]), dtype=confusion.dtype, device=confusion.device
    )
    rows.index_add_(0, index, confusion)
    folded = torch.zeros(
        (new_num_classes, new_num_classes), dtype=confusion.dtype, device=confusion.device
    )
    folded.index_add_(1, index, rows)
    return folded
