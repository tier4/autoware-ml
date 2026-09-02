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

"""Partial-ignore support for partially annotated classes.

Some dataset scenes are annotated for every class except a subset. Training on
such frames must not punish background predictions of the un-annotated classes
as false positives. The per-frame ``annotation_status`` flag marks whether the
frame's scene carries those annotations; when it is ``False``, classification
weights for the ignored class columns are zeroed on that frame's queries.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def resolve_partial_ignore_labels(
    class_names: Sequence[str] | None,
    partial_ignore_classes: Sequence[str] | None,
) -> list[int] | None:
    """Map partially annotated class names to label indices.

    Args:
        class_names: Ordered detector class names.
        partial_ignore_classes: Class names that are only partially annotated.

    Returns:
        Label indices of the partially annotated classes, or ``None`` when
        partial-ignore is disabled.
    """
    if not partial_ignore_classes:
        return None
    if class_names is None:
        raise ValueError("class_names is required when partial_ignore_classes is set.")
    name_to_index = {name: index for index, name in enumerate(class_names)}
    missing = [name for name in partial_ignore_classes if name not in name_to_index]
    if missing:
        raise ValueError(f"partial_ignore_classes {missing} not found in class_names.")
    return [name_to_index[name] for name in partial_ignore_classes]


def normalize_status_flags(value: object, batch_size: int) -> list[bool]:
    """Normalize per-sample annotation-status flags to a plain bool list.

    Accepts tensors, nested lists, or scalars produced by different collation
    paths.

    Args:
        value: Raw ``annotation_status`` value from the batch.
        batch_size: Expected number of samples.

    Returns:
        One bool per sample.

    Raises:
        ValueError: If the flags are missing or their count does not match
            ``batch_size`` — silently defaulting would train the ignored
            classes' background as false positives.
    """
    if value is None:
        raise ValueError(
            "annotation_status is required when partial_ignore_classes is configured; "
            "check the datamodule's annotation_status_field and collation_map."
        )
    flattened = _flatten_flags(value)
    if len(flattened) != batch_size:
        raise ValueError(
            f"annotation_status carries {len(flattened)} flags for a batch of "
            f"{batch_size} samples; check the collation_map strategy for the key."
        )
    return flattened


def mask_ignored_columns(
    weights: torch.Tensor,
    rows: torch.Tensor,
    ignore_labels: Sequence[int],
) -> None:
    """Zero the ignored class columns on the selected rows, in place.

    Args:
        weights: Classification weights ``(num_rows, num_classes)``.
        rows: Row indices belonging to un-annotated frames.
        ignore_labels: Label indices of the partially annotated classes.
    """
    if rows.numel() == 0:
        return
    columns = torch.as_tensor(ignore_labels, device=weights.device, dtype=torch.long)
    weights[rows[:, None], columns] = 0.0


def _flatten_flags(value: object) -> list[bool]:
    if torch.is_tensor(value):
        return [bool(item) for item in value.detach().cpu().flatten().tolist()]
    if isinstance(value, (list, tuple)):
        flattened: list[bool] = []
        for item in value:
            flattened.extend(_flatten_flags(item))
        return flattened
    return [bool(value)]
