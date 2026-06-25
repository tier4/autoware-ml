# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared 3D annotation interpretation helpers."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import Any

FilterAttributeSet = frozenset[tuple[str, str]]


def normalize_filter_attributes(
    filter_attributes: Iterable[Sequence[str]] | None,
) -> FilterAttributeSet:
    """Normalize configured class-attribute exclusions for repeated lookup."""
    if filter_attributes is None:
        return frozenset()
    return frozenset(
        (str(class_name), str(attribute)) for class_name, attribute in filter_attributes
    )


def resolve_detection_class(
    instance: Mapping[str, Any],
    *,
    class_names: Sequence[str],
    name_mapping: Mapping[str, str | None] | None,
    label_to_category: Mapping[int, str] | None = None,
    filter_attributes: Collection[tuple[str, str]] | None = None,
    min_num_lidar_points: int = 1,
    use_valid_flag: bool = True,
) -> str | None:
    """Resolve one stored instance into a detector class or reject it."""
    if "bbox_label_3d" in instance and int(instance["bbox_label_3d"]) < 0:
        return None
    if use_valid_flag and not bool(instance.get("bbox_3d_isvalid", True)):
        return None

    num_lidar_points = int(instance.get("num_lidar_pts", 0))
    if num_lidar_points < min_num_lidar_points:
        return None

    raw_name = instance.get("gt_nusc_name")
    stored_name = _resolve_stored_name(instance, label_to_category)
    if raw_name is None:
        raw_name = stored_name
    if raw_name is None:
        return None

    raw_name = str(raw_name)
    mapped_name = _map_name(raw_name, name_mapping)
    if stored_name is not None:
        mapped_stored_name = _map_name(stored_name, name_mapping)
        if mapped_name != mapped_stored_name:
            raise ValueError(
                "Annotation label disagreement: "
                f"gt_nusc_name={raw_name!r} maps to {mapped_name!r}, while "
                f"bbox_label_3d maps to source class {stored_name!r} and target "
                f"{mapped_stored_name!r}."
            )

    if mapped_name not in class_names:
        return None
    if _has_filtered_attribute(instance, raw_name, filter_attributes):
        return None
    return mapped_name


def _resolve_stored_name(
    instance: Mapping[str, Any],
    label_to_category: Mapping[int, str] | None,
) -> str | None:
    """Decode ``bbox_label_3d`` through the annotation file's class table."""
    if "bbox_label_3d" not in instance or label_to_category is None:
        return None
    label = int(instance["bbox_label_3d"])
    if label not in label_to_category:
        raise ValueError(f"bbox_label_3d={label} is absent from the annotation class table.")
    return str(label_to_category[label])


def _map_name(raw_name: str, name_mapping: Mapping[str, str | None] | None) -> str | None:
    """Map a source class name into the configured detector taxonomy."""
    if name_mapping is None:
        return raw_name
    mapped_name = name_mapping.get(raw_name, raw_name)
    return str(mapped_name) if mapped_name is not None else None


def _has_filtered_attribute(
    instance: Mapping[str, Any],
    raw_name: str,
    filter_attributes: Collection[tuple[str, str]] | None,
) -> bool:
    """Return whether the raw class and attributes match an exclusion rule."""
    if not filter_attributes:
        return False
    attributes = {str(attribute) for attribute in instance.get("gt_attrs", [])}
    return any((raw_name, attribute) in filter_attributes for attribute in attributes)
