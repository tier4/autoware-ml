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

"""Merge spatially-coupled object annotations (e.g. truck + trailer).

This transform reproduces the AWML info-generation behaviour where a sub-object
(such as a trailer) that overlaps or sits next to its target object (a truck) is
merged into a single, elongated target box. Unmatched sub-objects are left in
place and subsequently dropped by ``LoadAnnotations3D`` when their mapped class
is not in ``class_names`` - exactly as AWML's ``merge_objects`` + class filtering.

It runs on the raw ``instances`` list, before ``LoadAnnotations3D``, so the
sub-object class is still distinguishable from the target via ``name_mapping``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import pi
from typing import Any

import numpy as np
from shapely import affinity
from shapely.geometry import Polygon
from shapely.ops import unary_union

from autoware_ml.transforms.base import BaseTransform


class MergeObjects3D(BaseTransform):
    """Merge a sub-object box into a nearby target box into a single box.

    For each ``(target, [primary, secondary])`` rule, every ``primary`` box is
    matched against every ``secondary`` box; a pair that overlaps in BEV or whose
    front/back face centers are within ``distance_threshold`` is merged into one
    box labelled ``target``. Matching is greedy: each box participates in at most
    one merge. Unmatched boxes are left untouched.

    Required keys:
        instances: List of raw annotation dicts with ``bbox_3d`` (7 values:
            ``[cx, cy, cz, dx, dy, dz, yaw]``) and ``gt_nusc_name``.

    Generated keys:
        instances: Rewritten list with merged instances replacing matched pairs.
    """

    _required_keys = ["instances"]

    def __init__(
        self,
        *,
        merge_objects: Sequence[tuple[str, Sequence[str]]] | None = None,
        name_mapping: Mapping[str, str | None] | None = None,
        distance_threshold: float = 2.0,
        merge_type: str = "extend_longer",
    ) -> None:
        """Initialize the MergeObjects3D transform.

        Args:
            merge_objects: Rules ``(target, [primary_class, secondary_class])``
                using canonical class names after ``name_mapping``.
            name_mapping: Optional raw-to-canonical class-name mapping used to
                classify instances before matching.
            distance_threshold: Maximum front/back face-center distance in meters
                for boxes to be considered adjacent.
            merge_type: Box merge strategy, ``"extend_longer"`` or ``"union"``.
        """
        if merge_type not in {"extend_longer", "union"}:
            raise ValueError(f"merge_type must be 'extend_longer' or 'union', got {merge_type!r}.")
        merge_rules = [] if merge_objects is None else merge_objects
        self.merge_objects = []
        for index, rule in enumerate(merge_rules):
            if isinstance(rule, str) or not isinstance(rule, Sequence):
                raise TypeError(
                    "merge_objects entries must be two-item sequences, "
                    f"got {type(rule).__name__} at index {index}."
                )
            if len(rule) != 2:
                raise ValueError(
                    "merge_objects entries must contain [target, [primary, secondary]], "
                    f"got {list(rule)!r} at index {index}."
                )
            target, sources = rule
            if isinstance(sources, str) or not isinstance(sources, Sequence):
                raise TypeError(
                    "merge_objects sources must be two-item sequences, "
                    f"got {type(sources).__name__} at index {index}."
                )
            if len(sources) != 2:
                raise ValueError(
                    "merge_objects sources must contain [primary, secondary], "
                    f"got {sources!r} at index {index}."
                )
            primary, secondary = sources
            self.merge_objects.append((str(target), [str(primary), str(secondary)]))
        self.name_mapping = dict(name_mapping) if name_mapping is not None else None
        self.distance_threshold = float(distance_threshold)
        self.merge_type = merge_type

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Merge matched primary/secondary instance pairs within the sample.

        Args:
            input_dict: Sample dictionary holding the raw ``instances`` list.

        Returns:
            Updated sample dictionary whose ``instances`` list has each matched
            pair replaced by a single merged target instance, with unmatched
            instances left in place. Returned unchanged when there are no merge
            rules, no instances, or no pairs matched.
        """
        if not self.merge_objects:
            return input_dict

        instances = list(input_dict["instances"])
        if not instances:
            return input_dict

        canonical = [self._canonical_name(inst) for inst in instances]
        boxes = [np.asarray(inst["bbox_3d"], dtype=np.float64)[:7] for inst in instances]
        merge_function = (
            _merge_boxes_extend_longer if self.merge_type == "extend_longer" else _merge_boxes_union
        )

        consumed: set[int] = set()
        merged_instances: list[dict[str, Any]] = []
        for target, (primary, secondary) in self.merge_objects:
            primary_indices = [i for i, name in enumerate(canonical) if name == primary]
            secondary_indices = [i for i, name in enumerate(canonical) if name == secondary]
            for i in primary_indices:
                if i in consumed:
                    continue
                for j in secondary_indices:
                    if j in consumed or i == j:
                        continue
                    if _boxes_overlap(boxes[i], boxes[j]) or _boxes_proximity(
                        boxes[i], boxes[j], self.distance_threshold
                    ):
                        merged_instances.append(
                            self._merge_instances(
                                instances[i],
                                instances[j],
                                merge_function(boxes[i], boxes[j]),
                                target,
                            )
                        )
                        consumed.add(i)
                        consumed.add(j)
                        break

        if not consumed:
            return input_dict

        survivors = [inst for idx, inst in enumerate(instances) if idx not in consumed]
        input_dict["instances"] = merged_instances + survivors
        return input_dict

    def _canonical_name(self, instance: Mapping[str, Any]) -> str | None:
        """Resolve an instance's canonical class name via ``name_mapping``.

        Args:
            instance: Raw annotation dict carrying ``gt_nusc_name``.

        Returns:
            The canonical class name (raw name when ``name_mapping`` is ``None``),
            or ``None`` when the raw name maps to ``None``.

        Raises:
            KeyError: If ``gt_nusc_name`` is missing from the instance.
        """
        if "gt_nusc_name" not in instance:
            raise KeyError("MergeObjects3D requires every instance to contain 'gt_nusc_name'.")
        raw_name = instance["gt_nusc_name"]
        raw_name = str(raw_name)
        if self.name_mapping is None:
            return raw_name
        mapped = self.name_mapping.get(raw_name, raw_name)
        return str(mapped) if mapped is not None else None

    @staticmethod
    def _merge_instances(
        primary: Mapping[str, Any],
        secondary: Mapping[str, Any],
        merged_box: list[float],
        target: str,
    ) -> dict[str, Any]:
        """Build one merged instance from a matched primary/secondary pair."""
        merged = dict(primary)
        merged["bbox_3d"] = [float(value) for value in merged_box]
        merged["gt_nusc_name"] = target
        merged["num_lidar_pts"] = int(primary.get("num_lidar_pts", 0)) + int(
            secondary.get("num_lidar_pts", 0)
        )
        primary_velocity = np.asarray(primary.get("velocity", [0.0, 0.0]), dtype=np.float64)
        secondary_velocity = np.asarray(secondary.get("velocity", [0.0, 0.0]), dtype=np.float64)
        merged["velocity"] = ((primary_velocity + secondary_velocity) / 2.0).tolist()
        merged["gt_attrs"] = sorted(
            set(primary.get("gt_attrs", [])) | set(secondary.get("gt_attrs", []))
        )
        merged["bbox_3d_isvalid"] = bool(primary.get("bbox_3d_isvalid", True)) and bool(
            secondary.get("bbox_3d_isvalid", True)
        )
        # The merged box is a fresh target object; drop any stored per-box label
        # so downstream resolution keys off the new gt_nusc_name.
        merged.pop("bbox_label_3d", None)
        return merged


def _box_corners(box: np.ndarray) -> np.ndarray:
    """Return the four BEV corners of an oriented box ``[x, y, z, dx, dy, dz, yaw]``."""
    x, y, _, dx, dy, _, yaw = box
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    half_dx, half_dy = dx / 2.0, dy / 2.0
    return np.array(
        [
            [x - half_dx * cos_yaw + half_dy * sin_yaw, y - half_dx * sin_yaw - half_dy * cos_yaw],
            [x + half_dx * cos_yaw + half_dy * sin_yaw, y + half_dx * sin_yaw - half_dy * cos_yaw],
            [x + half_dx * cos_yaw - half_dy * sin_yaw, y + half_dx * sin_yaw + half_dy * cos_yaw],
            [x - half_dx * cos_yaw - half_dy * sin_yaw, y - half_dx * sin_yaw + half_dy * cos_yaw],
        ]
    )


def _boxes_overlap(box1: np.ndarray, box2: np.ndarray) -> bool:
    """Return whether two boxes overlap in the BEV plane."""
    return Polygon(_box_corners(box1)).intersects(Polygon(_box_corners(box2)))


def _boxes_proximity(box1: np.ndarray, box2: np.ndarray, distance_threshold: float) -> bool:
    """Return whether any front/back face centers are within the threshold."""

    def face_centers(box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the front and back face-center points along the box heading.

        Args:
            box: Oriented box ``[x, y, z, dx, dy, dz, yaw]``.

        Returns:
            Tuple ``(front, back)`` of ``(3,)`` face-center coordinates offset
            from the center by half the length ``dx`` along the yaw direction.
        """
        x, y, z, dx, _, _, yaw = box
        front = np.array([x + dx / 2.0 * np.cos(yaw), y + dx / 2.0 * np.sin(yaw), z])
        back = np.array([x - dx / 2.0 * np.cos(yaw), y - dx / 2.0 * np.sin(yaw), z])
        return front, back

    front1, back1 = face_centers(box1)
    front2, back2 = face_centers(box2)
    for a in (front1, back1):
        for b in (front2, back2):
            if np.linalg.norm(a - b) <= distance_threshold:
                return True
    return False


def _merge_boxes_extend_longer(box1: np.ndarray, box2: np.ndarray) -> list[float]:
    """Merge by elongating the larger box up to the far face of the smaller box."""

    def get_box_faces(box: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        x, y, _, dx, dy, _, yaw = box
        center = np.array([x, y])
        if dx >= dy:
            face1 = np.array([x + (dx / 2) * np.cos(yaw), y + (dx / 2) * np.sin(yaw)])
            face2 = np.array([x - (dx / 2) * np.cos(yaw), y - (dx / 2) * np.sin(yaw)])
        else:
            face1 = np.array(
                [x + (dy / 2) * np.cos(yaw + pi / 2), y + (dy / 2) * np.sin(yaw + pi / 2)]
            )
            face2 = np.array(
                [x - (dy / 2) * np.cos(yaw + pi / 2), y - (dy / 2) * np.sin(yaw + pi / 2)]
            )
        return center, face1, face2, dx, dy

    c1, c1_f1, c1_f2, dx1, dy1 = get_box_faces(box1)
    c2, c2_f1, c2_f2, dx2, dy2 = get_box_faces(box2)

    if dx1 * dy1 >= dx2 * dy2:
        larger_center, larger_f1, larger_f2, larger_dx, larger_dy, larger_box = (
            c1,
            c1_f1,
            c1_f2,
            dx1,
            dy1,
            box1,
        )
        smaller_center, smaller_f1, smaller_f2 = c2, c2_f1, c2_f2
    else:
        larger_center, larger_f1, larger_f2, larger_dx, larger_dy, larger_box = (
            c2,
            c2_f1,
            c2_f2,
            dx2,
            dy2,
            box2,
        )
        smaller_center, smaller_f1, smaller_f2 = c1, c1_f1, c1_f2

    # Far face of the smaller box relative to the larger box center.
    if np.linalg.norm(smaller_f1 - larger_center) > np.linalg.norm(smaller_f2 - larger_center):
        selected_smaller_face = smaller_f1
    else:
        selected_smaller_face = smaller_f2

    # Near face of the larger box relative to the smaller box center.
    if np.linalg.norm(larger_f1 - smaller_center) < np.linalg.norm(larger_f2 - smaller_center):
        selected_larger_face = larger_f1
    else:
        selected_larger_face = larger_f2

    axis_vector = selected_larger_face - larger_center
    axis_vector_normalized = axis_vector / np.linalg.norm(axis_vector)
    to_smaller_box = selected_smaller_face - larger_center
    projection_length = np.dot(to_smaller_box, axis_vector_normalized)
    projection_point = larger_center + projection_length * axis_vector_normalized

    elongation_vector = projection_point - selected_larger_face
    elongation_length = np.linalg.norm(elongation_vector)

    new_dx = larger_dx + elongation_length if larger_dx >= larger_dy else larger_dx
    new_dy = larger_dy + elongation_length if larger_dy > larger_dx else larger_dy
    new_center = larger_center + elongation_vector / 2.0

    new_z, new_dz = _merge_center_z_and_height(box1, box2)
    new_yaw = larger_box[6]

    return [new_center[0], new_center[1], new_z, new_dx, new_dy, new_dz, new_yaw]


def _merge_boxes_union(box1: np.ndarray, box2: np.ndarray) -> list[float]:
    """Merge via the minimum rotated rectangle covering both BEV footprints."""

    def shapely_box(box: np.ndarray) -> Polygon:
        x, y, _, dx, dy, _, yaw = box
        rect = Polygon([(-dx / 2, -dy / 2), (dx / 2, -dy / 2), (dx / 2, dy / 2), (-dx / 2, dy / 2)])
        rect = affinity.rotate(rect, yaw, origin=(0, 0), use_radians=True)
        return affinity.translate(rect, x, y)

    merged = unary_union([shapely_box(box1), shapely_box(box2)]).minimum_rotated_rectangle
    coords = list(merged.exterior.coords)[:-1]
    new_x = sum(point[0] for point in coords) / 4.0
    new_y = sum(point[1] for point in coords) / 4.0
    edge1 = float(np.hypot(coords[0][0] - coords[1][0], coords[0][1] - coords[1][1]))
    edge2 = float(np.hypot(coords[1][0] - coords[2][0], coords[1][1] - coords[2][1]))
    new_dx, new_dy = max(edge1, edge2), min(edge1, edge2)
    new_z, new_dz = _merge_center_z_and_height(box1, box2)
    if edge1 >= edge2:
        new_yaw = float(np.arctan2(coords[1][1] - coords[0][1], coords[1][0] - coords[0][0]))
    else:
        new_yaw = float(np.arctan2(coords[2][1] - coords[1][1], coords[2][0] - coords[1][0]))
    return [new_x, new_y, new_z, new_dx, new_dy, new_dz, new_yaw]


def _merge_center_z_and_height(box1: np.ndarray, box2: np.ndarray) -> tuple[float, float]:
    """Return center z and height spanning two center-based boxes."""
    bottom = min(box1[2] - box1[5] / 2.0, box2[2] - box2[5] / 2.0)
    top = max(box1[2] + box1[5] / 2.0, box2[2] + box2[5] / 2.0)
    height = top - bottom
    center_z = bottom + height / 2.0
    return float(center_z), float(height)
