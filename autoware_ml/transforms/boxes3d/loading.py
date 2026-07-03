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

"""3D bounding-box annotation loading transforms."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np

from autoware_ml.transforms.base import BaseTransform
from autoware_ml.transforms.boxes3d.annotations import (
    normalize_filter_attributes,
    resolve_detection_class,
)

logger = logging.getLogger(__name__)


class LoadAnnotations3D(BaseTransform):
    """Parse raw instance annotations into 3D bounding-box targets.

    Reads the ``instances`` list from the sample, applies an optional
    class-name mapping, filters by minimum lidar point count, and produces
    ``gt_boxes`` (Nx9, with velocity), ``gt_names``, ``gt_labels``, and
    ``gt_num_points``.

    When ``name_mapping`` is ``None``, class names are read from
    ``class_names`` in the sample dict.

    Required keys:
        instances: List of raw annotation dicts from the dataset.

    Optional keys:
        class_names: List of canonical class names used for label assignment
                     when ``name_mapping`` is ``None``.

    Generated keys:
        gt_boxes: Bounding boxes (N, 9) - 7 box params + 2 velocity components.
        gt_names: Canonical class names per box.
        gt_labels: Integer label indices per box.
        gt_num_points: Lidar point count per box.
    """

    _required_keys = ["instances"]
    _optional_keys = ["class_names", "label_to_category"]

    def __init__(
        self,
        *,
        name_mapping: Mapping[str, str | None] | None = None,
        filter_attributes: list[list[str]] | None = None,
        use_valid_flag: bool = True,
    ) -> None:
        """Initialize the LoadAnnotations3D transform.

        Args:
            name_mapping: Optional raw-to-canonical class-name mapping. Values set
                to ``None`` drop the corresponding raw class.
            filter_attributes: Attribute groups used to filter raw annotations.
            use_valid_flag: Whether to honor the raw ``bbox_3d_isvalid`` flag.
        """
        self.name_mapping = dict(name_mapping) if name_mapping is not None else None
        self.filter_attributes = normalize_filter_attributes(filter_attributes)
        self.use_valid_flag = use_valid_flag
        self._validated_class_names: set[tuple[str, ...]] = set()

    def apply_defaults(self, input_dict: dict[str, Any]) -> None:
        """Populate optional class-name metadata when it is absent.

        Args:
            input_dict: Sample dictionary updated in place.
        """
        input_dict.setdefault("class_names", [])

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert raw instance annotations into detection target arrays.

        Args:
            input_dict: Sample dictionary containing raw ``instances``.

        Returns:
            Updated sample dictionary with 3D box target keys.
        """
        instances = input_dict["instances"]
        class_names = input_dict.get("class_names", [])
        canonical_list = list(class_names)
        self._validate_name_mapping_targets(canonical_list)

        gt_boxes, gt_names, gt_num_points = [], [], []

        for inst in instances:
            canonical = resolve_detection_class(
                inst,
                class_names=canonical_list,
                name_mapping=self.name_mapping,
                label_to_category=input_dict.get("label_to_category"),
                filter_attributes=self.filter_attributes,
                use_valid_flag=self.use_valid_flag,
            )
            if canonical is None:
                continue

            num_pts = int(inst.get("num_lidar_pts", 0))
            box = list(inst["bbox_3d"])  # 7 values: cx cy cz dx dy dz yaw
            vel = list(inst.get("velocity", [0.0, 0.0]))
            vel = [0.0 if not np.isfinite(v) else float(v) for v in vel]
            gt_boxes.append(box + vel)
            gt_names.append(canonical)
            gt_num_points.append(num_pts)

        if gt_boxes:
            boxes_arr = np.array(gt_boxes, dtype=np.float32)
        else:
            boxes_arr = np.zeros((0, 9), dtype=np.float32)

        names_arr = np.array(gt_names, dtype=object)

        name_to_label = {n: i for i, n in enumerate(canonical_list)}
        gt_labels = np.array([name_to_label[n] for n in gt_names], dtype=np.int64)

        input_dict["gt_boxes"] = boxes_arr
        input_dict["gt_names"] = names_arr
        input_dict["gt_labels"] = gt_labels
        input_dict["gt_num_points"] = np.array(gt_num_points, dtype=np.int64)
        return input_dict

    def _validate_name_mapping_targets(self, class_names: list[str]) -> None:
        """Log mapping targets dropped because they are not detector classes.

        A ``name_mapping`` target absent from ``class_names`` is treated as an
        intentional drop (the AWML convention, e.g. mapping ``trailer`` to the
        non-target class ``trailer`` so standalone trailers are excluded). Such
        boxes are dropped downstream by ``resolve_detection_class``; this only
        surfaces them once per distinct class-name set.
        """
        if self.name_mapping is None:
            return
        class_name_key = tuple(str(name) for name in class_names)
        if class_name_key in self._validated_class_names:
            return
        self._validated_class_names.add(class_name_key)

        dropped_targets = sorted(
            {
                str(mapped_name)
                for mapped_name in self.name_mapping.values()
                if mapped_name is not None and str(mapped_name) not in class_name_key
            }
        )
        if dropped_targets:
            logger.info(
                "name_mapping targets not in class_names will be dropped: %s", dropped_targets
            )
