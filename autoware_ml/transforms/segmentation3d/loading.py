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

"""Segmentation-annotation loading transforms."""

from __future__ import annotations

from typing import Any

import numpy as np

from autoware_ml.transforms.base import BaseTransform


class LoadSegAnnotations3D(BaseTransform):
    """Load raw point-wise segmentation labels from metadata paths."""

    _required_keys = ["pts_semantic_mask_path"]

    def __init__(
        self,
        dtype: str = "uint8",
        label_mapping: dict[int, int] | None = None,
        max_label: int | None = None,
        class_mapping: dict[str, int] | None = None,
        ignore_index: int = -1,
    ) -> None:
        """Initialize the segmentation annotation loader.

        Args:
            dtype: Raw label dtype stored on disk.
            label_mapping: Optional raw-label to training-label mapping.
            max_label: Optional maximum raw label used to size the lookup table.
            class_mapping: Optional category-name to training-label mapping.
            ignore_index: Ignore label used for unknown categories.
        """
        self.dtype = np.dtype(dtype)
        self.label_mapping = label_mapping
        self.max_label = max_label
        self.class_mapping = class_mapping
        self.ignore_index = ignore_index

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Load point-wise semantic labels from the configured mask file.

        Args:
            input_dict: Sample metadata containing ``pts_semantic_mask_path``.

        Returns:
            Updated sample dictionary with ``pts_semantic_mask``.
        """
        labels = np.fromfile(input_dict["pts_semantic_mask_path"], dtype=self.dtype).astype(
            np.int64
        )
        idx_begin = input_dict.get("idx_begin")
        length = input_dict.get("length")

        if idx_begin is not None and length is not None:
            labels = labels[idx_begin : idx_begin + length]

        if self.class_mapping is not None and "pts_semantic_mask_categories" in input_dict:
            categories = input_dict["pts_semantic_mask_categories"]
            lookup_size = max(int(label) for label in categories.values()) + 1 if categories else 0
            lookup = np.full(lookup_size, fill_value=self.ignore_index, dtype=np.int64)
            for category_name, raw_label in categories.items():
                lookup[int(raw_label)] = self.class_mapping.get(
                    str(category_name), self.ignore_index
                )
            mapped = np.full(labels.shape, self.ignore_index, dtype=np.int64)
            valid = (labels >= 0) & (labels < lookup.shape[0])
            mapped[valid] = lookup[labels[valid]]
            labels = mapped
        elif self.label_mapping is not None:
            lookup_size = (
                self.max_label + 1 if self.max_label is not None else max(self.label_mapping) + 1
            )
            lookup = np.full(lookup_size, fill_value=self.ignore_index, dtype=np.int64)
            for source_label, target_label in self.label_mapping.items():
                lookup[int(source_label)] = int(target_label)
            mapped = np.full(labels.shape, self.ignore_index, dtype=np.int64)
            valid = (labels >= 0) & (labels < lookup.shape[0])
            mapped[valid] = lookup[labels[valid]]
            labels = mapped

        return {"pts_semantic_mask": labels}
