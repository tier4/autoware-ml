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

"""Utilities shared by detection2d transforms."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import torch
from torchvision.tv_tensors import BoundingBoxFormat, BoundingBoxes


def convert_boxes_to_tv_tensor(
    boxes: torch.Tensor,
    canvas_size: tuple[int, int],
    box_format: str = "XYXY",
) -> BoundingBoxes:
    """Wrap plain tensor boxes as ``torchvision`` bounding boxes."""
    return BoundingBoxes(
        boxes,
        format=getattr(BoundingBoxFormat, box_format.upper()),
        canvas_size=canvas_size,
    )


def clone_target(target: Mapping[str, Any]) -> dict[str, Any]:
    """Clone a detection target dictionary without sharing tensor storage."""
    return copy.deepcopy(dict(target))


def resolve_current_epoch(context: Any) -> int:
    """Resolve the current epoch from the dataset/datamodule pipeline context."""
    dataset = getattr(context, "dataset", None)
    datamodule = getattr(dataset, "owner_datamodule", None)
    if datamodule is None:
        return 0
    return getattr(datamodule, "current_epoch", 0)
