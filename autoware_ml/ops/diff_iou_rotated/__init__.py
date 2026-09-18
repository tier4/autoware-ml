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

"""Differentiable rotated IoU backed by a CUDA vertex-sorting kernel."""

from autoware_ml.ops.diff_iou_rotated.diff_iou_rotated import (
    box2corners,
    convex_hull_area,
    diff_iou_rotated_2d,
    diff_iou_rotated_3d,
    enclosing_area,
    enclosing_box_aligned,
    enclosing_box_smallest,
    oriented_box_intersection_2d,
)

__all__ = [
    "box2corners",
    "convex_hull_area",
    "diff_iou_rotated_2d",
    "diff_iou_rotated_3d",
    "enclosing_area",
    "enclosing_box_aligned",
    "enclosing_box_smallest",
    "oriented_box_intersection_2d",
]
