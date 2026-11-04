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

"""Fixtures building small Sparse4D heads, their inputs and their losses for the head tests."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from jaxtyping import Bool, Float, Int64, Integer

from autoware_ml.anchors.tests.fixtures import FixedAnchorRecipe, fill_anchors
from autoware_ml.dataclasses.batch.detection3d import Detection3DGTBatch
from autoware_ml.dataclasses.models.detection3d.head_outputs import Sparse4DHeadOutputs
from autoware_ml.models.sparsedrive.sparse4d_head import ANCHOR_DIM, Sparse4DHead
from autoware_ml.models.sparsedrive.lidar_stages import SparseStage
from autoware_ml.models.detection3d.tests.ptv3_detection3d_fixtures import POINT_CLOUD_RANGE


def make_anchor_recipe(
    num_anchor: int = 8, low: float = 0.0, high: float = 8.0
) -> FixedAnchorRecipe:
    """Synthetic Sparse4D anchors whose centers spread over ``[low, high)`` in x and y."""
    rng = np.random.default_rng(0)
    anchors = np.zeros((num_anchor, ANCHOR_DIM), dtype=np.float32)
    anchors[:, :2] = rng.uniform(low, high, size=(num_anchor, 2))
    anchors[:, 2] = rng.uniform(-1.0, 1.0, size=num_anchor)
    anchors[:, 3:6] = np.log(rng.uniform(0.5, 4.0, size=(num_anchor, 3)))
    anchors[:, 7] = 1.0
    return FixedAnchorRecipe({"anchors": anchors})


def build_sparse4d_head(anchor_recipe: FixedAnchorRecipe, **overrides: object) -> Sparse4DHead:
    """Return a small temporal Sparse4D head matching the 8 m test geometry, its anchors filled."""
    kwargs: dict[str, object] = {
        "num_classes": 2,
        "embed_dims": 32,
        "anchor_recipe": anchor_recipe,
        "num_anchor": 8,
        "point_cloud_range": POINT_CLOUD_RANGE,
        "grid_size": 1.0,
        "feature_strides": (1,),
        "num_decoder": 2,
        "num_heads": 4,
        "num_groups": 4,
        "feedforward_channels": 64,
        "num_learnable_pts": 2,
        "num_output": 8,
        "num_temp_instances": 2,
        "min_observed_points": 1,
        "track_birth_score": 0.0,
    }
    kwargs.update(overrides)
    return fill_anchors(Sparse4DHead(**kwargs))


def random_stage(
    grid_dims: Sequence[int],
    num_voxels: int,
    batch_size: int = 1,
    seed: int = 0,
    requires_grad: bool = False,
) -> SparseStage:
    """Random voxels of every sample inside a stage grid, with 32 feature channels."""
    generator = torch.Generator().manual_seed(seed)
    total = batch_size * num_voxels
    grid_coord = torch.stack(
        [torch.randint(0, size, (total,), generator=generator) for size in grid_dims], dim=1
    )
    feat = torch.randn((total, 32), generator=generator).requires_grad_(requires_grad)
    offset = torch.arange(1, batch_size + 1, dtype=torch.long) * num_voxels
    return SparseStage(feat, grid_coord, offset)


def gt_batch(
    boxes: Sequence[Float[torch.Tensor, "_num_gt_boxes box_dim"]],
    labels: Sequence[Integer[torch.Tensor, " _num_gt_boxes"]],
    instance_ids: Sequence[Int64[torch.Tensor, " _num_gt_boxes"]] | None = None,
    num_points: int = 5,
) -> Detection3DGTBatch:
    """Detection ground truth of a batch from the boxes of every sample.

    Args:
        boxes: Boxes ``(M, 9)`` of every sample.
        labels: Class of every box of every sample.
        instance_ids: Object identity of every box of every sample, the box index by default.
        num_points: Lidar points of every box.

    Returns:
        The padded ground truth batch.
    """
    max_gts = max([1, *(sample_boxes.shape[0] for sample_boxes in boxes)])
    batch_size = len(boxes)
    gt_bboxes_3d = torch.zeros((batch_size, max_gts, 9))
    gt_labels_3d = torch.zeros((batch_size, max_gts), dtype=torch.int32)
    gt_bboxes_instance_ids = torch.full((batch_size, max_gts), -1, dtype=torch.int64)
    for index, (sample_boxes, sample_labels) in enumerate(zip(boxes, labels, strict=True)):
        count = sample_boxes.shape[0]
        gt_bboxes_3d[index, :count] = sample_boxes
        gt_labels_3d[index, :count] = sample_labels.to(torch.int32)
        gt_bboxes_instance_ids[index, :count] = (
            torch.arange(count) if instance_ids is None else instance_ids[index]
        )
    return Detection3DGTBatch(
        gt_bboxes_3d=gt_bboxes_3d,
        gt_labels_3d=gt_labels_3d,
        gt_valid_bboxes=torch.tensor([sample.shape[0] for sample in boxes], dtype=torch.int32),
        gt_bboxes_num_points=torch.full((batch_size, max_gts), num_points, dtype=torch.int32),
        gt_bboxes_instance_ids=gt_bboxes_instance_ids,
    )


def forward_frame(
    head: Sparse4DHead,
    stages: Sequence[SparseStage],
    timestamp: float = 100.0,
    gt_detections: Detection3DGTBatch | None = None,
    scene_tokens: Sequence[str] | None = None,
    lidar2global: Float[torch.Tensor, "batch_size 4 4"] | None = None,
) -> Sparse4DHeadOutputs:
    """Run the head on one frame of every lane, each lane streaming its own scene by default."""
    batch_size = stages[0].offset.shape[0]
    return head(
        stages,
        timestamps_seconds=torch.full((batch_size,), timestamp, dtype=torch.float64),
        lidar2global=(
            torch.eye(4, dtype=torch.float64).repeat(batch_size, 1, 1)
            if lidar2global is None
            else lidar2global
        ),
        scene_tokens=(
            [f"scene_{lane}" for lane in range(batch_size)]
            if scene_tokens is None
            else scene_tokens
        ),
        gt_detections=gt_detections,
    )


def head_loss(
    head: Sparse4DHead,
    outputs: Sparse4DHeadOutputs,
    gt_detections: Detection3DGTBatch,
    lanes: Sequence[int],
    gt_annotated_class_masks: Bool[torch.Tensor, "num_lanes num_classes"] | None = None,
) -> dict[str, Float[torch.Tensor, ""]]:
    """Assign the outputs of the given lanes to their ground truth and compute the losses."""
    targets = head.assign(
        outputs,
        [gt_detections.valid_bboxes_3d()[lane] for lane in lanes],
        [gt_detections.valid_labels_3d()[lane] for lane in lanes],
        [gt_detections.valid_bboxes_instance_ids()[lane] for lane in lanes],
        lanes,
    )
    return head.loss_from_targets(outputs, targets, lanes, gt_annotated_class_masks)
