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

"""Builders of the typed model inputs the model tests feed in."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from autoware_ml.dataclasses.batch.detection3d import Detection3DGTBatch
from autoware_ml.dataclasses.batch.sample_batch import ModelGTBatch
from autoware_ml.dataclasses.batch.segmentation3d import Segmentation3DGTBatch
from autoware_ml.dataclasses.geometry.images import ImageGTBatch
from autoware_ml.dataclasses.geometry.point_clouds import PointCloudGTBatch
from autoware_ml.dataclasses.geometry.voxels import VoxelsData
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs


def build_detection_gt_batch(
    gt_boxes: Sequence[torch.Tensor], gt_labels: Sequence[torch.Tensor]
) -> Detection3DGTBatch:
    """Pad the boxes and labels of every sample into a detection ground truth batch.

    Args:
        gt_boxes: Boxes of every sample.
        gt_labels: Labels of every sample.

    Returns:
        The collated detection ground truth.
    """
    max_boxes = max(1, max(boxes.shape[0] for boxes in gt_boxes))
    box_size = gt_boxes[0].shape[1]
    device = gt_boxes[0].device
    padded_boxes = torch.zeros((len(gt_boxes), max_boxes, box_size), device=device)
    padded_labels = torch.full((len(gt_boxes), max_boxes), -1, dtype=torch.int32, device=device)
    for index, (boxes, labels) in enumerate(zip(gt_boxes, gt_labels)):
        padded_boxes[index, : boxes.shape[0]] = boxes
        padded_labels[index, : labels.shape[0]] = labels.to(torch.int32)
    valid = torch.tensor([boxes.shape[0] for boxes in gt_boxes], dtype=torch.int32, device=device)
    return Detection3DGTBatch(
        gt_bboxes_3d=padded_boxes,
        gt_labels_3d=padded_labels,
        gt_valid_bboxes=valid,
        gt_bboxes_num_points=torch.zeros(
            (len(gt_boxes), max_boxes), dtype=torch.int32, device=device
        ),
    )


def build_point_cloud_batch(
    points: Sequence[torch.Tensor], timestamp_difference_dim: int = -1
) -> PointCloudGTBatch:
    """Concatenate the points of every sample into a point cloud batch.

    Args:
        points: Points of every sample.
        timestamp_difference_dim: Column of the time lag, -1 for points without one.

    Returns:
        The collated point cloud.
    """
    batch_indices = torch.cat(
        [
            torch.full((sample.shape[0],), index, dtype=torch.int32, device=sample.device)
            for index, sample in enumerate(points)
        ]
    )
    return PointCloudGTBatch(
        points=torch.cat(list(points)),
        batch_indices=batch_indices,
        batch_size=len(points),
        timestamp_difference_dim=timestamp_difference_dim,
    )


def voxels_data_from_zyx(
    voxels: torch.Tensor, num_points: torch.Tensor, voxel_coords: torch.Tensor
) -> VoxelsData:
    """Wrap voxels whose coordinates are given as ``[batch, z, y, x]`` rows.

    Args:
        voxels: Padded voxel features.
        num_points: Point count of every voxel.
        voxel_coords: Sample and coordinate of every voxel.

    Returns:
        The voxels with their coordinates in (x, y, z) order.
    """
    return VoxelsData(
        voxels=voxels,
        coords=voxel_coords[:, [3, 2, 1]].to(torch.int32),
        num_points=num_points.to(torch.int32),
        batch_indices=voxel_coords[:, 0].to(torch.int32),
        point_voxel_indices=torch.zeros((0,), dtype=torch.int64),
        num_dropped_voxels=torch.zeros((), dtype=torch.int64),
    )


def build_batch_inputs(
    point_cloud: PointCloudGTBatch | None = None,
    detection: Detection3DGTBatch | None = None,
    segmentation: Segmentation3DGTBatch | None = None,
    images: ImageGTBatch | None = None,
    voxels: VoxelsData | None = None,
) -> ModelBatchInputs:
    """Build model inputs from the batches a test needs.

    Args:
        point_cloud: Point cloud batch.
        detection: Detection ground truth batch.
        segmentation: Segmentation ground truth batch.
        images: Image batch.
        voxels: Voxels a preprocessing layer would add.

    Returns:
        The model inputs.
    """
    batch = ModelGTBatch(
        point_cloud_gt_batch=point_cloud,
        detection3d_gt_batch=detection,
        segmentation3d_gt_batch=segmentation,
        image_gt_batch=images,
    )
    return ModelBatchInputs.from_gt_batch(batch).replace(voxels_data=voxels)
