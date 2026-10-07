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

"""Entry point of the runtime preprocessing, where the typed batch becomes model inputs.

The dataloader hands the models a ModelGTBatch. The batch adapter names every tensor of it
the way the model families read it, and the preprocessing layers derive the rest from there.
"""

from __future__ import annotations

from typing import Any

import torch

from autoware_ml.dataclasses.batch.sample_batch import ModelGTBatch
from autoware_ml.types.geometry import PointFieldIndex


class ModelGTBatchAdapter:
    """Name the tensors of a ModelGTBatch the way the models read them."""

    def __call__(self, batch: ModelGTBatch) -> dict[str, Any]:
        """Flatten the typed batch into the model inputs it carries.

        A task the batch does not carry adds no key, so a model that reads it raises a KeyError.

        Args:
            batch: Collated batch on the target device.

        Returns:
            Model inputs of every task the batch carries.
        """
        batch_inputs_dict: dict[str, Any] = {}
        if batch.point_cloud_gt_batch is not None:
            batch_inputs_dict |= self.point_cloud_inputs(batch)
        if batch.segmentation3d_gt_batch is not None:
            batch_inputs_dict["segment"] = batch.segmentation3d_gt_batch.gt_semantic_masks
        if batch.detection3d_gt_batch is not None:
            batch_inputs_dict |= self.detection3d_inputs(batch)
        if batch.image_gt_batch is not None:
            batch_inputs_dict |= self.image_inputs(batch)
        if batch.frame_meta_batch is not None:
            batch_inputs_dict["ego2global"] = batch.frame_meta_batch.ego2globals
            batch_inputs_dict["scene_token"] = batch.frame_meta_batch.scene_tokens
        return batch_inputs_dict

    @staticmethod
    def image_inputs(batch: ModelGTBatch) -> dict[str, Any]:
        """Name the camera tensors of the batch, grouped per sample.

        The collated tensors are laid out as (batch_size, num_cameras, ...), so iterating the
        leading dimension yields the cameras of one sample.

        Args:
            batch: Collated batch holding camera images.

        Returns:
            The images, their calibration, and the fused image the calibration classifier
            reads when the points have been projected onto them.
        """
        image_gt_batch = batch.image_gt_batch
        grouped: dict[str, Any] = {
            "img": list(image_gt_batch.images),
            "camera_intrinsics": list(image_gt_batch.camera_intrinsics),
            "lidar2cam": list(image_gt_batch.lidar2cams),
            "lidar2img": list(image_gt_batch.lidar2images),
            "img_aug_matrix": list(image_gt_batch.image_augmentation_matrices),
        }
        if image_gt_batch.depth_maps is not None:
            # The classifier reads one fused image per camera, so the batch and the camera
            # dimensions collapse into one
            fused = torch.cat([image_gt_batch.images, image_gt_batch.depth_maps], dim=2)
            grouped["fused_img"] = fused.flatten(0, 1)
        if image_gt_batch.calibration_statuses is not None:
            grouped["gt_calibration_status"] = image_gt_batch.calibration_statuses.flatten()
        return grouped

    @staticmethod
    def point_cloud_inputs(batch: ModelGTBatch) -> dict[str, Any]:
        """Name the point cloud tensors of the batch.

        Args:
            batch: Collated batch holding a point cloud.

        Returns:
            The concatenated points, their coordinates, the per sample split, and the point
            list the voxelizers consume.
        """
        point_cloud_gt_batch = batch.point_cloud_gt_batch
        counts = torch.bincount(
            point_cloud_gt_batch.batch_indices.long(), minlength=int(batch.infer_batch_size())
        )
        return {
            "coord": point_cloud_gt_batch.points[
                :, [PointFieldIndex.X, PointFieldIndex.Y, PointFieldIndex.Z]
            ],
            "feat": point_cloud_gt_batch.points,
            "offset": torch.cumsum(counts, dim=0),
            "batch_indices": point_cloud_gt_batch.batch_indices,
            "points": list(torch.split(point_cloud_gt_batch.points, counts.tolist())),
            "sample_count": int(counts.shape[0]),
            "time_lag_column": point_cloud_gt_batch.timestamp_difference_dim,
        }

    @staticmethod
    def detection3d_inputs(batch: ModelGTBatch) -> dict[str, Any]:
        """Name the 3D detection tensors of the batch, one entry per sample.

        The collated boxes are padded to the same length, the models read the valid boxes of
        every sample instead.

        Args:
            batch: Collated batch holding 3D detection ground truth.

        Returns:
            The boxes, the labels, and the point count of every sample.
        """
        detection3d_gt_batch = batch.detection3d_gt_batch
        valid_counts = detection3d_gt_batch.gt_valid_bboxes.tolist()
        return {
            "gt_boxes": [
                detection3d_gt_batch.gt_bboxes_3d[index, :count]
                for index, count in enumerate(valid_counts)
            ],
            "gt_labels": [
                detection3d_gt_batch.gt_labels_3d[index, :count].long()
                for index, count in enumerate(valid_counts)
            ],
            "gt_num_points": [
                detection3d_gt_batch.gt_bboxes_num_points[index, :count]
                for index, count in enumerate(valid_counts)
            ],
        }
