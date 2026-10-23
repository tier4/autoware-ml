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

"""Shared test case building camera-only samples seen by plain pinhole cameras."""

from __future__ import annotations

from collections.abc import Sequence
import unittest

import numpy as np
import torch

from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.geometry.bbox_3d.lidar_bbox3d import LidarBBoxes3D
from autoware_ml.geometry.cameras.base_images import BaseImages, compose_lidar2images
from autoware_ml.geometry.points.lidar_points import LiDARPoints
from autoware_ml.types.geometry import Box3DCenterCoordinateType, Box3DFieldIndex, PointFeatureName


class PinholeCameraTestCase(unittest.TestCase):
    """Builders of camera-only samples seen by plain pinhole cameras.

    The 2D annotation, camera order and camera-only augmentation tests inherit them: a camera
    looking along the lidar x axis with a known focal length, gravity-center boxes in front of
    it and a sample that carries the images (and optionally one point) but nothing else.
    """

    # A camera looking along the lidar x axis: image right is lidar -y, image down is lidar -z.
    LIDAR2CAM_FORWARD = torch.tensor(
        [[0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )

    def build_sample(
        self, camera_image_data: BaseImages | None, with_points: bool = False
    ) -> ModelGTSample:
        """Build a sample holding images and optionally a one point cloud."""
        point_cloud_data = (
            LiDARPoints(
                points=torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
                point_feature_names=[PointFeatureName.X, PointFeatureName.Y, PointFeatureName.Z],
                timestamp=0.0,
            )
            if with_points
            else None
        )
        return ModelGTSample(
            lidar_point_cloud_samples=None,
            image_samples=None,
            point_cloud_data=point_cloud_data,
            camera_image_data=camera_image_data,
            detection3d_gt_bboxes_3d=None,
            segmentation3d_gt_sample=None,
        )

    def build_pinhole_images(
        self, lidar2cams: Sequence[torch.Tensor], width: int, height: int
    ) -> BaseImages:
        """Images of pinhole cameras with fx = fy = 100 and the principal point at the center."""
        num_cameras = len(lidar2cams)
        camera_intrinsics = torch.eye(3).repeat(num_cameras, 1, 1)
        camera_intrinsics[:, 0, 0] = camera_intrinsics[:, 1, 1] = 100.0
        camera_intrinsics[:, 0, 2] = width / 2.0
        camera_intrinsics[:, 1, 2] = height / 2.0
        lidar2cams_tensor = torch.stack(list(lidar2cams))
        return BaseImages(
            images=torch.zeros((num_cameras, 3, height, width)),
            timestamps=torch.zeros(num_cameras, dtype=torch.float64),
            camera_intrinsics=camera_intrinsics,
            camera_names=[f"CAM_{index}" for index in range(num_cameras)],
            lidar2images=compose_lidar2images(camera_intrinsics, lidar2cams_tensor),
            lidar2cams=lidar2cams_tensor,
            distortion_models=[""] * num_cameras,
            distortion_coefficients=[torch.zeros(0)] * num_cameras,
            augmented_camera_intrinsics=camera_intrinsics.clone(),
            image_augmentation_matrices=BaseImages.identity_image_augmentation_matrices(
                camera_intrinsics
            ),
        )

    def build_boxes(self, rows: Sequence[Sequence[float]], labels: Sequence[int]) -> LidarBBoxes3D:
        """Gravity-center boxes from ``[x, y, z, l, w, h, yaw]`` rows."""
        params = np.zeros((len(rows), len(Box3DFieldIndex)), dtype=np.float32)
        if len(rows):
            params[:, :7] = np.asarray(rows, dtype=np.float32)
        return LidarBBoxes3D.from_numpy(
            bbox_params=params,
            bbox_labels=np.asarray(labels, dtype=np.int32),
            bbox_label_names=["car"] * len(rows),
            bbox_num_lidar_points=np.full(len(rows), 10, dtype=np.int32),
            bbox_center_coordinate_type=Box3DCenterCoordinateType.GRAVITY_CENTER,
        )
