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

"""End to end smoke tests of the camera pipeline, from the record table to a training step."""

from __future__ import annotations

from pathlib import Path
import tempfile
from types import MappingProxyType
import unittest

import cv2
import numpy as np
import polars as pl
import torch

from autoware_ml.databases.schemas.dataset_schemas import DatasetTableSchema
from autoware_ml.datamodule.t4dataset.dataset import T4Dataset
from autoware_ml.models.calibration_status.calibration_status import (
    CalibrationStatusClassifier,
)
from autoware_ml.models.common.backbones.resnet import ResNet18
from autoware_ml.models.common.heads.linear_cls_head import LinearClsHead
from autoware_ml.models.common.necks.global_average_pooling import GlobalAveragePooling
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor
from autoware_ml.datamodule.tests.corpus import SCENE, write_corpus
from autoware_ml.transforms.base import TransformsCompose
from autoware_ml.transforms.camera.distortion import UndistortImage
from autoware_ml.transforms.camera.resize import CropAndScale, ResizeCropFlipRotImage
from autoware_ml.transforms.camera.loading import LoadImagesFromFile
from autoware_ml.transforms.camera_lidar.camera_lidar import (
    Affine,
    AxisActivation,
    CalibrationMisalignment,
    LidarCameraFusion,
    MagnitudeRange,
)
from autoware_ml.transforms.image.image import PhotometricDistortion
from autoware_ml.transforms.point_cloud.geometry import CropBoxInner, GlobalRotScaleTrans
from autoware_ml.transforms.point_cloud.loading import LoadPointsFromFile

CAMERAS = ("CAM_FRONT", "CAM_BACK")
IMAGE_SIZE = (48, 64)
EGO_BOX = (-1.5, -1.0, -0.1, 1.5, 1.0, 1.0)


def write_images(root: Path, records: pl.DataFrame) -> pl.DataFrame:
    """Write one image per camera per record and attach the image frames to the table."""
    image_frames_column = []
    for record_index in range(records.height):
        camera_frames = []
        for camera_index, camera in enumerate(CAMERAS):
            relative = f"{SCENE}/data/{camera}/{record_index}.jpg"
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(path),
                np.random.default_rng(record_index).integers(
                    0, 255, size=(*IMAGE_SIZE, 3), dtype=np.uint8
                ),
            )
            camera_intrinsic = np.eye(3, dtype=np.float32)
            camera_intrinsic[0, 0] = 40.0
            camera_intrinsic[1, 1] = 40.0
            camera_intrinsic[0, 2] = IMAGE_SIZE[1] / 2.0
            camera_intrinsic[1, 2] = IMAGE_SIZE[0] / 2.0
            # The camera looks along the lidar x axis, so the points fall in front of it
            lidar2cam = np.array(
                [[0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0, 0, 0, 1]],
                dtype=np.float32,
            )
            camera_matrix = np.eye(4, dtype=np.float32)
            camera_matrix[:3, :3] = camera_intrinsic
            camera_frames.append(
                {
                    "image_frame_id": f"image-{record_index}-{camera_index}",
                    "image_keyframe": True,
                    "image_sensor_id": camera,
                    "image_sensor_channel_name": camera,
                    "image_timestamp_seconds": float(record_index),
                    "image_path": str(path),
                    "image_height": IMAGE_SIZE[0],
                    "image_width": IMAGE_SIZE[1],
                    "cam2img": camera_intrinsic.tolist(),
                    "image_distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
                    "image_distortion_model": "plumb_bob",
                    "image_sensor_to_ego_pose_matrix": np.eye(4).tolist(),
                    "image_frame_ego_pose_to_global_matrix": np.eye(4).tolist(),
                    "lidar2cam": lidar2cam.tolist(),
                    "lidar2img": (camera_matrix @ lidar2cam).tolist(),
                }
            )
        # The frames of every camera at the sample time, the records carry no camera sweep
        image_frames_column.append([camera_frames])
    return records.with_columns(
        pl.Series(
            DatasetTableSchema.IMAGE_FRAMES.name,
            image_frames_column,
            dtype=DatasetTableSchema.IMAGE_FRAMES.dtype,
        )
    )


def build_dataset(root: Path, records: pl.DataFrame, transforms: TransformsCompose) -> T4Dataset:
    """Build a camera dataset over the synthetic record table."""
    return T4Dataset(
        lidar_intensity_scale=255.0,
        database_root_path=str(root),
        max_num_3d_gt_bboxes=0,
        dataset_records_dataframe=records,
        transforms=transforms,
        dataset_tasks=MappingProxyType({}),
        det3d_supervised=True,
        seg3d_supervised=True,
        camera_names=CAMERAS,
    )


class TestCameraPipelineSmoke(unittest.TestCase):
    """The camera transforms carry the images and their calibration into the models."""

    def setUp(self) -> None:
        """Write a two frame corpus with images."""
        torch.manual_seed(0)
        np.random.seed(0)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.records = write_images(self.root, write_corpus(self.root, num_records=2))

    def test_multiview_pipeline_collates_the_images_and_their_calibration(self) -> None:
        transforms = TransformsCompose(
            pipeline=[
                LoadPointsFromFile(use_dim=(0, 1, 2, 3)),
                LoadImagesFromFile(),
                ResizeCropFlipRotImage(
                    target_size=[24, 32],
                    resize_range=[0.5, 0.5],
                    bottom_crop_ratio_range=[0.0, 0.0],
                    training=True,
                ),
                GlobalRotScaleTrans(
                    yaw_rot_range=(-0.2, 0.2),
                    scale_ratio_range=(0.95, 1.05),
                    translation_std=(0.1, 0.1, 0.05),
                ),
            ]
        )
        dataset = build_dataset(self.root, self.records, transforms)

        batch = dataset.collate_fn([dataset[0], dataset[1]])

        assert batch.image_gt_batch is not None
        self.assertEqual(batch.image_gt_batch.images.shape, (2, len(CAMERAS), 3, 24, 32))
        self.assertEqual(int(batch.infer_batch_size()), 2)

        batch_inputs = DataPreprocessor(preprocessor_modules=[])(batch, is_training=True)
        assert batch_inputs.image_data is not None
        self.assertEqual(batch_inputs.image_data.images.shape, (2, len(CAMERAS), 3, 24, 32))
        self.assertEqual(batch_inputs.image_data.lidar2images.shape, (2, len(CAMERAS), 4, 4))

    def test_calibration_pipeline_runs_a_training_step(self) -> None:
        transforms = TransformsCompose(
            pipeline=[
                LoadImagesFromFile(),
                LoadPointsFromFile(use_dim=(0, 1, 2, 3)),
                CropBoxInner(crop_box=EGO_BOX),
                UndistortImage(alpha=0.0),
                CalibrationMisalignment(
                    probability=0.5,
                    activate=AxisActivation(yaw=True),
                    yaw=MagnitudeRange(min_neg=4.0, max_neg=8.0, min_pos=4.0, max_pos=8.0),
                ),
                CropAndScale(probability=0.2, crop_ratio=0.8),
                Affine(probability=0.2, max_distortion=0.05),
                PhotometricDistortion(probability=0.2, brightness=0.1),
                LidarCameraFusion(max_depth=128.0, dilation_size=1, ego_box=list(EGO_BOX)),
            ]
        )
        dataset = build_dataset(self.root, self.records, transforms)
        batch = dataset.collate_fn([dataset[0], dataset[1]])
        batch_inputs = DataPreprocessor(preprocessor_modules=[])(batch, is_training=True)

        assert batch_inputs.image_data is not None
        assert batch_inputs.image_data.calibration_statuses is not None
        self.assertEqual(batch_inputs.image_data.fused_images().shape, (4, 5, *IMAGE_SIZE))
        self.assertEqual(batch_inputs.image_data.calibration_statuses.flatten().shape, (4,))

        model = CalibrationStatusClassifier(
            backbone=ResNet18(in_channels=5),
            neck=GlobalAveragePooling(),
            head=LinearClsHead(
                num_classes=2, in_channels=512, loss=torch.nn.CrossEntropyLoss(), topk=(1,)
            ),
            optimizer=lambda params: torch.optim.AdamW(params, lr=1e-3),
        )
        model.log_dict = lambda *args, **kwargs: None

        loss = model.training_step(batch_inputs, batch_idx=0)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))

    def test_the_projection_lands_the_points_on_the_images(self) -> None:
        transforms = TransformsCompose(
            pipeline=[
                LoadImagesFromFile(),
                LoadPointsFromFile(use_dim=(0, 1, 2, 3)),
                LidarCameraFusion(max_depth=128.0, dilation_size=1),
            ]
        )
        dataset = build_dataset(self.root, self.records, transforms)

        sample = dataset[0]

        assert sample.camera_image_data is not None
        depth_maps = sample.camera_image_data.depth_maps
        assert depth_maps is not None
        self.assertEqual(depth_maps.shape, (len(CAMERAS), 2, *IMAGE_SIZE))
        self.assertGreater(float(depth_maps[0, 0].max()), 0.0)
