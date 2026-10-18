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

"""Unit tests for the camera transforms running on ``ModelGTSample``."""

from __future__ import annotations

from collections.abc import Sequence
import math
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import torch

from autoware_ml.dataclasses.batch.detection2d import Detection2DGTSample
from autoware_ml.dataclasses.batch.frame_meta import FrameMetaBatch, FrameMetaSample
from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.dataclasses.geometry.images import ImageSample
from autoware_ml.geometry.bbox_3d.lidar_bbox3d import LidarBBoxes3D
from autoware_ml.geometry.cameras.base_images import BaseImages, compose_lidar2images
from autoware_ml.geometry.points.lidar_points import LiDARPoints
from autoware_ml.transforms.camera.annotations2d import LoadAnnotations2DFromBoxes3D
from autoware_ml.transforms.camera.distortion import UndistortImage
from autoware_ml.transforms.camera.geometry import CameraGlobalRotScaleTrans, ImageAug3D
from autoware_ml.transforms.camera.loading import LoadImagesFromFile
from autoware_ml.transforms.image.image import PhotometricDistortion
from autoware_ml.transforms.point_cloud.geometry import GlobalRotScaleTrans
from autoware_ml.types.geometry import Box3DCenterCoordinateType, Box3DFieldIndex, PointFeatureName


def build_images(
    height: int = 8,
    width: int = 12,
    num_cameras: int = 2,
    distortion_coefficients: Sequence[float] = (),
) -> BaseImages:
    """Build images of the given size with a plain pinhole calibration."""
    camera_intrinsics = torch.eye(3, dtype=torch.float32).repeat(num_cameras, 1, 1)
    camera_intrinsics[:, 0, 0] = float(width)
    camera_intrinsics[:, 1, 1] = float(height)
    camera_intrinsics[:, 0, 2] = width / 2.0
    camera_intrinsics[:, 1, 2] = height / 2.0
    lidar2cams = torch.eye(4, dtype=torch.float32).repeat(num_cameras, 1, 1)
    return BaseImages(
        images=torch.rand((num_cameras, 3, height, width), dtype=torch.float32),
        timestamps=torch.zeros(num_cameras, dtype=torch.float64),
        camera_intrinsics=camera_intrinsics,
        camera_names=[f"CAM_{index}" for index in range(num_cameras)],
        lidar2images=torch.eye(4, dtype=torch.float32).repeat(num_cameras, 1, 1),
        lidar2cams=lidar2cams,
        distortion_models=["plumb_bob" if distortion_coefficients else ""] * num_cameras,
        distortion_coefficients=[torch.tensor(distortion_coefficients, dtype=torch.float32)]
        * num_cameras,
        augmented_camera_intrinsics=camera_intrinsics.clone(),
        image_augmentation_matrices=BaseImages.identity_image_augmentation_matrices(
            camera_intrinsics
        ),
    )


def build_sample(camera_image_data: BaseImages, with_points: bool = False) -> ModelGTSample:
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


class TestImageAug3D(unittest.TestCase):
    """Resize, crop and flip of the images and of their calibration."""

    def test_resizes_to_the_final_dimension(self) -> None:
        transform = ImageAug3D(
            final_dim=[4, 6], resize_lim=[1.0, 1.0], bot_pct_lim=[0.0, 0.0], training=False
        )

        sample = transform(build_sample(build_images()))

        self.assertEqual(sample.camera_image_data.images.shape[-2:], (4, 6))

    def test_composes_the_pixel_affine_into_the_intrinsics(self) -> None:
        # Halving the image halves the focal length and the principal point
        transform = ImageAug3D(
            final_dim=[4, 6], resize_lim=[0.5, 0.5], bot_pct_lim=[0.0, 0.0], training=False
        )

        sample = transform(build_sample(build_images()))

        augmented = sample.camera_image_data.augmented_camera_intrinsics
        self.assertAlmostEqual(float(augmented[0, 0, 0]), 6.0, places=5)
        self.assertAlmostEqual(float(augmented[0, 1, 1]), 4.0, places=5)

    def test_composes_onto_the_intrinsics_of_earlier_image_transforms(self) -> None:
        # An earlier transform already halved the image, a second halving quarters it
        images = build_images()
        images = images.model_copy(
            update={"augmented_camera_intrinsics": images.augmented_camera_intrinsics * 0.5}
        )
        transform = ImageAug3D(
            final_dim=[2, 3], resize_lim=[0.5, 0.5], bot_pct_lim=[0.0, 0.0], training=False
        )

        sample = transform(build_sample(images))

        augmented = sample.camera_image_data.augmented_camera_intrinsics
        self.assertAlmostEqual(float(augmented[0, 0, 0]), 3.0, places=5)

    def test_records_the_augmentation_matrix(self) -> None:
        transform = ImageAug3D(
            final_dim=[4, 6], resize_lim=[0.5, 0.5], bot_pct_lim=[0.0, 0.0], training=False
        )

        sample = transform(build_sample(build_images()))

        affines = sample.camera_image_data.image_augmentation_pixel_affines()
        self.assertAlmostEqual(float(affines[0, 0, 0]), 0.5, places=5)
        self.assertAlmostEqual(float(affines[0, 1, 1]), 0.5, places=5)

    def test_keeps_lidar2image_consistent_with_the_intrinsics(self) -> None:
        transform = ImageAug3D(
            final_dim=[4, 6], resize_lim=[0.5, 0.5], bot_pct_lim=[0.0, 0.0], training=False
        )

        sample = transform(build_sample(build_images()))

        images = sample.camera_image_data
        expected = torch.eye(4, dtype=torch.float32).repeat(2, 1, 1)
        expected[:, :3, :3] = images.augmented_camera_intrinsics
        torch.testing.assert_close(images.lidar2images, expected @ images.lidar2cams)


class TestUndistortImage(unittest.TestCase):
    """Undistortion of the images and of their calibration."""

    def test_leaves_undistorted_images_untouched(self) -> None:
        images = build_images()
        transform = UndistortImage()

        sample = transform(build_sample(images))

        torch.testing.assert_close(sample.camera_image_data.images, images.images)

    def test_clears_the_distortion_it_removed(self) -> None:
        transform = UndistortImage()

        sample = transform(build_sample(build_images(distortion_coefficients=(0.1, 0.0, 0.0, 0.0))))

        images = sample.camera_image_data
        self.assertEqual([len(c) for c in images.distortion_coefficients], [0, 0])
        self.assertEqual(list(images.distortion_models), ["", ""])


class TestPhotometricDistortion(unittest.TestCase):
    """Color distortion of the images."""

    def test_identity_parameters_keep_the_images(self) -> None:
        images = build_images()
        transform = PhotometricDistortion(probability=1.0)

        sample = transform(build_sample(images))

        torch.testing.assert_close(
            sample.camera_image_data.images, images.images, atol=1e-5, rtol=0.0
        )

    def test_brightness_changes_the_images(self) -> None:
        images = build_images()
        transform = PhotometricDistortion(probability=1.0, brightness=0.5)

        sample = transform(build_sample(images))

        self.assertEqual(sample.camera_image_data.images.shape, images.images.shape)
        self.assertFalse(torch.equal(sample.camera_image_data.images, images.images))


class TestCameraFollowsTheLidarAugmentation(unittest.TestCase):
    """The lidar space augmentation updates the camera calibration."""

    def test_lidar2cam_takes_the_inverse_of_the_augmentation(self) -> None:
        # A pure scaling by two moves the points, so the cameras take the halving
        transform = GlobalRotScaleTrans(
            yaw_rot_range=[0.0, 0.0], scale_ratio_range=[2.0, 2.0], translation_std=None
        )

        sample = transform(build_sample(build_images(), with_points=True))

        lidar2cams = sample.camera_image_data.lidar2cams
        expected = torch.eye(4, dtype=torch.float32)
        expected[:3, :3] = torch.eye(3, dtype=torch.float32) * 0.5
        torch.testing.assert_close(lidar2cams[0], expected, atol=1e-6, rtol=0.0)

    def test_sample_without_camera_stays_without_camera(self) -> None:
        transform = GlobalRotScaleTrans(
            yaw_rot_range=[0.0, 0.0], scale_ratio_range=[1.0, 1.0], translation_std=None
        )

        sample = transform(build_sample(None, with_points=True))

        self.assertIsNone(sample.camera_image_data)


# A camera looking along the lidar x axis: image right is lidar -y, image down is lidar -z.
LIDAR2CAM_FORWARD = torch.tensor(
    [[0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
)


def build_pinhole_images(lidar2cams: Sequence[torch.Tensor], width: int, height: int) -> BaseImages:
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


def build_boxes(rows: Sequence[Sequence[float]], labels: Sequence[int]) -> LidarBBoxes3D:
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


def project_annotations(
    rows: Sequence[Sequence[float]],
    labels: Sequence[int] | None = None,
    lidar2cams: Sequence[torch.Tensor] = (LIDAR2CAM_FORWARD,),
    width: int = 160,
    height: int = 96,
) -> Detection2DGTSample:
    """Run LoadAnnotations2DFromBoxes3D on boxes seen by the given cameras."""
    sample = build_sample(build_pinhole_images(lidar2cams, width, height))._replace(
        detection3d_gt_bboxes_3d=build_boxes(rows, [0] * len(rows) if labels is None else labels)
    )
    return LoadAnnotations2DFromBoxes3D()(sample).detection2d_gt_sample


class TestLoadAnnotations2DFromBoxes3D(unittest.TestCase):
    """Projection of the 3D boxes onto every camera as auxiliary 2D targets."""

    def test_projects_a_box_in_front_of_the_camera(self) -> None:
        # A 2 m cube on the optical axis, 10 m ahead, spans 9-11 m in depth: its gravity
        # center projects onto the principal point and its near face sets the extent.
        annotations = project_annotations(
            [[10.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]], labels=[2], width=100, height=100
        )

        self.assertEqual(annotations.gt_labels_2d[0].tolist(), [2])
        self.assertEqual(annotations.gt_labels_2d[0].dtype, torch.int32)
        torch.testing.assert_close(annotations.gt_centers_2d[0][0], torch.tensor([50.0, 50.0]))
        extent = 100.0 / 9.0
        torch.testing.assert_close(
            annotations.gt_bboxes_2d[0][0],
            torch.tensor([50.0 - extent, 50.0 - extent, 50.0 + extent, 50.0 + extent]),
        )

    def test_drops_a_box_behind_the_camera(self) -> None:
        annotations = project_annotations([[-10.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]])

        self.assertEqual(tuple(annotations.gt_bboxes_2d[0].shape), (0, 4))
        self.assertEqual(tuple(annotations.gt_centers_2d[0].shape), (0, 2))
        self.assertEqual(tuple(annotations.gt_labels_2d[0].shape), (0,))

    def test_keeps_a_box_whose_center_leaves_the_image(self) -> None:
        # The center at y = 9 m projects to x = -10 while the near-right corners are inside:
        # the box stays a target with its extent clipped and its center clamped to the edge,
        # as in the reference recipe.
        annotations = project_annotations([[10.0, 9.0, 0.0, 2.0, 6.0, 2.0, 0.0]])

        x1, _, x2, _ = annotations.gt_bboxes_2d[0][0].tolist()
        self.assertEqual(x1, 0.0)
        self.assertTrue(0 < x2 <= 160)
        self.assertEqual(float(annotations.gt_centers_2d[0][0][0]), 0.0)

    def test_clips_a_partially_visible_box_to_the_image(self) -> None:
        # The center projects at x = 15, the near-left corners far outside the left edge.
        annotations = project_annotations([[10.0, 6.5, 0.0, 2.0, 6.0, 2.0, 0.0]])

        x1, _, x2, _ = annotations.gt_bboxes_2d[0][0].tolist()
        self.assertEqual(x1, 0.0)
        self.assertTrue(0 < x2 <= 160)
        self.assertTrue(0 <= float(annotations.gt_centers_2d[0][0][0]) < 160)

    def test_follows_the_box_yaw(self) -> None:
        # A 4 m x 1 m box turned by 90 degrees shows its long side to the camera.
        straight = project_annotations([[10.0, 0.0, 0.0, 4.0, 1.0, 1.0, 0.0]])
        turned = project_annotations([[10.0, 0.0, 0.0, 4.0, 1.0, 1.0, math.pi / 2]])

        def width(annotations: Detection2DGTSample) -> float:
            x1, _, x2, _ = annotations.gt_bboxes_2d[0][0].tolist()
            return x2 - x1

        self.assertGreater(width(turned), 2.0 * width(straight))

    def test_assigns_every_box_to_the_cameras_that_see_it(self) -> None:
        # The second camera looks backward.
        backward = LIDAR2CAM_FORWARD @ torch.diag(torch.tensor([-1.0, -1.0, 1.0, 1.0]))
        annotations = project_annotations(
            [[10.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0], [-10.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]],
            labels=[0, 1],
            lidar2cams=(LIDAR2CAM_FORWARD, backward),
        )

        self.assertEqual(annotations.gt_labels_2d[0].tolist(), [0])
        self.assertEqual(annotations.gt_labels_2d[1].tolist(), [1])

    def test_handles_a_frame_without_boxes(self) -> None:
        annotations = project_annotations([])

        self.assertEqual(tuple(annotations.gt_bboxes_2d[0].shape), (0, 4))
        self.assertEqual(tuple(annotations.gt_centers_2d[0].shape), (0, 2))

    def test_needs_the_3d_boxes(self) -> None:
        sample = build_sample(build_pinhole_images([LIDAR2CAM_FORWARD], 160, 96))

        with self.assertRaises(KeyError):
            LoadAnnotations2DFromBoxes3D()(sample)


class TestLoadImagesFromFileShuffleOrder(unittest.TestCase):
    """Train-time shuffling of the camera order."""

    def setUp(self) -> None:
        """Write one image per camera whose pixels and focal length name the camera."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.image_samples = []
        for index in range(5):
            path = Path(directory.name) / f"CAM_{index}.png"
            cv2.imwrite(str(path), np.full((4, 6, 3), index * 10, dtype=np.uint8))
            camera_intrinsic = torch.eye(3) * (index + 1)
            camera_intrinsic[2, 2] = 1.0
            self.image_samples.append(
                ImageSample(
                    image_path=str(path),
                    camera_name=f"CAM_{index}",
                    timestamp=float(index),
                    camera_intrinsic=camera_intrinsic,
                    lidar2cam=torch.eye(4),
                    lidar2image=torch.eye(4),
                    distortion_model="",
                    distortion_coefficients=torch.zeros(0),
                )
            )

    def load(self, loader: LoadImagesFromFile) -> ModelGTSample:
        """Load the images of the five cameras through the given loader."""
        return loader(build_sample(None)._replace(image_samples=self.image_samples))

    def test_every_camera_field_follows_the_shuffled_order(self) -> None:
        loader = LoadImagesFromFile(shuffle_order=True)
        names = [f"CAM_{index}" for index in range(5)]
        np.random.seed(3)
        shuffled_seen = False
        for _ in range(8):
            sample = self.load(loader)
            images = sample.camera_image_data
            self.assertEqual(sorted(images.camera_names), names)
            self.assertEqual(
                [image.camera_name for image in sample.image_samples], list(images.camera_names)
            )
            shuffled_seen |= list(images.camera_names) != names
            for position, name in enumerate(images.camera_names):
                index = names.index(name)
                self.assertAlmostEqual(
                    float(images.images[position].mean()) * 255.0, index * 10, places=3
                )
                self.assertEqual(float(images.camera_intrinsics[position, 0, 0]), index + 1)
                self.assertEqual(float(images.timestamps[position]), float(index))
        self.assertTrue(shuffled_seen)

    def test_keeps_the_record_order_by_default(self) -> None:
        sample = self.load(LoadImagesFromFile())

        self.assertEqual(
            list(sample.camera_image_data.camera_names), [f"CAM_{index}" for index in range(5)]
        )


def _frame_meta(pose: torch.Tensor) -> FrameMetaSample:
    """Frame metadata at ``pose``; fills the sample time when the metadata carries one."""
    fields = {"ego2global": pose, "scene_token": "scene"}
    if "timestamp" in FrameMetaSample._fields:
        fields["timestamp"] = 0.0
    return FrameMetaSample(**fields)


class TestCameraGlobalRotScaleTrans(unittest.TestCase):
    """The global 3D augmentation of a camera-only sample."""

    def build_camera_sample(self) -> ModelGTSample:
        """A camera-only sample with one box in front of a forward-looking camera."""
        return build_sample(build_pinhole_images([LIDAR2CAM_FORWARD], 160, 96))._replace(
            detection3d_gt_bboxes_3d=build_boxes([[10.0, 1.0, 0.5, 4.0, 2.0, 1.5, 0.3]], [0])
        )

    def augment(self, sample: ModelGTSample, seed: int) -> ModelGTSample:
        """Apply a random rotation, scale and translation."""
        np.random.seed(seed)
        return CameraGlobalRotScaleTrans(
            yaw_rot_range=[-0.5, 0.5],
            scale_ratio_range=[0.9, 1.1],
            translation_std=[0.5, 0.5, 0.2],
        )(sample)

    def test_moves_the_boxes_and_records_the_augmentation_without_points(self) -> None:
        sample = self.build_camera_sample()
        raw_center = torch.cat([sample.detection3d_gt_bboxes_3d.center[0], torch.ones(1)])

        augmented = self.augment(sample, seed=0)

        transformation = augmented.lidar_transformation_sample.transformation_matrix
        self.assertIsNone(augmented.point_cloud_data)
        torch.testing.assert_close(
            augmented.detection3d_gt_bboxes_3d.center[0], (transformation @ raw_center)[:3]
        )

    def test_needs_the_images_loaded_first(self) -> None:
        sample = self.build_camera_sample()._replace(camera_image_data=None)

        with self.assertRaises(KeyError):
            CameraGlobalRotScaleTrans(yaw_rot_range=[0.0, 0.0], scale_ratio_range=[1.0, 1.0])(
                sample
            )

    def test_refuses_a_sample_with_points(self) -> None:
        sample = build_sample(
            build_pinhole_images([LIDAR2CAM_FORWARD], 160, 96), with_points=True
        )._replace(
            detection3d_gt_bboxes_3d=build_boxes([[10.0, 1.0, 0.5, 4.0, 2.0, 1.5, 0.3]], [0])
        )

        with self.assertRaisesRegex(ValueError, "GlobalRotScaleTrans"):
            self.augment(sample, seed=0)

    def test_keeps_the_2d_targets(self) -> None:
        # The boxes and the calibration move together, so the projection does not change.
        sample = self.build_camera_sample()
        before = LoadAnnotations2DFromBoxes3D()(sample).detection2d_gt_sample

        after = LoadAnnotations2DFromBoxes3D()(self.augment(sample, seed=1)).detection2d_gt_sample

        self.assertEqual(before.gt_labels_2d[0].tolist(), after.gt_labels_2d[0].tolist())
        torch.testing.assert_close(before.gt_bboxes_2d[0], after.gt_bboxes_2d[0], atol=1e-3, rtol=0)
        torch.testing.assert_close(
            before.gt_centers_2d[0], after.gt_centers_2d[0], atol=1e-3, rtol=0
        )

    def test_frames_augmented_on_their_own_stay_aligned_through_the_folded_poses(self) -> None:
        # One parked car seen from two ego poses. Every frame draws its own augmentation,
        # and the frame metadata folds it into the pose, so a temporal model maps the box of
        # the previous frame onto the box of the current one through the folded poses alone.
        global_box = torch.tensor([3012.0, -1180.0, 0.5, 1.0])
        poses = []
        for translation, yaw in (([3000.0, -1200.0, 0.0], 0.3), ([3002.0, -1198.5, 0.0], 0.35)):
            pose = torch.eye(4)
            pose[:2, :2] = torch.tensor(
                [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
            )
            pose[:3, 3] = torch.tensor(translation)
            poses.append(pose)
        augmented = []
        for seed, pose in enumerate(poses):
            local_box = (torch.linalg.inv(pose) @ global_box)[:3]
            sample = build_sample(build_pinhole_images([LIDAR2CAM_FORWARD], 160, 96))._replace(
                detection3d_gt_bboxes_3d=build_boxes(
                    [[*local_box.tolist(), 4.0, 2.0, 1.5, 0.0]], [0]
                ),
                frame_meta=_frame_meta(pose),
            )
            augmented.append(self.augment(sample, seed=seed))
        folded = FrameMetaBatch.collate_gt_samples(
            [sample.frame_meta for sample in augmented],
            [sample.lidar_transformation_sample for sample in augmented],
        ).ego2globals

        previous_center = torch.cat(
            [augmented[0].detection3d_gt_bboxes_3d.center[0], torch.ones(1)]
        )
        mapped = torch.linalg.inv(folded[1]) @ folded[0] @ previous_center

        torch.testing.assert_close(
            mapped[:3], augmented[1].detection3d_gt_bboxes_3d.center[0], atol=1e-2, rtol=0
        )
