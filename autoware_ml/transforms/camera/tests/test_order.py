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

"""Unit tests for the camera order augmentation."""

from __future__ import annotations

from typing import Any
import unittest

import numpy as np
import torch

from autoware_ml.dataclasses.batch.detection2d import Detection2DGTSample
from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.dataclasses.geometry.images import ImageSample
from autoware_ml.geometry.cameras.base_images import BaseImages
from autoware_ml.transforms.camera.annotations2d import LoadAnnotations2DFromBoxes3D
from autoware_ml.transforms.camera.order import ShuffleCameraOrder
from autoware_ml.transforms.camera.resize import ResizeMultiviewImages
from autoware_ml.transforms.camera.tests.pinhole_camera_test_case import PinholeCameraTestCase
from autoware_ml.types.geometry import Box2DFieldIndex

NUM_CAMERAS = 5


def tagged(offset: float, *shape: int) -> torch.Tensor:
    """One tensor of ``shape`` per camera, filled with the camera index plus ``offset``."""
    tags = torch.arange(NUM_CAMERAS, dtype=torch.float32) + offset
    return tags.view(-1, *([1] * len(shape))).expand(NUM_CAMERAS, *shape).clone()


def same(first: Any, second: Any) -> bool:
    """Whether two per-camera entries, tensors or plain values, are equal."""
    if isinstance(first, torch.Tensor):
        return torch.equal(first, second)
    return first == second


def drawn_order(sample: ModelGTSample) -> list[int]:
    """Index of the input camera every position of a shuffled sample holds."""
    return [int(name.removeprefix("CAM_")) for name in sample.camera_image_data.camera_names]


def per_camera_fields(sample: ModelGTSample) -> dict[str, Any]:
    """Every per-camera field of a sample by name: the BaseImages fields, records and 2D boxes."""
    fields = {name: getattr(sample.camera_image_data, name) for name in BaseImages.model_fields}
    fields["image_samples"] = sample.image_samples
    fields["gt_bboxes_2d"] = sample.detection2d_gt_sample.gt_bboxes_2d
    fields["gt_labels_2d"] = sample.detection2d_gt_sample.gt_labels_2d
    return fields


class TestShuffleCameraOrder(PinholeCameraTestCase):
    """Train-time shuffling of the camera order of a sample."""

    def build_tagged_sample(self, with_depth_maps: bool = True) -> ModelGTSample:
        """A sample of five cameras whose every per-camera field carries its camera index."""
        camera_image_data = BaseImages(
            images=tagged(0.0, 3, 4, 6),
            depth_maps=tagged(10.0, 2, 4, 6) if with_depth_maps else None,
            timestamps=tagged(20.0).to(torch.float64),
            camera_intrinsics=tagged(30.0, 3, 3),
            camera_names=[f"CAM_{camera}" for camera in range(NUM_CAMERAS)],
            lidar2images=tagged(40.0, 4, 4),
            lidar2cams=tagged(50.0, 4, 4),
            distortion_models=[f"model_{camera}" for camera in range(NUM_CAMERAS)],
            distortion_coefficients=[
                torch.full((camera + 1,), float(camera)) for camera in range(NUM_CAMERAS)
            ],
            augmented_camera_intrinsics=tagged(60.0, 3, 3),
            image_augmentation_matrices=tagged(70.0, 4, 4),
            noises=tagged(80.0, 4, 4),
            calibration_statuses=torch.arange(NUM_CAMERAS, dtype=torch.int64),
        )
        image_samples = [
            ImageSample(
                image_path=f"CAM_{camera}.png",
                camera_name=f"CAM_{camera}",
                timestamp=float(camera),
                camera_intrinsic=torch.eye(3),
                lidar2cam=torch.eye(4),
                lidar2image=torch.eye(4),
                distortion_model="",
                distortion_coefficients=torch.zeros(0),
            )
            for camera in range(NUM_CAMERAS)
        ]
        # Camera c holds c + 1 boxes of class c.
        detection2d_gt_sample = Detection2DGTSample(
            gt_bboxes_2d=[
                torch.full((camera + 1, len(Box2DFieldIndex)), float(camera))
                for camera in range(NUM_CAMERAS)
            ],
            gt_labels_2d=[
                torch.full((camera + 1,), camera, dtype=torch.int32)
                for camera in range(NUM_CAMERAS)
            ],
        )
        return self.build_sample(camera_image_data)._replace(
            image_samples=image_samples, detection2d_gt_sample=detection2d_gt_sample
        )

    def test_covers_every_field_of_the_camera_image_data(self) -> None:
        """
        Input: the fields the transform reorders and the fields of BaseImages.
        Expected: every field of BaseImages holds one entry per camera, so a field left out
        would keep the old order next to the shuffled images.
        Check: the two sets of field names are equal.
        """
        self.assertEqual(set(ShuffleCameraOrder.PER_CAMERA_FIELDS), set(BaseImages.model_fields))

    def test_every_per_camera_field_follows_the_drawn_permutation(self) -> None:
        """
        Input: a sample of five cameras whose every per-camera field (every BaseImages field,
        the image records and the 2D boxes and labels) carries the index of its camera.
        Expected: one permutation is drawn and every per-camera field follows it, so every
        camera keeps its pixels, depth maps, calibration, augmentation, record and 2D boxes.
        Check: read the drawn order off the camera names, then compare every field at every
        position with the input entry of the camera that moved there.
        """
        sample = self.build_tagged_sample()
        np.random.seed(3)

        shuffled = ShuffleCameraOrder()(sample)

        order = drawn_order(shuffled)
        self.assertEqual(sorted(order), list(range(NUM_CAMERAS)))
        self.assertNotEqual(order, list(range(NUM_CAMERAS)))
        before, after = per_camera_fields(sample), per_camera_fields(shuffled)
        for name, values in before.items():
            with self.subTest(field=name):
                self.assertEqual(len(after[name]), NUM_CAMERAS)
                for position, camera in enumerate(order):
                    self.assertTrue(same(after[name][position], values[camera]))

    def test_draws_the_permutation_from_numpy(self) -> None:
        """
        Input: shuffles run after seeding np.random.
        Expected: the permutation is drawn from np.random like the other augmentations, so
        seeding numpy, as every dataloader worker is seeded, reproduces it, and every sample
        draws its own.
        Check: the drawn order equals np.random.permutation under the same seed, and ten
        samples draw more than one order.
        """
        np.random.seed(11)
        expected = np.random.permutation(NUM_CAMERAS).tolist()
        np.random.seed(11)

        orders = [drawn_order(ShuffleCameraOrder()(self.build_tagged_sample())) for _ in range(10)]

        self.assertEqual(orders[0], expected)
        self.assertGreater(len({tuple(order) for order in orders}), 1)

    def test_keeps_the_order_with_probability_zero(self) -> None:
        """
        Input: the tagged sample through the transform with probability 0.
        Expected: the transform is skipped, so the cameras keep the order of their records.
        Check: the returned sample is the input sample itself.
        """
        sample = self.build_tagged_sample()

        self.assertIs(ShuffleCameraOrder(probability=0.0)(sample), sample)

    def test_needs_the_loaded_images(self) -> None:
        """
        Input: a sample whose images are not loaded yet.
        Expected: the transform reorders the loaded camera image data, so it runs after
        LoadImagesFromFile.
        Check: a KeyError naming camera_image_data is raised.
        """
        sample = self.build_tagged_sample()._replace(camera_image_data=None)

        with self.assertRaisesRegex(KeyError, "camera_image_data"):
            ShuffleCameraOrder()(sample)

    def test_rejects_a_field_of_another_number_of_cameras(self) -> None:
        """
        Input: a sample of five cameras whose image records or 2D boxes cover only four.
        Expected: those cannot follow a permutation of five cameras, and leaving them in place
        would pair a camera with the record or the boxes of another one.
        Check: a ValueError naming the field is raised.
        """
        sample = self.build_tagged_sample()
        detection2d_gt_sample = sample.detection2d_gt_sample
        cases = {
            "image_samples": sample._replace(image_samples=sample.image_samples[:4]),
            "gt_bboxes_2d": sample._replace(
                detection2d_gt_sample=detection2d_gt_sample._replace(
                    gt_bboxes_2d=detection2d_gt_sample.gt_bboxes_2d[:4]
                )
            ),
        }
        for name, case in cases.items():
            with self.subTest(field=name), self.assertRaisesRegex(ValueError, name):
                ShuffleCameraOrder()(case)

    def test_may_run_before_or_after_an_image_space_transform(self) -> None:
        """
        Input: the tagged sample, without depth maps, through ResizeMultiviewImages and the
        shuffle in both orders under the same seed.
        Expected: the image-space transform treats every camera alike and the shuffle moves
        its outputs together with the camera, so the order of the two does not matter.
        Check: every field of the two camera image data is equal.
        """
        resize = ResizeMultiviewImages(target_size=[8, 12])
        sample = self.build_tagged_sample(with_depth_maps=False)
        np.random.seed(5)
        resized_first = ShuffleCameraOrder()(resize(sample)).camera_image_data
        np.random.seed(5)
        shuffled_first = resize(ShuffleCameraOrder()(sample)).camera_image_data

        for name in BaseImages.model_fields:
            with self.subTest(field=name):
                before, after = getattr(resized_first, name), getattr(shuffled_first, name)
                if before is None:
                    self.assertIsNone(after)
                    continue
                for first, second in zip(before, after, strict=True):
                    self.assertTrue(same(first, second))

    def test_may_run_before_or_after_the_2d_projection(self) -> None:
        """
        Input: four boxes, one in front of each of a forward, a backward, a left and a right
        camera, projected by LoadAnnotations2DFromBoxes3D before and after the shuffle under
        the same seed.
        Expected: the shuffle reorders the 2D boxes together with the calibration they were
        projected with, so projecting first or shuffling first gives every camera the same
        targets.
        Check: the order is not the identity, and the 2D boxes and labels of every camera are
        equal in both orders.
        """
        turn_left = torch.tensor(
            [[0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0, 0, 0, 1.0]]
        )
        lidar2cams = [
            self.LIDAR2CAM_FORWARD,
            self.LIDAR2CAM_FORWARD @ torch.diag(torch.tensor([-1.0, -1.0, 1.0, 1.0])),
            self.LIDAR2CAM_FORWARD @ turn_left,
            self.LIDAR2CAM_FORWARD @ turn_left.T,
        ]
        boxes = self.build_boxes(
            [
                [10.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0],
                [-10.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0],
                [0.0, 10.0, 0.0, 2.0, 2.0, 2.0, 0.0],
                [0.0, -10.0, 0.0, 2.0, 2.0, 2.0, 0.0],
            ],
            [0, 1, 2, 3],
        )
        sample = self.build_sample(self.build_pinhole_images(lidar2cams, 160, 96))._replace(
            detection3d_gt_bboxes_3d=boxes
        )
        project = LoadAnnotations2DFromBoxes3D()
        np.random.seed(0)
        projected_first = ShuffleCameraOrder()(project(sample))
        np.random.seed(0)
        shuffled_first = project(ShuffleCameraOrder()(sample))

        self.assertNotEqual(drawn_order(projected_first), list(range(4)))
        self.assertEqual(
            [labels.tolist() for labels in projected_first.detection2d_gt_sample.gt_labels_2d],
            [[camera] for camera in drawn_order(projected_first)],
        )
        for first, second in zip(
            projected_first.detection2d_gt_sample, shuffled_first.detection2d_gt_sample
        ):
            for camera_first, camera_second in zip(first, second, strict=True):
                torch.testing.assert_close(camera_first, camera_second)


if __name__ == "__main__":
    unittest.main()
