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

"""Unit tests for the camera image loading transform."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torchvision.io import write_png

from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.dataclasses.geometry.images import ImageSample
from autoware_ml.geometry.cameras.base_images import BaseImages
from autoware_ml.transforms.camera.loading import LoadImagesFromFile
from autoware_ml.types.geometry import ImageChannel

# Red varies along the rows only and green along the columns only, so a vertical flip, a
# horizontal flip, a transpose, a rotation or a swap of the color channels each changes the
# loaded tensor. The image is wider than tall so a transposed load also changes the shape.
# Both gradients span ``[0, 255]``, which pins the unit-range scaling at both ends.
ROW_GRADIENT = torch.tensor([0, 85, 170, 255], dtype=torch.uint8)
COLUMN_GRADIENT = torch.tensor([0, 51, 102, 153, 204, 255], dtype=torch.uint8)
IMAGE_HEIGHT = ROW_GRADIENT.numel()
IMAGE_WIDTH = COLUMN_GRADIENT.numel()


def build_image_pixels(camera_marker: int) -> torch.Tensor:
    """Build the rgb pixels of a test camera image.

    Args:
        camera_marker: Value filling the blue channel, distinct per camera so a test can tell
            which camera an image was loaded from.

    Returns:
        torch.Tensor: uint8 (3, IMAGE_HEIGHT, IMAGE_WIDTH) rgb image whose red channel is a
            vertical gradient, green channel a horizontal gradient and blue channel the marker.
    """
    red = ROW_GRADIENT.view(IMAGE_HEIGHT, 1).expand(IMAGE_HEIGHT, IMAGE_WIDTH)
    green = COLUMN_GRADIENT.view(1, IMAGE_WIDTH).expand(IMAGE_HEIGHT, IMAGE_WIDTH)
    blue = torch.full((IMAGE_HEIGHT, IMAGE_WIDTH), camera_marker, dtype=torch.uint8)
    return torch.stack([red, green, blue])


class TestLoadImagesFromFile(unittest.TestCase):
    """Unit tests for the LoadImagesFromFile transform."""

    def setUp(self) -> None:
        """Create the temporary directory the test images are written to."""
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.image_root = Path(self._temporary_directory.name)
        self.addCleanup(self._temporary_directory.cleanup)
        self.camera_markers = {"camera0": 10, "camera1": 20}

    def build_image_sample(self, camera_name: str, timestamp: float = 1.5) -> ImageSample:
        """Write the structured PNG of the camera to disk and build the sample pointing at it.

        Args:
            camera_name: Name of the camera the sample belongs to, one of ``camera_markers``.
            timestamp: Timestamp of the sample.

        Returns:
            ImageSample referencing the image written to the temporary directory.
        """
        image_path = self.image_root / f"{camera_name}.png"
        write_png(build_image_pixels(self.camera_markers[camera_name]), str(image_path))
        return ImageSample(
            image_path=str(image_path),
            camera_name=camera_name,
            timestamp=timestamp,
            camera_intrinsic=torch.tensor(
                [[100.0, 0.0, 3.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]],
                dtype=torch.float32,
            ),
            lidar2cam=torch.eye(4, dtype=torch.float32),
            lidar2image=torch.eye(4, dtype=torch.float32),
            distortion_model="plumb_bob",
            distortion_coefficients=torch.zeros(5, dtype=torch.float32),
        )

    def build_model_gt_sample(self, image_samples: list[ImageSample]) -> ModelGTSample:
        """Build a sample holding only the given image metadata.

        Args:
            image_samples: Image metadata the loading transform reads.

        Returns:
            ModelGTSample with no camera image data loaded yet.
        """
        return ModelGTSample(
            lidar_point_cloud_samples=None,
            image_samples=image_samples,
            point_cloud_data=None,
            camera_image_data=None,
            detection3d_gt_bboxes_3d=None,
            segmentation3d_gt_sample=None,
        )

    def expected_images(self, camera_names: list[str], normalize_to_unit: bool) -> torch.Tensor:
        """Build the float32 tensor the loader must serve for the cameras, in the given order.

        Args:
            camera_names: Names of the cameras in loading order.
            normalize_to_unit: Whether the loader divides the pixels by 255.

        Returns:
            torch.Tensor: float32 (num_cameras, 3, IMAGE_HEIGHT, IMAGE_WIDTH) expected images.
        """
        pixels = torch.stack(
            [build_image_pixels(self.camera_markers[name]) for name in camera_names]
        ).to(torch.float32)
        if normalize_to_unit:
            return pixels / 255.0
        return pixels

    def test_defaults_serve_rgb_images_in_unit_range(self) -> None:
        """Test that the defaults load rgb images divided by 255 in sample order."""
        transform = LoadImagesFromFile()

        self.assertEqual(transform.color_type, ImageChannel.RGB)
        self.assertTrue(transform.normalize_to_unit)

    def test_missing_image_samples_key(self) -> None:
        """Test that missing 'image_samples' raises KeyError."""
        sample = self.build_model_gt_sample([])._replace(image_samples=None)

        with self.assertRaises(KeyError):
            LoadImagesFromFile()(sample)

    def test_empty_image_samples_raise(self) -> None:
        """Test that a sample without any image sample raises ValueError."""
        with self.assertRaises(ValueError):
            LoadImagesFromFile()(self.build_model_gt_sample([]))

    def test_loads_every_camera_in_sample_order(self) -> None:
        """Test that every camera is stacked along the leading dimension with exact pixels."""
        sample = self.build_model_gt_sample(
            [self.build_image_sample("camera0"), self.build_image_sample("camera1")]
        )

        output = LoadImagesFromFile(normalize_to_unit=False)(sample)

        assert output.camera_image_data is not None
        camera_image_data = output.camera_image_data
        self.assertIsInstance(camera_image_data, BaseImages)
        self.assertEqual(camera_image_data.camera_names, ["camera0", "camera1"])
        self.assertEqual(camera_image_data.images.shape, (2, 3, IMAGE_HEIGHT, IMAGE_WIDTH))
        self.assertEqual(camera_image_data.images.dtype, torch.float32)
        self.assertTrue(
            torch.equal(
                camera_image_data.images,
                self.expected_images(["camera0", "camera1"], normalize_to_unit=False),
            )
        )
        self.assertEqual(camera_image_data.timestamps.shape, (2,))
        self.assertEqual(camera_image_data.timestamps.dtype, torch.float64)
        self.assertEqual(camera_image_data.camera_intrinsics.shape, (2, 3, 3))
        self.assertEqual(camera_image_data.lidar2cams.shape, (2, 4, 4))
        self.assertEqual(camera_image_data.lidar2images.shape, (2, 4, 4))
        self.assertEqual(camera_image_data.distortion_models, ["plumb_bob", "plumb_bob"])
        self.assertEqual(len(camera_image_data.distortion_coefficients), 2)

    def test_rgb_channel_order_and_pixel_layout(self) -> None:
        """Test that the channels are served as rgb in (channel, height, width) layout."""
        sample = self.build_model_gt_sample([self.build_image_sample("camera0")])

        output = LoadImagesFromFile(normalize_to_unit=False)(sample)

        assert output.camera_image_data is not None
        image = output.camera_image_data.images[0]
        expected_red = ROW_GRADIENT.to(torch.float32).view(IMAGE_HEIGHT, 1)
        expected_green = COLUMN_GRADIENT.to(torch.float32).view(1, IMAGE_WIDTH)
        # Channel 0 is red: a vertical gradient, constant along every row.
        self.assertTrue(torch.equal(image[0], expected_red.expand(IMAGE_HEIGHT, IMAGE_WIDTH)))
        # Channel 1 is green: a horizontal gradient, constant along every column.
        self.assertTrue(torch.equal(image[1], expected_green.expand(IMAGE_HEIGHT, IMAGE_WIDTH)))
        # Channel 2 is blue: the uniform marker of camera0.
        self.assertTrue(torch.equal(image[2], torch.full((IMAGE_HEIGHT, IMAGE_WIDTH), 10.0)))
        # A bgr load, a flip along either axis or a transposed load would not match.
        self.assertFalse(torch.equal(image, image.flip(0)))
        self.assertFalse(torch.equal(image, image.flip(1)))
        self.assertFalse(torch.equal(image, image.flip(2)))
        self.assertNotEqual(image.shape, image.transpose(1, 2).shape)

    def test_normalize_to_unit(self) -> None:
        """Test that pixel values are divided by 255 only when requested."""
        sample = self.build_model_gt_sample(
            [self.build_image_sample("camera0"), self.build_image_sample("camera1")]
        )

        normalized = LoadImagesFromFile(normalize_to_unit=True)(sample)
        raw = LoadImagesFromFile(normalize_to_unit=False)(sample)

        assert normalized.camera_image_data is not None
        assert raw.camera_image_data is not None
        self.assertEqual(normalized.camera_image_data.images.dtype, torch.float32)
        self.assertEqual(raw.camera_image_data.images.dtype, torch.float32)
        self.assertTrue(
            torch.equal(
                normalized.camera_image_data.images,
                self.expected_images(["camera0", "camera1"], normalize_to_unit=True),
            )
        )
        self.assertTrue(
            torch.equal(
                raw.camera_image_data.images,
                self.expected_images(["camera0", "camera1"], normalize_to_unit=False),
            )
        )
        # The gradients span the full uint8 range, so the unit range is pinned at both ends.
        self.assertEqual(normalized.camera_image_data.images.min().item(), 0.0)
        self.assertEqual(normalized.camera_image_data.images.max().item(), 1.0)
        self.assertEqual(raw.camera_image_data.images.min().item(), 0.0)
        self.assertEqual(raw.camera_image_data.images.max().item(), 255.0)

    def test_metadata_carried_over(self) -> None:
        """Test that the calibration metadata of the sample is carried over."""
        sample = self.build_model_gt_sample([self.build_image_sample("camera0", timestamp=2.5)])
        assert sample.image_samples is not None
        image_sample = sample.image_samples[0]

        output = LoadImagesFromFile()(sample)

        assert output.camera_image_data is not None
        camera_image_data = output.camera_image_data
        self.assertEqual(camera_image_data.distortion_models, ["plumb_bob"])
        self.assertTrue(
            torch.allclose(camera_image_data.timestamps, torch.tensor([2.5], dtype=torch.float64))
        )
        self.assertTrue(torch.equal(camera_image_data.distortion_coefficients[0], torch.zeros(5)))
        self.assertTrue(
            torch.equal(camera_image_data.camera_intrinsics[0], image_sample.camera_intrinsic)
        )
        # Both are set only once the augmentation transforms have run.
        self.assertIsNone(camera_image_data.noises)
        self.assertIsNone(camera_image_data.depth_maps)
        self.assertTrue(
            torch.equal(
                camera_image_data.augmented_camera_intrinsics,
                camera_image_data.camera_intrinsics,
            )
        )
        self.assertTrue(
            torch.equal(
                camera_image_data.image_augmentation_matrices,
                torch.eye(4, dtype=torch.float32).repeat(1, 1, 1),
            )
        )


class TestLoadImagesFromFileShuffleOrder(unittest.TestCase):
    """Train-time shuffling of the camera order."""

    def setUp(self) -> None:
        """Write one image per camera whose pixels and focal length name the camera."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.image_samples = []
        for index in range(5):
            path = Path(directory.name) / f"CAM_{index}.png"
            write_png(torch.full((3, 4, 6), index * 10, dtype=torch.uint8), str(path))
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
        return loader(
            ModelGTSample(
                lidar_point_cloud_samples=None,
                image_samples=self.image_samples,
                point_cloud_data=None,
                camera_image_data=None,
                detection3d_gt_bboxes_3d=None,
                segmentation3d_gt_sample=None,
            )
        )

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


if __name__ == "__main__":
    unittest.main()
