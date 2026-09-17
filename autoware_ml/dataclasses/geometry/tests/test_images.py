"""Unit tests for the ImageGTBatch and ImageSample named tuples."""

from __future__ import annotations

import unittest

import torch

from autoware_ml.dataclasses.geometry.images import ImageGTBatch, ImageSample
from autoware_ml.geometry.cameras.base_images import BaseImages


class ImageGTBatchTestCase(unittest.TestCase):
    """Shared fixtures for the ImageGTBatch tests."""

    NUM_CHANNELS = 3
    HEIGHT = 4
    WIDTH = 6

    def setUp(self) -> None:
        """Build two two-camera samples, without and with depth maps."""
        self.device = torch.device("cpu")
        self.num_cameras = 2
        self.samples = [
            self.make_images(self.num_cameras, fill=1.0),
            self.make_images(self.num_cameras, fill=2.0),
        ]
        self.samples_with_depth = [
            self.make_images(self.num_cameras, fill=1.0, with_depth=True),
            self.make_images(self.num_cameras, fill=2.0, with_depth=True),
        ]

    def make_images(self, num_cameras: int, fill: float, with_depth: bool = False) -> BaseImages:
        """
        Build a BaseImages sample whose per-camera tensors all carry ``fill`` so a test can tell
        which sample a collated slice came from.
        """
        image_shape = (num_cameras, self.NUM_CHANNELS, self.HEIGHT, self.WIDTH)
        depth_shape = (num_cameras, 1, self.HEIGHT, self.WIDTH)
        intrinsics = torch.eye(3, dtype=torch.float32).repeat(num_cameras, 1, 1) * fill
        return BaseImages(
            images=torch.full(image_shape, fill, dtype=torch.float32),
            depth_maps=torch.full(depth_shape, fill, dtype=torch.float32) if with_depth else None,
            timestamps=torch.full((num_cameras,), fill, dtype=torch.float64),
            camera_intrinsics=intrinsics,
            camera_names=[f"camera{i}" for i in range(num_cameras)],
            lidar2images=torch.full((num_cameras, 4, 4), fill, dtype=torch.float32),
            lidar2cams=torch.full((num_cameras, 4, 4), fill + 0.5, dtype=torch.float32),
            distortion_models=["plumb_bob"] * num_cameras,
            distortion_coefficients=[torch.zeros(5, dtype=torch.float32)] * num_cameras,
            augmented_camera_intrinsics=intrinsics.clone(),
            image_augmentation_matrices=torch.eye(4, dtype=torch.float32).repeat(num_cameras, 1, 1)
            * fill,
        )


class TestImageGTBatchFields(ImageGTBatchTestCase):
    """The named tuple contract of ImageGTBatch."""

    def test_field_order(self) -> None:
        self.assertEqual(
            ImageGTBatch._fields,
            (
                "images",
                "depth_maps",
                "camera_intrinsics",
                "image_augmentation_matrices",
                "lidar2images",
                "lidar2cams",
            ),
        )

    def test_is_immutable(self) -> None:
        batch = ImageGTBatch.collate_gt_samples(self.samples)

        with self.assertRaises(AttributeError):
            batch.images = torch.zeros(1)  # type: ignore


class TestImageGTBatchCollate(ImageGTBatchTestCase):
    """ImageGTBatch.collate_gt_samples."""

    def test_empty_sequence_returns_none(self) -> None:
        self.assertIsNone(ImageGTBatch.collate_gt_samples([]))

    def test_stacks_samples_along_batch_dimension(self) -> None:
        batch = ImageGTBatch.collate_gt_samples(self.samples)

        self.assertIsNotNone(batch)
        batch_size, num_cameras = len(self.samples), self.num_cameras
        assert batch is not None
        self.assertEqual(
            batch.images.shape,
            (batch_size, num_cameras, self.NUM_CHANNELS, self.HEIGHT, self.WIDTH),
        )
        self.assertEqual(batch.camera_intrinsics.shape, (batch_size, num_cameras, 3, 3))
        self.assertEqual(batch.image_augmentation_matrices.shape, (batch_size, num_cameras, 4, 4))
        self.assertEqual(batch.lidar2images.shape, (batch_size, num_cameras, 4, 4))
        self.assertEqual(batch.lidar2cams.shape, (batch_size, num_cameras, 4, 4))
        self.assertIsNone(batch.depth_maps)

        for index, sample in enumerate(self.samples):
            self.assertTrue(torch.equal(batch.images[index], sample.images))
            self.assertTrue(torch.equal(batch.camera_intrinsics[index], sample.camera_intrinsics))
            self.assertTrue(
                torch.equal(
                    batch.image_augmentation_matrices[index], sample.image_augmentation_matrices
                )
            )
            self.assertTrue(torch.equal(batch.lidar2images[index], sample.lidar2images))
            self.assertTrue(torch.equal(batch.lidar2cams[index], sample.lidar2cams))

    def test_mismatched_camera_counts_raise(self) -> None:
        samples = [self.make_images(2, fill=1.0), self.make_images(3, fill=2.0)]

        with self.assertRaises(RuntimeError):
            ImageGTBatch.collate_gt_samples(samples)

    def test_single_sample_adds_batch_dimension(self) -> None:
        sample = self.make_images(3, fill=4.0)

        batch = ImageGTBatch.collate_gt_samples([sample])
        assert batch is not None
        self.assertEqual(batch.images.shape, (1, 3, self.NUM_CHANNELS, self.HEIGHT, self.WIDTH))
        self.assertTrue(torch.equal(batch.images[0], sample.images))
        self.assertTrue(torch.equal(batch.camera_intrinsics[0], sample.camera_intrinsics))
        self.assertTrue(torch.equal(batch.lidar2images[0], sample.lidar2images))
        self.assertTrue(torch.equal(batch.lidar2cams[0], sample.lidar2cams))
        self.assertTrue(
            torch.equal(batch.image_augmentation_matrices[0], sample.image_augmentation_matrices)
        )

    def test_stacks_depth_maps_when_every_sample_carries_them(self) -> None:
        batch = ImageGTBatch.collate_gt_samples(self.samples_with_depth)
        assert batch is not None
        assert batch.depth_maps is not None
        self.assertEqual(
            batch.depth_maps.shape,
            (len(self.samples_with_depth), self.num_cameras, 1, self.HEIGHT, self.WIDTH),
        )
        self.assertTrue(torch.all(batch.depth_maps[0] == 1.0))
        self.assertTrue(torch.all(batch.depth_maps[1] == 2.0))

    def test_mixed_depth_maps_raise(self) -> None:
        samples = [self.samples_with_depth[0], self.samples[1]]

        with self.assertRaisesRegex(ValueError, "1 out of 2 samples with depth"):
            ImageGTBatch.collate_gt_samples(samples)

    def test_output_dtypes(self) -> None:
        batch = ImageGTBatch.collate_gt_samples(self.samples_with_depth)
        assert batch is not None
        for tensor in batch:
            assert tensor is not None
            self.assertEqual(tensor.dtype, torch.float32)


class TestImageGTBatchToDevice(ImageGTBatchTestCase):
    """ImageGTBatch.to_device."""

    def setUp(self) -> None:
        """Collate the fixture samples once, with and without depth maps."""
        super().setUp()
        self.batch_with_depth = ImageGTBatch.collate_gt_samples(self.samples_with_depth)
        self.batch_without_depth = ImageGTBatch.collate_gt_samples(self.samples)

    def test_returns_new_batch_with_equal_tensors(self) -> None:
        moved = self.batch_with_depth.to_device(self.device)  # type: ignore

        self.assertIsInstance(moved, ImageGTBatch)
        self.assertIsNot(moved, self.batch_with_depth)
        assert self.batch_with_depth is not None
        for original, result in zip(self.batch_with_depth, moved):
            self.assertTrue(torch.equal(original, result))  # type: ignore

    def test_preserves_none_depth_maps(self) -> None:
        moved = self.batch_without_depth.to_device(self.device)  # type: ignore

        self.assertIsNone(moved.depth_maps)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required to move tensors to GPU")
    def test_moves_every_tensor_to_cuda(self) -> None:
        moved = self.batch_with_depth.to_device(torch.device("cuda"))  # type: ignore

        for tensor in moved:
            assert tensor is not None
            self.assertEqual(tensor.device.type, "cuda")

        assert self.batch_with_depth is not None
        for tensor in self.batch_with_depth:
            assert tensor is not None
            self.assertEqual(tensor.device.type, "cpu")


class ImageSampleTestCase(unittest.TestCase):
    """Shared fixtures for the ImageSample tests."""

    def setUp(self) -> None:
        """Build the field values of one calibrated camera image and the sample holding them."""
        self.fields = dict(
            image_path="/data/camera0/000001.jpg",
            camera_name="camera0",
            timestamp=1700000000.25,
            camera_intrinsic=torch.eye(3, dtype=torch.float32),
            lidar2cam=torch.eye(4, dtype=torch.float32),
            lidar2image=torch.eye(4, dtype=torch.float32) * 2.0,
            distortion_model="plumb_bob",
            distortion_coefficients=torch.tensor([0.1, -0.2, 0.0, 0.0, 0.05], dtype=torch.float32),
        )
        self.sample = ImageSample(**self.fields)  # type: ignore

    def make_sample(self, **overrides) -> ImageSample:
        """Build an ImageSample from the fixture fields with some of them overridden."""
        return ImageSample(**{**self.fields, **overrides})


class TestImageSample(ImageSampleTestCase):
    """The named tuple contract of ImageSample."""

    def test_field_order(self) -> None:
        self.assertEqual(
            ImageSample._fields,
            (
                "image_path",
                "camera_name",
                "timestamp",
                "camera_intrinsic",
                "lidar2cam",
                "lidar2image",
                "distortion_model",
                "distortion_coefficients",
            ),
        )

    def test_holds_given_fields(self) -> None:
        self.assertEqual(self.sample.image_path, "/data/camera0/000001.jpg")
        self.assertEqual(self.sample.camera_name, "camera0")
        self.assertEqual(self.sample.timestamp, 1700000000.25)
        self.assertEqual(self.sample.camera_intrinsic.shape, (3, 3))
        self.assertEqual(self.sample.lidar2cam.shape, (4, 4))
        self.assertTrue(torch.equal(self.sample.lidar2image, torch.eye(4) * 2.0))
        self.assertEqual(self.sample.distortion_model, "plumb_bob")
        self.assertEqual(self.sample.distortion_coefficients.shape, (5,))

    def test_accepts_empty_distortion_coefficients(self) -> None:
        sample = self.make_sample(
            distortion_model="", distortion_coefficients=torch.empty(0, dtype=torch.float32)
        )

        self.assertEqual(sample.distortion_model, "")
        self.assertEqual(sample.distortion_coefficients.numel(), 0)

    def test_is_immutable(self) -> None:
        with self.assertRaises(AttributeError):
            self.sample.camera_name = "camera1"  # type: ignore

    def test_asdict_round_trip(self) -> None:
        rebuilt = ImageSample(**self.sample._asdict())

        self.assertEqual(rebuilt.image_path, self.sample.image_path)
        self.assertEqual(rebuilt.camera_name, self.sample.camera_name)
        self.assertEqual(rebuilt.timestamp, self.sample.timestamp)
        self.assertIs(rebuilt.camera_intrinsic, self.sample.camera_intrinsic)
        self.assertIs(rebuilt.lidar2cam, self.sample.lidar2cam)
        self.assertIs(rebuilt.lidar2image, self.sample.lidar2image)
        self.assertEqual(rebuilt.distortion_model, self.sample.distortion_model)
        self.assertIs(rebuilt.distortion_coefficients, self.sample.distortion_coefficients)


if __name__ == "__main__":
    unittest.main()
