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
        """Build two two-camera samples, plain and with every optional field filled in."""
        self.device = torch.device("cpu")
        self.num_cameras = 2
        self.samples = [
            self.make_images(self.num_cameras, fill=1.0),
            self.make_images(self.num_cameras, fill=2.0),
        ]
        self.samples_with_depth = [
            self.make_images(self.num_cameras, fill=1.0, with_depth=True, with_statuses=True),
            self.make_images(self.num_cameras, fill=2.0, with_depth=True, with_statuses=True),
        ]

    def make_images(
        self,
        num_cameras: int,
        fill: float,
        with_depth: bool = False,
        with_statuses: bool = False,
    ) -> BaseImages:
        """
        Build a BaseImages sample whose per-camera tensors all carry ``fill`` so a test can tell
        which sample a collated slice came from. ``lidar2cams`` carries ``fill + 0.5`` so it can
        be told apart from ``lidar2images``.
        """
        image_shape = (num_cameras, self.NUM_CHANNELS, self.HEIGHT, self.WIDTH)
        depth_shape = (num_cameras, 1, self.HEIGHT, self.WIDTH)
        intrinsics = torch.eye(3, dtype=torch.float32).repeat(num_cameras, 1, 1) * fill
        return BaseImages(
            images=torch.full(image_shape, fill, dtype=torch.float32),
            depth_maps=torch.full(depth_shape, fill, dtype=torch.float32) if with_depth else None,
            calibration_statuses=torch.ones(num_cameras, dtype=torch.int64)
            if with_statuses
            else None,
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

    def collate(self, samples: list[BaseImages]) -> ImageGTBatch:
        """
        Collate the given samples into a batch.

        The samples are never empty here, so the None branch of ``collate_gt_samples`` is
        asserted away to narrow the return type for the callers.
        """
        batch = ImageGTBatch.collate_gt_samples(samples)
        assert batch is not None
        return batch


class TestImageGTBatchFields(ImageGTBatchTestCase):
    """The named tuple contract of ImageGTBatch."""

    def test_field_order(self) -> None:
        """
        Input: the class itself.
        Expected: the tuple exposes its seven fields in the documented order, since downstream
        code unpacks the batch positionally.
        Check: compare ``_fields`` against the expected names.
        """
        self.assertEqual(
            ImageGTBatch._fields,
            (
                "images",
                "depth_maps",
                "camera_intrinsics",
                "image_augmentation_matrices",
                "lidar2images",
                "lidar2cams",
                "calibration_statuses",
            ),
        )

    def test_is_immutable(self) -> None:
        """
        Input: a batch collated from the fixture samples.
        Expected: named tuple fields cannot be reassigned after construction.
        Check: assigning to ``images`` raises AttributeError.
        """
        batch = self.collate(self.samples)

        with self.assertRaises(AttributeError):
            batch.images = torch.zeros(1)


class TestImageGTBatchCollate(ImageGTBatchTestCase):
    """ImageGTBatch.collate_gt_samples."""

    def test_empty_sequence_returns_none(self) -> None:
        """
        Input: an empty sample sequence.
        Expected: there is nothing to collate, so the result is None instead of an empty batch.
        Check: the return value is None.
        """
        self.assertIsNone(ImageGTBatch.collate_gt_samples([]))

    def test_stacks_samples_along_batch_dimension(self) -> None:
        """
        Input: the two fixture samples with two cameras each and no depth maps.
        Expected: every tensor field gains a leading batch dimension of 2 ahead of the camera
        dimension, slice ``i`` of each field equals sample ``i``'s tensor, and the depth maps
        stay None.
        Check: compare the shape of every field, then compare each batch slice against its
        source sample with ``torch.equal``.
        """
        batch = self.collate(self.samples)

        batch_size, num_cameras = len(self.samples), self.num_cameras
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
        """
        Input: a two-camera sample next to a three-camera sample.
        Expected: stacking requires an identical camera count per sample, so the collate fails
        instead of silently producing a ragged batch.
        Check: ``torch.stack`` raises RuntimeError.
        """
        samples = [self.make_images(2, fill=1.0), self.make_images(3, fill=2.0)]

        with self.assertRaises(RuntimeError):
            ImageGTBatch.collate_gt_samples(samples)

    def test_single_sample_adds_batch_dimension(self) -> None:
        """
        Input: a single three-camera sample.
        Expected: a batch of size 1 is still batched, so the images become
        ``(1, 3, C, H, W)`` and slice 0 of every field equals the sample's tensor.
        Check: compare the image shape, then compare slice 0 of each field with ``torch.equal``.
        """
        sample = self.make_images(3, fill=4.0)

        batch = self.collate([sample])

        self.assertEqual(batch.images.shape, (1, 3, self.NUM_CHANNELS, self.HEIGHT, self.WIDTH))
        self.assertTrue(torch.equal(batch.images[0], sample.images))
        self.assertTrue(torch.equal(batch.camera_intrinsics[0], sample.camera_intrinsics))
        self.assertTrue(torch.equal(batch.lidar2images[0], sample.lidar2images))
        self.assertTrue(torch.equal(batch.lidar2cams[0], sample.lidar2cams))
        self.assertTrue(
            torch.equal(batch.image_augmentation_matrices[0], sample.image_augmentation_matrices)
        )

    def test_stacks_depth_maps_when_every_sample_carries_them(self) -> None:
        """
        Input: the two fixture samples that both carry depth maps filled with 1.0 and 2.0.
        Expected: the depth maps are stacked to ``(2, num_cameras, 1, H, W)`` with each batch
        slice holding its sample's fill value.
        Check: compare the shape and assert every element of each slice equals its fill.
        """
        batch = self.collate(self.samples_with_depth)

        assert batch.depth_maps is not None
        self.assertEqual(
            batch.depth_maps.shape,
            (len(self.samples_with_depth), self.num_cameras, 1, self.HEIGHT, self.WIDTH),
        )
        self.assertTrue(torch.all(batch.depth_maps[0] == 1.0))
        self.assertTrue(torch.all(batch.depth_maps[1] == 2.0))

    def test_mixed_depth_maps_raise(self) -> None:
        """
        Input: one sample with depth maps and one without.
        Expected: a batch must carry depth for all samples or none, so the mix is rejected.
        Check: a ValueError reporting "1 out of 2 samples with depth" is raised.
        """
        samples = [self.make_images(self.num_cameras, fill=1.0, with_depth=True), self.samples[1]]

        with self.assertRaisesRegex(ValueError, "1 out of 2 samples with depth"):
            ImageGTBatch.collate_gt_samples(samples)

    def test_stacks_calibration_statuses_when_every_sample_carries_them(self) -> None:
        """
        Input: the two fixture samples that both carry a calibration status per camera.
        Expected: the statuses are stacked to ``(2, num_cameras)`` so every camera of the batch
        keeps the status the misalignment augmentation wrote for it.
        Check: read the shape and the values of the stacked statuses.
        """
        batch = self.collate(self.samples_with_depth)

        assert batch.calibration_statuses is not None
        self.assertEqual(
            batch.calibration_statuses.shape, (len(self.samples_with_depth), self.num_cameras)
        )
        self.assertTrue(torch.all(batch.calibration_statuses == 1))

    def test_calibration_statuses_stay_none_when_no_sample_carries_them(self) -> None:
        """
        Input: the two fixture samples built before the misalignment augmentation runs.
        Expected: the batch reports no calibration status rather than inventing one.
        Check: the collated statuses are None.
        """
        batch = self.collate(self.samples)

        self.assertIsNone(batch.calibration_statuses)

    def test_mixed_calibration_statuses_raise(self) -> None:
        """
        Input: one sample with a calibration status per camera and one without.
        Expected: a batch must carry the status for all samples or none, so the mix is
        rejected instead of collating a status tensor shorter than the batch.
        Check: a ValueError reporting "1 out of 2 samples with calibration statuses" is raised.
        """
        samples = [
            self.make_images(self.num_cameras, fill=1.0, with_statuses=True),
            self.samples[1],
        ]

        with self.assertRaisesRegex(ValueError, "1 out of 2 samples with calibration statuses"):
            ImageGTBatch.collate_gt_samples(samples)

    def test_output_dtypes(self) -> None:
        """
        Input: the fixture samples with every optional field filled in, so every field is a
        tensor.
        Expected: the image fields keep the float32 dtype of their source while the calibration
        status keeps its integer labels.
        Check: iterate the image fields and read the status dtype on its own.
        """
        batch = self.collate(self.samples_with_depth)

        for tensor in batch[: batch._fields.index("calibration_statuses")]:
            assert tensor is not None
            self.assertEqual(tensor.dtype, torch.float32)
        assert batch.calibration_statuses is not None
        self.assertEqual(batch.calibration_statuses.dtype, torch.int64)


class TestImageGTBatchToDevice(ImageGTBatchTestCase):
    """ImageGTBatch.to_device."""

    def setUp(self) -> None:
        """Collate the fixture samples once, with and without the optional fields."""
        super().setUp()
        self.batch_with_depth = self.collate(self.samples_with_depth)
        self.batch_without_depth = self.collate(self.samples)

    def test_returns_new_batch_with_equal_tensors(self) -> None:
        """
        Input: the fixture batch with every optional field, moved to the CPU it already lives
        on.
        Expected: a new ImageGTBatch is returned rather than the same object, and every field
        holds the same values.
        Check: assert the identity differs and compare every field pair with ``torch.equal``.
        """
        moved = self.batch_with_depth.to_device(self.device)

        self.assertIsInstance(moved, ImageGTBatch)
        self.assertIsNot(moved, self.batch_with_depth)
        for original, result in zip(self.batch_with_depth, moved):
            assert original is not None and result is not None
            self.assertTrue(torch.equal(original, result))

    def test_preserves_none_depth_maps(self) -> None:
        """
        Input: the fixture batch without depth maps.
        Expected: a None depth field is passed through instead of being moved like a tensor.
        Check: the moved batch's depth maps are None.
        """
        moved = self.batch_without_depth.to_device(self.device)

        self.assertIsNone(moved.depth_maps)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required to move tensors to GPU")
    def test_moves_every_tensor_to_cuda(self) -> None:
        """
        Input: the fixture batch with every optional field, moved to CUDA.
        Expected: every field of the result lives on CUDA while the original stays on CPU,
        since named tuples are immutable and the move must not alias.
        Check: read ``device.type`` of every field on both batches.
        """
        moved = self.batch_with_depth.to_device(torch.device("cuda"))

        for tensor in moved:
            assert tensor is not None
            self.assertEqual(tensor.device.type, "cuda")
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
        self.sample = ImageSample(**self.fields)

    def make_sample(self, **overrides) -> ImageSample:
        """Build an ImageSample from the fixture fields with some of them overridden."""
        return ImageSample(**{**self.fields, **overrides})


class TestImageSample(ImageSampleTestCase):
    """The named tuple contract of ImageSample."""

    def test_field_order(self) -> None:
        """
        Input: the class itself.
        Expected: the tuple exposes its eight fields in the documented order, matching the
        dataset record layout.
        Check: compare ``_fields`` against the expected names.
        """
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
        """
        Input: the fixture sample with a plumb_bob camera and five distortion coefficients.
        Expected: every field is stored as given, without conversion.
        Check: read each scalar back, compare matrix shapes, and compare the doubled identity
        ``lidar2image`` with ``torch.equal``.
        """
        self.assertEqual(self.sample.image_path, "/data/camera0/000001.jpg")
        self.assertEqual(self.sample.camera_name, "camera0")
        self.assertEqual(self.sample.timestamp, 1700000000.25)
        self.assertEqual(self.sample.camera_intrinsic.shape, (3, 3))
        self.assertEqual(self.sample.lidar2cam.shape, (4, 4))
        self.assertTrue(torch.equal(self.sample.lidar2image, torch.eye(4) * 2.0))
        self.assertEqual(self.sample.distortion_model, "plumb_bob")
        self.assertEqual(self.sample.distortion_coefficients.shape, (5,))

    def test_accepts_empty_distortion_coefficients(self) -> None:
        """
        Input: the fixture fields with an empty distortion model and a zero-length coefficient
        tensor, as a pre-undistorted image is recorded.
        Expected: the sample accepts the empty values without complaint.
        Check: the model string is empty and the coefficient tensor has zero elements.
        """
        sample = self.make_sample(
            distortion_model="", distortion_coefficients=torch.empty(0, dtype=torch.float32)
        )

        self.assertEqual(sample.distortion_model, "")
        self.assertEqual(sample.distortion_coefficients.numel(), 0)

    def test_is_immutable(self) -> None:
        """
        Input: the fixture sample.
        Expected: named tuple fields cannot be reassigned after construction.
        Check: assigning to ``camera_name`` raises AttributeError.
        """
        with self.assertRaises(AttributeError):
            self.sample.camera_name = "camera1"

    def test_asdict_round_trip(self) -> None:
        """
        Input: the fixture sample.
        Expected: ``_asdict`` yields keyword arguments that rebuild an equivalent sample, with
        tensor fields shared rather than copied.
        Check: compare the scalar fields for equality and the tensor fields for identity.
        """
        rebuilt = ImageSample(**self.sample._asdict())

        self.assertEqual(rebuilt.image_path, self.sample.image_path)
        self.assertEqual(rebuilt.camera_name, self.sample.camera_name)
        self.assertEqual(rebuilt.timestamp, self.sample.timestamp)
        self.assertIs(rebuilt.camera_intrinsic, self.sample.camera_intrinsic)
        self.assertIs(rebuilt.lidar2cam, self.sample.lidar2cam)
        self.assertIs(rebuilt.lidar2image, self.sample.lidar2image)
        self.assertEqual(rebuilt.distortion_model, self.sample.distortion_model)
        self.assertIs(rebuilt.distortion_coefficients, self.sample.distortion_coefficients)


class TestImageGTBatchCameraGeometry(ImageGTBatchTestCase):
    """The collated calibration describes the augmented images."""

    def test_intrinsics_back_project_what_lidar2images_projects(self) -> None:
        """
        Input: one camera whose image was resized by 0.5, so its augmented intrinsics are half
        the raw ones and lidar2images is built from the augmented ones.
        Expected: back projecting a projected lidar point with the collated intrinsics and
        lidar2cams recovers the point, so the camera to BEV lift and the depth guidance of a
        model describe the same image.
        Check: the round trip of the point (10, 0, 20) in camera coordinates.
        """
        sample = self.make_images(1, fill=1.0)
        raw = torch.tensor([[[1000.0, 0.0, 800.0], [0.0, 1000.0, 450.0], [0.0, 0.0, 1.0]]])
        resize = torch.diag(torch.tensor([0.5, 0.5, 1.0])).unsqueeze(0)
        augmented = resize @ raw
        lidar2cams = torch.eye(4).unsqueeze(0)
        lidar2images = torch.eye(4).unsqueeze(0)
        lidar2images[:, :3, :3] = augmented
        sample = sample.model_copy(
            update={
                "camera_intrinsics": raw,
                "augmented_camera_intrinsics": augmented,
                "lidar2cams": lidar2cams,
                "lidar2images": lidar2images,
            }
        )

        batch = self.collate([sample])

        point = torch.tensor([10.0, 0.0, 20.0, 1.0])
        pixel = batch.lidar2images[0, 0] @ point
        pixel = pixel[:3] / pixel[2]
        ray = torch.linalg.inv(batch.camera_intrinsics[0, 0]) @ pixel
        recovered = torch.linalg.inv(batch.lidar2cams[0, 0])[:3, :3] @ (ray * point[2])
        torch.testing.assert_close(recovered, point[:3])


if __name__ == "__main__":
    unittest.main()
