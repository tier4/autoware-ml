"""Unit tests for the FrameMetaSample and FrameMetaBatch named tuples."""

from __future__ import annotations

import math
import unittest

import torch

from autoware_ml.dataclasses.batch.frame_meta import FrameMetaBatch, FrameMetaSample
from autoware_ml.dataclasses.geometry.transformation import LiDARTransformationSample
from autoware_ml.types.geometry import TransformationName


class FrameMetaTestCase(unittest.TestCase):
    """Shared fixtures for the FrameMetaSample and FrameMetaBatch tests."""

    def setUp(self) -> None:
        """Build two frames whose map poses are translations of 1 and 2 along x."""
        self.device = torch.device("cpu")
        self.samples = [
            FrameMetaSample(self.make_ego2global(1.0), "db/scene_a/0"),
            FrameMetaSample(self.make_ego2global(2.0), "db/scene_b/1"),
        ]

    def make_ego2global(self, x: float, yaw_rad: float = 0.0) -> torch.Tensor:
        """4x4 map pose that rotates by ``yaw_rad`` about z and then translates by ``x``."""
        matrix = torch.eye(4, dtype=torch.float32)
        matrix[:3, :3] = self.rotation_z(yaw_rad)
        matrix[0, 3] = x
        return matrix

    def rotation_z(self, angle_rad: float) -> torch.Tensor:
        """3x3 rotation about the z axis."""
        cos, sin = math.cos(angle_rad), math.sin(angle_rad)
        return torch.tensor(
            [[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32
        )

    def make_augmentation(
        self,
        yaw_rad: float = 0.0,
        scale_factor: float = 1.0,
        translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> LiDARTransformationSample:
        """Lidar-space augmentation composed of a z rotation, a scale and a translation."""
        return LiDARTransformationSample.create_lidar_transformation_sample(
            rotation_matrix=self.rotation_z(yaw_rad),
            scale_factor=scale_factor,
            translation_vector=torch.tensor([translation], dtype=torch.float32),
            transformation_order=[
                TransformationName.ROTATION,
                TransformationName.SCALING,
                TransformationName.TRANSLATION,
            ],
        )

    def apply(self, matrix: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
        """Apply a 4x4 homogeneous transformation to a 3D point."""
        homogeneous = torch.cat([point, torch.ones(1, dtype=point.dtype)])
        return (matrix @ homogeneous)[:3]


class TestFrameMetaSample(FrameMetaTestCase):
    """The named tuple contract of FrameMetaSample."""

    def test_field_order(self) -> None:
        """
        Input: the class itself.
        Expected: the tuple exposes ``ego2global`` then ``scene_token``, since the dataset
        builds it positionally.
        Check: compare ``_fields`` against the expected names.
        """
        self.assertEqual(FrameMetaSample._fields, ("ego2global", "scene_token"))

    def test_holds_given_fields(self) -> None:
        """
        Input: the first fixture sample.
        Expected: the pose and scene token are stored as given.
        Check: compare the pose with ``torch.equal`` and read the token back.
        """
        sample = self.samples[0]

        self.assertTrue(torch.equal(sample.ego2global, self.make_ego2global(1.0)))
        self.assertEqual(sample.scene_token, "db/scene_a/0")

    def test_is_immutable(self) -> None:
        """
        Input: the first fixture sample.
        Expected: named tuple fields cannot be reassigned after construction.
        Check: assigning to ``scene_token`` raises AttributeError.
        """
        with self.assertRaises(AttributeError):
            self.samples[0].scene_token = "other"


class TestFrameMetaBatchFields(FrameMetaTestCase):
    """The named tuple contract of FrameMetaBatch."""

    def test_field_order(self) -> None:
        """
        Input: the class itself.
        Expected: the tuple exposes ``ego2globals`` then ``scene_tokens``.
        Check: compare ``_fields`` against the expected names.
        """
        self.assertEqual(FrameMetaBatch._fields, ("ego2globals", "scene_tokens"))

    def test_is_immutable(self) -> None:
        """
        Input: a batch collated from the fixture samples without augmentation.
        Expected: named tuple fields cannot be reassigned after construction.
        Check: assigning to ``scene_tokens`` raises AttributeError.
        """
        batch = FrameMetaBatch.collate_gt_samples(self.samples, [None, None])

        with self.assertRaises(AttributeError):
            batch.scene_tokens = []


class TestFrameMetaBatchCollate(FrameMetaTestCase):
    """FrameMetaBatch.collate_gt_samples without lidar augmentation."""

    def test_stacks_poses_and_tokens_in_batch_order(self) -> None:
        """
        Input: the two fixture samples, neither augmented.
        Expected: the poses are stacked into ``(2, 4, 4)`` unchanged and the tokens are listed
        in the same order, so row ``i`` of both fields describes sample ``i``.
        Check: compare the shape, each pose with ``torch.equal`` and the token list.
        """
        batch = FrameMetaBatch.collate_gt_samples(self.samples, [None, None])

        self.assertEqual(tuple(batch.ego2globals.shape), (2, 4, 4))
        self.assertTrue(torch.equal(batch.ego2globals[0], self.samples[0].ego2global))
        self.assertTrue(torch.equal(batch.ego2globals[1], self.samples[1].ego2global))
        self.assertEqual(list(batch.scene_tokens), ["db/scene_a/0", "db/scene_b/1"])

    def test_output_dtype(self) -> None:
        """
        Input: the fixture samples with float32 poses.
        Expected: the stacked poses keep float32, as the metrics consume them.
        Check: read the dtype of ``ego2globals``.
        """
        batch = FrameMetaBatch.collate_gt_samples(self.samples, [None, None])

        self.assertEqual(batch.ego2globals.dtype, torch.float32)

    def test_single_sample(self) -> None:
        """
        Input: one fixture sample.
        Expected: the batch has a single row holding that sample's pose and token.
        Check: compare the shape, the pose and the token list of length 1.
        """
        batch = FrameMetaBatch.collate_gt_samples(self.samples[:1], [None])

        self.assertEqual(tuple(batch.ego2globals.shape), (1, 4, 4))
        self.assertTrue(torch.equal(batch.ego2globals[0], self.samples[0].ego2global))
        self.assertEqual(list(batch.scene_tokens), ["db/scene_a/0"])

    def test_empty_sequence_raises(self) -> None:
        """
        Input: empty frame metadata and augmentation sequences.
        Expected: an empty batch cannot be stacked, so collation refuses it.
        Check: ``ValueError`` is raised.
        """
        with self.assertRaises(ValueError):
            FrameMetaBatch.collate_gt_samples([], [])

    def test_length_mismatch_raises(self) -> None:
        """
        Input: two frame metadata samples but a single augmentation entry.
        Expected: every sample needs its own augmentation slot (possibly ``None``), so the
        mismatch is rejected instead of silently dropping a sample.
        Check: ``ValueError`` is raised.
        """
        with self.assertRaises(ValueError):
            FrameMetaBatch.collate_gt_samples(self.samples, [None])


class TestFrameMetaBatchLidarAugmentation(FrameMetaTestCase):
    """FrameMetaBatch.collate_gt_samples following the lidar-space augmentation."""

    def test_identity_augmentation_keeps_pose(self) -> None:
        """
        Input: one sample with an identity lidar augmentation.
        Expected: inverting the identity changes nothing, so the pose is kept.
        Check: compare the collated pose with the input using ``assert_close``.
        """
        batch = FrameMetaBatch.collate_gt_samples(self.samples[:1], [self.make_augmentation()])

        torch.testing.assert_close(batch.ego2globals[0], self.samples[0].ego2global)

    def test_augmented_pose_is_ego2global_times_inverse(self) -> None:
        """
        Input: one sample whose pose is a yaw of 30 degrees and x offset of 1, augmented by a
        yaw of 90 degrees, a scale of 2 and a translation of ``(1, -2, 0.5)``.
        Expected: the collated pose equals ``ego2global @ inv(T)``.
        Check: build that product explicitly and compare with ``assert_close``.
        """
        ego2global = self.make_ego2global(1.0, yaw_rad=math.radians(30.0))
        augmentation = self.make_augmentation(
            yaw_rad=math.pi / 2, scale_factor=2.0, translation=(1.0, -2.0, 0.5)
        )
        expected = ego2global @ torch.linalg.inv(augmentation.transformation_matrix)

        batch = FrameMetaBatch.collate_gt_samples(
            [FrameMetaSample(ego2global, "db/scene/0")], [augmentation]
        )

        torch.testing.assert_close(batch.ego2globals[0], expected)

    def test_augmented_point_projects_to_same_map_position(self) -> None:
        """
        Input: a raw lidar point ``(3, 4, 5)``, a sample pose with yaw 30 degrees and x offset 1,
        and an augmentation with yaw 90 degrees, scale 2 and translation ``(1, -2, 0.5)``.
        Expected: the point the model sees is ``p_aug = T @ p_raw``, and projecting it with the
        collated pose lands on the same map position as projecting ``p_raw`` with the original
        pose, so region and collision filters keep working on augmented samples.
        Check: compare both projections with ``assert_close``.
        """
        ego2global = self.make_ego2global(1.0, yaw_rad=math.radians(30.0))
        augmentation = self.make_augmentation(
            yaw_rad=math.pi / 2, scale_factor=2.0, translation=(1.0, -2.0, 0.5)
        )
        raw_point = torch.tensor([3.0, 4.0, 5.0], dtype=torch.float32)
        augmented_point = self.apply(augmentation.transformation_matrix, raw_point)

        batch = FrameMetaBatch.collate_gt_samples(
            [FrameMetaSample(ego2global, "db/scene/0")], [augmentation]
        )

        expected_map_point = self.apply(ego2global, raw_point)
        map_point = self.apply(batch.ego2globals[0], augmented_point)
        torch.testing.assert_close(map_point, expected_map_point)

    def test_translation_only_augmentation_shifts_origin_back(self) -> None:
        """
        Input: a sample at map offset ``x = 1`` augmented by a pure translation of ``(2, 0, 0)``.
        Expected: the augmented lidar origin sits at ``(2, 0, 0)`` in the raw frame, so the
        collated pose maps the augmented origin to ``(1 - 2, 0, 0) = (-1, 0, 0)``, and the
        rotation block stays identity.
        Check: project the origin and compare the rotation block with ``assert_close``.
        """
        augmentation = self.make_augmentation(translation=(2.0, 0.0, 0.0))

        batch = FrameMetaBatch.collate_gt_samples(self.samples[:1], [augmentation])

        origin = self.apply(batch.ego2globals[0], torch.zeros(3, dtype=torch.float32))
        torch.testing.assert_close(origin, torch.tensor([-1.0, 0.0, 0.0]))
        torch.testing.assert_close(batch.ego2globals[0, :3, :3], torch.eye(3))

    def test_mixed_batch_only_adjusts_augmented_samples(self) -> None:
        """
        Input: the two fixture samples, the first augmented by a 90 degree yaw and the second
        left ``None``.
        Expected: only the first pose is composed with the inverse rotation while the second
        is kept as given, and the tokens stay in batch order.
        Check: compare each pose against its expectation and read the token list.
        """
        augmentation = self.make_augmentation(yaw_rad=math.pi / 2)
        expected_first = self.samples[0].ego2global @ torch.linalg.inv(
            augmentation.transformation_matrix
        )

        batch = FrameMetaBatch.collate_gt_samples(self.samples, [augmentation, None])

        torch.testing.assert_close(batch.ego2globals[0], expected_first)
        self.assertTrue(torch.equal(batch.ego2globals[1], self.samples[1].ego2global))
        self.assertEqual(list(batch.scene_tokens), ["db/scene_a/0", "db/scene_b/1"])

    def test_composed_augmentation_matches_sequential_inverse(self) -> None:
        """
        Input: a rotation augmentation composed on top of a translation augmentation, as the
        transform pipeline produces when several lidar-space transforms run in sequence.
        Expected: the collated pose equals ``ego2global @ inv(T_rot @ T_trans)``, which is the
        same as undoing the rotation first and then the translation.
        Check: compare against ``ego2global @ inv(T_trans) @ inv(T_rot)`` with ``assert_close``.
        """
        translation = self.make_augmentation(translation=(1.0, 2.0, 3.0))
        rotation = self.make_augmentation(yaw_rad=math.pi / 4)
        composed = rotation.create_composed_lidar_transformation_sample(translation)
        ego2global = self.samples[0].ego2global
        expected = (
            ego2global
            @ torch.linalg.inv(translation.transformation_matrix)
            @ torch.linalg.inv(rotation.transformation_matrix)
        )

        batch = FrameMetaBatch.collate_gt_samples(self.samples[:1], [composed])

        torch.testing.assert_close(batch.ego2globals[0], expected)

    def test_augmentation_is_cast_to_pose_dtype(self) -> None:
        """
        Input: a float32 pose and an augmentation whose matrix is float64.
        Expected: the augmentation is cast to the pose's dtype before inversion, so the batch
        stays float32 instead of failing on a dtype mismatch or being promoted.
        Check: read the dtype and compare against the float32 product.
        """
        augmentation = self.make_augmentation(translation=(2.0, 0.0, 0.0))
        augmentation = augmentation._replace(
            transformation_matrix=augmentation.transformation_matrix.to(torch.float64)
        )
        expected = self.samples[0].ego2global @ torch.linalg.inv(
            augmentation.transformation_matrix.to(torch.float32)
        )

        batch = FrameMetaBatch.collate_gt_samples(self.samples[:1], [augmentation])

        self.assertEqual(batch.ego2globals.dtype, torch.float32)
        torch.testing.assert_close(batch.ego2globals[0], expected)


class TestFrameMetaBatchToDevice(FrameMetaTestCase):
    """FrameMetaBatch.to_device."""

    def setUp(self) -> None:
        """Collate the fixture samples once without augmentation."""
        super().setUp()
        self.batch = FrameMetaBatch.collate_gt_samples(self.samples, [None, None])

    def test_returns_new_batch_with_equal_fields(self) -> None:
        """
        Input: the fixture batch, moved to the CPU it already lives on.
        Expected: a new FrameMetaBatch is returned rather than the same object, the poses hold
        the same values, and the scene tokens are passed through untouched.
        Check: assert the identity differs, compare the poses and the token list.
        """
        moved = self.batch.to_device(self.device)

        self.assertIsInstance(moved, FrameMetaBatch)
        self.assertIsNot(moved, self.batch)
        self.assertTrue(torch.equal(moved.ego2globals, self.batch.ego2globals))
        self.assertEqual(list(moved.scene_tokens), list(self.batch.scene_tokens))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required to move tensors to GPU")
    def test_moves_poses_to_cuda_and_keeps_tokens(self) -> None:
        """
        Input: the fixture batch, moved to CUDA.
        Expected: the poses of the result live on CUDA while the original stays on CPU, and the
        scene tokens are unchanged since they are plain strings.
        Check: read ``device.type`` of both batches and compare the token lists.
        """
        moved = self.batch.to_device(torch.device("cuda"))

        self.assertEqual(moved.ego2globals.device.type, "cuda")
        self.assertEqual(self.batch.ego2globals.device.type, "cpu")
        self.assertEqual(list(moved.scene_tokens), list(self.batch.scene_tokens))


if __name__ == "__main__":
    unittest.main()
