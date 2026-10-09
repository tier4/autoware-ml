"""Unit tests for the LiDARTransformationSample named tuple."""

from __future__ import annotations

import math
import unittest

import torch

from autoware_ml.dataclasses.geometry.transformation import LiDARTransformationSample
from autoware_ml.types.geometry import TransformationName


class LiDARTransformationSampleTestCase(unittest.TestCase):
    """Shared fixtures for the LiDARTransformationSample tests."""

    def setUp(self) -> None:
        """Build the identity inputs and a unit point on the x axis the tests transform."""
        self.identity_rotation = torch.eye(3, dtype=torch.float32)
        self.zero_translation = torch.zeros((1, 3), dtype=torch.float32)
        self.identity_sample = LiDARTransformationSample(
            transformation_matrix=torch.eye(4, dtype=torch.float32), transformation_order=[]
        )
        self.unit_x = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)

    def rotation_z(self, angle_rad: float) -> torch.Tensor:
        """3x3 rotation about the z axis."""
        cos, sin = math.cos(angle_rad), math.sin(angle_rad)
        return torch.tensor(
            [[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32
        )

    def apply(self, matrix: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
        """Apply a 4x4 homogeneous transformation to a 3D point."""
        homogeneous = torch.cat([point, torch.ones(1, dtype=torch.float32)])
        return (matrix @ homogeneous)[:3]

    def make_sample(
        self,
        rotation_matrix: torch.Tensor | None = None,
        scale_factor: float = 1.0,
        translation_vector: torch.Tensor | None = None,
        transformation_order: list[TransformationName] | None = None,
    ) -> LiDARTransformationSample:
        """Build a sample through the factory, defaulting every input to the identity."""
        return LiDARTransformationSample.create_lidar_transformation_sample(
            rotation_matrix=(
                self.identity_rotation if rotation_matrix is None else rotation_matrix
            ),
            scale_factor=scale_factor,
            translation_vector=(
                self.zero_translation if translation_vector is None else translation_vector
            ),
            transformation_order=[] if transformation_order is None else transformation_order,
        )


class TestLiDARTransformationSampleFields(LiDARTransformationSampleTestCase):
    """The named tuple contract of LiDARTransformationSample."""

    def test_field_order(self) -> None:
        """
        Input: the class itself.
        Expected: the tuple exposes the matrix first and the order second, since downstream
        code unpacks the sample positionally.
        Check: compare ``_fields`` against the expected names.
        """
        self.assertEqual(
            LiDARTransformationSample._fields, ("transformation_matrix", "transformation_order")
        )

    def test_holds_given_fields(self) -> None:
        """
        Input: an identity matrix and a one-element order list passed to the constructor.
        Expected: the constructor stores both objects as given, without copying.
        Check: assert identity of both fields with the objects passed in.
        """
        matrix = torch.eye(4, dtype=torch.float32)
        order = [TransformationName.ROTATION]

        sample = LiDARTransformationSample(transformation_matrix=matrix, transformation_order=order)

        self.assertIs(sample.transformation_matrix, matrix)
        self.assertIs(sample.transformation_order, order)

    def test_is_immutable(self) -> None:
        """
        Input: the identity fixture sample.
        Expected: named tuple fields cannot be reassigned after construction.
        Check: assigning to ``transformation_order`` raises AttributeError.
        """
        with self.assertRaises(AttributeError):
            self.identity_sample.transformation_order = [TransformationName.SCALING]


class TestCreateLiDARTransformationSample(LiDARTransformationSampleTestCase):
    """LiDARTransformationSample.create_lidar_transformation_sample."""

    def test_identity_inputs_give_identity_matrix(self) -> None:
        """
        Input: identity rotation, scale 1, zero translation and an empty order.
        Expected: the factory produces the 4x4 identity and an empty order.
        Check: compare the matrix with ``torch.eye(4)`` and the order with an empty list.
        """
        sample = self.make_sample()

        self.assertIsInstance(sample, LiDARTransformationSample)
        self.assertTrue(torch.equal(sample.transformation_matrix, torch.eye(4)))
        self.assertEqual(list(sample.transformation_order), [])

    def test_builds_homogeneous_matrix_from_rotation_scale_translation(self) -> None:
        """
        Input: a 90 degree rotation about z, scale 2, translation ``(1, 2, 3)`` as a ``(1, 3)``
        tensor, and a three-step order.
        Expected: the top-left 3x3 block is the rotation scaled by 2, the last column holds the
        translation, the bottom row is ``(0, 0, 0, 1)``, the dtype is float32 and the order is
        stored unchanged.
        Check: compare each block of the 4x4 matrix separately and compare the order list.
        """
        rotation = self.rotation_z(math.pi / 2)
        translation = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
        order = [
            TransformationName.ROTATION,
            TransformationName.SCALING,
            TransformationName.TRANSLATION,
        ]

        sample = self.make_sample(
            rotation_matrix=rotation,
            scale_factor=2.0,
            translation_vector=translation,
            transformation_order=order,
        )

        matrix = sample.transformation_matrix
        self.assertEqual(matrix.shape, (4, 4))
        self.assertEqual(matrix.dtype, torch.float32)
        self.assertTrue(torch.allclose(matrix[:3, :3], rotation * 2.0))
        self.assertTrue(torch.equal(matrix[:3, 3], translation.reshape(3)))
        self.assertTrue(torch.equal(matrix[3], torch.tensor([0.0, 0.0, 0.0, 1.0])))
        self.assertEqual(list(sample.transformation_order), order)

    def test_matrix_applies_scaled_rotation_then_translation(self) -> None:
        """
        Input: a 90 degree rotation about z, scale 2 and a translation of 10 along x, applied
        to the unit point ``(1, 0, 0)``.
        Expected: the point is rotated onto the y axis, doubled to ``(0, 2, 0)``, then shifted
        to ``(10, 2, 0)``, which confirms the rotation and scale act before the translation.
        Check: apply the matrix to the point and compare with ``torch.allclose``.
        """
        sample = self.make_sample(
            rotation_matrix=self.rotation_z(math.pi / 2),
            scale_factor=2.0,
            translation_vector=torch.tensor([[10.0, 0.0, 0.0]], dtype=torch.float32),
            transformation_order=[TransformationName.ROTATION],
        )

        result = self.apply(sample.transformation_matrix, self.unit_x)

        self.assertTrue(torch.allclose(result, torch.tensor([10.0, 2.0, 0.0]), atol=1e-6))

    def test_accepts_flat_translation_vector(self) -> None:
        """
        Input: a translation given as a flat ``(3,)`` tensor instead of ``(1, 3)``.
        Expected: the factory reshapes it and writes ``(4, 5, 6)`` into the last column.
        Check: compare the translation column with ``torch.equal``.
        """
        sample = self.make_sample(
            translation_vector=torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32),
            transformation_order=[TransformationName.TRANSLATION],
        )

        self.assertTrue(
            torch.equal(sample.transformation_matrix[:3, 3], torch.tensor([4.0, 5.0, 6.0]))
        )

    def test_does_not_alias_inputs(self) -> None:
        """
        Input: the identity fixtures with scale 3, followed by an in-place write into the
        resulting matrix.
        Expected: the factory copies the rotation and translation into a fresh matrix, so
        writing into the result leaves the fixture inputs untouched.
        Check: after the write, the rotation still has a 1 at ``(0, 0)`` and the translation
        still sums to 0.
        """
        sample = self.make_sample(scale_factor=3.0)
        sample.transformation_matrix[0, 0] = 99.0

        self.assertEqual(self.identity_rotation[0, 0].item(), 1.0)
        self.assertEqual(self.zero_translation.sum().item(), 0.0)


class TestCreateComposedLiDARTransformationSample(LiDARTransformationSampleTestCase):
    """LiDARTransformationSample.create_composed_lidar_transformation_sample."""

    def setUp(self) -> None:
        """Build a 90 degree rotation as the previous step and a unit x shift as the current."""
        super().setUp()
        self.previous = self.make_sample(
            rotation_matrix=self.rotation_z(math.pi / 2),
            transformation_order=[TransformationName.ROTATION],
        )
        self.current = self.make_sample(
            translation_vector=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
            transformation_order=[TransformationName.TRANSLATION],
        )

    def test_matrix_is_current_times_previous(self) -> None:
        """
        Input: the rotation as the previous step and the translation as the current step.
        Expected: the composed matrix is ``current @ previous``, the right-to-left product that
        applies the previous transformation first.
        Check: compute the product by hand and compare with ``torch.equal``.
        """
        composed = self.current.create_composed_lidar_transformation_sample(self.previous)

        expected = self.current.transformation_matrix @ self.previous.transformation_matrix
        self.assertTrue(torch.equal(composed.transformation_matrix, expected))

    def test_previous_transformation_is_applied_first(self) -> None:
        """
        Input: the composed rotation-then-translation applied to the unit point ``(1, 0, 0)``.
        Expected: applying the composed matrix once equals applying the previous matrix and
        then the current one, and the point lands at ``(1, 1, 0)`` because it is rotated onto
        the y axis before being shifted along x. Had the order been reversed the result would
        be ``(0, 2, 0)``.
        Check: compare the composed result against the sequential result and against the
        expected point with ``torch.allclose``.
        """
        composed = self.current.create_composed_lidar_transformation_sample(self.previous)

        composed_result = self.apply(composed.transformation_matrix, self.unit_x)
        sequential_result = self.apply(
            self.current.transformation_matrix,
            self.apply(self.previous.transformation_matrix, self.unit_x),
        )

        self.assertTrue(torch.allclose(composed_result, sequential_result, atol=1e-6))
        # Rotate (1, 0, 0) to (0, 1, 0), then translate by +1 along x.
        self.assertTrue(torch.allclose(composed_result, torch.tensor([1.0, 1.0, 0.0]), atol=1e-6))

    def test_order_lists_previous_then_current(self) -> None:
        """
        Input: the rotation step composed with the translation step.
        Expected: the order records the previous step first and the current step second, as a
        plain list, mirroring the application order of the matrices.
        Check: compare the order with ``[ROTATION, TRANSLATION]`` and assert it is a list.
        """
        composed = self.current.create_composed_lidar_transformation_sample(self.previous)

        self.assertEqual(
            composed.transformation_order,
            [TransformationName.ROTATION, TransformationName.TRANSLATION],
        )
        self.assertIsInstance(composed.transformation_order, list)

    def test_order_accepts_tuple_sequences(self) -> None:
        """
        Input: two identity samples whose orders are tuples rather than lists.
        Expected: any sequence type is accepted and the composed order is a list holding the
        previous then the current entries.
        Check: compare the composed order with ``[SCALING, HORIZONTAL_FLIP]``.
        """
        previous = self.identity_sample._replace(transformation_order=(TransformationName.SCALING,))
        current = self.identity_sample._replace(
            transformation_order=(TransformationName.HORIZONTAL_FLIP,)
        )

        composed = current.create_composed_lidar_transformation_sample(previous)

        self.assertEqual(
            composed.transformation_order,
            [TransformationName.SCALING, TransformationName.HORIZONTAL_FLIP],
        )

    def test_does_not_mutate_operands(self) -> None:
        """
        Input: the rotation and translation steps, with copies of their matrices taken first.
        Expected: composition returns a new sample and leaves both operands' matrices and
        orders exactly as they were.
        Check: assert the result is neither operand, compare both matrices with their copies,
        and compare both orders with their original single entries.
        """
        previous_matrix = self.previous.transformation_matrix.clone()
        current_matrix = self.current.transformation_matrix.clone()

        composed = self.current.create_composed_lidar_transformation_sample(self.previous)

        self.assertIsNot(composed, self.current)
        self.assertIsNot(composed, self.previous)
        self.assertTrue(torch.equal(self.previous.transformation_matrix, previous_matrix))
        self.assertTrue(torch.equal(self.current.transformation_matrix, current_matrix))
        self.assertEqual(list(self.previous.transformation_order), [TransformationName.ROTATION])
        self.assertEqual(list(self.current.transformation_order), [TransformationName.TRANSLATION])

    def test_composing_with_identity_keeps_matrix(self) -> None:
        """
        Input: the rotation step composed with the identity sample as the previous step.
        Expected: multiplying by the identity leaves the matrix unchanged and the empty order
        contributes nothing, so the order is just ``[ROTATION]``.
        Check: compare the matrix with the rotation step's matrix and compare the order.
        """
        composed = self.previous.create_composed_lidar_transformation_sample(self.identity_sample)

        self.assertTrue(
            torch.equal(composed.transformation_matrix, self.previous.transformation_matrix)
        )
        self.assertEqual(composed.transformation_order, [TransformationName.ROTATION])

    def test_chaining_three_transformations(self) -> None:
        """
        Input: the rotation, then the translation, then a scale-by-2 step, composed pairwise.
        Expected: chaining composes to ``third @ current @ previous`` and the order lists the
        three steps in application order.
        Check: compute the triple product by hand and compare with ``torch.allclose``, then
        compare the order with ``[ROTATION, TRANSLATION, SCALING]``.
        """
        third = self.make_sample(
            scale_factor=2.0, transformation_order=[TransformationName.SCALING]
        )

        first_two = self.current.create_composed_lidar_transformation_sample(self.previous)
        composed = third.create_composed_lidar_transformation_sample(first_two)

        expected = (
            third.transformation_matrix
            @ self.current.transformation_matrix
            @ self.previous.transformation_matrix
        )
        self.assertTrue(torch.allclose(composed.transformation_matrix, expected))
        self.assertEqual(
            composed.transformation_order,
            [
                TransformationName.ROTATION,
                TransformationName.TRANSLATION,
                TransformationName.SCALING,
            ],
        )


if __name__ == "__main__":
    unittest.main()
