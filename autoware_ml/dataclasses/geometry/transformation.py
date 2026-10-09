from __future__ import annotations

from typing import NamedTuple, Sequence

from jaxtyping import Float32
from torch import Tensor
import torch

from autoware_ml.types.geometry import TransformationName


class LiDARTransformationSample(NamedTuple):
    """
    Named tuple to represent a lidar transformation, for example, 4x4 matrices applied
    during data augmentation. This is used when we need to reverse the transformations applied
    to the point cloud and bounding boxes, for example, when doing LSS.

    Attributes:
        rotation_matrix: 3x3 rotation matrix applied.
        scale: Scale factor applied.
        translation_vector: 1x3 translation vector applied.
        transformation_order: Order of the transformations applied.
    """

    transformation_matrix: Float32[Tensor, "4 4"]  # 4x4 transformation matrix applied.
    transformation_order: Sequence[TransformationName]  # Order of the transformations applied.

    @classmethod
    def create_lidar_transformation_sample(
        cls,
        rotation_matrix: Float32[Tensor, "3 3"],
        scale_factor: float,
        translation_vector: Float32[Tensor, "1 3"],
        transformation_order: Sequence[TransformationName],
    ) -> LiDARTransformationSample:
        """
        Create a LiDARTransformationSample from rotation, scale, translation and the
          transformation order.

        Args:
            rotation_matrix (Tensor.float32, (3, 3)): The rotation matrix to apply.
            scale_factor (float): The scale factor to apply.
            translation_vector (Tensor.float32, (1, 3)): The translation vector to apply.
            transformation_order (Sequence[TransformationName]): The order of the transformations applied.

        Returns:
            LiDARTransformationSample: A named tuple containing the 4x4 transformation matrix
                and the order of transformations applied.
        """
        transformation_matrix = torch.eye(4, dtype=torch.float32)
        transformation_matrix[:3, :3] = rotation_matrix * scale_factor
        transformation_matrix[:3, 3] = translation_vector.reshape(3)

        return cls(
            transformation_matrix=transformation_matrix, transformation_order=transformation_order
        )

    def create_composed_lidar_transformation_sample(
        self, previous_lidar_transformation_sample: LiDARTransformationSample
    ) -> LiDARTransformationSample:
        """
        Update the lidar transformation sample of the LiDARTransformationSample with a previous transformation.

        Args:
            previous_lidar_transformation_sample (LiDARTransformationSample): The previous lidar transformation sample to compose with.

        Returns:
            LiDARTransformationSample: A new LiDARTransformationSample with the updated lidar transformation sample.
        """
        # Compose the new transformation with the existing one, right to left multiplication,
        # i.e., the previous transformation is applied first, then the current one.
        updated_transformation_matrix = (
            self.transformation_matrix @ previous_lidar_transformation_sample.transformation_matrix
        )
        updated_transformation_order = list(
            previous_lidar_transformation_sample.transformation_order
        ) + list(self.transformation_order)

        return LiDARTransformationSample(
            transformation_matrix=updated_transformation_matrix,
            transformation_order=updated_transformation_order,
        )
