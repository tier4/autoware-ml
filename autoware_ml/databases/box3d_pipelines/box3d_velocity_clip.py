from typing import Sequence
import numpy as np

from autoware_ml.types.geometry import Box3DFieldIndex

from autoware_ml.databases.box3d_pipelines.box3d_pipeline import Box3DPipeline
from autoware_ml.databases.schemas.box3d_schemas import Box3DDataModel


class Box3DVelocityNormClip(Box3DPipeline):
    """
    Pipeline to clip the ground plane velocity norm (ground plane speed in m/s)
    of the 3D bounding boxes.
    """

    def __init__(self, max_ground_plane_speed: float):
        """
        Initialize Box3DVelocityNormClip.
        Args:
          max_ground_plane_speed: Maximum ground plane speed to clip the velocity norm.
        """
        super().__init__()
        self.max_ground_plane_speed = max_ground_plane_speed

    def __str__(self) -> str:
        """
        String representation of the pipeline, used for logging.

        Returns:
          str: String representation of the pipeline.
        """
        return f"{self.__class__.__name__}(max_ground_plane_speed={self.max_ground_plane_speed})"

    def __call__(self, boxes3d_data_model: Sequence[Box3DDataModel]) -> Sequence[Box3DDataModel]:
        """
        Clip the ground plane velocity norm (ground plane speed in m/s) of the 3D bounding boxes.
        """
        if not len(boxes3d_data_model):
            return boxes3d_data_model

        ground_plane_velocities = np.asarray(
            [
                (
                    box3d.box3d_params[Box3DFieldIndex.VELOCITY_X],
                    box3d.box3d_params[Box3DFieldIndex.VELOCITY_Y],
                )
                for box3d in boxes3d_data_model
            ]
        )

        ground_plane_speeds = np.linalg.norm(ground_plane_velocities, axis=1)

        mask_indices = np.where(ground_plane_speeds > self.max_ground_plane_speed)[0]

        # Clip velocity_x and velocity_y by scaling them down to the max_ground_plane_speed
        # while keeping the direction the same
        ground_plane_velocities[mask_indices, 0] = ground_plane_velocities[mask_indices, 0] * (
            self.max_ground_plane_speed / ground_plane_speeds[mask_indices]
        )
        ground_plane_velocities[mask_indices, 1] = ground_plane_velocities[mask_indices, 1] * (
            self.max_ground_plane_speed / ground_plane_speeds[mask_indices]
        )

        # Assign velocity_x and velocity_y back to the boxes3d_data_model
        new_boxes3d_data_model = []
        for box3d_data_model, velocity in zip(boxes3d_data_model, ground_plane_velocities):
            new_box3d_params = box3d_data_model.box3d_params
            new_box3d_params[Box3DFieldIndex.VELOCITY_X] = velocity[0]
            new_box3d_params[Box3DFieldIndex.VELOCITY_Y] = velocity[1]

            new_box3d = box3d_data_model.create_new_data_model(
                box3d_params=new_box3d_params,
            )
            new_boxes3d_data_model.append(new_box3d)

        return new_boxes3d_data_model
