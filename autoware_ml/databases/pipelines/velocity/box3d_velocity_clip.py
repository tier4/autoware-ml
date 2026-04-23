import numpy as np

from autoware_ml.common.enums.enums import Box3DFieldIndex

from autoware_ml.databases.pipelines.box3d_pipeline import Boxes3DPipeline
from autoware_ml.databases.t4dataset.t4sample_records import Boxes3DMetadata


class Box3DVelocityNormClip(Boxes3DPipeline):
    """
    Pipeline to clip the velocity norm of the 3D bounding boxes.
    """

    def __init__(self, velocity_norm_threshold: float):
        super().__init__()
        self.velocity_norm_threshold = velocity_norm_threshold

    def __call__(self, boxes_3d_metadata: Boxes3DMetadata) -> Boxes3DMetadata:
        """
        Clip the velocity norm of the 3D bounding boxes.
        """
        bev_speeds = boxes_3d_metadata.get_bev_speeds()

        # Get the indices of the boxes whose velocity norm is greater than the threshold
        mask_indices = np.where(bev_speeds > self.velocity_norm_threshold)[0]
        # Assign the new
        velocity_x = boxes_3d_metadata.boxes_3d_arrays[:, Box3DFieldIndex.VELOCITY_X]
        velocity_y = boxes_3d_metadata.boxes_3d_arrays[:, Box3DFieldIndex.VELOCITY_Y]

        velocity_x[mask_indices] = velocity_x[mask_indices] * (
            self.velocity_norm_threshold / bev_speeds[mask_indices]
        )
        velocity_y[mask_indices] = velocity_y[mask_indices] * (
            self.velocity_norm_threshold / bev_speeds[mask_indices]
        )

        boxes_3d_arrays = boxes_3d_metadata.boxes_3d_arrays
        boxes_3d_arrays[:, Box3DFieldIndex.VELOCITY_X] = velocity_x
        boxes_3d_arrays[:, Box3DFieldIndex.VELOCITY_Y] = velocity_y

        return Boxes3DMetadata(
            boxes_3d_arrays=boxes_3d_arrays,
            boxes_3d_instance_ids=boxes_3d_metadata.boxes_3d_instance_ids,
            boxes_3d_dataset_label_names=boxes_3d_metadata.boxes_3d_dataset_label_names,
            boxes_3d_label_names=boxes_3d_metadata.boxes_3d_label_names,
            boxes_3d_label_indices=boxes_3d_metadata.boxes_3d_label_indices,
            boxes_3d_num_lidar_pointclouds=boxes_3d_metadata.boxes_3d_num_lidar_pointclouds,
            boxes_3d_num_radar_pointclouds=boxes_3d_metadata.boxes_3d_num_radar_pointclouds,
            boxes_3d_valid=boxes_3d_metadata.boxes_3d_valid,
            boxes_3d_attributes=boxes_3d_metadata.boxes_3d_attributes,
        )
