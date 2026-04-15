from typing import MappingProxyType

from autoware_ml.databases.pipelines.box3d_pipeline import Boxes3DPipeline
from autoware_ml.databases.t4dataset.t4sample_records import Boxes3DMetadata


class Box3DLabelRemmaper(Boxes3DPipeline):
    """
    Pipeline to remap the label names of the 3D bounding boxes to another label name.
    """

    def __init__(self, label_remapping: MappingProxyType[str, str]):
        super().__init__()
        self.label_remapping = label_remapping

    def __call__(self, boxes_3d_metadata: Boxes3DMetadata) -> Boxes3DMetadata:
        """
        Remap the label names of the 3D bounding boxes to another label name.
        """
        boxed_3d_label_names = []
        for box_3d_label_name in boxes_3d_metadata.boxed_3d_label_names:
            if box_3d_label_name in self.label_remapping:
                boxed_3d_label_names.append(self.label_remapping[box_3d_label_name])
            else:
                boxed_3d_label_names.append(box_3d_label_name)

        return Boxes3DMetadata(
            boxes_3d_arrays=boxes_3d_metadata.boxes_3d_arrays,
            boxed_3d_dataset_label_names=boxes_3d_metadata.boxed_3d_dataset_label_names,
            boxed_3d_label_names=boxed_3d_label_names,
            boxed_3d_label_indices=boxes_3d_metadata.boxed_3d_label_indices,
            boxes_3d_instance_ids=boxes_3d_metadata.boxes_3d_instance_ids,
            boxes_3d_num_lidar_pointclouds=boxes_3d_metadata.boxes_3d_num_lidar_pointclouds,
            boxed_3d_num_radar_pointclouds=boxes_3d_metadata.boxed_3d_num_radar_pointclouds,
            boxes_3d_valid=boxes_3d_metadata.boxes_3d_valid,
            boxes_3d_attributes=boxes_3d_metadata.boxes_3d_attributes,
        )


class Box3DLabelIndicesRemapper(Boxes3DPipeline):
    """
    Pipeline to remap the label indices of the 3D bounding boxes to another label index.
    """

    def __init__(self, label_index_remapping: MappingProxyType[str, int]):
        """
        Initialize Box3DLabelIndicesRemapper.

        Args:
          label_index_remapping ({label_name: label_index}): Mapping of the label name to the label index.
        """

        super().__init__()
        self.label_index_remapping = label_index_remapping

    def __call__(self, boxes_3d_metadata: Boxes3DMetadata) -> Boxes3DMetadata:
        """Inherits, check the super class."""

        boxed_3d_label_indices = []
        for box_3d_label_index in boxes_3d_metadata.boxed_3d_label_indices:
            if box_3d_label_index in self.label_index_remapping:
                boxed_3d_label_indices.append(self.label_index_remapping[box_3d_label_index])
            else:
                boxed_3d_label_indices.append(box_3d_label_index)

        return Boxes3DMetadata(
            boxes_3d_arrays=boxes_3d_metadata.boxes_3d_arrays,
            boxed_3d_dataset_label_names=boxes_3d_metadata.boxed_3d_dataset_label_names,
            boxed_3d_label_names=boxes_3d_metadata.boxed_3d_label_names,
            boxed_3d_label_indices=boxed_3d_label_indices,
            boxes_3d_instance_ids=boxes_3d_metadata.boxes_3d_instance_ids,
            boxes_3d_num_lidar_pointclouds=boxes_3d_metadata.boxes_3d_num_lidar_pointclouds,
            boxed_3d_num_radar_pointclouds=boxes_3d_metadata.boxed_3d_num_radar_pointclouds,
            boxes_3d_valid=boxes_3d_metadata.boxes_3d_valid,
            boxes_3d_attributes=boxes_3d_metadata.boxes_3d_attributes,
        )
