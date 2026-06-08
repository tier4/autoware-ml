from types import MappingProxyType

from autoware_ml.databases.box3d_pipelines.box3d_pipeline import Box3DPipeline
from autoware_ml.databases.schemas.box3d_datamodel import Boxes3DDataModel


class Box3DLabelRemapper(Box3DPipeline):
    """
    Pipeline to remap the label names of the 3D bounding boxes to another label name.
    """

    def __init__(self, label_remapper: MappingProxyType[str, str]):
        super().__init__()
        self.label_remapper = label_remapper

    def __str__(self) -> str:
        """
        String representation of the pipeline, used for logging.

        Returns:
          str: String representation of the pipeline.
        """
        return f"{self.__class__.__name__}(label_remapper={self.label_remapper})"

    def __call__(self, boxes3d_metadata: Boxes3DDataModel) -> Boxes3DDataModel:
        """
        Remap the label names of the 3D bounding boxes to another label name.
        """
        boxed_3d_label_names = []
        for box_3d_label_name in boxes3d_metadata.boxed_3d_label_names:
            if box_3d_label_name in self.label_remapper:
                boxed_3d_label_names.append(self.label_remapper[box_3d_label_name])
            else:
                boxed_3d_label_names.append(box_3d_label_name)

        return Boxes3DDataModel(
            boxes_3d_arrays=boxes3d_metadata.boxes_3d_arrays,
            boxed_3d_dataset_label_names=boxes3d_metadata.boxed_3d_dataset_label_names,
            boxed_3d_label_names=boxed_3d_label_names,
            boxed_3d_label_indices=boxes3d_metadata.boxes_3d_label_indices,
            boxes_3d_instance_ids=boxes3d_metadata.boxes_3d_instance_ids,
            boxes_3d_num_lidar_pointclouds=boxes3d_metadata.boxes_3d_num_lidar_pointclouds,
            boxed_3d_num_radar_pointclouds=boxes3d_metadata.boxed_3d_num_radar_pointclouds,
            boxes_3d_valid=boxes3d_metadata.boxes_3d_valid,
            boxes_3d_attributes=boxes3d_metadata.boxes_3d_attributes,
        )


class Box3DLabelIndicesRemapper(Box3DPipeline):
    """
    Pipeline to remap the label indices of the 3D bounding boxes to another label index.
    """

    def __init__(self, label_index_remapper: MappingProxyType[str, int]):
        """
        Initialize Box3DLabelIndicesRemapper.

        Args:
          label_index_remapper ({label_name: label_index}): Mapping of the label name to the label index.
        """

        super().__init__()
        self.label_index_remapper = label_index_remapper

    def __call__(self, boxes3d_metadata: Boxes3DDataModel) -> Boxes3DDataModel:
        """Inherits, check the super class."""

        boxed_3d_label_indices = []
        for box_3d_label_index in boxes3d_metadata.boxes_3d_label_indices:
            if box_3d_label_index in self.label_index_remapping:
                boxed_3d_label_indices.append(self.label_index_remapping[box_3d_label_index])
            else:
                boxed_3d_label_indices.append(box_3d_label_index)

        return Boxes3DDataModel(
            boxes_3d_arrays=boxes3d_metadata.boxes_3d_arrays,
            boxed_3d_dataset_label_names=boxes3d_metadata.boxes_3d_dataset_label_names,
            boxed_3d_label_names=boxes3d_metadata.boxes_3d_label_names,
            boxed_3d_label_indices=boxed_3d_label_indices,
            boxes_3d_instance_ids=boxes3d_metadata.boxes_3d_instance_ids,
            boxes_3d_num_lidar_pointclouds=boxes3d_metadata.boxes_3d_num_lidar_pointclouds,
            boxes_3d_num_radar_pointclouds=boxes3d_metadata.boxes_3d_num_radar_pointclouds,
            boxes_3d_valid=boxes3d_metadata.boxes_3d_valid,
            boxes_3d_attributes=boxes3d_metadata.boxes_3d_attributes,
        )
